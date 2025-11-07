import { NextRequest, NextResponse } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'

const CLASSES = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File

    if (!file) {
      return NextResponse.json({ success: false, error: 'No file uploaded' }, { status: 400 })
    }

    const startTime = Date.now()

    // Convert file to base64
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)
    const base64 = buffer.toString('base64')

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!)

    // Classification prompt
    const classificationPrompt = `You are an expert pathologist specializing in cervical cytology. Analyze this microscopy image and classify the cells you observe into ONE of these five categories:

1. **Dyskeratotic**: Abnormal keratin production, often associated with HPV infection
2. **Koilocytotic**: Cells showing HPV-related changes with perinuclear halos
3. **Metaplastic**: Cells undergoing transformation, often benign
4. **Parabasal**: Immature squamous cells from basal layers
5. **Superficial-Intermediate**: Mature squamous cells from upper layers

IMPORTANT: You MUST respond with ONLY valid JSON in this exact format (no extra text before or after):
{
  "classification": "<one of the 5 cell types exactly as written above>",
  "confidence": <number between 0-100>,
  "probabilities": {
    "Dyskeratotic": <0-100>,
    "Koilocytotic": <0-100>,
    "Metaplastic": <0-100>,
    "Parabasal": <0-100>,
    "Superficial-Intermediate": <0-100>
  },
  "reasoning": "<brief 2-3 sentence explanation>"
}

Base your classification on visible features. Even if uncertain, you MUST provide your best classification. Probabilities should sum to approximately 100.`

    // Call Gemini Vision for classification
    const model = genAI.getGenerativeModel({
      model: 'gemini-2.0-flash-exp',
      generationConfig: {
        responseMimeType: 'application/json'
      }
    })

    const imagePart = {
      inlineData: {
        data: base64,
        mimeType: file.type
      }
    }

    const classificationResult = await model.generateContent([classificationPrompt, imagePart])
    const classificationText = classificationResult.response.text()
    const classificationData = JSON.parse(classificationText)

    const predictedClass = classificationData.classification
    const confidence = parseFloat(classificationData.confidence) / 100.0
    const probabilitiesPercent = classificationData.probabilities
    const reasoning = classificationData.reasoning || ''

    // Convert probabilities to 0-1 scale
    const probabilities: Record<string, number> = {}
    for (const [key, value] of Object.entries(probabilitiesPercent)) {
      probabilities[key] = parseFloat(value as string) / 100.0
    }

    const processingTime = (Date.now() - startTime) / 1000

    // Generate explanation
    const explanationPrompt = `You are a medical AI assistant. Write a comprehensive medical explanation in MARKDOWN format (NOT JSON) for healthcare professionals.

Classification Result:
- Predicted Cell Type: ${predictedClass}
- Confidence: ${(confidence * 100).toFixed(1)}%
- Initial Reasoning: ${reasoning}

All Probabilities:
${Object.entries(probabilities).map(([cls, prob]) => `- ${cls}: ${(prob * 100).toFixed(1)}%`).join('\n')}

Write a markdown-formatted explanation with these sections:

## Cell Type Explanation
Explain what ${predictedClass} cells are (2-3 sentences).

## Clinical Significance
What this finding indicates and its clinical importance.

## Model Confidence
Why the model is confident (${(confidence * 100).toFixed(1)}%) based on the probabilities and image features.

## Considerations & Recommendations
Important clinical points and recommendations for follow-up.

Keep it concise, professional, and actionable. Use proper markdown formatting with headers (##), bold (**text**), and lists. Max 200 words. DO NOT use JSON format - use plain markdown text.`

    const explanationModel = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-exp' })
    const explanationResult = await explanationModel.generateContent(explanationPrompt)
    let aiExplanation = explanationResult.response.text()

    // If Gemini returns JSON, convert to markdown
    if (aiExplanation.trim().startsWith('{')) {
      try {
        const jsonExplanation = JSON.parse(aiExplanation)
        const markdownParts: string[] = []
        for (const [key, value] of Object.entries(jsonExplanation)) {
          if (typeof value === 'object' && value !== null) {
            for (const [subkey, subvalue] of Object.entries(value)) {
              markdownParts.push(`## ${subkey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}\n${subvalue}\n`)
            }
          } else {
            markdownParts.push(`## ${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}\n${value}\n`)
          }
        }
        aiExplanation = markdownParts.join('\n')
      } catch {
        // Keep original if parsing fails
      }
    }

    return NextResponse.json({
      success: true,
      prediction: predictedClass,
      confidence: confidence,
      probabilities: probabilities,
      processing_time: `${processingTime.toFixed(3)}s`,
      model_name: 'Gemini 2.0 Flash Vision',
      ai_explanation: aiExplanation
    })

  } catch (error: any) {
    console.error('Prediction error:', error)

    // Handle rate limiting
    if (error.message?.includes('429') || error.message?.includes('Resource exhausted')) {
      return NextResponse.json({
        success: false,
        error: 'Gemini API rate limit reached. Please wait a few moments and try again.'
      })
    }

    return NextResponse.json({
      success: false,
      error: `Prediction failed: ${error.message || 'Unknown error'}`
    }, { status: 500 })
  }
}
