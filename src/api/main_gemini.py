import os
import time
import json
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

app = FastAPI(title="Cervical Cell Classification API (Gemini Vision)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

CELL_DESCRIPTIONS = {
    'Dyskeratotic': 'Abnormal keratin production, often associated with HPV infection',
    'Koilocytotic': 'Cells showing HPV-related changes with perinuclear halos',
    'Metaplastic': 'Cells undergoing transformation, often benign',
    'Parabasal': 'Immature squamous cells from basal layers',
    'Superficial-Intermediate': 'Mature squamous cells from upper layers'
}

@app.get("/")
async def root():
    return {"message": "Cervical Cell Classification API (Gemini Vision)", "status": "running"}

@app.get("/model-info")
async def model_info():
    return {
        "model_loaded": True,
        "model_name": "Gemini 2.0 Flash Vision",
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "api_based": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        start_time = time.time()

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Prepare prompt for classification
        classification_prompt = f"""You are an expert pathologist specializing in cervical cytology. Analyze this microscopy image and classify the cells you observe into ONE of these five categories:

1. **Dyskeratotic**: Abnormal keratin production, often associated with HPV infection
2. **Koilocytotic**: Cells showing HPV-related changes with perinuclear halos
3. **Metaplastic**: Cells undergoing transformation, often benign
4. **Parabasal**: Immature squamous cells from basal layers
5. **Superficial-Intermediate**: Mature squamous cells from upper layers

IMPORTANT: You MUST respond with ONLY valid JSON in this exact format (no extra text before or after):
{{
  "classification": "<one of the 5 cell types exactly as written above>",
  "confidence": <number between 0-100>,
  "probabilities": {{
    "Dyskeratotic": <0-100>,
    "Koilocytotic": <0-100>,
    "Metaplastic": <0-100>,
    "Parabasal": <0-100>,
    "Superficial-Intermediate": <0-100>
  }},
  "reasoning": "<brief 2-3 sentence explanation>"
}}

Base your classification on visible features:
- Cell morphology and shape
- Nuclear characteristics (size, color, structure)
- Cytoplasmic features (color, texture)
- Presence of halos or abnormal keratinization
- Cell maturity indicators

Even if the image is unclear or you're uncertain, you MUST provide your best classification with probability estimates. The probabilities should sum to approximately 100."""

        # Use Gemini Vision to classify
        print("[INFO] Classifying image with Gemini Vision...")
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content([classification_prompt, image])

        # Parse response (now guaranteed to be JSON)
        response_text = response.text.strip()
        print(f"[DEBUG] Raw response: {response_text[:200]}...")

        classification_data = json.loads(response_text)

        predicted_class = classification_data['classification']
        confidence = float(classification_data['confidence']) / 100.0
        probabilities_percent = classification_data['probabilities']
        reasoning = classification_data.get('reasoning', '')

        # Convert probabilities to 0-1 scale
        probabilities = {k: float(v)/100.0 for k, v in probabilities_percent.items()}

        processing_time = time.time() - start_time
        print(f"[OK] Classification complete: {predicted_class} ({confidence*100:.1f}%)")

        # Generate detailed explanation
        print("[INFO] Generating detailed medical explanation...")
        explanation_prompt = f"""You are a medical AI assistant. Write a comprehensive medical explanation in MARKDOWN format (NOT JSON) for healthcare professionals.

Classification Result:
- Predicted Cell Type: {predicted_class}
- Confidence: {confidence*100:.1f}%
- Initial Reasoning: {reasoning}

All Probabilities:
{chr(10).join([f'- {cls}: {prob*100:.1f}%' for cls, prob in probabilities.items()])}

Write a markdown-formatted explanation with these sections:

## Cell Type Explanation
Explain what {predicted_class} cells are (2-3 sentences).

## Clinical Significance
What this finding indicates and its clinical importance.

## Model Confidence
Why the model is confident ({confidence*100:.1f}%) based on the probabilities and image features.

## Considerations & Recommendations
Important clinical points and recommendations for follow-up.

Keep it concise, professional, and actionable. Use proper markdown formatting with headers (##), bold (**text**), and lists. Max 200 words. DO NOT use JSON format - use plain markdown text."""

        # Create a new model instance without JSON mode for text explanation
        explanation_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        explanation_response = explanation_model.generate_content(explanation_prompt)
        ai_explanation = explanation_response.text.strip()

        # If Gemini still returns JSON despite instructions, convert it to readable text
        if ai_explanation.startswith('{'):
            try:
                json_explanation = json.loads(ai_explanation)
                # Extract and format the text content from JSON
                markdown_parts = []
                for key, value in json_explanation.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            markdown_parts.append(f"## {subkey.replace('_', ' ').title()}\n{subvalue}\n")
                    else:
                        markdown_parts.append(f"## {key.replace('_', ' ').title()}\n{value}\n")
                ai_explanation = '\n'.join(markdown_parts)
            except:
                pass  # If parsing fails, keep original text

        print(f"[OK] Explanation generated: {len(ai_explanation)} characters")

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "processing_time": f"{processing_time:.3f}s",
            "model_name": "Gemini 2.0 Flash Vision",
            "ai_explanation": ai_explanation
        }

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        print(f"Raw response: {response_text}")

        # Check if Gemini refused to classify (likely not a cervical cell image)
        if "unable to process" in response_text.lower() or "cannot analyze" in response_text.lower():
            return {
                "success": False,
                "error": "The uploaded image does not appear to be a cervical cell microscopy image. Please upload a valid microscopy image of cervical cells."
            }

        return {"success": False, "error": f"Failed to parse classification response. The AI model could not provide a classification for this image."}
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        error_msg = str(e)

        # Handle rate limiting
        if "429" in error_msg or "Resource exhausted" in error_msg:
            return {
                "success": False,
                "error": "Gemini API rate limit reached. Please wait a few moments and try again."
            }

        # Handle other API errors
        if "api" in error_msg.lower() or "quota" in error_msg.lower():
            return {
                "success": False,
                "error": f"API error: {error_msg}"
            }

        return {"success": False, "error": f"Prediction failed: {error_msg}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
