'use client'

import { useEffect, useState, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Activity, Home, Upload, CheckCircle, Sparkles, BarChart3 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

function ResultsContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [prediction, setPrediction] = useState<any>(null)

  // Function to convert JSON explanation to markdown
  const parseExplanation = (explanation: string): string => {
    try {
      // Check if explanation is JSON
      if (explanation.trim().startsWith('{')) {
        const jsonData = JSON.parse(explanation)
        let markdown = ''

        // Recursively extract text from JSON structure
        const extractText = (obj: any, level = 2) => {
          for (const [key, value] of Object.entries(obj)) {
            const title = key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim()
            markdown += `${'#'.repeat(level)} ${title}\n\n`

            if (typeof value === 'object' && value !== null) {
              extractText(value, level + 1)
            } else {
              markdown += `${value}\n\n`
            }
          }
        }

        extractText(jsonData)
        return markdown
      }
    } catch (e) {
      // Not JSON, return as is
    }
    return explanation
  }

  useEffect(() => {
    // Try to get data from sessionStorage
    const storedData = sessionStorage.getItem('predictionData')
    if (storedData) {
      try {
        const parsedData = JSON.parse(storedData)
        console.log('Received prediction data:', parsedData)
        console.log('AI Explanation:', parsedData.ai_explanation)

        // Parse JSON explanation if needed
        if (parsedData.ai_explanation) {
          parsedData.ai_explanation = parseExplanation(parsedData.ai_explanation)
        }

        setPrediction(parsedData)
        // Clear the data after loading
        sessionStorage.removeItem('predictionData')
      } catch (e) {
        console.error('Failed to parse prediction data:', e)
      }
    }
  }, [])

  const cellTypes = [
    { name: 'Dyskeratotic', color: '#ef4444', desc: 'Abnormal keratin production, often associated with HPV infection' },
    { name: 'Koilocytotic', color: '#f97316', desc: 'Cells showing HPV-related changes with perinuclear halos' },
    { name: 'Metaplastic', color: '#eab308', desc: 'Cells undergoing transformation, often benign' },
    { name: 'Parabasal', color: '#22c55e', desc: 'Immature squamous cells from basal layers' },
    { name: 'Superficial-Intermediate', color: '#3b82f6', desc: 'Mature squamous cells from upper layers' }
  ]

  if (!prediction) {
    return (
      <div style={{ minHeight: '100vh', background: '#f5f7fa', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <Activity style={{ width: '3rem', height: '3rem', color: '#9ca3af', margin: '0 auto 1rem' }} />
          <p style={{ color: '#6b7280', fontSize: '1.125rem' }}>Loading results...</p>
        </div>
      </div>
    )
  }

  const predictedCellType = cellTypes.find(c => c.name === prediction.prediction)

  return (
    <div style={{ minHeight: '100vh', background: '#f5f7fa' }}>
      {/* Header */}
      <header style={{
        background: 'white',
        borderBottom: '1px solid #e5e7eb',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
      }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.25rem 2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <Activity style={{ width: '2.5rem', height: '2.5rem', color: '#4f46e5' }} />
              <div>
                <h1 style={{ fontSize: '1.75rem', fontWeight: '700', color: '#1f2937', margin: 0 }}>
                  Cervical Cell Classifier
                </h1>
                <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: 0 }}>
                  AI-Powered Medical Image Analysis
                </p>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                onClick={() => router.push('/upload')}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: 'white',
                  color: '#4f46e5',
                  borderRadius: '0.5rem',
                  fontWeight: '600',
                  border: '1px solid #4f46e5',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#4f46e5'
                  e.currentTarget.style.color = 'white'
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'white'
                  e.currentTarget.style.color = '#4f46e5'
                }}
              >
                <Upload style={{ width: '1.25rem', height: '1.25rem' }} />
                New Analysis
              </button>
              <button
                onClick={() => router.push('/')}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: 'white',
                  color: '#6b7280',
                  borderRadius: '0.5rem',
                  fontWeight: '600',
                  border: '1px solid #e5e7eb',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.background = '#f9fafb'}
                onMouseOut={(e) => e.currentTarget.style.background = 'white'}
              >
                <Home style={{ width: '1.25rem', height: '1.25rem' }} />
                Home
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Progress Steps */}
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        padding: '2rem 2rem 0'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '1rem',
          marginBottom: '1rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '0.75rem 1.25rem',
            background: 'white',
            borderRadius: '0.5rem',
            border: '1px solid #e5e7eb'
          }}>
            <div style={{
              width: '1.75rem',
              height: '1.75rem',
              borderRadius: '50%',
              background: '#22c55e',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: '700',
              fontSize: '0.875rem'
            }}>✓</div>
            <span style={{ fontWeight: '600', color: '#6b7280' }}>Upload Image</span>
          </div>

          <div style={{
            width: '3rem',
            height: '2px',
            background: '#22c55e'
          }}></div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '0.75rem 1.25rem',
            background: '#4f46e5',
            borderRadius: '0.5rem',
            color: 'white'
          }}>
            <div style={{
              width: '1.75rem',
              height: '1.75rem',
              borderRadius: '50%',
              background: 'white',
              color: '#4f46e5',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: '700',
              fontSize: '0.875rem'
            }}>2</div>
            <span style={{ fontWeight: '600' }}>View Results</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '1rem 2rem' }}>
        {/* Success Banner */}
        <div style={{
          background: 'white',
          borderRadius: '0.75rem',
          padding: '1rem 1.5rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          border: '1px solid #e5e7eb',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem'
        }}>
          <div style={{
            width: '3rem',
            height: '3rem',
            borderRadius: '0.5rem',
            background: '#d1fae5',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0
          }}>
            <CheckCircle style={{ width: '1.5rem', height: '1.5rem', color: '#22c55e' }} />
          </div>
          <div style={{ flex: 1 }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#1f2937', marginBottom: '0.25rem' }}>
              Analysis Complete
            </h2>
            <p style={{ fontSize: '0.8125rem', color: '#6b7280', margin: 0 }}>
              Review the classification results below.
            </p>
          </div>
          {prediction.processing_time && (
            <div style={{
              padding: '0.5rem 1rem',
              background: '#f9fafb',
              borderRadius: '0.375rem',
              textAlign: 'center',
              border: '1px solid #e5e7eb'
            }}>
              <p style={{ fontSize: '0.625rem', color: '#6b7280', margin: '0 0 0.125rem 0', textTransform: 'uppercase', fontWeight: '600' }}>
                Time
              </p>
              <p style={{ fontSize: '0.875rem', fontWeight: '700', color: '#1f2937', margin: 0 }}>
                {prediction.processing_time}
              </p>
            </div>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1rem' }}>
          {/* Left Column - Main Result */}
          <div>
            {/* Main Prediction Card */}
            <div style={{
              background: predictedCellType?.color || '#4f46e5',
              borderRadius: '1rem',
              padding: '2rem',
              color: 'white',
              marginBottom: '1rem',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', marginBottom: '0.75rem' }}>
                <CheckCircle style={{ width: '1.125rem', height: '1.125rem' }} />
                <p style={{ fontSize: '0.75rem', fontWeight: '600', opacity: 0.95, margin: 0, textTransform: 'uppercase' }}>
                  Predicted Classification
                </p>
              </div>
              <h3 style={{ fontSize: '2rem', fontWeight: '700', marginBottom: '1rem', lineHeight: 1.2 }}>
                {prediction.prediction}
              </h3>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '3rem', fontWeight: '700', lineHeight: 1 }}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
                <span style={{ fontSize: '1rem', opacity: 0.9 }}>confidence</span>
              </div>
              <p style={{ fontSize: '0.875rem', opacity: 0.9, lineHeight: 1.5, margin: 0 }}>
                {predictedCellType?.desc}
              </p>
            </div>

            {/* Cell Info Card */}
            <div style={{
              background: 'white',
              borderRadius: '1rem',
              padding: '1.25rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e7eb'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Activity style={{ width: '1rem', height: '1rem', color: '#4f46e5' }} />
                <h3 style={{ fontSize: '1rem', fontWeight: '700', color: '#1f2937', margin: 0 }}>
                  Cell Type Information
                </h3>
              </div>
              <div style={{ fontSize: '0.8125rem', color: '#6b7280', lineHeight: 1.5 }}>
                <p style={{ margin: '0 0 0.625rem 0' }}>
                  <strong style={{ color: '#1f2937' }}>Classification:</strong> {prediction.prediction}
                </p>
                <p style={{ margin: 0 }}>
                  <strong style={{ color: '#1f2937' }}>Description:</strong> {predictedCellType?.desc}
                </p>
              </div>
            </div>
          </div>

          {/* Right Column - Detailed Analysis */}
          <div>
            {/* All Probabilities */}
            <div style={{
              background: 'white',
              borderRadius: '1rem',
              padding: '1.25rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e7eb',
              marginBottom: '1rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <BarChart3 style={{ width: '1rem', height: '1rem', color: '#4f46e5' }} />
                <h3 style={{ fontSize: '1rem', fontWeight: '700', color: '#1f2937', margin: 0 }}>
                  All Classification Probabilities
                </h3>
              </div>
              <div>
                {prediction.probabilities && Object.entries(prediction.probabilities as Record<string, number>)
                  .sort((a, b) => b[1] - a[1])
                  .map(([className, prob]) => {
                    const percentage = (prob * 100).toFixed(1)
                    const cellType = cellTypes.find(c => c.name === className)
                    const isTop = className === prediction.prediction

                    return (
                      <div key={className} style={{
                        padding: '0.75rem',
                        borderRadius: '0.375rem',
                        background: isTop ? '#eef2ff' : '#f9fafb',
                        border: isTop ? '1px solid #c7d2fe' : '1px solid #e5e7eb',
                        marginBottom: '0.5rem'
                      }}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          marginBottom: '0.5rem'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <div style={{
                              width: '0.75rem',
                              height: '0.75rem',
                              borderRadius: '0.1875rem',
                              background: cellType?.color,
                              flexShrink: 0
                            }}></div>
                            <span style={{
                              fontWeight: '600',
                              fontSize: '0.8125rem',
                              color: isTop ? '#4f46e5' : '#374151'
                            }}>
                              {className}
                            </span>
                          </div>
                          <span style={{
                            fontWeight: '700',
                            fontSize: '0.875rem',
                            color: isTop ? '#4f46e5' : '#6b7280'
                          }}>
                            {percentage}%
                          </span>
                        </div>
                        <div style={{
                          width: '100%',
                          background: '#e5e7eb',
                          borderRadius: '9999px',
                          height: '0.375rem',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            height: '100%',
                            background: cellType?.color || '#6b7280',
                            borderRadius: '9999px',
                            width: `${percentage}%`,
                            transition: 'width 0.5s ease-out'
                          }}></div>
                        </div>
                      </div>
                    )
                  })}
              </div>
            </div>

            {/* AI Explanation */}
            <div style={{
              background: 'white',
              borderRadius: '1rem',
              padding: '1.25rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e7eb'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Sparkles style={{ width: '1rem', height: '1rem', color: '#4f46e5' }} />
                <h3 style={{ fontSize: '1rem', fontWeight: '700', color: '#1f2937', margin: 0 }}>
                  AI Medical Explanation
                </h3>
              </div>
              <div style={{
                fontSize: '0.8125rem',
                color: '#374151',
                lineHeight: 1.6
              }}
              className="markdown-content">
                {prediction.ai_explanation ? (
                  <ReactMarkdown>{prediction.ai_explanation}</ReactMarkdown>
                ) : (
                  'AI explanation is loading or unavailable. Please check the API logs.'
                )}
              </div>
            </div>

            <style jsx global>{`
              .markdown-content h1,
              .markdown-content h2,
              .markdown-content h3 {
                font-weight: 700;
                color: #1f2937;
                margin-top: 1.5rem;
                margin-bottom: 0.75rem;
              }
              .markdown-content h1 {
                font-size: 1.5rem;
              }
              .markdown-content h2 {
                font-size: 1.25rem;
              }
              .markdown-content h3 {
                font-size: 1.125rem;
              }
              .markdown-content p {
                margin-bottom: 1rem;
              }
              .markdown-content ul,
              .markdown-content ol {
                margin-left: 1.5rem;
                margin-bottom: 1rem;
              }
              .markdown-content li {
                margin-bottom: 0.5rem;
              }
              .markdown-content strong {
                font-weight: 600;
                color: #1f2937;
              }
              .markdown-content code {
                background: #f3f4f6;
                padding: 0.125rem 0.375rem;
                border-radius: 0.25rem;
                font-family: monospace;
                font-size: 0.875em;
              }
            `}</style>
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{
          marginTop: '1rem',
          display: 'flex',
          justifyContent: 'center',
          gap: '0.75rem'
        }}>
          <button
            onClick={() => router.push('/upload')}
            style={{
              padding: '0.75rem 1.5rem',
              background: '#4f46e5',
              color: 'white',
              borderRadius: '0.5rem',
              fontWeight: '600',
              fontSize: '0.875rem',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'background 0.2s'
            }}
            onMouseOver={(e) => e.currentTarget.style.background = '#4338ca'}
            onMouseOut={(e) => e.currentTarget.style.background = '#4f46e5'}
          >
            <Upload style={{ width: '1rem', height: '1rem' }} />
            Analyze Another Image
          </button>
          <button
            onClick={() => router.push('/')}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'white',
              color: '#6b7280',
              borderRadius: '0.5rem',
              fontWeight: '600',
              fontSize: '0.875rem',
              border: '1px solid #e5e7eb',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'background 0.2s'
            }}
            onMouseOver={(e) => e.currentTarget.style.background = '#f9fafb'}
            onMouseOut={(e) => e.currentTarget.style.background = 'white'}
          >
            <Home style={{ width: '1rem', height: '1rem' }} />
            Back to Home
          </button>
        </div>
      </main>

    </div>
  )
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div style={{ minHeight: '100vh', background: '#f5f7fa', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <Activity style={{ width: '3rem', height: '3rem', color: '#9ca3af', margin: '0 auto 1rem' }} />
          <p style={{ color: '#6b7280', fontSize: '1.125rem' }}>Loading results...</p>
        </div>
      </div>
    }>
      <ResultsContent />
    </Suspense>
  )
}
