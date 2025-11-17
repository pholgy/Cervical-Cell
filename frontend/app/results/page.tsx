'use client'

import { useEffect, useState, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Activity, Home, Upload, CheckCircle, Sparkles, BarChart3 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

function ResultsContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [prediction, setPrediction] = useState<any>(null)

  const styles = `
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

    * { font-family: 'DM Sans', sans-serif; }
    .display-font { font-family: 'Outfit', sans-serif; }

    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideInRight {
      from { opacity: 0; transform: translateX(-30px); }
      to { opacity: 1; transform: translateX(0); }
    }

    @keyframes glow-pulse {
      0%, 100% { box-shadow: 0 0 20px rgba(219, 39, 119, 0.3); }
      50% { box-shadow: 0 0 40px rgba(219, 39, 119, 0.5); }
    }

    .card-lift { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
    .card-lift:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(219, 39, 119, 0.12); }

    .gradient-text {
      background: linear-gradient(135deg, #db2777 0%, #be185d 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .animate-fade-in { animation: fadeInUp 0.6s ease-out; }
  `

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
      <div style={{ minHeight: '100vh', background: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <style>{styles}</style>
        <div style={{ textAlign: 'center' }}>
          <Activity style={{ width: '3rem', height: '3rem', color: '#db2777', margin: '0 auto 1rem' }} />
          <p style={{ color: '#6b7280', fontSize: '1.125rem', fontFamily: "'DM Sans', sans-serif" }}>Loading results...</p>
        </div>
      </div>
    )
  }

  const predictedCellType = cellTypes.find(c => c.name === prediction.prediction)

  return (
    <div style={{ minHeight: '100vh', background: '#ffffff', overflowX: 'hidden' }}>
      <style>{styles}</style>

      {/* Header */}
      <header style={{
        background: 'rgba(255, 255, 255, 0.7)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(219, 39, 119, 0.1)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        boxShadow: '0 1px 20px rgba(0,0,0,0.03)'
      }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.25rem 2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{
                width: '2.5rem',
                height: '2.5rem',
                borderRadius: '0.75rem',
                background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Activity style={{ width: '1.25rem', height: '1.25rem', color: 'white' }} />
              </div>
              <div>
                <h1 style={{ fontSize: '1.375rem', fontWeight: '700', color: '#1f2937', margin: '0.25rem 0 0 0', letterSpacing: '-0.5px' }} className="display-font">
                  Peakily
                </h1>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                onClick={() => router.push('/upload')}
                style={{
                  padding: '0.75rem 1.75rem',
                  background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
                  color: 'white',
                  borderRadius: '0.75rem',
                  fontWeight: '600',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  fontSize: '0.9375rem',
                  boxShadow: '0 4px 15px rgba(219, 39, 119, 0.3)',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)'
                  e.currentTarget.style.boxShadow = '0 8px 25px rgba(219, 39, 119, 0.4)'
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)'
                  e.currentTarget.style.boxShadow = '0 4px 15px rgba(219, 39, 119, 0.3)'
                }}
              >
                <Upload style={{ width: '1rem', height: '1rem' }} />
                New Analysis
              </button>
              <button
                onClick={() => router.push('/')}
                style={{
                  padding: '0.75rem 1.75rem',
                  background: 'white',
                  color: '#6b7280',
                  borderRadius: '0.75rem',
                  fontWeight: '600',
                  border: '1px solid rgba(219, 39, 119, 0.2)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  fontSize: '0.9375rem',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#fdf2f8'
                  e.currentTarget.style.color = '#db2777'
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'white'
                  e.currentTarget.style.color = '#6b7280'
                }}
              >
                <Home style={{ width: '1rem', height: '1rem' }} />
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
        padding: '3rem 2rem 2rem'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '1.5rem',
          marginBottom: '2rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '0.875rem 1.5rem',
            background: 'white',
            borderRadius: '0.875rem',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            boxShadow: '0 2px 8px rgba(34, 197, 94, 0.08)'
          }}>
            <div style={{
              width: '2rem',
              height: '2rem',
              borderRadius: '50%',
              background: '#22c55e',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: '700',
              fontSize: '0.9375rem'
            }}>✓</div>
            <span style={{ fontWeight: '600', color: '#16a34a', fontSize: '0.9375rem' }}>Upload</span>
          </div>

          <div style={{
            width: '4rem',
            height: '2px',
            background: 'linear-gradient(90deg, #22c55e 0%, #db2777 100%)'
          }}></div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '0.875rem 1.5rem',
            background: 'linear-gradient(135deg, rgba(219, 39, 119, 0.1) 0%, rgba(219, 39, 119, 0.05) 100%)',
            borderRadius: '0.875rem',
            border: '1px solid rgba(219, 39, 119, 0.2)',
            boxShadow: '0 2px 8px rgba(219, 39, 119, 0.08)'
          }}>
            <div style={{
              width: '2rem',
              height: '2rem',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: '700',
              fontSize: '0.9375rem'
            }}>2</div>
            <span style={{ fontWeight: '600', color: '#db2777', fontSize: '0.9375rem' }}>Results</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '2rem' }}>
        {/* Success Banner */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.03) 100%)',
          borderRadius: '1.25rem',
          padding: '1.5rem 2rem',
          boxShadow: '0 2px 12px rgba(34, 197, 94, 0.08)',
          border: '1px solid rgba(34, 197, 94, 0.2)',
          marginBottom: '2rem',
          display: 'flex',
          alignItems: 'center',
          gap: '1.5rem'
        }}>
          <div style={{
            width: '3.5rem',
            height: '3.5rem',
            borderRadius: '1rem',
            background: '#dcfce7',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0
          }}>
            <CheckCircle style={{ width: '1.75rem', height: '1.75rem', color: '#16a34a' }} />
          </div>
          <div style={{ flex: 1 }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#1f2937', marginBottom: '0.375rem', margin: 0 }} className="display-font">
              Analysis Complete
            </h2>
            <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: 0 }}>
              Your cervical cell classification has been processed with high confidence
            </p>
          </div>
          {prediction.processing_time && (
            <div style={{
              padding: '0.75rem 1.25rem',
              background: 'white',
              borderRadius: '0.75rem',
              textAlign: 'center',
              border: '1px solid rgba(34, 197, 94, 0.2)',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
            }}>
              <p style={{ fontSize: '0.625rem', color: '#6b7280', margin: '0 0 0.25rem 0', textTransform: 'uppercase', fontWeight: '600' }}>
                Processing
              </p>
              <p style={{ fontSize: '0.95rem', fontWeight: '700', color: '#16a34a', margin: 0 }}>
                {prediction.processing_time}
              </p>
            </div>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '2rem' }}>
          {/* Left Column - Main Result */}
          <div>
            {/* Main Prediction Card */}
            <div style={{
              background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
              borderRadius: '1.5rem',
              padding: '2.5rem 2rem',
              color: 'white',
              marginBottom: '1.5rem',
              boxShadow: '0 12px 32px rgba(219, 39, 119, 0.2)',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{
                position: 'absolute',
                top: '-50%',
                right: '-10%',
                width: '300px',
                height: '300px',
                background: 'rgba(255, 255, 255, 0.08)',
                borderRadius: '50%',
                pointerEvents: 'none'
              }} />

              <div style={{ position: 'relative', zIndex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                  <CheckCircle style={{ width: '1.25rem', height: '1.25rem' }} />
                  <p style={{ fontSize: '0.8125rem', fontWeight: '600', opacity: 0.95, margin: 0, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Predicted Classification
                  </p>
                </div>
                <h3 style={{ fontSize: '2.25rem', fontWeight: '700', marginBottom: '1.5rem', lineHeight: 1.2, letterSpacing: '-0.5px' }} className="display-font">
                  {prediction.prediction}
                </h3>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginBottom: '1.5rem' }}>
                  <span style={{ fontSize: '3.5rem', fontWeight: '800', lineHeight: 1 }} className="display-font">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                  <span style={{ fontSize: '1.125rem', opacity: 0.95, fontWeight: '500' }}>confidence</span>
                </div>
                <p style={{ fontSize: '0.95rem', opacity: 0.95, lineHeight: 1.6, margin: 0 }}>
                  {predictedCellType?.desc}
                </p>
              </div>
            </div>

            {/* Cancer Risk Card */}
            {prediction.cancer_risk_level && (
              <div style={{
                background: prediction.cancer_risk_level === 'HIGH' ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.03) 100%)' : prediction.cancer_risk_level === 'MODERATE' ? 'linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.03) 100%)' : 'linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.03) 100%)',
                borderRadius: '1.25rem',
                padding: '1.75rem 1.5rem',
                boxShadow: `0 4px 12px ${prediction.cancer_risk_level === 'HIGH' ? 'rgba(239, 68, 68, 0.08)' : prediction.cancer_risk_level === 'MODERATE' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(34, 197, 94, 0.08)'}`,
                border: `1px solid ${prediction.cancer_risk_level === 'HIGH' ? 'rgba(239, 68, 68, 0.2)' : prediction.cancer_risk_level === 'MODERATE' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(34, 197, 94, 0.2)'}`,
                marginBottom: '1.5rem',
                position: 'relative'
              }} className="card-lift">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
                  <div style={{
                    width: '2.5rem',
                    height: '2.5rem',
                    borderRadius: '0.75rem',
                    background: prediction.cancer_risk_level === 'HIGH' ? 'rgba(239, 68, 68, 0.1)' : prediction.cancer_risk_level === 'MODERATE' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(34, 197, 94, 0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '1.25rem'
                  }}>
                    {prediction.cancer_risk_level === 'HIGH' ? '⚠' : prediction.cancer_risk_level === 'MODERATE' ? '!' : '✓'}
                  </div>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#1f2937', margin: 0 }} className="display-font">
                    Cancer Risk Level
                  </h3>
                </div>
                <div style={{ fontSize: '0.875rem', color: '#374151', lineHeight: 1.6 }}>
                  <p style={{ margin: '0 0 0.75rem 0' }}>
                    <strong style={{ color: '#1f2937', display: 'block', marginBottom: '0.25rem' }}>Risk Assessment:</strong>
                    <span style={{
                      fontSize: '1.5rem',
                      fontWeight: '800',
                      color: prediction.cancer_risk_level === 'HIGH' ? '#dc2626' : prediction.cancer_risk_level === 'MODERATE' ? '#d97706' : '#16a34a',
                      display: 'block'
                    }} className="display-font">
                      {prediction.cancer_risk_level}
                    </span>
                  </p>
                  <p style={{ margin: '1rem 0 0.75rem 0' }}>
                    <strong style={{ color: '#1f2937' }}>Risk Score:</strong> {prediction.cancer_risk_percentage}%
                  </p>
                  <p style={{ margin: 0, fontSize: '0.8125rem', color: '#6b7280', lineHeight: 1.5 }}>
                    {prediction.cancer_risk_justification}
                  </p>
                </div>
              </div>
            )}

            {/* Cell Info Card */}
            <div style={{
              background: 'white',
              borderRadius: '1.25rem',
              padding: '1.75rem 1.5rem',
              boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
              border: '1px solid rgba(219, 39, 119, 0.1)'
            }} className="card-lift">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
                <div style={{
                  width: '2.5rem',
                  height: '2.5rem',
                  borderRadius: '0.75rem',
                  background: 'rgba(219, 39, 119, 0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Activity style={{ width: '1.25rem', height: '1.25rem', color: '#db2777' }} />
                </div>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#1f2937', margin: 0 }} className="display-font">
                  Classification Details
                </h3>
              </div>
              <div style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: 1.6 }}>
                <p style={{ margin: '0 0 1rem 0' }}>
                  <strong style={{ color: '#1f2937', display: 'block', marginBottom: '0.25rem', fontSize: '0.8125rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Cell Type</strong>
                  <span style={{ color: '#db2777', fontWeight: '600', fontSize: '1rem' }}>{prediction.prediction}</span>
                </p>
                <p style={{ margin: 0 }}>
                  <strong style={{ color: '#1f2937', display: 'block', marginBottom: '0.25rem', fontSize: '0.8125rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Description</strong>
                  <span>{predictedCellType?.desc}</span>
                </p>
              </div>
            </div>
          </div>

          {/* Right Column - Detailed Analysis */}
          <div>
            {/* All Probabilities */}
            <div style={{
              background: 'white',
              borderRadius: '1.25rem',
              padding: '1.75rem 1.5rem',
              boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
              border: '1px solid rgba(219, 39, 119, 0.1)',
              marginBottom: '1.5rem'
            }} className="card-lift">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                <div style={{
                  width: '2.5rem',
                  height: '2.5rem',
                  borderRadius: '0.75rem',
                  background: 'rgba(219, 39, 119, 0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <BarChart3 style={{ width: '1.25rem', height: '1.25rem', color: '#db2777' }} />
                </div>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#1f2937', margin: 0 }} className="display-font">
                  Probability Distribution
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
                        padding: '0.875rem',
                        borderRadius: '0.75rem',
                        background: isTop ? 'linear-gradient(135deg, rgba(219, 39, 119, 0.1) 0%, rgba(219, 39, 119, 0.05) 100%)' : '#f9fafb',
                        border: isTop ? '1px solid rgba(219, 39, 119, 0.2)' : '1px solid rgba(0,0,0,0.04)',
                        marginBottom: '0.75rem'
                      }}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          marginBottom: '0.625rem'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
                            <div style={{
                              width: '0.75rem',
                              height: '0.75rem',
                              borderRadius: '0.25rem',
                              background: cellType?.color,
                              flexShrink: 0
                            }}></div>
                            <span style={{
                              fontWeight: '600',
                              fontSize: '0.8125rem',
                              color: isTop ? '#db2777' : '#374151'
                            }}>
                              {className}
                            </span>
                          </div>
                          <span style={{
                            fontWeight: '700',
                            fontSize: '0.9rem',
                            color: isTop ? '#db2777' : '#6b7280'
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
                            transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
                          }}></div>
                        </div>
                      </div>
                    )
                  })}
              </div>
            </div>

            {/* Model Performance Metrics */}
            {prediction.model_metrics && (
              <div style={{
                background: 'white',
                borderRadius: '1.25rem',
                padding: '1.75rem 1.5rem',
                boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
                border: '1px solid rgba(219, 39, 119, 0.1)',
                marginBottom: '1.5rem'
              }} className="card-lift">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                  <div style={{
                    width: '2.5rem',
                    height: '2.5rem',
                    borderRadius: '0.75rem',
                    background: 'rgba(219, 39, 119, 0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <BarChart3 style={{ width: '1.25rem', height: '1.25rem', color: '#db2777' }} />
                  </div>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#1f2937', margin: 0 }} className="display-font">
                    Model Performance
                  </h3>
                </div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: '1rem'
                }}>
                  <div style={{
                    padding: '1rem 1.125rem',
                    background: 'linear-gradient(135deg, rgba(219, 39, 119, 0.08) 0%, rgba(219, 39, 119, 0.03) 100%)',
                    borderRadius: '0.875rem',
                    border: '1px solid rgba(219, 39, 119, 0.15)',
                    textAlign: 'center'
                  }}>
                    <p style={{ fontSize: '0.6875rem', color: '#6b7280', margin: '0 0 0.5rem 0', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Accuracy
                    </p>
                    <p style={{ fontSize: '1.75rem', fontWeight: '800', color: '#db2777', margin: 0 }} className="display-font">
                      {prediction.model_metrics.accuracy_percentage}%
                    </p>
                  </div>
                  <div style={{
                    padding: '1rem 1.125rem',
                    background: 'linear-gradient(135deg, rgba(219, 39, 119, 0.08) 0%, rgba(219, 39, 119, 0.03) 100%)',
                    borderRadius: '0.875rem',
                    border: '1px solid rgba(219, 39, 119, 0.15)',
                    textAlign: 'center'
                  }}>
                    <p style={{ fontSize: '0.6875rem', color: '#6b7280', margin: '0 0 0.5rem 0', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Sensitivity
                    </p>
                    <p style={{ fontSize: '1.75rem', fontWeight: '800', color: '#db2777', margin: 0 }} className="display-font">
                      {prediction.model_metrics.sensitivity_percentage}%
                    </p>
                  </div>
                  <div style={{
                    padding: '1rem 1.125rem',
                    background: 'linear-gradient(135deg, rgba(219, 39, 119, 0.08) 0%, rgba(219, 39, 119, 0.03) 100%)',
                    borderRadius: '0.875rem',
                    border: '1px solid rgba(219, 39, 119, 0.15)',
                    textAlign: 'center'
                  }}>
                    <p style={{ fontSize: '0.6875rem', color: '#6b7280', margin: '0 0 0.5rem 0', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Specificity
                    </p>
                    <p style={{ fontSize: '1.75rem', fontWeight: '800', color: '#db2777', margin: 0 }} className="display-font">
                      {prediction.model_metrics.specificity_percentage}%
                    </p>
                  </div>
                  <div style={{
                    padding: '1rem 1.125rem',
                    background: 'linear-gradient(135deg, rgba(219, 39, 119, 0.08) 0%, rgba(219, 39, 119, 0.03) 100%)',
                    borderRadius: '0.875rem',
                    border: '1px solid rgba(219, 39, 119, 0.15)',
                    textAlign: 'center'
                  }}>
                    <p style={{ fontSize: '0.6875rem', color: '#6b7280', margin: '0 0 0.5rem 0', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Precision
                    </p>
                    <p style={{ fontSize: '1.75rem', fontWeight: '800', color: '#db2777', margin: 0 }} className="display-font">
                      {prediction.model_metrics.precision_percentage}%
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* AI Explanation */}
            <div style={{
              background: 'white',
              borderRadius: '1.25rem',
              padding: '1.75rem 1.5rem',
              boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
              border: '1px solid rgba(219, 39, 119, 0.1)'
            }} className="card-lift">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                <div style={{
                  width: '2.5rem',
                  height: '2.5rem',
                  borderRadius: '0.75rem',
                  background: 'rgba(219, 39, 119, 0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Sparkles style={{ width: '1.25rem', height: '1.25rem', color: '#db2777' }} />
                </div>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#1f2937', margin: 0 }} className="display-font">
                  Medical Analysis
                </h3>
              </div>
              <div style={{
                fontSize: '0.875rem',
                color: '#374151',
                lineHeight: 1.7
              }}
              className="markdown-content">
                {prediction.ai_explanation ? (
                  <ReactMarkdown>{prediction.ai_explanation}</ReactMarkdown>
                ) : (
                  <p style={{ color: '#6b7280', fontStyle: 'italic' }}>AI explanation is loading or unavailable. Please check the API logs.</p>
                )}
              </div>
            </div>

            <style jsx global>{`
              .markdown-content h1,
              .markdown-content h2,
              .markdown-content h3 {
                font-weight: 700;
                color: #1f2937;
                margin-top: 1.25rem;
                margin-bottom: 0.75rem;
                font-family: 'Outfit', sans-serif;
                letter-spacing: -0.5px;
              }
              .markdown-content h1 {
                font-size: 1.375rem;
              }
              .markdown-content h2 {
                font-size: 1.125rem;
              }
              .markdown-content h3 {
                font-size: 1rem;
              }
              .markdown-content p {
                margin-bottom: 0.875rem;
              }
              .markdown-content ul,
              .markdown-content ol {
                margin-left: 1.5rem;
                margin-bottom: 0.875rem;
                padding: 0;
              }
              .markdown-content li {
                margin-bottom: 0.375rem;
              }
              .markdown-content strong {
                font-weight: 700;
                color: #db2777;
              }
              .markdown-content code {
                background: #fdf2f8;
                color: #db2777;
                padding: 0.125rem 0.375rem;
                border-radius: 0.25rem;
                font-family: 'Courier New', monospace;
                font-size: 0.875em;
                font-weight: 500;
              }
            `}</style>
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{
          marginTop: '3rem',
          paddingTop: '2rem',
          borderTop: '1px solid rgba(219, 39, 119, 0.1)',
          display: 'flex',
          justifyContent: 'center',
          gap: '1rem'
        }}>
          <button
            onClick={() => router.push('/upload')}
            style={{
              padding: '1rem 2.25rem',
              background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
              color: 'white',
              borderRadius: '0.875rem',
              fontWeight: '700',
              fontSize: '1rem',
              border: 'none',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.625rem',
              boxShadow: '0 4px 15px rgba(219, 39, 119, 0.3)',
              transition: 'all 0.3s ease',
              letterSpacing: '-0.5px'
            }}
            className="display-font"
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)'
              e.currentTarget.style.boxShadow = '0 8px 25px rgba(219, 39, 119, 0.4)'
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = '0 4px 15px rgba(219, 39, 119, 0.3)'
            }}
          >
            <Upload style={{ width: '1.125rem', height: '1.125rem' }} />
            Analyze Another
          </button>
          <button
            onClick={() => router.push('/')}
            style={{
              padding: '1rem 2.25rem',
              background: 'white',
              color: '#6b7280',
              borderRadius: '0.875rem',
              fontWeight: '700',
              fontSize: '1rem',
              border: '1px solid rgba(219, 39, 119, 0.2)',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.625rem',
              transition: 'all 0.3s ease',
              letterSpacing: '-0.5px'
            }}
            className="display-font"
            onMouseOver={(e) => {
              e.currentTarget.style.background = '#fdf2f8'
              e.currentTarget.style.color = '#db2777'
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'white'
              e.currentTarget.style.color = '#6b7280'
            }}
          >
            <Home style={{ width: '1.125rem', height: '1.125rem' }} />
            Home
          </button>
        </div>
      </main>

    </div>
  )
}

export default function ResultsPage() {
  const styles = `
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    .display-font { font-family: 'Outfit', sans-serif; }
  `

  return (
    <Suspense fallback={
      <div style={{ minHeight: '100vh', background: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <style>{styles}</style>
        <div style={{ textAlign: 'center' }}>
          <Activity style={{ width: '3rem', height: '3rem', color: '#db2777', margin: '0 auto 1rem' }} />
          <p style={{ color: '#6b7280', fontSize: '1.125rem', fontFamily: "'DM Sans', sans-serif" }}>Loading results...</p>
        </div>
      </div>
    }>
      <ResultsContent />
    </Suspense>
  )
}
