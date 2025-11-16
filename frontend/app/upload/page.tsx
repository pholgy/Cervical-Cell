'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import axios from 'axios'
import { Upload, Activity, Loader2, XCircle, Home, UserPlus, ArrowRight } from 'lucide-react'

export default function UploadPage() {
  const router = useRouter()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showRegistration, setShowRegistration] = useState(false)
  const [pendingPrediction, setPendingPrediction] = useState<any>(null)
  const [patientName, setPatientName] = useState('')
  const [patientPhone, setPatientPhone] = useState('')
  const [registerLoading, setRegisterLoading] = useState(false)
  const [registerError, setRegisterError] = useState<string | null>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setError(null)

      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handlePredict = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      // Use Next.js API route
      const response = await axios.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      console.log('API Response:', response.data)
      console.log('Has AI explanation?', !!response.data.ai_explanation)
      console.log('AI explanation content:', response.data.ai_explanation)

      // Check if the prediction was successful
      if (response.data.success === false) {
        setError(response.data.error || 'Classification failed. Please try again.')
        return
      }

      // Store prediction data temporarily and show registration modal
      setPendingPrediction(response.data)
      setShowRegistration(true)
    } catch (err) {
      setError('Failed to get prediction. Make sure API is running on port 8000.')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRegister = async () => {
    if (!patientName.trim() || !patientPhone.trim()) {
      setRegisterError('Please fill in all fields')
      return
    }

    setRegisterLoading(true)
    setRegisterError(null)

    try {
      const response = await fetch('/api/patients', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: patientName.trim(), phone: patientPhone.trim() }),
      })

      const data = await response.json()

      if (data.success) {
        // Store patient info in sessionStorage
        sessionStorage.setItem('patientInfo', JSON.stringify({
          name: patientName.trim(),
          phone: patientPhone.trim()
        }))

        // Store prediction data and navigate to results
        sessionStorage.setItem('predictionData', JSON.stringify(pendingPrediction))
        router.push('/results')
      } else {
        setRegisterError(data.error || 'Registration failed')
      }
    } catch (err) {
      setRegisterError('Failed to register. Please try again.')
      console.error('Error:', err)
    } finally {
      setRegisterLoading(false)
    }
  }

  const cellTypes = [
    { name: 'Dyskeratotic', color: '#ef4444', desc: 'Abnormal keratin' },
    { name: 'Koilocytotic', color: '#f97316', desc: 'HPV changes' },
    { name: 'Metaplastic', color: '#eab308', desc: 'Cell transformation' },
    { name: 'Parabasal', color: '#22c55e', desc: 'Immature cells' },
    { name: 'Superficial-Intermediate', color: '#3b82f6', desc: 'Mature cells' }
  ]

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
              <Activity style={{ width: '2.5rem', height: '2.5rem', color: '#db2777' }} />
              <div>
                <h1 style={{ fontSize: '1.75rem', fontWeight: '700', color: '#1f2937', margin: 0 }}>
                  Cervical Cell Classifier
                </h1>
                <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: 0 }}>
                  AI-Powered Medical Image Analysis
                </p>
              </div>
            </div>
            <button
              onClick={() => router.push('/')}
              style={{
                padding: '0.75rem 1.5rem',
                background: 'white',
                color: '#db2777',
                borderRadius: '0.5rem',
                fontWeight: '600',
                border: '1px solid #db2777',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                transition: 'all 0.2s'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = '#db2777'
                e.currentTarget.style.color = 'white'
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = 'white'
                e.currentTarget.style.color = '#db2777'
              }}
            >
              <Home style={{ width: '1.25rem', height: '1.25rem' }} />
              Home
            </button>
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
            background: '#db2777',
            borderRadius: '0.5rem',
            color: 'white'
          }}>
            <div style={{
              width: '1.75rem',
              height: '1.75rem',
              borderRadius: '50%',
              background: 'white',
              color: '#db2777',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: '700',
              fontSize: '0.875rem'
            }}>1</div>
            <span style={{ fontWeight: '600' }}>Upload Image</span>
          </div>

          <div style={{
            width: '3rem',
            height: '2px',
            background: '#d1d5db'
          }}></div>

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
              background: '#f3f4f6',
              color: '#9ca3af',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: '700',
              fontSize: '0.875rem'
            }}>2</div>
            <span style={{ fontWeight: '600', color: '#6b7280' }}>View Results</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.5rem 2rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem' }}>

          {/* Left Column - Upload */}
          <div>
            <div style={{
              background: 'white',
              borderRadius: '1rem',
              padding: '1.5rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e7eb'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Upload style={{ width: '1.25rem', height: '1.25rem', color: '#db2777' }} />
                <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#1f2937', margin: 0 }}>
                  Upload Cell Image
                </h2>
              </div>

              <p style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '1rem', lineHeight: 1.5 }}>
                Upload cervical cell image for AI classification. PNG, JPG, BMP (max 10MB).
              </p>

              {!preview ? (
                <label style={{ display: 'block', cursor: 'pointer' }}>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                  />
                  <div style={{
                    border: '2px dashed #d1d5db',
                    borderRadius: '0.75rem',
                    padding: '2.5rem 1.5rem',
                    textAlign: 'center',
                    transition: 'all 0.2s',
                    background: '#f9fafb'
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.borderColor = '#db2777'
                    e.currentTarget.style.background = '#fdf2f8'
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.borderColor = '#d1d5db'
                    e.currentTarget.style.background = '#f9fafb'
                  }}>
                    <Upload style={{ width: '3rem', height: '3rem', color: '#9ca3af', margin: '0 auto 0.75rem' }} />
                    <p style={{ fontSize: '1rem', fontWeight: '600', color: '#374151', margin: '0 0 0.25rem 0' }}>
                      Click to upload image
                    </p>
                    <p style={{ fontSize: '0.8125rem', color: '#6b7280', margin: 0 }}>
                      PNG, JPG, BMP (max 10MB)
                    </p>
                  </div>
                </label>
              ) : (
                <div>
                  <div style={{
                    borderRadius: '0.75rem',
                    overflow: 'hidden',
                    background: '#f3f4f6',
                    border: '1px solid #e5e7eb',
                    marginBottom: '1rem'
                  }}>
                    <img
                      src={preview}
                      alt="Preview"
                      style={{ width: '100%', height: '16rem', objectFit: 'contain' }}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <label style={{ flex: 1 }}>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileSelect}
                        style={{ display: 'none' }}
                      />
                      <div style={{
                        width: '100%',
                        padding: '0.875rem',
                        background: 'white',
                        color: '#374151',
                        borderRadius: '0.5rem',
                        fontWeight: '600',
                        textAlign: 'center',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        border: '1px solid #e5e7eb'
                      }}
                      onMouseOver={(e) => e.currentTarget.style.background = '#f9fafb'}
                      onMouseOut={(e) => e.currentTarget.style.background = 'white'}>
                        Change Image
                      </div>
                    </label>

                    <button
                      onClick={handlePredict}
                      disabled={loading}
                      style={{
                        flex: 2,
                        padding: '0.875rem 1.5rem',
                        background: '#db2777',
                        color: 'white',
                        borderRadius: '0.5rem',
                        fontWeight: '600',
                        border: 'none',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.5rem',
                        opacity: loading ? 0.6 : 1,
                        transition: 'background 0.2s'
                      }}
                      onMouseOver={(e) => !loading && (e.currentTarget.style.background = '#be185d')}
                      onMouseOut={(e) => e.currentTarget.style.background = '#db2777'}
                    >
                      {loading ? (
                        <>
                          <Loader2 style={{ width: '1.25rem', height: '1.25rem', animation: 'spin 1s linear infinite' }} />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Activity style={{ width: '1.25rem', height: '1.25rem' }} />
                          Classify Cell
                        </>
                      )}
                    </button>
                  </div>

                  {error && (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.75rem',
                      padding: '1rem',
                      background: '#fef2f2',
                      border: '1px solid #fecaca',
                      borderRadius: '0.5rem',
                      color: '#b91c1c',
                      marginTop: '1rem'
                    }}>
                      <XCircle style={{ width: '1.25rem', height: '1.25rem', flexShrink: 0 }} />
                      <p style={{ fontSize: '0.875rem', margin: 0 }}>{error}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Reference */}
          <div>
            <div style={{
              background: 'white',
              borderRadius: '1rem',
              padding: '1.5rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: '1px solid #e5e7eb'
            }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#1f2937', marginBottom: '0.5rem' }}>
                Cell Type Reference
              </h3>
              <p style={{ fontSize: '0.8125rem', color: '#6b7280', marginBottom: '1rem', lineHeight: 1.5 }}>
                AI model can identify these five cervical cell types:
              </p>
              <div>
                {cellTypes.map((cell) => (
                  <div key={cell.name} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    padding: '0.625rem',
                    borderRadius: '0.5rem',
                    transition: 'background 0.2s',
                    marginBottom: '0.5rem'
                  }}
                  onMouseOver={(e) => e.currentTarget.style.background = '#f9fafb'}
                  onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                    <div style={{
                      width: '2.5rem',
                      height: '2.5rem',
                      borderRadius: '0.375rem',
                      background: cell.color,
                      flexShrink: 0
                    }}></div>
                    <div style={{ flex: 1 }}>
                      <p style={{ fontWeight: '600', color: '#1f2937', fontSize: '0.8125rem', margin: '0 0 0.125rem 0' }}>
                        {cell.name}
                      </p>
                      <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0, lineHeight: 1.3 }}>
                        {cell.desc}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div style={{
                marginTop: '1rem',
                padding: '1rem',
                background: '#fdf2f8',
                borderRadius: '0.625rem',
                border: '1px solid #fbcfe8'
              }}>
                <h4 style={{ fontSize: '0.8125rem', fontWeight: '700', color: '#db2777', marginBottom: '0.5rem' }}>
                  How Classification Works
                </h4>
                <ul style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0, paddingLeft: '1.125rem', lineHeight: 1.5 }}>
                  <li>Upload cervical cell image</li>
                  <li>AI analyzes cell morphology</li>
                  <li>Get classification scores</li>
                  <li>View AI medical insights</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </main>


      {/* Registration Modal */}
      {showRegistration && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '1rem'
        }}>
          <div style={{
            background: 'white',
            borderRadius: '1rem',
            padding: '2rem',
            maxWidth: '500px',
            width: '100%',
            boxShadow: '0 25px 50px rgba(0, 0, 0, 0.25)'
          }}>
            <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
              <div style={{
                width: '3.5rem',
                height: '3.5rem',
                background: '#fdf2f8',
                borderRadius: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1rem'
              }}>
                <UserPlus style={{ width: '1.75rem', height: '1.75rem', color: '#db2777' }} />
              </div>
              <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1f2937', marginBottom: '0.5rem' }}>
                Patient Registration
              </h2>
              <p style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: 1.5 }}>
                Please enter patient information to view results
              </p>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#374151',
                marginBottom: '0.5rem'
              }}>
                Patient Name
              </label>
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="Enter patient name"
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  fontSize: '0.9375rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.5rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#db2777'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#d1d5db'}
              />
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#374151',
                marginBottom: '0.5rem'
              }}>
                Phone Number
              </label>
              <input
                type="tel"
                value={patientPhone}
                onChange={(e) => setPatientPhone(e.target.value)}
                placeholder="Enter phone number"
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  fontSize: '0.9375rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.5rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#db2777'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#d1d5db'}
              />
            </div>

            {registerError && (
              <div style={{
                padding: '0.75rem',
                background: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: '0.5rem',
                color: '#b91c1c',
                fontSize: '0.8125rem',
                marginBottom: '1.25rem'
              }}>
                {registerError}
              </div>
            )}

            <button
              onClick={handleRegister}
              disabled={registerLoading}
              style={{
                width: '100%',
                padding: '0.875rem',
                background: '#db2777',
                color: 'white',
                borderRadius: '0.5rem',
                fontWeight: '600',
                fontSize: '0.9375rem',
                border: 'none',
                cursor: registerLoading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                opacity: registerLoading ? 0.6 : 1,
                transition: 'background 0.2s'
              }}
              onMouseOver={(e) => !registerLoading && (e.currentTarget.style.background = '#be185d')}
              onMouseOut={(e) => e.currentTarget.style.background = '#db2777'}
            >
              {registerLoading ? (
                <>
                  <Loader2 style={{ width: '1.125rem', height: '1.125rem', animation: 'spin 1s linear infinite' }} />
                  Saving...
                </>
              ) : (
                <>
                  View Results
                  <ArrowRight style={{ width: '1.125rem', height: '1.125rem' }} />
                </>
              )}
            </button>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
