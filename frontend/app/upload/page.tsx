'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import axios from 'axios'
import { Upload, Activity, Loader2, XCircle, Home, UserPlus, ArrowRight, UploadCloud } from 'lucide-react'

export default function UploadPage() {
  const router = useRouter()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showRegistration, setShowRegistration] = useState(false)
  const [pendingPrediction, setPendingPrediction] = useState<any>(null)
  const [patientIdCard, setPatientIdCard] = useState('')
  const [patientName, setPatientName] = useState('')
  const [patientSurname, setPatientSurname] = useState('')
  const [patientPhone, setPatientPhone] = useState('')
  const [patientDateOfBirth, setPatientDateOfBirth] = useState('')
  const [patientGender, setPatientGender] = useState('')
  const [registerLoading, setRegisterLoading] = useState(false)
  const [registerError, setRegisterError] = useState<string | null>(null)

  const styles = `
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    .display-font { font-family: 'Outfit', sans-serif; }
  `

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
      const response = await axios.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      if (response.data.success === false) {
        setError(response.data.error || 'Classification failed. Please try again.')
        return
      }

      setPendingPrediction(response.data)
      setShowRegistration(true)
    } catch (err) {
      setError('Failed to get prediction. Please try again.')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRegister = async () => {
    // Validate required fields
    if (!patientIdCard.trim() || !patientName.trim() || !patientSurname.trim() || !patientDateOfBirth.trim()) {
      setRegisterError('Please fill in all required fields (ID Card, Name, Surname, Date of Birth)')
      return
    }

    // Validate ID Card format (13 digits)
    if (!/^\d{13}$/.test(patientIdCard.trim())) {
      setRegisterError('ID Card must be 13 digits')
      return
    }

    setRegisterLoading(true)
    setRegisterError(null)

    try {
      const response = await fetch('/api/patients', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          idCard: patientIdCard.trim(),
          name: patientName.trim(),
          surname: patientSurname.trim(),
          phone: patientPhone.trim(),
          dateOfBirth: patientDateOfBirth.trim(),
          gender: patientGender.trim()
        }),
      })

      const data = await response.json()

      if (data.success) {
        sessionStorage.setItem('patientInfo', JSON.stringify({
          idCard: patientIdCard.trim(),
          name: patientName.trim(),
          surname: patientSurname.trim(),
          phone: patientPhone.trim(),
          dateOfBirth: patientDateOfBirth.trim(),
          gender: patientGender.trim()
        }))
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
    <div style={{ minHeight: '100vh', background: '#ffffff' }}>
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
              <h1 style={{ fontSize: '1.375rem', fontWeight: '700', color: '#1f2937', margin: 0, letterSpacing: '-0.5px' }} className="display-font">
                Peakily
              </h1>
            </div>
            <button
              onClick={() => router.push('/')}
              style={{
                padding: '0.75rem 1.5rem',
                background: 'white',
                color: '#db2777',
                borderRadius: '0.75rem',
                fontWeight: '600',
                border: '1px solid #db2777',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                transition: 'all 0.3s ease'
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
              <Home style={{ width: '1rem', height: '1rem' }} />
              Home
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '3rem 2rem' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 320px',
          gap: '2rem',
          alignItems: 'start'
        }}>
          {/* Upload Section */}
          <div>
            <div style={{
              background: 'white',
              borderRadius: '1.5rem',
              padding: '3rem',
              boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
              border: '1px solid rgba(219, 39, 119, 0.1)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                <div style={{
                  width: '3rem',
                  height: '3rem',
                  borderRadius: '1rem',
                  background: 'rgba(219, 39, 119, 0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <UploadCloud style={{ width: '1.5rem', height: '1.5rem', color: '#db2777' }} />
                </div>
                <div>
                  <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1f2937', margin: 0 }} className="display-font">
                    Upload Cell Image
                  </h2>
                  <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: '0.25rem 0 0 0' }}>
                    PNG, JPG, BMP (max 10MB)
                  </p>
                </div>
              </div>

              {!preview ? (
                <label style={{ display: 'block', cursor: 'pointer' }}>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                  />
                  <div style={{
                    border: '2px dashed rgba(219, 39, 119, 0.3)',
                    borderRadius: '1.25rem',
                    padding: '3rem 1.5rem',
                    textAlign: 'center',
                    transition: 'all 0.3s ease',
                    background: 'rgba(219, 39, 119, 0.02)',
                    cursor: 'pointer'
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.borderColor = '#db2777'
                    e.currentTarget.style.background = 'rgba(219, 39, 119, 0.05)'
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(219, 39, 119, 0.3)'
                    e.currentTarget.style.background = 'rgba(219, 39, 119, 0.02)'
                  }}>
                    <UploadCloud style={{ width: '3.5rem', height: '3.5rem', color: '#db2777', margin: '0 auto 1rem' }} />
                    <p style={{ fontSize: '1.125rem', fontWeight: '600', color: '#1f2937', margin: '0 0 0.5rem 0' }}>
                      Click to upload image
                    </p>
                    <p style={{ fontSize: '0.875rem', color: '#9ca3af', margin: 0 }}>
                      or drag and drop
                    </p>
                  </div>
                </label>
              ) : (
                <div>
                  <div style={{
                    borderRadius: '1.25rem',
                    overflow: 'hidden',
                    background: '#f3f4f6',
                    border: '1px solid #e5e7eb',
                    marginBottom: '1.5rem'
                  }}>
                    <img
                      src={preview}
                      alt="Preview"
                      style={{ width: '100%', height: '300px', objectFit: 'cover' }}
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
                        padding: '1rem',
                        background: 'white',
                        color: '#374151',
                        borderRadius: '0.75rem',
                        fontWeight: '600',
                        textAlign: 'center',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        border: '1px solid #e5e7eb'
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.background = '#f9fafb'
                        e.currentTarget.style.borderColor = '#db2777'
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.background = 'white'
                        e.currentTarget.style.borderColor = '#e5e7eb'
                      }}>
                        Change Image
                      </div>
                    </label>

                    <button
                      onClick={handlePredict}
                      disabled={loading}
                      style={{
                        flex: 2,
                        padding: '1rem 1.5rem',
                        background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
                        color: 'white',
                        borderRadius: '0.75rem',
                        fontWeight: '700',
                        border: 'none',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.5rem',
                        opacity: loading ? 0.6 : 1,
                        transition: 'all 0.3s ease',
                        boxShadow: '0 4px 15px rgba(219, 39, 119, 0.3)'
                      }}
                      onMouseOver={(e) => !loading && (e.currentTarget.style.transform = 'translateY(-2px)')}
                      onMouseOut={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
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
                      borderRadius: '0.75rem',
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

          {/* Sidebar - Cell Reference */}
          <div style={{
            background: 'white',
            borderRadius: '1.5rem',
            padding: '1.5rem',
            boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
            border: '1px solid rgba(219, 39, 119, 0.1)',
            position: 'sticky',
            top: '80px',
            height: 'fit-content'
          }}>
            <h3 style={{ fontSize: '1rem', fontWeight: '700', color: '#1f2937', marginBottom: '1rem' }} className="display-font">
              Cell Types
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {cellTypes.map((cell) => (
                <div key={cell.name} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  padding: '0.75rem',
                  borderRadius: '0.75rem',
                  background: '#f9fafb',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'rgba(219, 39, 119, 0.08)'
                  e.currentTarget.style.transform = 'translateX(4px)'
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = '#f9fafb'
                  e.currentTarget.style.transform = 'translateX(0)'
                }}>
                  <div style={{
                    width: '2rem',
                    height: '2rem',
                    borderRadius: '0.5rem',
                    background: cell.color,
                    flexShrink: 0
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ fontWeight: '600', color: '#1f2937', fontSize: '0.8125rem', margin: 0 }}>
                      {cell.name}
                    </p>
                    <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: '0.125rem 0 0 0' }}>
                      {cell.desc}
                    </p>
                  </div>
                </div>
              ))}
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
          background: 'rgba(0, 0, 0, 0.6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '1rem',
          backdropFilter: 'blur(4px)'
        }}>
          <div style={{
            background: 'white',
            borderRadius: '1.5rem',
            padding: '2rem',
            maxWidth: '500px',
            width: '100%',
            boxShadow: '0 25px 50px rgba(0, 0, 0, 0.25)'
          }}>
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
              <div style={{
                width: '3.5rem',
                height: '3.5rem',
                background: 'rgba(219, 39, 119, 0.1)',
                borderRadius: '1rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1rem'
              }}>
                <UserPlus style={{ width: '1.75rem', height: '1.75rem', color: '#db2777' }} />
              </div>
              <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1f2937', marginBottom: '0.5rem' }} className="display-font">
                Patient Registration
              </h2>
              <p style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: 1.5 }}>
                Enter patient information to view results
              </p>
            </div>

            {/* ID Card (Required) */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#374151',
                marginBottom: '0.375rem'
              }}>
                ID Card Number <span style={{ color: '#db2777' }}>*</span>
              </label>
              <input
                type="text"
                value={patientIdCard}
                onChange={(e) => setPatientIdCard(e.target.value.replace(/\D/g, ''))}
                placeholder="เลขบัตรประชาชน (13 digits)"
                maxLength={13}
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  fontSize: '0.95rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.75rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#db2777'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#d1d5db'}
              />
            </div>

            {/* Name (Required) */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#374151',
                marginBottom: '0.375rem'
              }}>
                First Name <span style={{ color: '#db2777' }}>*</span>
              </label>
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="ชื่อ"
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  fontSize: '0.95rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.75rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#db2777'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#d1d5db'}
              />
            </div>

            {/* Surname (Required) */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#374151',
                marginBottom: '0.375rem'
              }}>
                Surname <span style={{ color: '#db2777' }}>*</span>
              </label>
              <input
                type="text"
                value={patientSurname}
                onChange={(e) => setPatientSurname(e.target.value)}
                placeholder="นามสกุล"
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  fontSize: '0.95rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.75rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#db2777'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#d1d5db'}
              />
            </div>

            {/* Phone Number (Optional) */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#374151',
                marginBottom: '0.375rem'
              }}>
                Phone Number
              </label>
              <input
                type="tel"
                value={patientPhone}
                onChange={(e) => setPatientPhone(e.target.value)}
                placeholder="เบอร์โทร (optional)"
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  fontSize: '0.95rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.75rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#db2777'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#d1d5db'}
              />
            </div>

            {/* Date of Birth (Required) */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#374151',
                marginBottom: '0.375rem'
              }}>
                Date of Birth <span style={{ color: '#db2777' }}>*</span>
              </label>
              <input
                type="date"
                value={patientDateOfBirth}
                onChange={(e) => setPatientDateOfBirth(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  fontSize: '0.95rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.75rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#db2777'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#d1d5db'}
              />
            </div>

            {/* Gender (Optional) */}
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#374151',
                marginBottom: '0.375rem'
              }}>
                Gender
              </label>
              <select
                value={patientGender}
                onChange={(e) => setPatientGender(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  fontSize: '0.95rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.75rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box',
                  backgroundColor: 'white',
                  cursor: 'pointer'
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = '#db2777'}
                onBlur={(e) => e.currentTarget.style.borderColor = '#d1d5db'}
              >
                <option value="">Select Gender</option>
                <option value="Male">Male (ชาย)</option>
                <option value="Female">Female (หญิง)</option>
                <option value="Other">Other (อื่น)</option>
              </select>
            </div>

            {registerError && (
              <div style={{
                padding: '0.75rem',
                background: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: '0.75rem',
                color: '#b91c1c',
                fontSize: '0.8125rem',
                marginBottom: '1.5rem'
              }}>
                {registerError}
              </div>
            )}

            <button
              onClick={handleRegister}
              disabled={registerLoading}
              style={{
                width: '100%',
                padding: '1rem',
                background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
                color: 'white',
                borderRadius: '0.75rem',
                fontWeight: '700',
                fontSize: '1rem',
                border: 'none',
                cursor: registerLoading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                opacity: registerLoading ? 0.6 : 1,
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 15px rgba(219, 39, 119, 0.3)'
              }}
              onMouseOver={(e) => !registerLoading && (e.currentTarget.style.transform = 'translateY(-2px)')}
              onMouseOut={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
            >
              {registerLoading ? (
                <>
                  <Loader2 style={{ width: '1rem', height: '1rem', animation: 'spin 1s linear infinite' }} />
                  Saving...
                </>
              ) : (
                <>
                  View Results
                  <ArrowRight style={{ width: '1rem', height: '1rem' }} />
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
