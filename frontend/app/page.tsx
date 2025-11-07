'use client'

import { Activity, Upload, Brain, Sparkles, ArrowRight, CheckCircle } from 'lucide-react'
import { useRouter } from 'next/navigation'

export default function Home() {
  const router = useRouter()

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Deep learning model trained on thousands of cervical cell images for accurate classification'
    },
    {
      icon: Activity,
      title: 'Real-time Classification',
      description: 'Get instant results with confidence scores and probability distributions'
    },
    {
      icon: Sparkles,
      title: 'Medical Insights',
      description: 'AI-generated explanations help understand clinical significance of each classification'
    }
  ]

  const steps = [
    { number: 1, title: 'Upload Image', desc: 'Upload a cervical cell microscopy image' },
    { number: 2, title: 'AI Analysis', desc: 'Our model analyzes the cell morphology' },
    { number: 3, title: 'View Results', desc: 'Get classification with medical insights' }
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
            <button
              onClick={() => router.push('/upload')}
              style={{
                padding: '0.75rem 1.5rem',
                background: '#4f46e5',
                color: 'white',
                borderRadius: '0.5rem',
                fontWeight: '600',
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
              Get Started
              <ArrowRight style={{ width: '1.25rem', height: '1.25rem' }} />
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '4rem 2rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '5rem' }}>
          <div style={{
            display: 'inline-block',
            padding: '0.5rem 1.25rem',
            background: '#eef2ff',
            borderRadius: '2rem',
            color: '#4f46e5',
            fontSize: '0.875rem',
            fontWeight: '600',
            marginBottom: '2rem',
            border: '1px solid #c7d2fe'
          }}>
            Welcome to Cervical Cell Classification System
          </div>

          <h1 style={{
            fontSize: '3.5rem',
            fontWeight: '800',
            color: '#1f2937',
            marginBottom: '1.5rem',
            lineHeight: 1.2
          }}>
            Advanced AI for<br />Cervical Cell Analysis
          </h1>

          <p style={{
            fontSize: '1.25rem',
            color: '#6b7280',
            marginBottom: '3rem',
            maxWidth: '700px',
            margin: '0 auto 3rem',
            lineHeight: 1.6
          }}>
            Powered by deep learning and computer vision, our system classifies cervical cells with high accuracy and provides clinical insights powered by AI.
          </p>

          <button
            onClick={() => router.push('/upload')}
            style={{
              padding: '1.25rem 2.5rem',
              background: '#4f46e5',
              color: 'white',
              borderRadius: '0.5rem',
              fontWeight: '600',
              fontSize: '1.125rem',
              border: 'none',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.75rem',
              transition: 'background 0.2s',
              boxShadow: '0 4px 12px rgba(79, 70, 229, 0.3)'
            }}
            onMouseOver={(e) => e.currentTarget.style.background = '#4338ca'}
            onMouseOut={(e) => e.currentTarget.style.background = '#4f46e5'}
          >
            <Upload style={{ width: '1.5rem', height: '1.5rem' }} />
            Start Classification
          </button>
        </div>

        {/* How It Works */}
        <div style={{
          background: 'white',
          borderRadius: '1rem',
          padding: '3rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          marginBottom: '4rem',
          border: '1px solid #e5e7eb'
        }}>
          <h2 style={{
            fontSize: '2rem',
            fontWeight: '700',
            color: '#1f2937',
            textAlign: 'center',
            marginBottom: '3rem'
          }}>
            How It Works
          </h2>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '2rem'
          }}>
            {steps.map((step) => (
              <div key={step.number} style={{ textAlign: 'center' }}>
                <div style={{
                  width: '4rem',
                  height: '4rem',
                  background: '#4f46e5',
                  borderRadius: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 1.5rem'
                }}>
                  <span style={{
                    fontSize: '1.75rem',
                    fontWeight: '700',
                    color: 'white'
                  }}>
                    {step.number}
                  </span>
                </div>
                <h3 style={{
                  fontSize: '1.25rem',
                  fontWeight: '700',
                  color: '#1f2937',
                  marginBottom: '0.75rem'
                }}>
                  {step.title}
                </h3>
                <p style={{
                  fontSize: '1rem',
                  color: '#6b7280',
                  lineHeight: 1.6
                }}>
                  {step.desc}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Features */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '1.5rem',
          marginBottom: '4rem'
        }}>
          {features.map((feature) => {
            const Icon = feature.icon
            return (
              <div
                key={feature.title}
                style={{
                  background: 'white',
                  borderRadius: '1rem',
                  padding: '2rem',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                  border: '1px solid #e5e7eb',
                  transition: 'box-shadow 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.boxShadow = '0 10px 25px rgba(0,0,0,0.1)'}
                onMouseOut={(e) => e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)'}
              >
                <div style={{
                  width: '3.5rem',
                  height: '3.5rem',
                  background: '#eef2ff',
                  borderRadius: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1.5rem'
                }}>
                  <Icon style={{ width: '1.75rem', height: '1.75rem', color: '#4f46e5' }} />
                </div>
                <h3 style={{
                  fontSize: '1.25rem',
                  fontWeight: '700',
                  color: '#1f2937',
                  marginBottom: '0.75rem'
                }}>
                  {feature.title}
                </h3>
                <p style={{
                  fontSize: '1rem',
                  color: '#6b7280',
                  lineHeight: 1.6
                }}>
                  {feature.description}
                </p>
              </div>
            )
          })}
        </div>

        {/* Cell Types */}
        <div style={{
          background: 'white',
          borderRadius: '1rem',
          padding: '3rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          border: '1px solid #e5e7eb'
        }}>
          <h2 style={{
            fontSize: '2rem',
            fontWeight: '700',
            color: '#1f2937',
            textAlign: 'center',
            marginBottom: '1rem'
          }}>
            Detectable Cell Types
          </h2>
          <p style={{
            fontSize: '1.125rem',
            color: '#6b7280',
            textAlign: 'center',
            marginBottom: '3rem',
            maxWidth: '700px',
            margin: '0 auto 3rem'
          }}>
            Our system can accurately identify and classify five distinct cervical cell types
          </p>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1.5rem'
          }}>
            {[
              { name: 'Dyskeratotic', color: '#ef4444', desc: 'Abnormal keratin production, often associated with HPV infection' },
              { name: 'Koilocytotic', color: '#f97316', desc: 'Cells showing HPV-related changes with perinuclear halos' },
              { name: 'Metaplastic', color: '#eab308', desc: 'Cells undergoing transformation, often benign' },
              { name: 'Parabasal', color: '#22c55e', desc: 'Immature squamous cells from basal layers' },
              { name: 'Superficial-Intermediate', color: '#3b82f6', desc: 'Mature squamous cells from upper layers' }
            ].map((cell) => (
              <div
                key={cell.name}
                style={{
                  padding: '1.5rem',
                  borderRadius: '0.75rem',
                  background: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  transition: 'border-color 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.borderColor = cell.color}
                onMouseOut={(e) => e.currentTarget.style.borderColor = '#e5e7eb'}
              >
                <div style={{
                  width: '3rem',
                  height: '3rem',
                  borderRadius: '0.5rem',
                  background: cell.color,
                  marginBottom: '1rem'
                }}></div>
                <h3 style={{
                  fontSize: '1.125rem',
                  fontWeight: '700',
                  color: '#1f2937',
                  marginBottom: '0.5rem'
                }}>
                  {cell.name}
                </h3>
                <p style={{
                  fontSize: '0.9375rem',
                  color: '#6b7280',
                  lineHeight: 1.6
                }}>
                  {cell.desc}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div style={{
          textAlign: 'center',
          marginTop: '5rem',
          padding: '3rem',
          background: '#eef2ff',
          borderRadius: '1rem',
          border: '1px solid #c7d2fe'
        }}>
          <h2 style={{
            fontSize: '2rem',
            fontWeight: '700',
            color: '#1f2937',
            marginBottom: '1rem'
          }}>
            Ready to Get Started?
          </h2>
          <p style={{
            fontSize: '1.125rem',
            color: '#6b7280',
            marginBottom: '2rem'
          }}>
            Upload your cervical cell image and get instant AI-powered analysis
          </p>
          <button
            onClick={() => router.push('/upload')}
            style={{
              padding: '1.25rem 2.5rem',
              background: '#4f46e5',
              color: 'white',
              borderRadius: '0.5rem',
              fontWeight: '600',
              fontSize: '1.125rem',
              border: 'none',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.75rem',
              transition: 'background 0.2s'
            }}
            onMouseOver={(e) => e.currentTarget.style.background = '#4338ca'}
            onMouseOut={(e) => e.currentTarget.style.background = '#4f46e5'}
          >
            <Upload style={{ width: '1.5rem', height: '1.5rem' }} />
            Upload Image Now
          </button>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        marginTop: '4rem',
        padding: '2rem',
        background: 'white',
        borderTop: '1px solid #e5e7eb'
      }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', textAlign: 'center' }}>
          <p style={{ color: '#6b7280', fontSize: '0.875rem', margin: 0 }}>
            Cervical Cell Classification System • Powered by Deep Learning & Computer Vision • Model: EfficientNetB3 • AI: Gemini 2.0 Flash
          </p>
        </div>
      </footer>
    </div>
  )
}
