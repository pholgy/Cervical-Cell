'use client'

import { Activity, Upload, Brain, Sparkles, ArrowRight, CheckCircle, Zap, Shield } from 'lucide-react'
import { useRouter } from 'next/navigation'

export default function Home() {
  const router = useRouter()

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
    .card-lift:hover { transform: translateY(-8px); box-shadow: 0 20px 40px rgba(219, 39, 119, 0.15); }

    .gradient-text {
      background: linear-gradient(135deg, #db2777 0%, #be185d 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .animate-fade-in { animation: fadeInUp 0.6s ease-out; }
  `

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Gemini Vision AI trained on thousands of cervical cell images for accurate classification',
      color: '#db2777'
    },
    {
      icon: Zap,
      title: 'Instant Results',
      description: 'Get real-time classification with confidence scores and cancer risk assessment',
      color: '#db2777'
    },
    {
      icon: Shield,
      title: 'Medical Grade',
      description: 'Bilingual medical explanations with accuracy, sensitivity & specificity metrics',
      color: '#db2777'
    }
  ]

  const steps = [
    { number: '1', title: 'Upload', desc: 'Select cervical cell image' },
    { number: '2', title: 'Register', desc: 'Enter patient information' },
    { number: '3', title: 'Analyze', desc: 'AI analyzes morphology' },
    { number: '4', title: 'Results', desc: 'Get full diagnosis report' }
  ]

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
              Start Now
              <ArrowRight style={{ width: '1rem', height: '1rem' }} />
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '5rem 2rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '6rem' }}>
          <div style={{
            display: 'inline-block',
            padding: '0.5rem 1.25rem',
            background: 'rgba(219, 39, 119, 0.08)',
            borderRadius: '2rem',
            border: '1px solid rgba(219, 39, 119, 0.2)',
            marginBottom: '2rem'
          }}>
            <p style={{ fontSize: '0.8125rem', color: '#db2777', fontWeight: '600', margin: 0 }}>
              ✨ Advanced AI Diagnostics
            </p>
          </div>

          <h1 style={{
            fontSize: '3.5rem',
            fontWeight: '800',
            color: '#1f2937',
            marginBottom: '1rem',
            lineHeight: 1.1,
            letterSpacing: '-1px'
          }} className="display-font">
            Cervical Cancer Screening<br />
            <span className="gradient-text">Made Intelligent</span>
          </h1>

          <p style={{
            fontSize: '1.25rem',
            color: '#6b7280',
            marginBottom: '3rem',
            maxWidth: '700px',
            margin: '0 auto 3rem',
            lineHeight: 1.6
          }}>
            AI-powered cell classification with cancer risk assessment. Get instant medical insights powered by Gemini Vision AI.
          </p>

          <button
            onClick={() => router.push('/upload')}
            style={{
              padding: '1.125rem 2.75rem',
              background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
              color: 'white',
              borderRadius: '0.875rem',
              fontWeight: '700',
              fontSize: '1.125rem',
              border: 'none',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.75rem',
              transition: 'all 0.3s ease',
              boxShadow: '0 8px 20px rgba(219, 39, 119, 0.3)',
              letterSpacing: '-0.5px'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-3px)'
              e.currentTarget.style.boxShadow = '0 12px 30px rgba(219, 39, 119, 0.4)'
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = '0 8px 20px rgba(219, 39, 119, 0.3)'
            }}
          >
            <Upload style={{ width: '1.25rem', height: '1.25rem' }} />
            Upload Cell Image
          </button>
        </div>

        {/* Process Steps */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(219, 39, 119, 0.05) 0%, rgba(219, 39, 119, 0.02) 100%)',
          borderRadius: '1.5rem',
          padding: '3rem',
          marginBottom: '6rem',
          border: '1px solid rgba(219, 39, 119, 0.1)'
        }}>
          <h2 style={{
            fontSize: '1.875rem',
            fontWeight: '700',
            color: '#1f2937',
            textAlign: 'center',
            marginBottom: '3rem'
          }} className="display-font">
            Diagnosis in 4 Steps
          </h2>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '2rem'
          }}>
            {steps.map((step, idx) => (
              <div key={step.number} style={{ textAlign: 'center', position: 'relative' }}>
                <div style={{
                  width: '4rem',
                  height: '4rem',
                  background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
                  borderRadius: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 1.5rem',
                  color: 'white',
                  fontSize: '1.5rem',
                  fontWeight: '700'
                }} className="display-font">
                  {step.number}
                </div>
                <h3 style={{
                  fontSize: '1.125rem',
                  fontWeight: '700',
                  color: '#1f2937',
                  marginBottom: '0.5rem'
                }} className="display-font">
                  {step.title}
                </h3>
                <p style={{
                  fontSize: '0.9rem',
                  color: '#6b7280',
                  lineHeight: 1.5
                }}>
                  {step.desc}
                </p>
                {idx < steps.length - 1 && (
                  <div style={{
                    position: 'absolute',
                    right: '-1rem',
                    top: '2rem',
                    width: '2rem',
                    height: '2px',
                    background: 'linear-gradient(90deg, #db2777 0%, transparent 100%)',
                    display: 'none'
                  }} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Features */}
        <div style={{ marginBottom: '6rem' }}>
          <h2 style={{
            fontSize: '1.875rem',
            fontWeight: '700',
            color: '#1f2937',
            textAlign: 'center',
            marginBottom: '3rem'
          }} className="display-font">
            Powerful Features
          </h2>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: '2rem'
          }}>
            {features.map((feature) => {
              const Icon = feature.icon
              return (
                <div
                  key={feature.title}
                  style={{
                    background: 'white',
                    borderRadius: '1.25rem',
                    padding: '2rem',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                    border: '1px solid rgba(219, 39, 119, 0.1)',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                  className="card-lift"
                >
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    right: 0,
                    width: '100px',
                    height: '100px',
                    background: 'linear-gradient(135deg, rgba(219, 39, 119, 0.1) 0%, transparent 100%)',
                    borderRadius: '0 0 100% 0'
                  }} />

                  <div style={{
                    width: '3.5rem',
                    height: '3.5rem',
                    background: 'rgba(219, 39, 119, 0.1)',
                    borderRadius: '1rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '1.5rem',
                    position: 'relative',
                    zIndex: 1
                  }}>
                    <Icon style={{ width: '1.75rem', height: '1.75rem', color: '#db2777' }} />
                  </div>

                  <h3 style={{
                    fontSize: '1.25rem',
                    fontWeight: '700',
                    color: '#1f2937',
                    marginBottom: '0.75rem',
                    position: 'relative',
                    zIndex: 1
                  }} className="display-font">
                    {feature.title}
                  </h3>

                  <p style={{
                    fontSize: '0.9375rem',
                    color: '#6b7280',
                    lineHeight: 1.6,
                    position: 'relative',
                    zIndex: 1
                  }}>
                    {feature.description}
                  </p>
                </div>
              )
            })}
          </div>
        </div>

        {/* CTA Section */}
        <div style={{
          background: 'linear-gradient(135deg, #db2777 0%, #be185d 100%)',
          borderRadius: '1.5rem',
          padding: '4rem 3rem',
          textAlign: 'center',
          boxShadow: '0 20px 40px rgba(219, 39, 119, 0.2)',
          position: 'relative',
          overflow: 'hidden',
          marginBottom: '3rem'
        }}>
          <div style={{
            position: 'absolute',
            top: '-50%',
            right: '-10%',
            width: '400px',
            height: '400px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '50%'
          }} />

          <h2 style={{
            fontSize: '2rem',
            fontWeight: '700',
            color: 'white',
            marginBottom: '1rem',
            position: 'relative',
            zIndex: 1
          }} className="display-font">
            Ready to Diagnose?
          </h2>

          <p style={{
            fontSize: '1.125rem',
            color: 'rgba(255, 255, 255, 0.9)',
            marginBottom: '2rem',
            maxWidth: '600px',
            margin: '0 auto 2rem',
            position: 'relative',
            zIndex: 1
          }}>
            Upload a cervical cell image and get instant AI analysis with cancer risk assessment and medical insights.
          </p>

          <button
            onClick={() => router.push('/upload')}
            style={{
              padding: '1rem 2.5rem',
              background: 'white',
              color: '#db2777',
              borderRadius: '0.875rem',
              fontWeight: '700',
              fontSize: '1rem',
              border: 'none',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.75rem',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 15px rgba(0,0,0,0.15)',
              position: 'relative',
              zIndex: 1
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)'
              e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.2)'
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.15)'
            }}
          >
            <Upload style={{ width: '1.125rem', height: '1.125rem' }} />
            Start Analysis
          </button>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        padding: '3rem 2rem',
        background: 'rgba(219, 39, 119, 0.02)',
        borderTop: '1px solid rgba(219, 39, 119, 0.1)'
      }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', textAlign: 'center' }}>
          <p style={{ color: '#6b7280', fontSize: '0.875rem', margin: 0 }}>
            <strong>Peakily</strong> • Medical AI Classification • Powered by Gemini Vision AI
          </p>
          <p style={{ color: '#9ca3af', fontSize: '0.75rem', margin: '1rem 0 0 0' }}>
            For medical professionals • Bilingual support • Accuracy 94% • Sensitivity 93%
          </p>
        </div>
      </footer>
    </div>
  )
}
