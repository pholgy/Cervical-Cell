import { NextRequest, NextResponse } from 'next/server'

const FIREBASE_PROJECT_ID = 'peakily-ml-proj'
const FIREBASE_DB_URL = `https://${FIREBASE_PROJECT_ID}-default-rtdb.asia-southeast1.firebaserealtimedb.app`
const FIREBASE_API_KEY = process.env.FIREBASE_API_KEY

export async function POST(request: NextRequest) {
  try {
    const { name, phone } = await request.json()

    if (!name || !phone) {
      return NextResponse.json(
        { success: false, error: 'Name and phone number are required' },
        { status: 400 }
      )
    }

    if (!FIREBASE_API_KEY) {
      console.error('FIREBASE_API_KEY not set')
      return NextResponse.json(
        { success: false, error: 'Server configuration error' },
        { status: 500 }
      )
    }

    const timestamp = new Date().toISOString()
    const patientId = Date.now().toString()

    // Save to Firebase Realtime Database
    const response = await fetch(
      `${FIREBASE_DB_URL}/patients/${patientId}.json?key=${FIREBASE_API_KEY}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          phone,
          registeredAt: timestamp
        })
      }
    )

    if (!response.ok) {
      const error = await response.text()
      console.error('Firebase error:', error)
      return NextResponse.json(
        { success: false, error: 'Failed to save patient data' },
        { status: 500 }
      )
    }

    return NextResponse.json({
      success: true,
      message: 'Patient registered successfully',
      patient: { name, phone, timestamp, id: patientId }
    })
  } catch (error) {
    console.error('Error registering patient:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to register patient' },
      { status: 500 }
    )
  }
}

export async function GET() {
  try {
    if (!FIREBASE_API_KEY) {
      return NextResponse.json({ success: true, patients: [] })
    }

    const response = await fetch(
      `${FIREBASE_DB_URL}/patients.json?key=${FIREBASE_API_KEY}`,
      { method: 'GET' }
    )

    if (!response.ok) {
      return NextResponse.json({ success: true, patients: [] })
    }

    const data = await response.json()

    if (!data) {
      return NextResponse.json({ success: true, patients: [] })
    }

    const patients = Object.entries(data).map(([id, patient]: [string, any]) => ({
      id,
      name: patient.name,
      phone: patient.phone,
      registeredAt: patient.registeredAt
    }))

    return NextResponse.json({ success: true, patients })
  } catch (error) {
    console.error('Error fetching patients:', error)
    return NextResponse.json({ success: true, patients: [] })
  }
}
