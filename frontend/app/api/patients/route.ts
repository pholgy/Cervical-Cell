import { NextRequest, NextResponse } from 'next/server'

const FIREBASE_DB_URL = 'https://peakily-ml-proj-default-rtdb.firebaseio.com'
const FIREBASE_API_KEY = process.env.FIREBASE_API_KEY

export async function POST(request: NextRequest) {
  try {
    const { idCard, name, surname, phone, dateOfBirth, gender } = await request.json()

    // Validate required fields
    if (!idCard || !name || !surname || !dateOfBirth) {
      return NextResponse.json(
        { success: false, error: 'ID Card, Name, Surname, and Date of Birth are required' },
        { status: 400 }
      )
    }

    // Validate ID Card format (13 digits)
    if (!/^\d{13}$/.test(idCard)) {
      return NextResponse.json(
        { success: false, error: 'ID Card must be 13 digits' },
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
    // Use ID Card as patient identifier
    const patientId = idCard

    // Save to Firebase Realtime Database
    const response = await fetch(
      `${FIREBASE_DB_URL}/patients/${patientId}.json?key=${FIREBASE_API_KEY}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          idCard,
          name,
          surname,
          phone: phone || '',
          dateOfBirth,
          gender: gender || '',
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
      patient: { idCard, name, surname, phone, dateOfBirth, gender, registeredAt: timestamp }
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
      idCard: patient.idCard,
      name: patient.name,
      surname: patient.surname,
      phone: patient.phone || '',
      dateOfBirth: patient.dateOfBirth,
      gender: patient.gender || '',
      registeredAt: patient.registeredAt
    }))

    return NextResponse.json({ success: true, patients })
  } catch (error) {
    console.error('Error fetching patients:', error)
    return NextResponse.json({ success: true, patients: [] })
  }
}
