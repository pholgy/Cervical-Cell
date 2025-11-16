import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

const CSV_FILE_PATH = path.join(process.cwd(), 'patients.csv')

export async function POST(request: NextRequest) {
  try {
    const { name, phone } = await request.json()

    if (!name || !phone) {
      return NextResponse.json(
        { success: false, error: 'Name and phone number are required' },
        { status: 400 }
      )
    }

    // Create CSV header if file doesn't exist
    let fileExists = false
    try {
      await fs.access(CSV_FILE_PATH)
      fileExists = true
    } catch {
      // File doesn't exist, will create it
    }

    // Format the data
    const timestamp = new Date().toISOString()
    const escapedName = name.replace(/"/g, '""')
    const escapedPhone = phone.replace(/"/g, '""')
    const csvLine = `"${escapedName}","${escapedPhone}","${timestamp}"\n`

    if (!fileExists) {
      // Create file with header
      const header = '"Name","Phone","Registered At"\n'
      await fs.writeFile(CSV_FILE_PATH, header + csvLine, 'utf8')
    } else {
      // Append to existing file
      await fs.appendFile(CSV_FILE_PATH, csvLine, 'utf8')
    }

    return NextResponse.json({
      success: true,
      message: 'Patient registered successfully',
      patient: { name, phone, timestamp }
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
    let fileExists = false
    try {
      await fs.access(CSV_FILE_PATH)
      fileExists = true
    } catch {
      // File doesn't exist
    }

    if (!fileExists) {
      return NextResponse.json({ success: true, patients: [] })
    }

    const csvContent = await fs.readFile(CSV_FILE_PATH, 'utf8')
    const lines = csvContent.trim().split('\n')

    if (lines.length <= 1) {
      return NextResponse.json({ success: true, patients: [] })
    }

    // Parse CSV (skip header)
    const patients = lines.slice(1).map(line => {
      const matches = line.match(/"([^"]*)","([^"]*)","([^"]*)"/);
      if (matches) {
        return {
          name: matches[1],
          phone: matches[2],
          registeredAt: matches[3]
        }
      }
      return null
    }).filter(Boolean)

    return NextResponse.json({ success: true, patients })
  } catch (error) {
    console.error('Error fetching patients:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to fetch patients' },
      { status: 500 }
    )
  }
}
