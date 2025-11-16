import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Cervical Cell Classifier',
  description: 'AI-powered cervical cell classification system',
  icons: {
    icon: '/icon.jpg',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
