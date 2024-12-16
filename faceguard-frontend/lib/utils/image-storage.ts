export async function uploadStudentImage(file: File, studentName: string): Promise<string> {
  try {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('studentName', studentName)

    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.error || 'Error uploading image')
    }

    const data = await response.json()
    return data.path
  } catch (error) {
    console.error('Error uploading image:', error)
    throw error
  }
}

// Helper function to ensure consistent path formatting
export function formatStudentName(firstName: string, lastName: string): string {
  return `${firstName}_${lastName}`.trim().replace(/\s+/g, '_')
} 