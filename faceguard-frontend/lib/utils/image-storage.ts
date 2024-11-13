import { writeFile } from 'fs/promises'
import path from 'path'

const IMAGES_DIR = path.join(process.cwd(), 'public', 'attendance-images')

export async function saveBase64Image(
  base64Data: string, 
  name: string, 
  timestamp: string
): Promise<string> {
  try {
    // Create a URL-friendly filename
    const sanitizedName = name.toLowerCase().replace(/[^a-z0-9]/g, '-')
    const filename = `${sanitizedName}-${timestamp}.jpg`
    
    // Create relative path for database storage
    const relativePath = `/attendance-images/${filename}`
    
    // Create absolute path for file writing
    const absolutePath = path.join(IMAGES_DIR, filename)
    
    // Remove base64 header if present
    const base64Image = base64Data.replace(/^data:image\/\w+;base64,/, '')
    
    // Convert base64 to buffer
    const imageBuffer = Buffer.from(base64Image, 'base64')
    
    // Write file
    await writeFile(absolutePath, imageBuffer)
    
    return relativePath
  } catch (error) {
    console.error('Error saving image:', error)
    throw new Error('Failed to save image')
  }
} 