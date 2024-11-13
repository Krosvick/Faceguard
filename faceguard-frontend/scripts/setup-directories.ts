import { mkdir } from 'fs/promises'
import path from 'path'

async function setupDirectories() {
  const imagesDir = path.join(process.cwd(), 'public', 'attendance-images')
  
  try {
    await mkdir(imagesDir, { recursive: true })
    console.log('Created attendance-images directory')
  } catch (error) {
    if ((error as any).code !== 'EEXIST') {
      console.error('Error creating directories:', error)
    }
  }
}

setupDirectories() 