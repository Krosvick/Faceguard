import { writeFile, mkdir, access } from 'fs/promises'
import { NextRequest, NextResponse } from 'next/server'
import path from 'path'

// Get absolute paths
const ROOT_DIR = process.cwd()
const BACKEND_DIR = path.join(ROOT_DIR, '..', 'backend')
const DATASETS_DIR = path.join(BACKEND_DIR, 'face-recognition', 'datasets')

// Path to the face recognition datasets directory
const UPLOAD_DIR = path.join(DATASETS_DIR, 'new_persons')
const PUBLIC_DIR = path.join(ROOT_DIR, 'public', 'uploads', 'students')

export async function POST(request: NextRequest) {
  try {
    console.log('\n🔵 API UPLOAD START ----------------')
    
    const formData = await request.formData()
    const files = formData.getAll('file') as Blob[]
    const studentName = formData.get('studentName') as string
    
    console.log('📥 Received request:', {
      studentName,
      numberOfFiles: files.length
    })
    
    if (!files.length || !studentName) {
      console.error('❌ Missing data:', { hasFiles: !!files.length, studentName })
      return NextResponse.json(
        { error: 'No se encontraron archivos o nombre del estudiante' },
        { status: 400 }
      )
    }

    // Create student directories
    const studentDirBackend = path.join(UPLOAD_DIR, studentName)
    const studentDirPublic = path.join(PUBLIC_DIR, studentName)
    
    console.log('📁 Creating directories:', {
      backend: studentDirBackend,
      public: studentDirPublic
    })

    // Ensure both directories exist
    await mkdir(studentDirBackend, { recursive: true })
    await mkdir(studentDirPublic, { recursive: true })

    const uploadedPaths = []

    // Process all files
    for (const [index, file] of files.entries()) {
      console.log(`\n📄 Processing file ${index + 1}/${files.length}`)
      
      // Create unique filename
      const uniqueName = `${Date.now()}-${Math.random().toString(36).substring(2)}`
      const filename = uniqueName + '.jpg'

      // Define paths for both locations
      const backendPath = path.join(studentDirBackend, filename)
      const publicPath = path.join(studentDirPublic, filename)

      console.log('💾 Saving to:', {
        backend: backendPath,
        public: publicPath
      })

      // Convert file to buffer
      const bytes = await file.arrayBuffer()
      const buffer = Buffer.from(bytes)

      // Save in backend directory (for face recognition)
      try {
        await writeFile(backendPath, buffer)
        console.log('✅ Saved to backend directory')
      } catch (error) {
        console.error('❌ Error saving to backend:', error)
        throw error
      }

      // Save in public directory (for web display)
      try {
        await writeFile(publicPath, buffer)
        console.log('✅ Saved to public directory')
      } catch (error) {
        console.error('❌ Error saving to public:', error)
        throw error
      }

      // Add public path to response
      const webPath = `/uploads/students/${studentName}/${filename}`
      uploadedPaths.push(webPath)
      console.log('✅ File processed:', webPath)
    }

    // Verify backend directory structure
    try {
      const backendFiles = await readdir(studentDirBackend)
      console.log('\n📂 Backend directory contents:', {
        directory: studentDirBackend,
        files: backendFiles
      })
    } catch (error) {
      console.error('❌ Error reading backend directory:', error)
    }

    console.log('\n🎉 All files processed successfully!')
    console.log('📝 Paths:', uploadedPaths)
    console.log('🔵 API UPLOAD END ----------------\n')

    return NextResponse.json({ paths: uploadedPaths })

  } catch (error) {
    console.error('\n❌ Upload error:', error)
    return NextResponse.json(
      { error: 'Error al subir los archivos', details: error.message },
      { status: 500 }
    )
  }
}

// Helper function to check if a directory exists
async function exists(path: string): Promise<boolean> {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

// Helper function to read directory contents
async function readdir(path: string): Promise<string[]> {
  const { readdir } = await import('fs/promises')
  return readdir(path)
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '10mb'
    }
  }
} 