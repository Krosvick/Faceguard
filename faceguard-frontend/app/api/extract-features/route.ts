import { NextRequest, NextResponse } from 'next/server'
import path from 'path'
import { exec } from 'child_process'
import { promisify } from 'util'
import { readdir, copyFile, mkdir } from 'fs/promises'

const execAsync = promisify(exec)

// Define paths
const ROOT_DIR = process.cwd()
const PUBLIC_DIR = path.join(ROOT_DIR, 'public', 'uploads', 'students')
const BACKEND_DIR = path.join(ROOT_DIR, '..', 'backend')
const DATASETS_DIR = path.join(BACKEND_DIR, 'face-recognition', 'datasets')
const NEW_PERSONS_DIR = path.join(DATASETS_DIR, 'new_persons')

export async function POST(request: NextRequest) {
  try {
    console.log('\n🔵 EXTRACT FEATURES START ----------------')

    // Step 1: Ensure all images are in the backend directory
    console.log('📂 Copying images to backend...')
    
    // Get all student directories from public uploads
    const studentDirs = await readdir(PUBLIC_DIR)
    
    // Process each student directory
    for (const studentName of studentDirs) {
      if (studentName === '.gitkeep') continue // Skip .gitkeep file

      const publicStudentDir = path.join(PUBLIC_DIR, studentName)
      const backendStudentDir = path.join(NEW_PERSONS_DIR, studentName)

      // Create backend student directory if it doesn't exist
      await mkdir(backendStudentDir, { recursive: true })

      // Get all images from public directory
      const images = await readdir(publicStudentDir)
      
      console.log(`Processing ${images.length} images for student: ${studentName}`)

      // Copy each image to backend
      for (const image of images) {
        const sourcePath = path.join(publicStudentDir, image)
        const targetPath = path.join(backendStudentDir, image)
        await copyFile(sourcePath, targetPath)
        console.log(`✅ Copied: ${image}`)
      }
    }

    console.log('✨ All images copied to backend successfully')

    // Step 2: Run the Python script
    console.log('\n🤖 Running face recognition model...')
    
    const pythonExecutable = 'C:\\Users\\gunda\\miniconda3\\envs\\face-guard\\python.exe'
    const addPersonsScript = path.join(BACKEND_DIR, 'face-recognition', 'add_persons.py')
    const backupDir = path.join(DATASETS_DIR, 'backup')
    const facesSaveDir = path.join(DATASETS_DIR, 'data')
    const featuresPath = path.join(DATASETS_DIR, 'face_features', 'feature.npz')

    const command = `${pythonExecutable} ${addPersonsScript} --backup-dir "${backupDir}" --add-persons-dir "${NEW_PERSONS_DIR}" --faces-save-dir "${facesSaveDir}" --features-path "${featuresPath}"`

    const { stdout, stderr } = await execAsync(command)
    
    if (stderr) {
      console.error('❌ Python script error:', stderr)
      return NextResponse.json({ 
        success: false, 
        error: 'Error executing Python script',
        details: stderr 
      })
    }

    console.log('✅ Python script output:', stdout)
    console.log('🔵 EXTRACT FEATURES END ----------------\n')

    return NextResponse.json({ 
      success: true,
      details: stdout
    })

  } catch (error) {
    console.error('❌ Error:', error)
    return NextResponse.json({ 
      success: false, 
      error: 'Error in extract features process',
      details: error.message
    })
  }
} 