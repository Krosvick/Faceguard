import fs from 'fs'
import path from 'path'

const ROOT_DIR = process.cwd()
const BACKEND_DIR = path.join(ROOT_DIR, '..', 'backend')
const DATASETS_DIR = path.join(BACKEND_DIR, 'face-recognition', 'datasets')

const directories = {
  public: path.join(ROOT_DIR, 'public', 'uploads', 'students'),
  newPersons: path.join(DATASETS_DIR, 'new_persons'),
  backup: path.join(DATASETS_DIR, 'backup'),
  data: path.join(DATASETS_DIR, 'data'),
  faceFeatures: path.join(DATASETS_DIR, 'face_features')
}

console.log('\n🔍 Verifying directory structure...\n')

Object.entries(directories).forEach(([name, dir]) => {
  const exists = fs.existsSync(dir)
  console.log(`${exists ? '✅' : '❌'} ${name}:`, dir)
  
  if (!exists) {
    try {
      fs.mkdirSync(dir, { recursive: true })
      console.log(`  📁 Created directory`)
    } catch (error) {
      console.error(`  ❌ Error creating directory:`, error)
    }
  }
})

console.log('\n✨ Directory verification complete\n') 