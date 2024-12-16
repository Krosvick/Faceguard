"use client"

import { useSupabase } from "@/utils/supabase/client"
import { Card, CardBody, CardHeader, Input, Table, TableHeader, TableBody, TableColumn, TableRow, TableCell, Button, Chip } from "@nextui-org/react"
import { useState, useEffect } from "react"
import { Search, Trash2, Plus, Cog } from "lucide-react"
import { uploadStudentImage, formatStudentName } from "@/lib/utils/image-storage"
import Link from "next/link"

interface Student {
  id: number
  first_name: string
  last_name: string
  photos: string[]
}

export default function StudentListPage() {
  const { supabase } = useSupabase()
  const [students, setStudents] = useState<Student[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [search, setSearch] = useState("")
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isExtracting, setIsExtracting] = useState(false)

  useEffect(() => {
    async function fetchStudents() {
      try {
        // Get current university's ID
        const { data: { user } } = await supabase.auth.getUser()
        const { data: universityData } = await supabase
          .from('universities')
          .select('id')
          .eq('auth_user_id', user?.id)
          .single()

        if (!universityData?.id) return

        // Get all students for this university
        const { data: studentsData, error } = await supabase
          .from('students')
          .select('*')
          .eq('university_id', universityData.id)
          .order('first_name', { ascending: true })

        if (error) throw error

        setStudents(studentsData)
      } catch (error) {
        console.error('Error fetching students:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchStudents()
  }, [supabase])

  const handleDeletePhoto = async (studentId: number, photoIndex: number) => {
    try {
      const student = students.find(s => s.id === studentId)
      if (!student) return

      const newPhotos = student.photos.filter((_, index) => index !== photoIndex)

      // Update student record
      const { error } = await supabase
        .from('students')
        .update({ photos: newPhotos })
        .eq('id', studentId)

      if (error) throw error

      // Update UI
      setStudents(prev => prev.map(s => 
        s.id === studentId ? { ...s, photos: newPhotos } : s
      ))

    } catch (error) {
      console.error('Error deleting photo:', error)
    }
  }

  const handleAddPhotos = async (studentId: number, files: FileList) => {
    console.log('🎯 Add Photos Triggered:', {
      studentId,
      numberOfFiles: files.length
    })
    
    try {
      const student = students.find(s => s.id === studentId)
      if (!student) return

      console.log('Adding photos to student:', {
        studentId,
        firstName: student.first_name,
        lastName: student.last_name,
        numberOfFiles: files.length
      })

      setSelectedStudent(student)
      setIsUploading(true)

      // Use the helper function for consistent formatting
      const formattedName = formatStudentName(student.first_name, student.last_name)
      console.log('Formatted student name:', formattedName)

      // Upload all new photos
      const uploadPromises = Array.from(files).map(file => 
        uploadStudentImage(file, formattedName)
      )
      const newPhotoPaths = await Promise.all(uploadPromises)
      console.log('New photos uploaded:', newPhotoPaths)

      // Update student record with combined photos
      const updatedPhotos = [...student.photos, ...newPhotoPaths]
      const { error } = await supabase
        .from('students')
        .update({ photos: updatedPhotos })
        .eq('id', studentId)

      if (error) throw error

      // Update UI
      setStudents(prev => prev.map(s => 
        s.id === studentId ? { ...s, photos: updatedPhotos } : s
      ))

    } catch (error) {
      console.error('Error adding photos:', error)
    } finally {
      setIsUploading(false)
      setSelectedStudent(null)
    }
  }

  const handleExtractFeatures = async () => {
    try {
      setIsExtracting(true)

      const response = await fetch('/api/extract-features', {
        method: 'POST'
      })

      const data = await response.json()

      if (!data.success) {
        throw new Error(data.error || 'Error al extraer características')
      }

      // Show success message (you might want to add a toast notification here)
      alert('Características extraídas exitosamente')

    } catch (error) {
      console.error('Error extracting features:', error)
      alert('Error al extraer características')
    } finally {
      setIsExtracting(false)
    }
  }

  const filteredStudents = students.filter(student => 
    `${student.first_name} ${student.last_name}`.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="container mx-auto p-4 space-y-6">
      <Card>
        <CardHeader className="flex flex-col gap-1 px-6 pt-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold">Estudiantes Registrados</h1>
              <p className="text-sm text-default-500">
                {students.length} estudiantes en total
              </p>
            </div>
            <div className="flex gap-4 items-center">
              <Input
                placeholder="Buscar estudiante..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                startContent={<Search size={18} />}
                className="w-64"
              />
              <Button 
                color="secondary"
                isLoading={isExtracting}
                onPress={handleExtractFeatures}
                startContent={!isExtracting && <Cog size={20} />}
              >
                {isExtracting ? "Creando modelo..." : "Crear Modelo de Reconocimiento"}
              </Button>
              <Link href="/students">
                <Button 
                  color="primary"
                  startContent={<Plus size={20} />}
                >
                  Agregar Estudiantes
                </Button>
              </Link>
            </div>
          </div>
        </CardHeader>
        <CardBody className="px-6 py-4">
          <Table>
            <TableHeader>
              <TableColumn>NOMBRE</TableColumn>
              <TableColumn>FOTOS DE REFERENCIA</TableColumn>
              <TableColumn>ACCIONES</TableColumn>
            </TableHeader>
            <TableBody
              emptyContent={isLoading ? "Cargando..." : "No hay estudiantes registrados"}
              isLoading={isLoading}
            >
              {filteredStudents.map(student => (
                <TableRow key={student.id}>
                  <TableCell>{student.first_name} {student.last_name}</TableCell>
                  <TableCell>
                    <div className="flex gap-2">
                      {student.photos.map((photoPath, index) => (
                        <div key={index} className="relative group">
                          <img
                            src={photoPath}
                            alt={`Foto de ${student.first_name}`}
                            className="w-12 h-12 object-cover rounded"
                          />
                          <Button
                            isIconOnly
                            size="sm"
                            color="danger"
                            variant="flat"
                            onPress={() => handleDeletePhoto(student.id, index)}
                            className="absolute -top-2 -right-2 hidden group-hover:flex"
                          >
                            <Trash2 size={14} />
                          </Button>
                        </div>
                      ))}
                      <label className="flex items-center justify-center w-12 h-12 border-2 border-dashed rounded cursor-pointer hover:border-primary group">
                        <input
                          type="file"
                          multiple
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => {
                            if (e.target.files?.length) {
                              handleAddPhotos(student.id, e.target.files)
                              e.target.value = '' // Reset input
                            }
                          }}
                        />
                        <Plus 
                          size={20} 
                          className="text-default-400 group-hover:text-primary"
                        />
                      </label>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <span className="text-small text-default-500">
                        {student.photos.length} foto{student.photos.length !== 1 ? 's' : ''}
                      </span>
                      {isUploading && selectedStudent?.id === student.id && (
                        <Chip color="warning" size="sm">
                          Subiendo...
                        </Chip>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardBody>
      </Card>
    </div>
  )
} 