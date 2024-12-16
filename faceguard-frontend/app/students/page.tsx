"use client"

import { useSupabase } from "@/utils/supabase/client"
import { Card, CardBody, CardHeader, Button, Input, Table, TableHeader, TableBody, TableColumn, TableRow, TableCell, Chip } from "@nextui-org/react"
import { useState } from "react"
import { uploadStudentImage, formatStudentName } from "@/lib/utils/image-storage"
import { Plus, X } from "lucide-react"

interface StudentPhoto {
  id: string
  file: File
}

interface StudentUpload {
  id: string
  name: string
  photos: StudentPhoto[]
  status: 'pending' | 'uploading' | 'success' | 'error'
  error?: string
}

export default function StudentsPage() {
  const { supabase } = useSupabase()
  const [students, setStudents] = useState<StudentUpload[]>([])
  const [isUploading, setIsUploading] = useState(false)

  const handleAddStudent = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const form = e.currentTarget
    const formData = new FormData(form)
    const name = formData.get('name') as string
    const files = Array.from(form.photo.files)

    if (!name || files.length === 0) return

    // Validate files are images
    for (const file of files) {
      if (!file.type.startsWith('image/')) {
        alert('Por favor, selecciona solo archivos de imagen válidos')
        return
      }
    }

    const photos = files.map(file => ({
      id: Math.random().toString(36).substring(2),
      file
    }))

    setStudents(prev => [...prev, {
      id: Math.random().toString(36).substring(2),
      name,
      photos,
      status: 'pending'
    }])

    form.reset()
  }

  const handleRemovePhoto = (studentId: string, photoId: string) => {
    setStudents(prev => prev.map(student => {
      if (student.id !== studentId) return student
      return {
        ...student,
        photos: student.photos.filter(photo => photo.id !== photoId)
      }
    }))
  }

  const handleUploadAll = async () => {
    if (isUploading) return
    setIsUploading(true)

    try {
      // Get university ID
      const { data: { user } } = await supabase.auth.getUser()
      const { data: universityData } = await supabase
        .from('universities')
        .select('id')
        .eq('auth_user_id', user?.id)
        .single()

      if (!universityData?.id) {
        throw new Error('Universidad no encontrada')
      }

      // Upload students one by one
      for (const student of students) {
        if (student.status === 'success') continue

        try {
          setStudents(prev => prev.map(s => 
            s.id === student.id ? { ...s, status: 'uploading' } : s
          ))

          // Split name and format consistently
          const nameParts = student.name.split(' ')
          const firstName = nameParts[0]
          const lastName = nameParts.slice(1).join(' ')
          const formattedName = formatStudentName(firstName, lastName)

          console.log('Uploading photos for student:', {
            originalName: student.name,
            formattedName,
            numberOfPhotos: student.photos.length
          })

          // Upload all photos
          const uploadPromises = student.photos.map(photo => 
            uploadStudentImage(photo.file, formattedName)
          )
          const photoPaths = await Promise.all(uploadPromises)
          
          console.log('Photos uploaded successfully:', {
            formattedName,
            paths: photoPaths
          })

          // Create student record with photo paths
          console.log('Creating student record:', {
            firstName,
            lastName,
            photoPaths
          })

          const { error: studentError } = await supabase
            .from('students')
            .insert({
              university_id: universityData.id,
              first_name: firstName,
              last_name: lastName,
              photos: photoPaths
            })

          if (studentError) throw studentError

          setStudents(prev => prev.map(s => 
            s.id === student.id ? { ...s, status: 'success' } : s
          ))
        } catch (error) {
          console.error('Error uploading student:', error)
          setStudents(prev => prev.map(s => 
            s.id === student.id ? { ...s, status: 'error', error: 'Error al subir estudiante' } : s
          ))
        }
      }
    } catch (error) {
      console.error('Error in upload process:', error)
    } finally {
      setIsUploading(false)
    }
  }

  const handleRemoveStudent = (id: string) => {
    setStudents(prev => prev.filter(s => s.id !== id))
  }

  const pendingStudents = students.filter(s => s.status === 'pending' || s.status === 'error')

  return (
    <div className="container mx-auto p-4 space-y-6">
      <Card>
        <CardHeader className="flex flex-col gap-1 px-6 pt-6">
          <h1 className="text-2xl font-bold">Agregar Estudiantes</h1>
          <p className="text-sm text-default-500">
            Agrega los estudiantes que deseas registrar
          </p>
        </CardHeader>
        <CardBody className="px-6 py-4">
          <form onSubmit={handleAddStudent} className="flex gap-4">
            <Input
              name="name"
              label="Nombre del Estudiante"
              labelPlacement="outside"
              placeholder="Ingresa el nombre completo"
              required
              classNames={{
                label: "pb-1",
                inputWrapper: "h-12",
              }}
            />
            <Input
              type="file"
              name="photo"
              label="Fotos"
              labelPlacement="outside"
              accept="image/*"
              required
              multiple
              classNames={{
                label: "pb-1",
                inputWrapper: "h-12",
              }}
            />
            <Button
              type="submit"
              color="primary"
              className="h-12 px-8 self-end"
              startContent={<Plus size={20} />}
            >
              Agregar
            </Button>
          </form>
        </CardBody>
      </Card>

      {students.length > 0 && (
        <Card>
          <CardHeader className="flex justify-between items-center px-6 pt-6">
            <div>
              <h2 className="text-xl font-bold">Lista de Estudiantes</h2>
              <p className="text-sm text-default-500">
                {pendingStudents.length} estudiantes pendientes
              </p>
            </div>
            <Button
              color="primary"
              isLoading={isUploading}
              onPress={handleUploadAll}
              isDisabled={pendingStudents.length === 0}
            >
              Subir Todos
            </Button>
          </CardHeader>
          <CardBody className="px-6 py-4">
            <Table>
              <TableHeader>
                <TableColumn>NOMBRE</TableColumn>
                <TableColumn>FOTOS</TableColumn>
                <TableColumn>ESTADO</TableColumn>
                <TableColumn>ACCIONES</TableColumn>
              </TableHeader>
              <TableBody>
                {students.map(student => (
                  <TableRow key={student.id}>
                    <TableCell>{student.name}</TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-2">
                        {student.photos.map(photo => (
                          <div key={photo.id} className="relative group">
                            <img
                              src={URL.createObjectURL(photo.file)}
                              alt={`Foto de ${student.name}`}
                              className="w-12 h-12 object-cover rounded"
                            />
                            {student.status !== 'success' && (
                              <button
                                onClick={() => handleRemovePhoto(student.id, photo.id)}
                                className="absolute -top-2 -right-2 bg-danger-100 text-danger rounded-full p-0.5 hidden group-hover:block"
                              >
                                <X size={14} />
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Chip
                        color={
                          student.status === 'success' ? "success" :
                          student.status === 'error' ? "danger" :
                          student.status === 'uploading' ? "warning" :
                          "default"
                        }
                        variant="flat"
                        size="sm"
                      >
                        {
                          student.status === 'success' ? "Completado" :
                          student.status === 'error' ? "Error" :
                          student.status === 'uploading' ? "Subiendo..." :
                          "Pendiente"
                        }
                      </Chip>
                    </TableCell>
                    <TableCell>
                      {student.status !== 'success' && (
                        <Button
                          color="danger"
                          variant="light"
                          size="sm"
                          onPress={() => handleRemoveStudent(student.id)}
                        >
                          Eliminar
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardBody>
        </Card>
      )}
    </div>
  )
} 