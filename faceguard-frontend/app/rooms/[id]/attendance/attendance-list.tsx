"use client"

import { Card, CardBody, Button, Chip } from "@nextui-org/react"
import { useSupabase } from "@/utils/supabase/client"
import { useEffect, useState } from "react"
import { Clock, UserPlus } from "lucide-react"
import AddTeacherModal from "./add-teacher-modal"

interface Room {
  id: number
  room: string
  university_id: number
  teacher_id: number | null
}

interface Teacher {
  id: number
  first_name: string
  last_name: string
  email: string
}

interface Attendance {
  id: number
  student_name: string
  confidence: number
  quality: number
  timestamp: string
  image_path: string
  room_id: number
}

interface AttendanceListProps {
  roomId: string
}

const AttendanceList = ({ roomId }: AttendanceListProps) => {
  const { supabase } = useSupabase()
  const [attendances, setAttendances] = useState<Attendance[]>([])
  const [roomDetails, setRoomDetails] = useState<Room | null>(null)
  const [teacher, setTeacher] = useState<Teacher | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isTeacherModalOpen, setIsTeacherModalOpen] = useState(false)

  const fetchData = async () => {
    try {
      setIsLoading(true)
      setError(null)

      // Get room details with teacher info
      const { data: roomData, error: roomError } = await supabase
        .from('rooms')
        .select('*')
        .eq('id', roomId)
        .single()

      if (roomError) throw roomError

      if (roomData) {
        setRoomDetails(roomData)
        
        // If room has a teacher, get teacher details
        if (roomData.teacher_id) {
          const { data: teacherData, error: teacherError } = await supabase
            .from('teachers')
            .select('*')
            .eq('id', roomData.teacher_id)
            .single()

          if (teacherError) throw teacherError
          setTeacher(teacherData)
        } else {
          setTeacher(null)
        }

        // Get attendance records
        const { data: attendanceData, error: attendanceError } = await supabase
          .from('attendance')
          .select('id, student_name, confidence, quality, timestamp, image_path, room_id')
          .eq('room_id', roomData.id)
          .order('timestamp', { ascending: false })

        if (attendanceError) throw attendanceError

        if (attendanceData) {
          setAttendances(attendanceData)
        }
      }
    } catch (error) {
      console.error('Error fetching data:', error)
      setError('Error al cargar los datos')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [roomId, supabase])

  function formatDateTime(dateString: string) {
    return new Date(dateString).toLocaleString('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const formatStudentName = (name: string) => {
    if (name.startsWith('Desconocido_')) {
      return 'Persona Desconocida'
    }
    return name
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-[200px]">
        <p className="text-xl text-default-500">Cargando asistencia...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex justify-center items-center min-h-[200px]">
        <p className="text-danger">{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {roomDetails && (
        <Card className="bg-white/50 backdrop-blur-sm">
          <CardBody className="p-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-primary/10">
                  <Clock className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold">Sala {roomDetails.room}</h2>
                  <div className="flex items-center gap-2">
                    <p className="text-default-500">
                      {attendances.length} registros de asistencia
                    </p>
                    {teacher && (
                      <Chip size="sm" variant="flat" color="primary">
                        Profesor: {teacher.first_name} {teacher.last_name}
                      </Chip>
                    )}
                  </div>
                </div>
              </div>
              <Button
                color="primary"
                variant="flat"
                startContent={<UserPlus className="w-4 h-4" />}
                onPress={() => setIsTeacherModalOpen(true)}
              >
                {teacher ? "Cambiar Profesor" : "Asignar Profesor"}
              </Button>
            </div>
          </CardBody>
        </Card>
      )}
      
      <div className="grid gap-4">
        {attendances.map((attendance) => (
          <Card key={attendance.id} className="bg-white/50 backdrop-blur-sm">
            <CardBody className="p-6">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div className="space-y-1">
                  <h3 className="text-lg font-semibold">
                    {formatStudentName(attendance.student_name)}
                  </h3>
                  <p className="text-default-500">
                    {formatDateTime(attendance.timestamp)}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Chip 
                    color={attendance.confidence >= 0.8 ? "success" : "warning"} 
                    variant="flat"
                  >
                    Confianza: {Math.round(attendance.confidence * 100)}%
                  </Chip>
                  <Chip 
                    color={attendance.quality >= 0.8 ? "success" : "warning"} 
                    variant="flat"
                  >
                    Calidad: {Math.round(attendance.quality * 100)}%
                  </Chip>
                  {attendance.student_name.startsWith('Desconocido_') && (
                    <Chip
                      color="danger"
                      variant="flat"
                    >
                      No Identificado
                    </Chip>
                  )}
                </div>
              </div>

              {attendance.image_path && (
                <div className="mt-4">
                  <div className="relative w-32 h-32 overflow-hidden rounded-lg">
                    <img 
                      src={attendance.image_path} 
                      alt={`Foto de ${attendance.student_name}`}
                      className="object-cover w-full h-full"
                      loading="lazy"
                    />
                  </div>
                </div>
              )}
            </CardBody>
          </Card>
        ))}

        {attendances.length === 0 && (
          <div className="flex justify-center items-center min-h-[200px]">
            <p className="text-xl text-default-500">
              No hay registros de asistencia para esta sala
            </p>
          </div>
        )}
      </div>

      <AddTeacherModal 
        isOpen={isTeacherModalOpen}
        onClose={() => setIsTeacherModalOpen(false)}
        roomId={roomId}
        currentTeacherId={roomDetails?.teacher_id}
        onAssign={() => {
          // Use the extracted function instead
          fetchData()
          setIsTeacherModalOpen(false)
        }}
      />
    </div>
  )
}

export default AttendanceList 