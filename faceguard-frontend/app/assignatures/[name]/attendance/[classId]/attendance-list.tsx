"use client"

import { Card, CardBody, CardHeader, Chip } from "@nextui-org/react"
import { useSupabase } from "@/utils/supabase/client"
import { useEffect, useState } from "react"
import { Clock } from "lucide-react"

interface Class {
  start_time: string
  end_time: string
  room: string
}

interface Attendance {
  id: number
  student_name: string
  confidence: number
  quality: number
  timestamp: string
  image_path: string
  room: string
  class_id: number
}

interface AttendanceListProps {
  classId: string
}

const AttendanceList = ({ classId }: AttendanceListProps) => {
  const { supabase } = useSupabase()
  const [attendances, setAttendances] = useState<Attendance[]>([])
  const [classDetails, setClassDetails] = useState<Class | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function getAttendance() {
      try {
        setIsLoading(true)
        setError(null)

        const { data: classData, error: classError } = await supabase
          .from('classes')
          .select('*')
          .eq('id', classId)
          .single()

        if (classError) throw classError

        if (classData) {
          setClassDetails(classData)
          
          const { data: attendanceData, error: attendanceError } = await supabase
            .from('attendance')
            .select('*')
            .eq('room', classData.room)
            .eq('class_id', classId)
            .order('timestamp', { ascending: false })

          if (attendanceError) throw attendanceError

          if (attendanceData) {
            setAttendances(attendanceData)
          }
        }
      } catch (error) {
        console.error('Error fetching attendance:', error)
        setError('Error al cargar la asistencia')
      } finally {
        setIsLoading(false)
      }
    }

    getAttendance()
  }, [classId, supabase])

  function formatDateTime(dateString: string) {
    return new Date(dateString).toLocaleString('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  function formatTime(dateString: string) {
    return new Date(dateString).toLocaleTimeString('es-ES', {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-[200px]">
        <p className="text-default-500">Cargando asistencia...</p>
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
      {classDetails && (
        <Card className="w-full bg-default-50">
          <CardBody className="py-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4" />
                <span>
                  {formatTime(classDetails.start_time)} - {formatTime(classDetails.end_time)}
                </span>
              </div>
              <Chip color="primary" variant="flat">
                Sala {classDetails.room}
              </Chip>
            </div>
          </CardBody>
        </Card>
      )}
      
      <div className="grid gap-4">
        {attendances.map((attendance) => (
          <Card key={attendance.id} className="w-full">
            <CardHeader className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <div className="space-y-1">
                <h3 className="text-lg font-semibold">{attendance.student_name}</h3>
                <p className="text-small text-default-500">
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
              </div>
            </CardHeader>
            <CardBody className="pt-2">
              <div className="flex items-center gap-4">
                {attendance.image_path && (
                  <div className="relative w-32 h-32 overflow-hidden rounded-lg">
                    <img 
                      src={attendance.image_path} 
                      alt={`Foto de ${attendance.student_name}`}
                      className="object-cover w-full h-full"
                      loading="lazy"
                    />
                  </div>
                )}
              </div>
            </CardBody>
          </Card>
        ))}
        {attendances.length === 0 && (
          <div className="flex justify-center items-center min-h-[200px]">
            <p className="text-default-500">
              No hay registros de asistencia para esta clase
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default AttendanceList 