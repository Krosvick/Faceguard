"use client"

import { Card, CardBody, Chip } from "@nextui-org/react"
import { useSupabase } from "@/utils/supabase/client"
import { useEffect, useState } from "react"

interface Attendance {
  id: number
  student_name: string
  confidence: number
  quality: number
  timestamp: string
  image_path: string
  room_id: number
}

export interface AttendanceListProps {
  classId: string
}

const AttendanceList = ({ classId }: AttendanceListProps) => {
  const { supabase } = useSupabase()
  const [attendances, setAttendances] = useState<Attendance[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function getAttendance() {
      try {
        setIsLoading(true)
        setError(null)

        const { data: attendanceData, error: attendanceError } = await supabase
          .from('attendance')
          .select('id, student_name, confidence, quality, timestamp, image_path, room_id')
          .eq('room_id', classId)
          .order('timestamp', { ascending: false })

        if (attendanceError) throw attendanceError

        if (attendanceData) {
          setAttendances(attendanceData)
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
    </div>
  )
}

export default AttendanceList 