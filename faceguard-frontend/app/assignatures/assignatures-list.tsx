"use client"

import { Card, CardBody, CardHeader, CardFooter, Button, Skeleton, Chip } from "@nextui-org/react"
import { useSupabase } from "@/utils/supabase/client"
import { useEffect, useState } from "react"
import Link from "next/link"
import { Book, Users, MapPin } from "lucide-react"

interface Class {
  start_time: string
  room: string
  attendance: { id: number }[]
}

interface Assignature {
  id: number
  name: string
  teacher_id: number
  classes: Class[]
}

interface ProcessedAssignature extends Assignature {
  _count: {
    classes: number
    students: number
  }
  latest_class?: Class
  rooms: string[]
}

const AssignaturesList = () => {
  const { supabase, session } = useSupabase()
  const [assignatures, setAssignatures] = useState<ProcessedAssignature[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function getAssignatures() {
      try {
        setIsLoading(true)
        setError(null)

        if (session?.user?.id) {
          const { data: teacher, error: teacherError } = await supabase
            .from('teachers')
            .select('id')
            .eq('auth_user_id', session.user.id)
            .single()

          if (teacherError) throw teacherError

          if (teacher) {
            const { data, error: assignaturesError } = await supabase
              .from('assignatures')
              .select(`
                *,
                classes (
                  start_time,
                  room,
                  attendance (
                    id
                  )
                )
              `)
              .eq('teacher_id', teacher.id)
              .order('name')

            if (assignaturesError) throw assignaturesError

            if (data) {
              const processedData = data.map(assignature => {
                const totalAttendance = assignature.classes?.reduce(
                  (sum: number, classItem: Class) => sum + (classItem.attendance?.length || 0), 
                  0
                ) || 0

                // Convert Set to Array to ensure compatibility
                const uniqueRooms = Array.from(new Set(assignature.classes?.map((c: Class) => c.room) || []))

                return {
                  ...assignature,
                  _count: {
                    classes: assignature.classes?.length || 0,
                    students: totalAttendance
                  },
                  latest_class: assignature.classes?.sort((a: Class, b: Class) => 
                    new Date(b.start_time).getTime() - new Date(a.start_time).getTime()
                  )[0],
                  rooms: uniqueRooms
                }
              })
              setAssignatures(processedData)
            }
          }
        }
      } catch (error) {
        console.error('Error fetching assignatures:', error)
        setError('Error al cargar las asignaturas')
      } finally {
        setIsLoading(false)
      }
    }

    getAssignatures()
  }, [session, supabase])

  function formatDate(dateString: string | undefined) {
    if (!dateString) return 'Sin clases'
    return new Date(dateString).toLocaleDateString('es-ES', {
      weekday: 'long',
      day: 'numeric',
      month: 'long'
    })
  }

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {[...Array(3)].map((_, i) => (
          <Card key={i} className="w-full hover:scale-[1.02] transition-transform">
            <CardHeader className="flex gap-3">
              <div className="w-full flex flex-col gap-2">
                <Skeleton className="w-4/5 rounded-lg">
                  <div className="h-6 rounded-lg bg-default-200"></div>
                </Skeleton>
                <Skeleton className="w-full rounded-lg">
                  <div className="h-4 rounded-lg bg-default-100"></div>
                </Skeleton>
              </div>
            </CardHeader>
            <CardBody className="py-2">
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2">
                  <Skeleton className="rounded-full w-4 h-4" />
                  <Skeleton className="w-24 rounded-lg">
                    <div className="h-4 rounded-lg bg-default-200"></div>
                  </Skeleton>
                </div>
                <div className="flex items-center gap-2">
                  <Skeleton className="rounded-full w-4 h-4" />
                  <Skeleton className="w-32 rounded-lg">
                    <div className="h-4 rounded-lg bg-default-200"></div>
                  </Skeleton>
                </div>
                <div className="flex gap-2">
                  <Skeleton className="w-16 rounded-lg">
                    <div className="h-6 rounded-lg bg-default-200"></div>
                  </Skeleton>
                  <Skeleton className="w-16 rounded-lg">
                    <div className="h-6 rounded-lg bg-default-200"></div>
                  </Skeleton>
                </div>
              </div>
            </CardBody>
            <CardFooter>
              <Skeleton className="w-full rounded-lg">
                <div className="h-10 rounded-lg bg-default-300"></div>
              </Skeleton>
            </CardFooter>
          </Card>
        ))}
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
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {assignatures.map((assignature) => (
        <Card key={assignature.id} className="hover:scale-[1.02] transition-transform">
          <CardHeader className="flex gap-3">
            <div className="flex flex-col">
              <p className="text-xl font-semibold">{assignature.name}</p>
              <p className="text-small text-default-500">
                {assignature.latest_class ? 
                  `Última clase: ${formatDate(assignature.latest_class.start_time)}` : 
                  'Sin clases programadas'}
              </p>
            </div>
          </CardHeader>
          <CardBody className="py-2">
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2 text-default-500">
                <Book className="w-4 h-4" />
                <span>{assignature._count.classes} clases</span>
              </div>
              <div className="flex items-center gap-2 text-default-500">
                <Users className="w-4 h-4" />
                <span>{assignature._count.students} registros de asistencia</span>
              </div>
              {assignature.rooms.length > 0 && (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2 text-default-500">
                    <MapPin className="w-4 h-4" />
                    <span>Salas:</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {assignature.rooms.map((room) => (
                      <Chip 
                        key={room} 
                        size="sm" 
                        variant="flat" 
                        color="primary"
                      >
                        {room}
                      </Chip>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardBody>
          <CardFooter>
            <Link 
              href={`/assignatures/${encodeURIComponent(assignature.name)}`}
              className="w-full"
            >
              <Button className="w-full" color="primary">
                Ver Detalles
              </Button>
            </Link>
          </CardFooter>
        </Card>
      ))}
      {assignatures.length === 0 && (
        <div className="col-span-full flex justify-center items-center min-h-[200px]">
          <p className="text-default-500">
            No hay asignaturas asignadas
          </p>
        </div>
      )}
    </div>
  )
}

export default AssignaturesList