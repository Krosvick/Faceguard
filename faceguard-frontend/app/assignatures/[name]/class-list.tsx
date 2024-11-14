"use client"

import { Card, CardBody, CardHeader, Button, Chip } from "@nextui-org/react"
import { useSupabase } from "@/utils/supabase/client"
import { useEffect, useState } from "react"
import Link from "next/link"
import { Clock } from "lucide-react"

interface Class {
  id: number
  start_time: string
  end_time: string
  room: string
  assignature_id: number
}

interface ClassListProps {
  assignatureName: string
}

const ClassList = ({ assignatureName }: ClassListProps) => {
  const { supabase } = useSupabase()
  const [classes, setClasses] = useState<Class[]>([])

  useEffect(() => {
    async function getClasses() {
      const { data: assignature } = await supabase
        .from('assignatures')
        .select('id')
        .eq('name', assignatureName)
        .single()

      if (assignature) {
        const { data: classes } = await supabase
          .from('classes')
          .select('*')
          .eq('assignature_id', assignature.id)
          .order('start_time', { ascending: false })

        if (classes) {
          setClasses(classes)
        }
      }
    }

    getClasses()
  }, [assignatureName, supabase])

  function formatDate(dateString: string) {
    const date = new Date(dateString)
    
    return date.toLocaleDateString('es-ES', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    })
  }

  function formatTime(dateString: string) {
    const date = new Date(dateString)
    
    return date.toLocaleTimeString('es-ES', {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  function isClassActive(startTime: string, endTime: string) {
    const now = new Date()
    const start = new Date(startTime)
    const end = new Date(endTime)
    
    return now >= start && now <= end
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {classes.map((classItem) => (
        <Link 
          key={classItem.id} 
          href={`/assignatures/${encodeURIComponent(assignatureName)}/attendance/${classItem.id}`}
        >
          <Card className="w-full hover:scale-105 transition-transform cursor-pointer">
            <CardHeader className="flex flex-col items-start gap-2">
              <div className="flex justify-between w-full items-center">
                <h3 className="text-lg font-semibold">
                  Sala {classItem.room}
                </h3>
                {isClassActive(classItem.start_time, classItem.end_time) && (
                  <Chip color="success" variant="flat">En curso</Chip>
                )}
              </div>
              <p className="text-small text-default-500">
                {formatDate(classItem.start_time)}
              </p>
            </CardHeader>
            <CardBody>
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2 text-default-500">
                  <Clock className="w-4 h-4" />
                  <span>{formatTime(classItem.start_time)} - {formatTime(classItem.end_time)}</span>
                </div>
                <Button color="primary" className="w-full">
                  Ver Asistencia
                </Button>
              </div>
            </CardBody>
          </Card>
        </Link>
      ))}
      {classes.length === 0 && (
        <p className="text-default-500 col-span-full text-center py-8">
          No hay clases registradas para {assignatureName}
        </p>
      )}
    </div>
  )
}

export default ClassList 