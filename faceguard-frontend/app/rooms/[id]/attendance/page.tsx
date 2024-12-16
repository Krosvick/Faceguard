"use client"

import { useParams } from "next/navigation"
import { Suspense } from "react"
import { Skeleton } from "@nextui-org/react"
import AttendanceList from "./attendance-list"

export default function RoomAttendancePage() {
  const params = useParams()
  const roomId = params.id as string

  return (
    <div className="min-h-screen bg-default-50">
      <div className="container mx-auto py-8 px-4">
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold mb-2">Registro de Asistencia</h1>
            <p className="text-lg text-default-500">
              Revisa el historial de asistencia de esta sala
            </p>
          </div>

          <Suspense fallback={<Skeleton className="h-[600px] rounded-lg" />}>
            <AttendanceList roomId={roomId} />
          </Suspense>
        </div>
      </div>
    </div>
  )
} 