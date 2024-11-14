"use client"

import { useParams } from "next/navigation"
import { Suspense } from "react"
import { Skeleton } from "@nextui-org/react"
import AttendanceList from "./attendance-list"

export default function AttendanceDetailPage() {
  const params = useParams()
  const classId = params.classId as string
  const assignatureName = decodeURIComponent(params.name as string)

  return (
    <div className="container py-8">
      <h1 className="text-2xl font-bold mb-2">Asistencia - {assignatureName}</h1>
      <p className="text-default-500 mb-6">Registro de asistencia de la clase</p>
      <Suspense fallback={<Skeleton className="h-48 rounded-lg w-full" />}>
        <AttendanceList classId={classId} />
      </Suspense>
    </div>
  )
} 