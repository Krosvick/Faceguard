"use client"

import { useParams } from "next/navigation"
import { Suspense } from "react"
import { Skeleton } from "@nextui-org/react"
import ClassList from "./class-list"
import { AddClassModal } from "@/components/class/add-class-modal"

export default function AssignatureDetailPage() {
  const params = useParams()
  const assignatureName = decodeURIComponent(params.name as string)

  return (
    <div className="container py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">{assignatureName}</h1>
        <AddClassModal 
          assignatureName={assignatureName}
          onClassAdded={() => {
            // This will trigger a refresh of the class list
            window.location.reload()
          }}
        />
      </div>
      <p className="text-default-500 mb-6">Listado de clases y asistencia</p>
      <Suspense fallback={<Skeleton className="h-48 rounded-lg w-full" />}>
        <ClassList assignatureName={assignatureName} />
      </Suspense>
    </div>
  )
} 