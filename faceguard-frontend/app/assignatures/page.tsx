"use client"

import { Suspense } from "react"
import AssignaturesList from "./assignatures-list"
import { Skeleton } from "@nextui-org/react"

export default function AssignaturesPage() {
  return (
    <div className="container py-8">
      <h1 className="text-2xl font-bold mb-6">Mis Asignaturas</h1>
      <Suspense fallback={<Skeleton className="h-48 rounded-lg w-full" />}>
        <AssignaturesList />
      </Suspense>
    </div>
  )
}