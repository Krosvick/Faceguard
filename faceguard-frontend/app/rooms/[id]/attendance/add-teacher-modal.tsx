"use client"

import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, Button, Select, SelectItem } from "@nextui-org/react"
import { useSupabase } from "@/utils/supabase/client"
import { useEffect, useState } from "react"

interface Teacher {
  id: number
  first_name: string
  last_name: string
  email: string
}

interface AddTeacherModalProps {
  isOpen: boolean
  onClose: () => void
  roomId: string
  currentTeacherId?: number | null
  onAssign: () => void
}

export default function AddTeacherModal({
  isOpen,
  onClose,
  roomId,
  currentTeacherId,
  onAssign
}: AddTeacherModalProps) {
  const { supabase } = useSupabase()
  const [teachers, setTeachers] = useState<Teacher[]>([])
  const [selectedTeacher, setSelectedTeacher] = useState<Set<string>>(new Set([]))
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    async function loadTeachers() {
      const { data, error } = await supabase
        .from('teachers')
        .select('*')
        .order('first_name')

      if (!error && data) {
        setTeachers(data)
      }
    }

    if (isOpen) {
      loadTeachers()
      if (currentTeacherId) {
        setSelectedTeacher(new Set([currentTeacherId.toString()]))
      } else {
        setSelectedTeacher(new Set([]))
      }
    }
  }, [isOpen, currentTeacherId, supabase])

  const handleAssign = async () => {
    try {
      setIsLoading(true)
      const selectedId = Array.from(selectedTeacher)[0]
      const { error } = await supabase
        .from('rooms')
        .update({ teacher_id: selectedId ? parseInt(selectedId) : null })
        .eq('id', roomId)

      if (error) throw error

      onAssign()
    } catch (error) {
      console.error('Error assigning teacher:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <ModalContent>
        <ModalHeader>Asignar Profesor</ModalHeader>
        <ModalBody>
          <Select
            label="Profesor"
            placeholder="Selecciona un profesor"
            selectedKeys={selectedTeacher}
            onSelectionChange={setSelectedTeacher}
            className="max-w-xs"
          >
            {teachers.map((teacher) => (
              <SelectItem key={teacher.id.toString()} textValue={`${teacher.first_name} ${teacher.last_name}`}>
                {teacher.first_name} {teacher.last_name}
              </SelectItem>
            ))}
          </Select>
        </ModalBody>
        <ModalFooter>
          <Button variant="light" onPress={onClose}>
            Cancelar
          </Button>
          <Button 
            color="primary" 
            onPress={handleAssign}
            isLoading={isLoading}
          >
            Asignar
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
} 