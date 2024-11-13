"use client"

import { 
  Modal, 
  ModalContent, 
  ModalHeader, 
  ModalBody, 
  ModalFooter,
  Button,
  Input,
  Select,
  SelectItem,
  useDisclosure
} from "@nextui-org/react"
import { Plus } from "lucide-react"
import { useState } from "react"
import { useSupabase } from "@/utils/supabase/client"

const ROOMS = ["LC6", "LP1", "LP2", "LP3", "LP4", "LP5"]

interface AddClassModalProps {
  assignatureName: string
  onClassAdded: () => void
}

export function AddClassModal({ assignatureName, onClassAdded }: AddClassModalProps) {
  const { isOpen, onOpen, onOpenChange } = useDisclosure()
  const { supabase } = useSupabase()
  const [isLoading, setIsLoading] = useState(false)
  const [formData, setFormData] = useState({
    date: "",
    startTime: "",
    endTime: "",
    room: ""
  })

  async function handleSubmit() {
    try {
      setIsLoading(true)

      const { data: assignature } = await supabase
        .from('assignatures')
        .select('id')
        .eq('name', assignatureName)
        .single()

      if (!assignature) throw new Error('Assignature not found')

      // Create ISO strings directly from the local date and time inputs
      const startDateTime = `${formData.date}T${formData.startTime}:00`
      const endDateTime = `${formData.date}T${formData.endTime}:00`

      // Create Date objects in local time
      const startDate = new Date(startDateTime)
      const endDate = new Date(endDateTime)

      // Convert to ISO strings (which will be in UTC)
      const startISO = startDate.toISOString()
      const endISO = endDate.toISOString()

      const { error } = await supabase
        .from('classes')
        .insert({
          assignature_id: assignature.id,
          start_time: startISO,
          end_time: endISO,
          room: formData.room
        })

      if (error) throw error

      setFormData({
        date: "",
        startTime: "",
        endTime: "",
        room: ""
      })
      
      onClassAdded()
      onOpenChange()

    } catch (error) {
      console.error('Error creating class:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      <Button 
        onPress={onOpen}
        color="primary" 
        endContent={<Plus className="w-4 h-4" />}
      >
        Agregar Clase
      </Button>

      <Modal 
        isOpen={isOpen} 
        onOpenChange={onOpenChange}
        placement="center"
        backdrop="blur"
      >
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader className="flex flex-col gap-1">
                Agregar Nueva Clase
              </ModalHeader>
              <ModalBody>
                <Input
                  type="date"
                  label="Fecha"
                  placeholder="Selecciona la fecha"
                  value={formData.date}
                  onChange={(e) => setFormData(prev => ({ ...prev, date: e.target.value }))}
                  isRequired
                />
                <div className="flex gap-2">
                  <Input
                    type="time"
                    label="Hora de inicio"
                    placeholder="Selecciona la hora de inicio"
                    value={formData.startTime}
                    onChange={(e) => setFormData(prev => ({ ...prev, startTime: e.target.value }))}
                    isRequired
                  />
                  <Input
                    type="time"
                    label="Hora de término"
                    placeholder="Selecciona la hora de término"
                    value={formData.endTime}
                    onChange={(e) => setFormData(prev => ({ ...prev, endTime: e.target.value }))}
                    isRequired
                  />
                </div>
                <Select
                  label="Sala"
                  placeholder="Selecciona la sala"
                  selectedKeys={formData.room ? [formData.room] : []}
                  onChange={(e) => setFormData(prev => ({ ...prev, room: e.target.value }))}
                  isRequired
                >
                  {ROOMS.map((room) => (
                    <SelectItem key={room} value={room}>
                      {room}
                    </SelectItem>
                  ))}
                </Select>
              </ModalBody>
              <ModalFooter>
                <Button color="danger" variant="light" onPress={onClose}>
                  Cancelar
                </Button>
                <Button 
                  color="primary" 
                  onPress={handleSubmit}
                  isLoading={isLoading}
                  isDisabled={!formData.date || !formData.startTime || !formData.endTime || !formData.room}
                >
                  Crear Clase
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </>
  )
} 