"use client"

import { useSupabase } from "@/utils/supabase/client"
import { Card, CardBody, CardHeader, Input, Button, Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure } from "@nextui-org/react"
import { useState, useEffect } from "react"
import { Plus, Trash2, Users } from "lucide-react"
import Link from "next/link"

interface Room {
  id: number
  room: string
  university_id: number
}

export default function RoomsPage() {
  const { supabase } = useSupabase()
  const [rooms, setRooms] = useState<Room[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [newRoom, setNewRoom] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const { isOpen, onOpen, onClose } = useDisclosure()

  useEffect(() => {
    async function fetchRooms() {
      try {
        // Get current university's ID
        const { data: { user } } = await supabase.auth.getUser()
        const { data: universityData } = await supabase
          .from('universities')
          .select('id')
          .eq('auth_user_id', user?.id)
          .single()

        if (!universityData?.id) return

        // Get all rooms for this university
        const { data: roomsData, error } = await supabase
          .from('rooms')
          .select('*')
          .eq('university_id', universityData.id)
          .order('room', { ascending: true })

        if (error) throw error

        setRooms(roomsData)
      } catch (error) {
        console.error('Error fetching rooms:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchRooms()
  }, [supabase])

  const handleAddRoom = async () => {
    try {
      setIsSubmitting(true)

      // Get university ID
      const { data: { user } } = await supabase.auth.getUser()
      const { data: universityData } = await supabase
        .from('universities')
        .select('id')
        .eq('auth_user_id', user?.id)
        .single()

      if (!universityData?.id) throw new Error('Universidad no encontrada')

      // Create room
      const { data: newRoomData, error } = await supabase
        .from('rooms')
        .insert({
          room: newRoom,
          university_id: universityData.id
        })
        .select()
        .single()

      if (error) throw error

      // Update UI
      setRooms(prev => [...prev, newRoomData])
      setNewRoom("")
      onClose()

    } catch (error) {
      console.error('Error adding room:', error)
      alert('Error al crear la sala')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDeleteRoom = async (roomId: number) => {
    try {
      const { error } = await supabase
        .from('rooms')
        .delete()
        .eq('id', roomId)

      if (error) throw error

      // Update UI
      setRooms(prev => prev.filter(room => room.id !== roomId))

    } catch (error) {
      console.error('Error deleting room:', error)
      alert('Error al eliminar la sala')
    }
  }

  return (
    <div className="min-h-screen bg-default-50">
      <div className="container mx-auto p-8 space-y-8 max-w-[1400px]">
        {/* Header Section */}
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold">Salas</h1>
          <p className="text-lg text-default-500">
            Gestiona las salas y revisa la asistencia
          </p>
        </div>

        {/* Main Content */}
        <div className="grid gap-8">
          {/* Stats and Actions Card */}
          <Card className="bg-white/50 backdrop-blur-sm">
            <CardBody className="p-8">
              <div className="flex justify-between items-center gap-10">
                <div className="space-y-1">
                  <h2 className="text-xl font-semibold">Resumen</h2>
                  <p className="text-default-500">
                    {rooms.length} {rooms.length === 1 ? 'sala registrada' : 'salas registradas'}
                  </p>
                </div>
                <Button 
                  color="primary"
                  onPress={onOpen}
                  startContent={<Plus size={20} />}
                  size="lg"
                  className="h-14 px-8"
                >
                  Agregar Nueva Sala
                </Button>
              </div>
            </CardBody>
          </Card>

          {/* Rooms Grid */}
          <div className="relative min-h-[400px]">
            {isLoading ? (
              <div className="flex justify-center items-center absolute inset-0">
                <p className="text-xl text-default-500">Cargando salas...</p>
              </div>
            ) : rooms.length === 0 ? (
              <Card className="bg-white/50 backdrop-blur-sm">
                <CardBody>
                  <div className="flex flex-col items-center justify-center py-16 gap-6">
                    <div className="p-6 rounded-full bg-primary/10">
                      <Users size={48} className="text-primary" />
                    </div>
                    <div className="text-center space-y-2">
                      <p className="text-xl text-default-600">No hay salas registradas</p>
                      <p className="text-default-500">Comienza agregando tu primera sala</p>
                    </div>
                    <Button 
                      color="primary" 
                      variant="flat"
                      onPress={onOpen}
                      startContent={<Plus size={20} />}
                      size="lg"
                    >
                      Agregar Primera Sala
                    </Button>
                  </div>
                </CardBody>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {rooms.map(room => (
                  <Card
                    key={room.id}
                    isPressable
                    as={Link}
                    href={`/rooms/${room.id}/attendance`}
                    className="bg-white/50 backdrop-blur-sm hover:bg-white/80 transition-all"
                  >
                    <CardBody className="p-6">
                      <div className="flex flex-col gap-6">
                        <div className="flex items-start justify-between">
                          <div className="p-4 rounded-lg bg-primary/10">
                            <Users size={28} className="text-primary" />
                          </div>
                          <Button
                            isIconOnly
                            color="danger"
                            variant="light"
                            onPress={(e) => {
                              e.preventDefault()
                              handleDeleteRoom(room.id)
                            }}
                            className="opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <Trash2 size={20} />
                          </Button>
                        </div>
                        <div>
                          <h3 className="text-xl font-semibold mb-1">{room.room}</h3>
                          <p className="text-default-500">
                            Click para ver asistencia
                          </p>
                        </div>
                      </div>
                    </CardBody>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Modal */}
      <Modal 
        isOpen={isOpen} 
        onClose={onClose}
        size="lg"
        classNames={{
          base: "max-w-[600px]"
        }}
      >
        <ModalContent>
          <ModalHeader className="flex flex-col gap-1">
            <h2 className="text-xl font-bold">Agregar Nueva Sala</h2>
            <p className="text-sm text-default-500">
              Ingresa los detalles de la nueva sala
            </p>
          </ModalHeader>
          <ModalBody className="py-6">
            <Input
              label="Nombre de la Sala"
              labelPlacement="outside"
              placeholder="Ej: Sala 101"
              value={newRoom}
              onChange={(e) => setNewRoom(e.target.value)}
              size="lg"
              classNames={{
                label: "text-default-700 font-medium",
                input: "h-12"
              }}
            />
          </ModalBody>
          <ModalFooter>
            <Button 
              variant="light" 
              onPress={onClose}
              size="lg"
            >
              Cancelar
            </Button>
            <Button 
              color="primary" 
              onPress={handleAddRoom}
              isLoading={isSubmitting}
              isDisabled={!newRoom.trim()}
              size="lg"
            >
              Agregar Sala
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  )
} 