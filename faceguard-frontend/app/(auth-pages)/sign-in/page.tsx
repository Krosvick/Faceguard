"use client"

import { Card, CardBody, CardHeader } from "@nextui-org/card"
import { Input } from "@nextui-org/input"
import { Button } from "@nextui-org/button"
import { signInAction } from "@/app/actions"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { useState } from "react"

export default function SignIn() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(formData: FormData) {
    try {
      setIsLoading(true)
      setError(null)
      const response = await signInAction(formData)
      
      if ('success' in response && response.success) {
        router.push('/')
        router.refresh()
      } else {
        if (response.error === 'Email not confirmed') {
          setError('Por favor, confirma tu correo electrónico antes de iniciar sesión.')
        } else if (response.error === 'Invalid credentials') {
          setError('Correo electrónico o contraseña incorrectos.')
        } else {
          setError('Error al iniciar sesión. Por favor, inténtalo de nuevo.')
        }
      }
    } catch (error) {
      console.error('Error during sign in:', error)
      setError('Error al iniciar sesión. Por favor, inténtalo de nuevo.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center p-4">
      <Card className="w-[400px]">
        <CardHeader className="flex flex-col gap-1 pb-2 px-6 pt-6">
          <h1 className="text-2xl font-bold">Iniciar Sesión</h1>
          <p className="text-sm text-default-500">
            Ingresa tus credenciales para continuar
          </p>
        </CardHeader>
        <CardBody className="px-6 py-4">
          <form action={handleSubmit} className="flex flex-col gap-4">
            <Input
              name="email"
              label="Correo electrónico"
              labelPlacement="outside"
              placeholder="Ingresa tu correo"
              type="email"
              isRequired
              classNames={{
                label: "pb-1",
                input: "h-12",
                inputWrapper: "h-12"
              }}
            />
            <Input
              name="password"
              label="Contraseña"
              labelPlacement="outside"
              type="password"
              placeholder="Ingresa tu contraseña"
              isRequired
              classNames={{
                label: "pb-1",
                input: "h-12",
                inputWrapper: "h-12"
              }}
            />
            {error && (
              <p className="text-danger text-sm text-center">{error}</p>
            )}
            <div className="flex justify-between items-center mt-2">
              <Link
                href="/forgot-password"
                className="text-sm text-primary hover:underline"
              >
                ¿Olvidaste tu contraseña?
              </Link>
              <Button
                type="submit"
                color="primary"
                isLoading={isLoading}
                className="h-12"
              >
                Iniciar Sesión
              </Button>
            </div>
          </form>
        </CardBody>
      </Card>
      <p className="mt-4 text-center text-sm text-gray-600">
        ¿No tienes una cuenta?{" "}
        <Link href="/sign-up" className="text-primary hover:underline">
          Regístrate
        </Link>
      </p>
    </div>
  )
}
