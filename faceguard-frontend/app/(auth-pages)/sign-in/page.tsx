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

  async function handleSubmit(formData: FormData) {
    try {
      setIsLoading(true)
      const response = await signInAction(formData)
      
      if ('success' in response && response.success) {
        // If sign in was successful, redirect to home
        router.push('/')
        router.refresh()
      } else {
        // Handle error case (the response will be a redirect with error message)
        console.error('Sign in failed:', response)
      }
    } catch (error) {
      console.error('Error during sign in:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-cente">
      <Card className="w-[400px]">
        <CardHeader className="flex flex-col gap-1 px-6">
          <h1 className="text-xl font-bold">Iniciar Sesión</h1>
          <p className="text-default-500 text-sm">
            Ingresa tus credenciales para continuar
          </p>
        </CardHeader>
        <CardBody>
          <form action={handleSubmit} className="flex flex-col gap-3">
            <Input
              name="email"
              label="Email"
              type="email"
              placeholder="Ingresa tu email"
              isRequired
            />
            <Input
              name="password"
              label="Contraseña"
              type="password"
              placeholder="Ingresa tu contraseña"
              isRequired
            />
            <div className="flex justify-between items-center">
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
