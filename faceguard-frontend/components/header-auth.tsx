"use client"

import { Button } from "@nextui-org/react"
import { useSupabase } from "@/utils/supabase/client"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Teacher } from "@/types/teacher"
import { Session } from '@supabase/supabase-js'

interface University {
  id: number
  name: string
  location: string
  auth_user_id: string
}

const HeaderAuth = () => {
  const { supabase } = useSupabase()
  const [session, setSession] = useState<Session | null>(null)
  const [userProfile, setUserProfile] = useState<Teacher | University | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  // Handle session changes
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session: initialSession } }) => {
      setSession(initialSession)
    })

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
    })

    return () => subscription.unsubscribe()
  }, [supabase.auth])

  // Handle user profile data
  useEffect(() => {
    let ignore = false

    async function getUserProfile() {
      try {
        if (!session?.user?.id) {
          if (!ignore) setUserProfile(null)
          return
        }

        // Try to get teacher profile
        const { data: teacherData, error: teacherError } = await supabase
          .from('teachers')
          .select('*')
          .eq('auth_user_id', session.user.id)
          .single()

        if (!teacherError && teacherData) {
          if (!ignore) setUserProfile(teacherData)
          return
        }

        // If not a teacher, try to get university profile
        const { data: universityData, error: universityError } = await supabase
          .from('universities')
          .select('*')
          .eq('auth_user_id', session.user.id)
          .single()

        if (!ignore) {
          if (universityError) {
            console.error('Error fetching user profile:', universityError)
            setUserProfile(null)
          } else {
            setUserProfile(universityData)
          }
        }
      } catch (error) {
        console.error('Error in getUserProfile:', error)
        if (!ignore) setUserProfile(null)
      }
    }

    getUserProfile()

    return () => {
      ignore = true
    }
  }, [session?.user?.id, supabase])

  const handleSignOut = async () => {
    try {
      setIsLoading(true)
      await supabase.auth.signOut()
      setUserProfile(null)
      router.push('/')
      router.refresh()
    } catch (error) {
      console.error('Error signing out:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // Show login/register buttons if no session
  if (!session?.user) {
    return (
      <div className="flex items-center gap-4">
        <Link href="/sign-in">
          <Button color="primary" variant="flat" size="sm">
            Iniciar Sesión
          </Button>
        </Link>
        <Link href="/sign-up">
          <Button color="primary" variant="bordered" size="sm">
            Registrarse
          </Button>
        </Link>
      </div>
    )
  }

  // Show loading state while fetching user data
  if (!userProfile) {
    return (
      <div className="flex items-center gap-4">
        <Button
          color="danger"
          variant="flat"
          size="sm"
          isLoading={isLoading}
          onClick={handleSignOut}
        >
          Cerrar Sesión
        </Button>
      </div>
    )
  }

  // Show full header when we have all data
  return (
    <div className="flex items-center gap-4">
      <span className="text-sm text-default-500">
        Bienvenido, {
          'first_name' in userProfile 
            ? `${userProfile.first_name} ${userProfile.last_name}`
            : userProfile.name
        }
      </span>
      {'first_name' in userProfile ? (
        <div className="flex gap-2">
          <Link href="/assignatures">
            <Button variant="bordered" size="sm">
              Mis Asignaturas
            </Button>
          </Link>
          <Link href="/rooms">
            <Button variant="bordered" size="sm">
              Mis Salas
            </Button>
          </Link>
        </div>
      ) : (
        <div className="flex gap-2">
          <Link href="/students">
            <Button variant="bordered" size="sm">
              Registrar Estudiantes
            </Button>
          </Link>
          <Link href="/students/list">
            <Button variant="bordered" size="sm">
              Ver Estudiantes
            </Button>
          </Link>
          <Link href="/rooms">
            <Button variant="bordered" size="sm">
              Gestionar Salas
            </Button>
          </Link>
        </div>
      )}
      <Button
        color="danger"
        variant="flat"
        size="sm"
        isLoading={isLoading}
        onClick={handleSignOut}
      >
        Cerrar Sesión
      </Button>
    </div>
  )
}

export default HeaderAuth
