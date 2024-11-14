"use client"

import { Button } from "@nextui-org/react"
import { useSupabase } from "@/utils/supabase/client"
import { useEffect, useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import Link from "next/link"
import { Teacher } from "@/types/teacher"
import { Session } from '@supabase/supabase-js'


const HeaderAuth = () => {
  const { supabase } = useSupabase()
  const [session, setSession] = useState<Session | null>(null)
  const [teacher, setTeacher] = useState<Teacher | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()
  const pathname = usePathname()

  // Handle session changes
  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session: initialSession } }) => {
      setSession(initialSession)
    })

    // Listen for session changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
    })

    return () => subscription.unsubscribe()
  }, [supabase.auth])

  // Handle teacher data
  useEffect(() => {
    let ignore = false

    async function getTeacherData() {
      try {
        if (!session?.user?.id) {
          if (!ignore) setTeacher(null)
          return
        }

        const { data, error } = await supabase
          .from('teachers')
          .select('*')
          .eq('auth_user_id', session.user.id)
          .single()

        if (!ignore) {
          if (error) {
            console.error('Error fetching teacher:', error)
            setTeacher(null)
          } else {
            setTeacher(data)
          }
        }
      } catch (error) {
        console.error('Error in getTeacherData:', error)
        if (!ignore) setTeacher(null)
      }
    }

    getTeacherData()

    return () => {
      ignore = true
    }
  }, [session?.user?.id, supabase])

  const handleSignOut = async () => {
    try {
      setIsLoading(true)
      await supabase.auth.signOut()
      setTeacher(null)
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

  // Show loading state while fetching teacher data
  if (!teacher) {
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
        Bienvenido, {teacher.first_name} {teacher.last_name}
      </span>
      <Link href="/assignatures">
        <Button variant="bordered" size="sm">
          Mis Asignaturas
        </Button>
      </Link>
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
