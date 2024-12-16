"use client";

import { FormMessage, Message } from "@/components/form-message";
import {
  Input,
  Select,
  SelectItem,
  Button,
  Link,
  Card,
  CardBody,
  CardHeader,
} from "@nextui-org/react";
import { useState } from "react";
import { useSupabase } from "@/utils/supabase/client";
import { useRouter } from "next/navigation";

export default function SignUpPage() {
  const router = useRouter();
  const { supabase } = useSupabase();
  const [rol, setRol] = useState<"universidad" | "profesor">("universidad");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [userEmail, setUserEmail] = useState<string>("");

  const handleSignUp = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    const formData = new FormData(e.currentTarget);
    const email = formData.get('email') as string;
    const password = formData.get('password') as string;

    try {
      // Create auth user
      const { data: authData, error: authError } = await supabase.auth.signUp({
        email,
        password,
      });

      if (authError) throw authError;
      if (!authData.user?.id) throw new Error('No se pudo crear el usuario');

      if (rol === "profesor") {
        // Create teacher record
        const { error: teacherError } = await supabase
          .from('teachers')
          .insert({
            first_name: formData.get('firstName') as string,
            last_name: formData.get('lastName') as string,
            email: email,
            auth_user_id: authData.user.id
          });

        if (teacherError) {
          console.error('Error creating teacher:', teacherError);
          throw new Error('Error al crear el profesor');
        }
      } else {
        // Create university record
        const { error: universityError } = await supabase
          .from('universities')
          .insert({
            name: formData.get('universityName') as string,
            location: formData.get('location') as string,
            auth_user_id: authData.user.id
          })
          .select()
          .single();

        if (universityError) {
          console.error('Error creating university:', universityError);
          throw new Error('Error al crear la universidad');
        }
      }

      // Show confirmation message instead of redirecting
      setUserEmail(email);
      setShowConfirmation(true);

    } catch (error) {
      console.error('Error signing up:', error);
      setError('Error al crear la cuenta. Por favor, inténtalo de nuevo.');
    } finally {
      setIsLoading(false);
    }
  };

  if (showConfirmation) {
    return (
      <div className="flex justify-center items-center p-4">
        <Card className="w-[400px]">
          <CardHeader className="flex flex-col gap-1 pb-2 px-6 pt-6">
            <h1 className="text-2xl font-bold">¡Registro Exitoso!</h1>
          </CardHeader>
          <CardBody className="px-6 py-4">
            <div className="flex flex-col gap-4 text-center">
              <p>
                Hemos enviado un correo de confirmación a:
                <br />
                <strong>{userEmail}</strong>
              </p>
              <p>
                Por favor, revisa tu bandeja de entrada y sigue las instrucciones
                para confirmar tu cuenta.
              </p>
              <p className="text-sm text-default-500">
                Una vez que confirmes tu correo, podrás{" "}
                <Link href="/sign-in" color="primary">
                  iniciar sesión
                </Link>
              </p>
            </div>
          </CardBody>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex justify-center items-center p-4">
      <Card className="w-[400px]">
        <CardHeader className="flex flex-col gap-1 pb-2 px-6 pt-6">
          <h1 className="text-2xl font-bold">Registro</h1>
          <p className="text-sm text-default-500">
            ¿Ya tienes una cuenta?{" "}
            <Link href="/sign-in" color="primary">
              Iniciar sesión
            </Link>
          </p>
        </CardHeader>
        <CardBody className="px-6 py-4">
          <form onSubmit={handleSignUp} className="flex flex-col gap-4">
            <Select 
              label="Rol"
              labelPlacement="outside"
              placeholder="Selecciona tu rol"
              name="role"
              defaultSelectedKeys={["universidad"]}
              onChange={(e) => setRol(e.target.value as "universidad" | "profesor")}
              classNames={{
                trigger: "h-12",
                label: "pb-1",
              }}
            >
              <SelectItem key="universidad" value="universidad">
                Universidad
              </SelectItem>
              <SelectItem key="profesor" value="profesor">
                Profesor
              </SelectItem>
            </Select>

            <Input
              label="Correo electrónico"
              labelPlacement="outside"
              name="email"
              placeholder="tu@ejemplo.com"
              type="email"
              isRequired
              classNames={{
                label: "pb-1",
                input: "h-12",
              }}
            />

            <Input
              label="Contraseña"
              labelPlacement="outside"
              name="password"
              type="password"
              placeholder="Ingresa tu contraseña"
              isRequired
              minLength={6}
              classNames={{
                label: "pb-1",
                input: "h-12",
              }}
            />

            {rol === "universidad" ? (
              <>
                <Input
                  label="Nombre de la Universidad"
                  labelPlacement="outside"
                  name="universityName"
                  placeholder="Ingresa el nombre de la universidad"
                  isRequired
                  classNames={{
                    label: "pb-1",
                    input: "h-12",
                  }}
                />
                <Input
                  label="Ubicación"
                  labelPlacement="outside"
                  name="location"
                  placeholder="Ciudad, País"
                  isRequired
                  classNames={{
                    label: "pb-1",
                    input: "h-12",
                  }}
                />
              </>
            ) : (
              <>
                <Input
                  label="Nombre"
                  labelPlacement="outside"
                  name="firstName"
                  placeholder="Ingresa tu nombre"
                  isRequired
                  classNames={{
                    label: "pb-1",
                    input: "h-12",
                  }}
                />
                <Input
                  label="Apellido"
                  labelPlacement="outside"
                  name="lastName"
                  placeholder="Ingresa tu apellido"
                  isRequired
                  classNames={{
                    label: "pb-1",
                    input: "h-12",
                  }}
                />
              </>
            )}

            <Button
              type="submit"
              color="primary"
              isLoading={isLoading}
              className="h-12 mt-2"
            >
              Registrarse
            </Button>

            {error && (
              <p className="text-danger text-sm text-center">{error}</p>
            )}
          </form>
        </CardBody>
      </Card>
    </div>
  );
}
