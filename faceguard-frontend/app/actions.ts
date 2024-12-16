"use server";

import { encodedRedirect } from "@/utils/utils";
import { createClient } from "@/utils/supabase/server";
import { headers } from "next/headers";
import { redirect } from "next/navigation";

export const signUpAction = async (formData: FormData) => {
  const email = formData.get("email")?.toString();
  const password = formData.get("password")?.toString();
  const role = formData.get("role")?.toString();
  
  const supabase = await createClient();
  const origin = (await headers()).get("origin");

  if (!email || !password || !role) {
    return encodedRedirect("error", "/sign-up", "Todos los campos son requeridos");
  }

  // Primero, crear el usuario de autenticación
  const { data: authData, error: authError } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: `${origin}/auth/callback`,
      data: {
        role: role
      }
    },
  });

  if (authError) {
    console.error(authError.code + " " + authError.message);
    return encodedRedirect("error", "/sign-up", "Error al crear la cuenta");
  }

  if (authData.user) {
    if (role === "universidad") {
      const universityName = formData.get("universityName")?.toString();
      const location = formData.get("location")?.toString();

      if (!universityName || !location) {
        return encodedRedirect("error", "/sign-up", "Todos los campos son requeridos");
      }

      const { error: dbError } = await supabase
        .from('universities')
        .insert([
          {
            name: universityName,
            location: location,
            email: email,
            auth_user_id: authData.user.id
          }
        ]);

      if (dbError) {
        console.error(dbError);
        return encodedRedirect("error", "/sign-up", "Error al crear el perfil de universidad");
      }
    } else if (role === "profesor") {
      const firstName = formData.get("firstName")?.toString();
      const lastName = formData.get("lastName")?.toString();

      if (!firstName || !lastName) {
        return encodedRedirect("error", "/sign-up", "Todos los campos son requeridos");
      }

      const { error: dbError } = await supabase
        .from('teachers')
        .insert([
          {
            first_name: firstName,
            last_name: lastName,
            email: email,
            auth_user_id: authData.user.id
          }
        ]);

      if (dbError) {
        console.error(dbError);
        return encodedRedirect("error", "/sign-up", "Error al crear el perfil de profesor");
      }
    }
  }

  return encodedRedirect(
    "success",
    "/sign-up",
    "¡Gracias por registrarte! Por favor, revisa tu correo electrónico para verificar tu cuenta.",
  );
};

export const signInAction = async (formData: FormData) => {
  const email = formData.get('email') as string
  const password = formData.get('password') as string
  const supabase = await createClient();

  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  })

  if (error) {
    if (error.message.includes('Email not confirmed')) {
      return { error: 'Email not confirmed' }
    }
    if (error.message.includes('Invalid login credentials')) {
      return { error: 'Invalid credentials' }
    }
    return { error: error.message }
  }

  return { success: true }
}

export const forgotPasswordAction = async (formData: FormData) => {
  const email = formData.get("email")?.toString();
  const supabase = await createClient();
  const origin = (await headers()).get("origin");
  const callbackUrl = formData.get("callbackUrl")?.toString();

  if (!email) {
    return encodedRedirect("error", "/forgot-password", "Email is required");
  }

  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${origin}/auth/callback?redirect_to=/reset-password`,
  });

  if (error) {
    console.error(error.message);
    return encodedRedirect(
      "error",
      "/forgot-password",
      "Could not reset password",
    );
  }

  if (callbackUrl) {
    return redirect(callbackUrl);
  }

  return encodedRedirect(
    "success",
    "/forgot-password",
    "Check your email for a link to reset your password.",
  );
};

export const resetPasswordAction = async (formData: FormData) => {
  const supabase = await createClient();

  const password = formData.get("password") as string;
  const confirmPassword = formData.get("confirmPassword") as string;

  if (!password || !confirmPassword) {
    encodedRedirect(
      "error",
      "/reset-password",
      "Password and confirm password are required",
    );
  }

  if (password !== confirmPassword) {
    encodedRedirect(
      "error",
      "/reset-password",
      "Passwords do not match",
    );
  }

  const { error } = await supabase.auth.updateUser({
    password: password,
  });

  if (error) {
    encodedRedirect(
      "error",
      "/reset-password",
      "Password update failed",
    );
  }

  encodedRedirect("success", "/reset-password", "Password updated");
};

export const signOutAction = async () => {
  const supabase = await createClient();
  await supabase.auth.signOut();
  return redirect("/");
};
