import { type NextRequest } from "next/server";
import { updateSession } from "@/utils/supabase/middleware";

export async function middleware(request: NextRequest) {
  const response = await updateSession(request);
  
  // Get the pathname
  const pathname = request.nextUrl.pathname;
  
  // Add auth routes to this array
  const authRoutes = ['/sign-in', '/sign-up', '/forgot-password'];
  
  // Check if it's an auth route
  const isAuthRoute = authRoutes.includes(pathname);
  
  // Get user from response headers
  const hasUser = response.headers.get('x-user-authenticated') === 'true';
  
  // Only redirect if trying to access auth pages while logged in
  if (hasUser && isAuthRoute) {
    return Response.redirect(new URL('/', request.url));
  }
  
  // Only redirect if trying to access protected routes while logged out
  if (!hasUser && pathname.startsWith('/assignatures')) {
    return Response.redirect(new URL('/sign-in', request.url));
  }
  
  return response;
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
