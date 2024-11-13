export interface Teacher {
    id: number
    first_name: string
    last_name: string
    email: string
    auth_user_id: string
  }
  
  export interface Assignature {
    id: number
    name: string
    teacher_id: number
  }