-- Table to store university information
CREATE TABLE universities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    auth_user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL
);

-- Table to store teacher information
CREATE TABLE teachers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    auth_user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE
);

-- Table to store class information
CREATE TABLE rooms (
    id SERIAL PRIMARY KEY,
    teacher_id INT REFERENCES teachers(id),
    university_id INT REFERENCES universities(id),
    room VARCHAR(10) NOT NULL
);

-- Table to store students information
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    university_id INT REFERENCES universities(id),
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    photos TEXT[] NOT NULL -- Array of photo paths
);

-- Table to store attendance records
CREATE TABLE attendance (
    id SERIAL PRIMARY KEY,
    room_id INT REFERENCES rooms(id),
    student_name VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL,
    quality FLOAT NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add foreign key relationship for students in attendance table
ALTER TABLE attendance
ADD CONSTRAINT fk_attendance_student
FOREIGN KEY (student_id) REFERENCES students(id); 