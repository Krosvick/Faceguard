-- Table to store university information
CREATE TABLE universities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255)
);

-- Table to store student information
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    university_id INT REFERENCES universities(id),
    face_embedding VECTOR(512)
);

-- Table to store teacher information
CREATE TABLE teachers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    auth_user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE
);

-- Table to store assignature information
CREATE TABLE assignatures (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    teacher_id INT REFERENCES teachers(id)
);

-- Table to store class information
CREATE TABLE classes (
    id SERIAL PRIMARY KEY,
    assignature_id INT REFERENCES assignatures(id),
    university_id INT REFERENCES universities(id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    room VARCHAR(10) NOT NULL
);

-- Table to store attendance records
CREATE TABLE attendance (
    id SERIAL PRIMARY KEY,
    room VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    quality FLOAT NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    student_name VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    class_id INT REFERENCES classes(id)
); 