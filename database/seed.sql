-- Insert a university
INSERT INTO universities (id, name) VALUES
(1, 'Universidad Arturo Prat');

-- Insert a teacher linked to an authenticated user
INSERT INTO teachers (id, first_name, last_name, email, auth_user_id) VALUES
(1, 'Gabriel', 'Icarte', 'gabrielicarte@unap.cl', '863ab821-8a6b-4bbf-a84a-b1b250992e6e');

-- Insert the assignature "T. de innovación"
INSERT INTO assignatures (id, name, teacher_id) VALUES
(1,'T. de innovación', 1);

-- Insert the class for "T. de innovación"
INSERT INTO classes (assignature_id, university_id, start_time, end_time, room) VALUES
(1, 1, '2023-01-03 08:00:00', '2023-01-03 10:15:00', 'LC6');

-- Insert attendance records for the class
INSERT INTO attendance (room, student_name, confidence, quality, timestamp) VALUES
('LC6', 'Juan Pérez', 0.95, 0.85, '2024-11-14 08:00:00');