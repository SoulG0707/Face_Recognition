-- Tạo cơ sở dữ liệu
CREATE DATABASE IF NOT EXISTS attendance_db;

-- Sử dụng cơ sở dữ liệu
USE attendance_db;

-- Tạo bảng students
CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    student_id VARCHAR(50) NOT NULL,
    major VARCHAR(255) NOT NULL
);

-- Tạo bảng attendance
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    student_id VARCHAR(50) NOT NULL,
    major VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    time TIME NOT NULL
);

-- Thêm một số dữ liệu mẫu vào bảng students
INSERT INTO students (name, student_id, major) VALUES ('Ngoc', '22205600', 'Artificial Intelligence');
INSERT INTO students (name, student_id, major) VALUES ('Nhi', '22203891', 'Artificial Intelligence');

-- Truy vấn dữ liệu
SELECT * FROM attendance;
SELECT * FROM students;
DELETE FROM students WHERE id > 0;
ALTER TABLE students AUTO_INCREMENT = 1;
