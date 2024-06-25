import os
import cv2
import time
import numpy as np
import face_recognition as fr
import pickle
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QMessageBox, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, QThread

EXCLUDE_NAMES = ['Unknown', 'HOD', 'Principal']

class VideoStream(QThread):
    def __init__(self, stream=0):
        super().__init__()
        self.video = cv2.VideoCapture(stream)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.video.isOpened():
            QMessageBox.critical(None, "Error", "Can't access the webcam stream.")
            exit(0)

        self.stopped = False

    def run(self):
        while not self.stopped:
            self.grabbed, self.frame = self.video.read()
        self.video.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def mark_attendance(name):
    today = time.strftime('%d_%m_%Y')
    record_file = f'Records/record_{today}.csv'
    if not os.path.exists(record_file):
        with open(record_file, 'w', encoding='utf-8') as f:
            f.write("Name,Time\n")
    with open(record_file, 'r', encoding='utf-8') as f:
        names = [line.split(',')[0].strip() for line in f.readlines()]
    if name not in names and name not in EXCLUDE_NAMES:
        current_time = time.strftime('%H:%M:%S')
        with open(record_file, 'a', encoding='utf-8') as f:
            f.write(f"{name},{current_time}\n")

class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance System")
        self.setGeometry(100, 100, 670, 550)

        self.init_ui()
        self.init_video_stream()
        self.faces = self.load_faces()
        self.threshold = 0.4

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setGeometry(80, 40, 640, 480)
        self.image_label.setStyleSheet("border: 1px solid black")

        self.btn_start = self.create_button("Start", self.start_video)
        self.btn_stop = self.create_button("Stop", self.stop_video)
        self.btn_mark_attendance = self.create_button("Mark Attendance", self.mark_attendance_manual)
        self.btn_quit = self.create_button("Quit", self.quit_app)
        self.entry_name = QLineEdit(self)
        self.entry_name.setFont(QFont('Arial', 10))
        self.entry_name.setPlaceholderText("Enter your name")

        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)
        button_layout.addWidget(self.btn_mark_attendance)
        button_layout.addWidget(self.btn_quit)

        layout.addWidget(self.image_label)
        layout.addWidget(self.entry_name)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def create_button(self, text, callback):
        button = QPushButton(text, self)
        button.setFont(QFont('Arial', 10))
        button.clicked.connect(callback)
        return button

    def init_video_stream(self):
        self.video_stream = VideoStream()

    def load_faces(self):
        with open('face_recognition_model.dat', 'rb') as f:
            return pickle.load(f)

    def start_video(self):
        self.video_stream.start()
        self.timer.start(30)

    def stop_video(self):
        self.video_stream.stop()
        self.timer.stop()
        self.display_black_screen()

    def quit_app(self):
        self.video_stream.stop()
        self.close()

    def mark_attendance_manual(self):
        name = self.entry_name.text().strip()
        if name:
            mark_attendance(name)
            QMessageBox.information(self, "Attendance", f"Attendance marked for {name}")
        else:
            QMessageBox.warning(self, "Warning", "Please enter a name")

    def update_frame(self):
        frame = self.video_stream.read()
        if frame is not None:
            face_locations = fr.face_locations(frame)
            unknown_face_encodings = fr.face_encodings(frame, face_locations)
            face_names = self.recognize_faces(unknown_face_encodings)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                self.draw_face_box(frame, top, right, bottom, left, name)
                mark_attendance(name)

            self.display_frame(frame)

    def recognize_faces(self, unknown_face_encodings):
        face_names = []
        for face_encoding in unknown_face_encodings:
            best_match_score = float('inf')
            best_match_name = "Unknown"
            for name, encodings in self.faces.items():
                face_distances = fr.face_distance(encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < self.threshold:
                    best_match_score = face_distances[best_match_index]
                    best_match_name = name
            face_names.append(best_match_name)
        return face_names

    def draw_face_box(self, frame, top, right, bottom, left, name):
        cv2.rectangle(frame, (left-20, top-20), (right+20, bottom+20), (0, 255, 0), 2)
        cv2.rectangle(frame, (left-20, bottom - 15), (right+20, bottom+20), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left - 20, bottom + 15), font, 0.85, (255, 255, 255), 2)

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        qimg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def display_black_screen(self):
        black_image = np.zeros((self.image_label.height(), self.image_label.width(), 3), dtype=np.uint8)
        qimg = QImage(black_image.data, black_image.shape[1], black_image.shape[0], black_image.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = AttendanceApp()
    main_window.show()
    sys.exit(app.exec_())
