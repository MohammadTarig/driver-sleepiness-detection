from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import time

# Paths for predictor and alarm files
predictor_path = r"absolute\path\to\shape_predictor\shape_predictor_68_face_landmarks.dat"
alarm_files = {
    "Alarm 1": r"absolute\path\to\Driver Sleepiness Detection App\audio\audio_file1.wav",
    "Alarm 2": r"absolute\path\to\Driver Sleepiness Detection App\audio\audio_file2.wav",
    "Alarm 3": r"absolute\path\to\Driver Sleepiness Detection App\audio\audio_file3.wav",
    "Alarm 4": r"absolute\path\to\Driver Sleepiness Detection App\audio\audio_file4.wav"
}

class AlarmThread(Thread):
    def __init__(self, alarm_path):
        super().__init__()
        self.alarm_path = alarm_path
        self._stop_event = False

    def run(self):
        if not self._stop_event:
            playsound.playsound(self.alarm_path)

    def stop(self):
        self._stop_event = True

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

class DriverSleepinessApp(App):
    def build(self):
        self.capture = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # Thresholds and frame counters
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 30
        self.MOUTH_AR_THRESH = 0.60
        self.MOUTH_CONSEC_FRAMES = 30
        self.EAR_COUNTER = 0
        self.MAR_COUNTER = 0
        self.ALARM_ON = False
        self.ALARM_DELAY = 10  # Delay before alarm can be triggered again
        self.last_alarm_time = 0

        # Facial landmarks
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # Kivy layout
        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)

        # Add start, stop, and pause/resume buttons
        button_layout = BoxLayout(size_hint_y=None, height=50)
        self.start_button = Button(text='Start')
        self.stop_button = Button(text='Stop', disabled=True)
        self.pause_resume_button = Button(text='Pause', disabled=True)
        self.start_button.bind(on_press=self.start)
        self.stop_button.bind(on_press=self.stop)
        self.pause_resume_button.bind(on_press=self.pause_resume)
        button_layout.add_widget(self.start_button)
        button_layout.add_widget(self.stop_button)
        button_layout.add_widget(self.pause_resume_button)
        layout.add_widget(button_layout)

        # Add spinner for alarm selection
        self.alarm_spinner = Spinner(
            text='Choose Alarm',
            values=list(alarm_files.keys()),
            size_hint=(None, None),
            size=(150, 44),
            pos_hint={'center_x': .5, 'center_y': .5}
        )
        self.alarm_spinner.bind(text=self.set_alarm)
        self.selected_alarm = alarm_files["Alarm 1"]
        layout.add_widget(self.alarm_spinner)

        self.paused = False  # Flag to check if the detection is paused

        return layout

    def set_alarm(self, spinner, text):
        self.selected_alarm = alarm_files[text]

    def start(self, instance):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update, 1.0 / 30.0)
            self.start_button.disabled = True
            self.stop_button.disabled = False
            self.pause_resume_button.disabled = False

    def stop(self, instance):
        if self.capture is not None:
            Clock.unschedule(self.update)
            self.capture.release()
            self.capture = None
            self.image.texture = None
            self.start_button.disabled = False
            self.stop_button.disabled = True
            self.pause_resume_button.disabled = True
            self.pause_resume_button.text = 'Pause'
            self.paused = False

    def pause_resume(self, instance):
        self.paused = not self.paused
        self.pause_resume_button.text = 'Resume' if self.paused else 'Pause'

    def update(self, dt):
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if not self.paused:  # Only run the detection logic if not paused
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                mouth = shape[self.mStart:self.mEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                mar = mouth_aspect_ratio(mouth)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)

                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                current_time = time.time()

                if ear < self.EYE_AR_THRESH:
                    self.EAR_COUNTER += 1
                    if self.EAR_COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        if not self.ALARM_ON and (current_time - self.last_alarm_time) > self.ALARM_DELAY:
                            self.ALARM_ON = True
                            self.last_alarm_time = current_time
                            self.alarm_thread = AlarmThread(self.selected_alarm)
                            self.alarm_thread.start()
                        cv2.putText(frame, "SLEEPINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                else:
                    self.EAR_COUNTER = 0
                    self.ALARM_ON = False

                if mar > self.MOUTH_AR_THRESH:
                    self.MAR_COUNTER += 1
                    if self.MAR_COUNTER >= self.MOUTH_CONSEC_FRAMES:
                        if not self.ALARM_ON and (current_time - self.last_alarm_time) > self.ALARM_DELAY:
                            self.ALARM_ON = True
                            self.last_alarm_time = current_time
                            self.alarm_thread = AlarmThread(self.selected_alarm)
                            self.alarm_thread.start()
                        cv2.putText(frame, "YAWNING ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                else:
                    self.MAR_COUNTER = 0
                    self.ALARM_ON = False

                # Update position and font size for EAR and MAR values
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        buf = cv2.flip(frame, 0).tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = image_texture


if __name__ == '__main__':
    DriverSleepinessApp().run()
