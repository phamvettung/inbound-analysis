from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, QDateTime
from PyQt5.QtGui import QImage, QPixmap
import psutil, time
import cv2, numpy as np
from M006_trackers import ParcelTracker
from M008_speed_estimator import SpeedEstimator

class SystemInfor(QThread):
    cpu = pyqtSignal(float)
    ram = pyqtSignal(float)

    def __init__(self):
        super().__init__()

    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent
            self.cpu.emit(cpu)
            self.ram.emit(ram)

    def stop(self):
        self.ThreadActive = False
        self.quit()

class MainWindow(QMainWindow):

    input_video_path = "M001_input_videos/input_video.mp4"
    output_video_path = "M005_output_videos/output_video.avi"
    parcel_detection_model_path = "M003_models/detect/v15_nano/best.pt"
    sensor_keypoint_model_path = "M003_models/pose/v2/best.pt"

    def __init__(self):      
        super().__init__()

        self.ui = uic.loadUi("M100_GUI/MainWindow.ui", self)     

        # System Infor
        self.resource_usage = SystemInfor()
        self.resource_usage.start()
        self.resource_usage.cpu.connect(self.get_cpu_usage)
        self.resource_usage.ram.connect(self.get_ram_usage)

        # Time
        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start()

        # OpenCv Videocapture
        self.cap = cv2.VideoCapture(self.input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duaration = 0.0
        self.frame_count = 0
        self.start_time = time.time()

        # Timer to read frames
        self.cv_timer = QTimer()
        self.cv_timer.timeout.connect(self.update_frame)
        self.cv_timer.start(30)  # ~30 fps

        # Parcel Tracker Initialize
        self.parcel_tracker = ParcelTracker(self.parcel_detection_model_path)

        # Speed Estimator Initialize
        self.speed_estimator = SpeedEstimator()





    def closeEvent(self, event):
            self.cap.release()  # release camera/video
            event.accept()


    parcel_dict_preframe = {}
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:

            frame = cv2.resize(frame, (640, 360))
            frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  # get frame index

            #parcel detection
            parcel_dict = self.parcel_tracker.detect_frame(frame, frame_id)

            #Speed Estimator
            self.frame_count += 1
            self.estimate_fps()
            parcel_dict_frames = [self.parcel_dict_preframe, parcel_dict]
            frame_speed = self.speed_estimator.estimate_speed_frame(parcel_dict_frames, self.duaration)




            # draw output
            frame = self.parcel_tracker.draw_bbox(frame, parcel_dict)  
            frame = self.speed_estimator.draw_text2(frame, frame_speed)

            self.show_image(frame)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
        
        if parcel_dict == None: return
        self.parcel_dict_preframe = parcel_dict


    def show_image(self, frame):
        # Convert frame from BGR(OpenCV) to RGB(Qt)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get image dimensions
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # Convert to QImage
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # Show on QLabel
        self.label_camera.setScaledContents(True)
        self.label_camera.setPixmap(QPixmap.fromImage(qimg))


    def estimate_fps(self):
        end_time = time.time()
        self.duaration = end_time - self.start_time
        self.fps = self.frame_count / self.duaration
        

    def clock(self):
        self.DateTime = QDateTime.currentDateTime()
        self.lcd_clock.display(self.DateTime.toString('hh:mm:ss'))
        self.lbl_fps.setText(str(np.round(self.fps, 1)))
 
    def get_ram_usage(self, ram):
        self.lbl_ram.setText(str(ram) + " %")
        if ram > 15: self.lbl_ram.setStyleSheet("color: rgb(23, 63, 95);")
        if ram > 25: self.lbl_ram.setStyleSheet("color: rgb(32, 99, 155);")
        if ram > 45: self.lbl_ram.setStyleSheet("color: rgb(60, 174, 163);")
        if ram > 65: self.lbl_ram.setStyleSheet("color: rgb(246, 213, 92);")
        if ram > 85: self.lbl_ram.setStyleSheet("color: rgb(237, 85, 59);")

    def get_cpu_usage(self, cpu):
        self.lbl_cpu.setText(str(cpu) + " %")
        if cpu > 15: self.lbl_cpu.setStyleSheet("color: rgb(23, 63, 95);")
        if cpu > 25: self.lbl_cpu.setStyleSheet("color: rgb(32, 99, 155);")
        if cpu > 45: self.lbl_cpu.setStyleSheet("color: rgb(60, 174, 163);")
        if cpu > 65: self.lbl_cpu.setStyleSheet("color: rgb(246, 213, 92);")
        if cpu > 85: self.lbl_cpu.setStyleSheet("color: rgb(237, 85, 59);")
