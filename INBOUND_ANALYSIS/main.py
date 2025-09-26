
from M004_utils import read_video, save_video
from M006_trackers import ParcelTracker, SensorDetector
from M008_speed_estimator import SpeedEstimator

from M100_GUI import MainWindow
from PyQt5.QtWidgets import QApplication

def main():
    input_video_path = "M001_input_videos/input_video.mp4"
    output_video_path = "M005_output_videos/output_video.avi"
    parcel_detection_model_path = "M003_models/detect/v15/best.pt"
    sensor_keypoint_model_path = "M003_models/pose/v2/best.pt"

    # read video
    video_frames = read_video(input_video_path)

    # detecting parcel
    parcel_tracker = ParcelTracker(parcel_detection_model_path)
    parcel_detections = parcel_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="M006_trackers/tracker_stub/parcel_detections.pkl")
    parcel_detections = parcel_tracker.interpolate_tracks(parcel_detections)
    
    # sensor detections
    sensor_detector = SensorDetector(sensor_keypoint_model_path)
    sensor_keypoint_detections = sensor_detector.detect_frames(video_frames, read_from_stub=True, stub_path="M006_trackers/tracker_stub/sensor_keypoint_detections.pkl")


    # estimate speeds of parcels
    speed_estimator = SpeedEstimator()
    speed_dt = speed_estimator.estimate_speed_frames(parcel_detections)


    # DRAW OUTPUT
    output_video_frames = parcel_tracker.draw_bboxes(video_frames, parcel_detections)
    output_video_frames = speed_estimator.draw_text(output_video_frames, speed_dt)
    output_video_frames = sensor_detector.draw_bbox(output_video_frames, sensor_keypoint_detections)

    # save videos
    save_video(output_video_frames, output_video_path)

if __name__ == '__main__':

    #main()
    
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
