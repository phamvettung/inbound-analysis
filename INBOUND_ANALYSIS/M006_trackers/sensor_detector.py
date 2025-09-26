from ultralytics import YOLO
import pickle
import cv2

class SensorDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def draw_bbox(self, video_frames, sensor_detections):
        output_video_frames = []

        for frame, sensor_dict in zip(video_frames, sensor_detections):

            # detections is already a list of dicts
            for det in sensor_dict:

                bbox = det['bbox']
                keypoints = det['keypoints']

                # draw bbox
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "sensor", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

                # draw keypoints
                for kp in keypoints:
                    x, y, conf = kp
                    if conf > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            output_video_frames.append(frame)

        return output_video_frames


    def detect_frame(self, frame):
        results = self.model.predict(frame,conf=0.15)[0]
        sensor_detections = []
        for box, keypoint_data in zip(results.boxes, results.keypoints.data):
            # bounding box [x1, y1, x2, y2]
            bbox = box.xyxy.cpu().numpy().flatten().tolist()

            # keypoints (x, y, conf)
            keypoints = keypoint_data.cpu().numpy().tolist()

            sensor_detections.append({
                "bbox": bbox,
                "keypoints": keypoints
            })
        return sensor_detections

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        sensor_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as file:
                sensor_detections = pickle.load(file)
            return sensor_detections
        
        for frame in frames:
            sensor_dict = self.detect_frame(frame)
            sensor_detections.append(sensor_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as file:
                pickle.dump(sensor_detections, file)

        return sensor_detections



        
    