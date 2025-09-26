from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np
from M004_utils import Sort

class ParcelTracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("cuda")
        self.skip_frames = 2
        self.max_age = 10
        self.last_detections  = {} # track_id → list of (frame_id, bbox)


    # interpolate parcel positions
    # frames: list of dicts, each dict: {track_id: [x1,y1,x2,y2]}
    # returns: list of dicts with interpolated bboxes
    def interpolate_tracks(self, frames):
        nframes = len(frames)

        # 1. Collect data per track
        track_data = {}
        for frame_idx, frame in enumerate(frames):
            for tid, bbox in frame.items():
                if tid not in track_data:
                    track_data[tid] = {}
                track_data[tid][frame_idx] = bbox

        # 2. Interpolate each track only between first and last detection
        interpolated_tracks = {}
        for tid, data in track_data.items():
            df = pd.DataFrame.from_dict(data, orient="index", columns=["x1", "y1", "x2", "y2"])
            start, end = df.index.min(), df.index.max()

            # reindex only within [start, end]
            df = df.reindex(range(start, end + 1))
            df_interp = df.interpolate(method="linear")

            interpolated_tracks[tid] = (start, end, df_interp)

        # 3. Rebuild back into frame-wise format
        new_frames = []
        for f in range(nframes):
            frame_dict = {}
            for tid, (start, end, df) in interpolated_tracks.items():
                if start <= f <= end:
                    row = df.loc[f]
                    if not row.isnull().any():
                        frame_dict[tid] = [row["x1"], row["y1"], row["x2"], row["y2"]]
            new_frames.append(frame_dict)

        return new_frames



    
    def draw_bbox(self, frame, parcel_dict):
        for track_id, bbox in parcel_dict.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(frame, f"Parcel ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
        return frame

    def draw_bboxes(self, video_frames, parcel_detections):
        output_video_frames = []

        for frame, parcel_dict in zip(video_frames, parcel_detections):
            #Draw bounding boxes
            for track_id, bbox in parcel_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Parcel ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)

        return output_video_frames
    


    def detect_frame_interpolate(self, frame, frame_id):
        parcel_dict = {}

        # run YOLO only on every skip_frames-th frame
        if frame_id % self.skip_frames == 0:
            results = self.model.track(frame, persist=True, imgsz=320, device=0)[0]
            id_name_dict = results.names

            for parcel in results.boxes:
                if parcel.id is None:
                    continue

                track_id = int(parcel.id.tolist()[0])
                bbox = parcel.xyxy.tolist()[0]
                object_cls_id = parcel.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]

                if object_cls_name == "parcel":
                    parcel_dict[track_id] = bbox
                    self.last_detections[track_id] = (frame_id, bbox)

        else:
            # Interpolate for skipped frames
            to_delete = []
            for track_id, (last_frame, last_bbox) in self.last_detections.items():
                # remove expired tracks
                if frame_id - last_frame > self.max_age:
                    to_delete.append(track_id)
                    continue

                # (optional) interpolate if you have next_frame
                parcel_dict[track_id] = last_bbox

            # delete expired tracks
            for tid in to_delete:
                del self.last_detections[tid]

        return parcel_dict
        

    def detect_frame(self, frame, frame_id):

        # if frame_id % self.skip_frames != 0:
        #     return {}

        results = self.model.track(frame, persist=True, imgsz=320, device=0)[0]
        id_name_dict = results.names

        parcel_dict = {}
        for parcel in results.boxes:

            if parcel.id is None:  # skip if tracker hasn't assigned ID yet
                continue

            track_id = int(parcel.id.tolist()[0])
            bbox = parcel.xyxy.tolist()[0]
            object_cls_id = parcel.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "parcel":
                parcel_dict[track_id] = bbox

        return parcel_dict

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        parcel_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as file:
                parcel_detections = pickle.load(file)
            return parcel_detections
        
        for frame in frames:
            parcel_dict = self.detect_frame(frame)
            parcel_detections.append(parcel_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as file:
                pickle.dump(parcel_detections, file)

        return parcel_detections