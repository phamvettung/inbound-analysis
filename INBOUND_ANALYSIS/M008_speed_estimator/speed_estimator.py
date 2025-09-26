import numpy as np
import cv2
import math
from M004_utils import get_center_of_bbox, compute_displacement_in_pixels, pixel_to_meter_scale, real_velocity

class SpeedEstimator():
    def __init__(self):
        self.FPS = 24
        self.CONVEYOR_WIDTH_MM = 700
        self.CONVEYOR_WIDTH_PIXELS = 640    



    def draw_text2(self, frame, speed_item):
        #Draw bounding boxes
        for track_id, item in speed_item.items():
            v = item[0]
            bbox = item[1]
            x1, y1, x2, y2 = bbox
            cv2.putText(frame, f"speed: {v} mm/s", (int(bbox[0]), int(bbox[3] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        return frame

    def draw_text(self, video_frames, speed_data):
        output_video_frames = []
        for frame, speed_item in zip(video_frames, speed_data):
            #Draw bounding boxes
            for track_id, item in speed_item.items():
                v = item[0]
                bbox = item[1]
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"speed: {v} mm/s", (int(bbox[0]), int(bbox[3] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            output_video_frames.append(frame)

        return output_video_frames



    def estimate_speed_frame(self, frames, fps):

        scale = pixel_to_meter_scale(700, 353)
        pre_frame = frames[0]
        cur_frame = frames[1]

        if pre_frame == [] and cur_frame == []:
            return

        frame_speeds = {}
        for track_id, bbox_now in cur_frame.items():
            if track_id not in pre_frame:
                continue  # object disappeared

            # get current and next centroid
            center_now = get_center_of_bbox(bbox_now)

            bbox_pre = pre_frame[track_id]
            center_pre = get_center_of_bbox(bbox_pre)

            # displacement in pixels
            d_px = compute_displacement_in_pixels(center_now, center_pre)

            # convert to mm/s
            v = real_velocity(d_px, fps, scale)
            
            frame_speeds[track_id] = np.round(v, 2), bbox_now

        return frame_speeds


    def estimate_speed_frames(self, frames):

        scale = pixel_to_meter_scale(700, 353)

        speed_data = []

        for t in range(len(frames) - 1):
            frame_now = frames[t - 1]
            frame_next = frames[t]
            frame_speeds = {}

            for track_id, bbox_now in frame_now.items():
                if track_id not in frame_next:
                    continue  # object disappeared

                # get current and next centroid
                center_now = get_center_of_bbox(bbox_now)

                bbox_next = frame_next[track_id]
                center_next = get_center_of_bbox(bbox_next)

                # displacement in pixels
                d_px = compute_displacement_in_pixels(center_now, center_next)

                # convert to mm/s
                v = real_velocity(d_px, 1/24, scale)
                
                frame_speeds[track_id] = np.round(v, 2), bbox_now

            speed_data.append(frame_speeds)

        return speed_data





