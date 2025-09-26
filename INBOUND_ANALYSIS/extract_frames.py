import cv2
import os

video_path = "M001_input_videos/boxes_video4.mp4"
output_folder = "dataset/boxes_frames_4"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join(output_folder, f"boxes_frame4_{frame_num:04d}.png")
    cv2.imwrite(frame_filename, frame)
    frame_num += 1

cap.release()
print(f"Done! Extracted {frame_num} frames to {output_folder}")