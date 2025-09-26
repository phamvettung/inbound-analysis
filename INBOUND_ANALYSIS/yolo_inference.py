from ultralytics import YOLO

model = YOLO('M003_models/detect/v14_yolov5/best.pt')

results = model.track('M001_input_videos/input_video.mp4', conf = 0.2, save = True)

print(results[0])
for box in results[0].boxes:
    print(box)

# # Loop through frames
# for result in results:
#     keypoints = result.keypoints  # keypoints tensor for all detected objects
#     boxes = result.boxes          # bounding boxes
#     probs = result.probs          # class probabilities (if applicable)

#     # Example: print keypoints
#     print(keypoints.xy)   # (x, y) format
#     print(keypoints.conf) # confidence for each keypoint

