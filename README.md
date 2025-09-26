# INBOUND ANALYSIS

## Introduction
This project Inbound statistics in a video to estimate speed of parcels, as well as weight conveyor speed and main conveyor speed. This project will detect parcels, sensor keypoint using YOLO. Tested on CUDA 12.1 using RTX 4050 GPU.

## Output Video
<div align="center">
    <img src="/INBOUND_ANALYSIS/M005_output_videos/output_0014.png" width="960" height="540" alt="Final Result"/>
    <p>A screenshot one of the output videos</p>
</div>

## Dataset
- Parcels detection: https://app.roboflow.com/boxesonconveyor/parcel_detection-2dnvd/15
- Sensor keypoints detection: https://app.roboflow.com/boxesonconveyor/sensor_keypoint-arspc/2

## Models used
- YOLO v8 large for video processing
- YOLO v8 nano + CUDA 12.1 + RTX 4050 GPU for real-time processing
- 
## Requirements
- python 3.12
- ultralytics
- pandas
- numpy
- opencv

