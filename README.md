# Multi-Object Detection and Persistent ID Tracking

This project focuses on detecting and tracking multiple objects in a sports video using computer vision techniques. The goal is to assign a consistent ID to each person (and the ball) throughout the video, even when there is movement, overlap, or partial visibility.

---

## Project Overview

In this project, I built a pipeline that takes a publicly available video and processes it frame by frame. It detects objects using YOLOv8 and then tracks them using DeepSORT, ensuring that each object keeps the same ID across frames.

The system is designed to handle real-world challenges such as:
- Fast movement  
- Occlusion (objects blocking each other)  
- Camera motion  
- Scale changes  

---

## Technologies Used

- Python  
- YOLOv8 (Ultralytics) for object detection  
- DeepSORT for tracking  
- OpenCV for video processing  
- yt-dlp for downloading the input video  

---

## Key Features

- Detects multiple players in the video  
- Tracks each player with a unique ID  
- Attempts to detect and track the ball  
- Maintains ID consistency across frames  
- Draws bounding boxes and trajectories  
- Saves a fully annotated output video  

---

## Project Structure

```
multi-object-tracking/
│
├── main.py
├── requirements.txt
├── README.md
├── screenshots/
├── output.mp4
│     
```

---

## Setup Instructions (Windows)

### Step 1: Create a Virtual Environment
```
python -m venv venv
```

### Step 2: Activate the Environment

Command Prompt:
```
venv\Scripts\activate
```

PowerShell:
```
venv\Scripts\Activate.ps1
```

### Step 3: Install Required Libraries
```
pip install -r requirements.txt
```

---

## How to Run the Project

```
python main.py
```

- The video will be downloaded automatically using the provided URL  
- Processing will start frame by frame  
- Press ESC to stop execution  

---

## Video Source

video_url = "https://youtube.com/shorts/ULx2G6UVQfQ?si=jI7vgM0yvGyWOxEQ"

---

## Output

The processed video is saved at:

```
output/output.mp4
```

It contains bounding boxes, IDs, and trajectories for tracked objects.

---

## Demo and Output Videos

Demo Video:  


Output Video:  
https://drive.google.com/file/d/1kNM1RBkkAsle2Bs4EF1fwtrAErytm7U2/view?usp=sharing

---

## How the System Works

1. Each frame of the video is passed through YOLOv8 to detect objects  
2. Only relevant classes (person and ball) are considered  
3. DeepSORT assigns IDs using:
   - Motion prediction (Kalman Filter)  
   - Appearance similarity  
4. The same ID is maintained across frames  
5. The results are drawn and saved into a new video  

---

## Assumptions

- Only players and ball are tracked  
- The video has a single camera view  
- Standard video quality is assumed  

---

## Limitations

- Ball detection is not always accurate because:
  - It is very small  
  - It moves very fast  
- Sometimes ID switching may happen in crowded scenes  

---

## Possible Improvements

- Train a custom model specifically for ball detection  
- Use more advanced trackers like StrongSORT  
- Add speed or movement analysis  
- Improve detection for small objects  

---

## Sample Results

Screenshots of the output are available

---

## Notes

- The YOLO model is downloaded automatically on first run  
- Internet connection is required for downloading the video  
- The project runs on CPU (GPU can improve speed)  

---


## Final Thoughts

This project demonstrates a practical approach to multi-object detection and tracking in real-world scenarios. It highlights both the strengths of modern detection models and the challenges involved in tracking small and fast-moving objects like a ball.
