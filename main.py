import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import yt_dlp


def download_video(url, output_path="input/input.mp4"):
    if os.path.exists(output_path):
        print("Video already exists")
        return output_path

    print("Downloading video...")

    ydl_opts = {
        'outtmpl': output_path,
        'format': 'mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print("Download complete")
    return output_path


video_url = "https://youtube.com/shorts/ULx2G6UVQfQ?si=jI7vgM0yvGyWOxEQ"
video_path = download_video(video_url)

model = YOLO("yolov8l.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

os.makedirs("output", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/output.mp4", fourcc, fps, (width, height))

print("Processing started... Press ESC to stop")

track_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, classes=[0, 32], stream=True)

    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            cls = int(cls)

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            if cls == 0:
                label = "Person"
            elif cls == 32:
                if conf < 0.35 or w * h < 100:
                    continue
                label = "Ball"
            else:
                continue

            detections.append(([x1, y1, w, h], float(conf), label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        label = track.get_det_class()

        color = (255, 0, 0) if label == "Person" else (0, 255, 255)

        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        cv2.putText(frame,
                    f"{label} ID: {track_id}",
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color, 2)

        center = ((l + r) // 2, (t + b) // 2)

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append(center)

        for i in range(1, len(track_history[track_id])):
            cv2.line(frame,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done. Output saved in output/output.mp4")
