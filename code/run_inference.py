import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO


# =========================
# Kalman Filter (2D motion)
# =========================
def init_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    return kf


def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# =========================
# Main inference pipeline
# =========================
def run_inference(
    video_path="videos/15.mov",
    model_path="models/best.pt",
    output_video="results/15.mp4",
    csv_path="annotations/15.csv",
    conf_thresh=0.2,
    ball_class_id=1
):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "x", "y", "visible"])

    kalman = init_kalman()
    last_pos = None
    trajectory = []
    frame_id = 0

    MAX_JUMP = int(0.25 * W)   # adaptive motion gate

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected = False
        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        results = model(frame, conf=conf_thresh, verbose=False)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            best_box = None
            best_conf = 0

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls != ball_class_id:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = box

            if best_box is not None:
                x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if last_pos is None or euclidean((cx, cy), last_pos) < MAX_JUMP:
                    kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                    trajectory.append((cx, cy))
                    writer.writerow([frame_id, cx, cy, 1])
                    last_pos = (cx, cy)
                    detected = True

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

        if not detected:
            trajectory.append((pred_x, pred_y))
            writer.writerow([frame_id, pred_x, pred_y, 0])
            cv2.circle(frame, (pred_x, pred_y), 4, (0,255,255), -1)

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (255,0,0), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    csv_file.close()

    print("✅ Inference complete")
    print("CSV:", csv_path)
    print("Video:", output_video)


if __name__ == "__main__":
    run_inference()
