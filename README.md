# Cricket Ball Detection & Trajectory Tracking

This project implements a complete computer vision pipeline to detect and track a cricket ball from videos recorded using a single, fixed camera.

## 🚀 Project Overview
The system performs high-precision detection and path estimation, ideal for sports analytics. 
* **Detection:** YOLO-based ball detection in every frame.
* **Extraction:** Precise centroid (x, y) coordinate calculation.
* **Estimation:** Kalman Filter integration to maintain tracking during occlusions.
* **Outputs:** Processed overlay videos and per-frame CSV data.

---

## 📂 Repository Structure
```text
EdgeFleet-Cricket-Ball-Tracking/
├── code/
│   ├── infer.py           # Inference logic
│   ├── train.py           # Training script
│   └── utils/
│       ├── centroid.py    # Centroid calculation logic
│       └── csv_writer.py  # CSV generation utilities
├── models/
│   └── best.pt            # Trained YOLO model weights
├── results/               # Processed output videos
├── annotations/           # Generated CSV annotation files
├── README.md
├── report.pdf
├── requirements.txt
└── .gitignore
```

## 🛠 Installation
Install all required dependencies using:
```text
pip install -r requirements.txt
```

## 🏃 Running Inference
The inference pipeline is executed via code/run_inference.py. To generate the trajectory and CSV files, run:
```text
python code/run_inference.py
```
Input: Reads videos from the videos/ directory.

## 📊 Output Format
CSV Annotations
Each CSV file contains per-frame detection information in the following format:

frame,x,y,visible

0,512.3,298.1,1

1,518.7,305.4,1

2,-1,-1,0

## 🧠 Methodology
1. Ball Detection
A YOLO-based object detection model trained on a custom dataset of ~10,000 images. The model is robust against varying lighting conditions and ball sizes.

2. Tracking & Trajectory
   A Kalman Filter is applied to the detected centroids to:

   Smooth noisy detections.

   Predict positions during occlusions or missed detections.

   Filter unrealistic jumps via motion-gating mechanisms.


## 📝 Notes & Assumptions
The camera must remain fixed (static).

The pipeline is designed for a single ball in the scene.

Supports both .mov and .mp4 formats.