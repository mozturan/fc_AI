<div align="center">

# fc_AI — Football Match Analysis with Computer Vision


<br/>

![11](https://github.com/mozturan/fc_AI/assets/89272933/8bf87054-c5ea-4a88-8b05-50cffa773b80)

<br/><br/>

![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-blue?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

</div>

---

## Overview

**fc_AI** is a modular computer vision pipeline for analysing real football match footage. Given a match video, the system automatically detects and tracks players, referees, and the ball — then extracts a rich set of tactical and physical metrics from the raw video signal.

The pipeline chains several distinct CV techniques together: object detection, unsupervised color-based team assignment, camera motion compensation, geometric perspective correction, and real-world speed & distance estimation.

---

## Features

- **Player, ball & referee detection** — custom-trained YOLOv5 model on real match footage
- **Team assignment** — K-Means clustering on jersey pixel color to automatically separate the two teams
- **Camera movement estimation** — optical flow to decouple true player motion from camera pan/tilt
- **Perspective transformation** — maps the 2D image plane to a top-down pitch representation for accurate real-world measurements
- **Speed & distance tracking** — per-player speed and total distance covered, estimated in real-world units
- **Custom visualisation** — annotated video output with bounding ellipses, team color indicators, speed overlays, and ball possession markers

---
## Pipeline Architecture

```
Raw Video
    │
    ▼
┌─────────────────────┐
│   YOLO Inference    │  ← Custom YOLOv5 trained on football dataset
│  (yolo_inference.py)│
└────────┬────────────┘
         │  detections (players, ball, referees)
         ▼
┌─────────────────────┐
│  Object Tracking    │  ← trackers/  (assigns consistent IDs across frames)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Team Assignment   │  ← assigners/  (K-Means on jersey crop pixels)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Camera Movement    │  ← camera/  (optical flow — removes camera drift)
│    Estimation       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Perspective       │  ← perspective/  (2D → top-down pitch transform)
│  Transformation     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Speed & Distance   │  ← utils/  (real-world metric calculation)
│    Calculation      │
└────────┬────────────┘
         │
         ▼
   Annotated Output Video
```

---

## Repository Structure

```
fc_AI/
├── main.py                  # Entry point — runs the full pipeline
├── yolo_inference.py        # YOLOv5 inference wrapper
├── assigners/               # Team & ball possession assignment (K-Means)
├── camera/                  # Camera movement estimation via optical flow
├── perspective/             # Perspective transformation to top-down view
├── trackers/                # Multi-object tracking across frames
├── utils/                   # Drawing utilities, speed & distance calculation
└── .gitignore
```

---

## Datasets

| Purpose | Source |
|---|---|
| Match footage for testing | [DFL Bundesliga Data Shootout — Kaggle](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips) |
| YOLOv5 training data | [Football Players Detection — Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) |

---
## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mozturan/fc_AI.git
cd fc_AI
```

### 2. Install dependencies

```bash
pip install ultralytics opencv-python numpy scikit-learn supervision
```

### 3. Add your input video

Place your match video in the project root, then update the input path in `main.py`.

### 4. Run the pipeline

```bash
python main.py
```

The annotated output video will be saved to the project directory.

---

## Techniques Used

| Technique | Purpose |
|---|---|
| **YOLOv5 (custom trained)** | Detect players, ball, and referees |
| **K-Means Clustering** | Segment jersey colors to assign team membership |
| **Optical Flow** | Estimate and compensate for camera movement |
| **Perspective Transformation** | Convert image-plane positions to real-world pitch coordinates |
| **Euclidean Distance Tracking** | Calculate per-player speed and distance across frames |

---

## Topics

`computer-vision` `deep-learning` `object-detection` `object-tracking`
`yolov5` `yolo` `ultralytics` `optical-flow` `kmeans` `supervision`
`sports-analytics` `football` `image-processing` `perspective-transform`

---

This is a reimplemented project of someone else's work.

---
<div align="center">
  <i>Bursa Uludağ University · Computer Engineering Department</i>
</div>

