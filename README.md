# Real-Time Campus Surveillance System

A production-oriented surveillance platform for real-time multi-camera monitoring, human pose estimation, and behavior analysis. This repository demonstrates a working `src/` implementation that integrates OpenVINO-optimized CPU inference, YOLOv8 pose detection, and an ONNX-based LSTM fight-risk model while providing a browser dashboard for live monitoring.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What’s Included](#whats-included)
- [Folder Structure](#folder-structure)
- [Key Components](#key-components)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Notes](#notes)

---

## Project Overview

This repository implements a campus surveillance dashboard that:

- reads multiple camera/video streams
- performs pose-based object detection using YOLOv8 and OpenVINO
- scores potential fight behavior using an ONNX LSTM model
- serves a live dashboard via Flask and Flask-SocketIO
- streams raw video frames over MJPEG with metadata overlays

The dashboard is started from `src/main.py`, and the core processing code is in `src/worker.py`, `src/detector.py`, and `src/fight_detector_onnx.py`.

---

## What’s Included

- `src/main.py` — dashboard server entry point
- `src/worker.py` — decoder and inference worker processes
- `src/detector.py` — YOLOv8 pose model wrapper with OpenVINO export
- `src/fight_detector_onnx.py` — per-person LSTM fight scoring
- `src/pose_model.py` — helper script for exporting YOLOv8 models
- `src/templates/dashboard.html` — dashboard UI
- `src/models/` — OpenVINO model assets and metadata
- `src/videos/` — test/reference video files
- `logs/` — runtime logging output
- `output/` — generated artifacts and outputs

---

## Folder Structure

```
real_time_vedio/
├── requirements.txt     # Python dependencies
├── README.md            # this documentation file
├── .gitignore           # ignore file
└── src/                 # current implementation
    ├── detector.py
    ├── fight_detector_onnx.py
    ├── main.py
    ├── monitor.py
    ├── pose_model.py
    ├── templates/
    │   └── dashboard.html
    ├── training/
    │   ├── convert_to_onnx.py
    │   └── inference_onnx.py
    ├── models/
    │   └── ..... # contains models used in this project
    │   
    ├── utils.py
    └── worker.py
```

---

## Key Components

- `src/main.py`
  - Starts the dashboard server on `http://localhost:5000`
  - Uses Flask + Flask-SocketIO
  - Serves MJPEG video feeds and lightweight metadata events

- `src/worker.py`
  - Manages decoder processes and shared-memory frame transport
  - Handles inference orchestration and pipeline state

- `src/detector.py`
  - Wraps YOLOv8 pose inference
  - Automatically exports the model to OpenVINO format if needed

- `src/fight_detector_onnx.py`
  - Maintains a rolling keypoint buffer per tracked person
  - Computes an ONNX-based fight probability score

- `src/pose_model.py`
  - Helper script to export a YOLOv8 pose model to OpenVINO format

- `src/templates/dashboard.html`
  - Browser UI used by the dashboard server

---

## How to Run

1. Activate the Python virtual environment:

```powershell
cd D:\Projects\real_time_vedio
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the dashboard server:

```powershell
python src/main.py
```

4. Open the dashboard:

```text
http://localhost:5000
```

---

## Dependencies

The repository uses the packages listed in `requirements.txt`, including:

- Flask
- Flask-SocketIO
- OpenVINO
- ultralytics
- onnxruntime
- opencv-python
- numpy
- psutil
- supervision

---

## Notes

- The current working implementation is in `src/`.
- `src/pose_model.py` is a helper script for model export rather than a runtime module.
- `src/training` contain helping scripts for convert model to onnx and test fight model working.
