# Classroom Attention Monitor

Real-time student attention monitoring system using computer vision. Two parallel threads run simultaneously: one processes the camera feed with three independent detectors (gaze direction, eye state, hand gesture), and a second drives a Finite State Machine that models a virtual lecture session.

## What it does

### Camera thread — three simultaneous detectors

**Gaze direction detector** (MediaPipe Face Detection):
- Detects keypoints for both eyes and the nose tip.
- Computes the angle between the eye-to-eye axis and the eyes-to-nose axis.
- Classifies gaze as *centered* (looking at the screen) or *deviated* (looking elsewhere) based on the angular threshold.

**Eye state detector** (MediaPipe Face Mesh):
- Extracts 468 facial landmarks.
- Crops both eye regions from the frame and binarizes them.
- Classifies eyes as *open* or *closed* by comparing the ratio of bright pixels.

**Hand gesture classifier** (MediaPipe Hands + TFLite):
- Detects hand landmarks and normalizes them relative to the bounding box center.
- Passes the 42-feature vector to a pre-trained TFLite `KeyPointClassifier`.
- Outputs one of: `OPEN`, `CLOSE`, `POINTER`, `OK`.

All three detectors run on the same frame in sequence, updating shared global boolean flags (`mirada_centrada`, `ollos_pechados`, `man_aberta`) used by the FSM thread.

### FSM thread — virtual lecture state machine

States:
1. **CLASS** — normal lecture; monitors attention via flags.
2. **PAUSE** — triggers if gaze deviates or eyes close for a sustained period.
3. **ASSERTION** — raised hand detected (`POINTER` or `OK` gesture); waits for teacher acknowledgment.
4. **END** — session complete.

Transitions are time-gated (configurable thresholds per state) to avoid reacting to momentary distractions.

## Project structure

```
classroom-attention-cv/
├── script.py              # Main script: detector initialization, threads, FSM logic
├── imports.py             # Shared imports and utility functions (landmark preprocessing)
├── landmarks.png          # Reference diagram of MediaPipe face landmark indices
├── model/
│   └── keypoint_classifier/
│       ├── keypoint_classifier.tflite     # Pre-trained gesture classifier
│       ├── keypoint_classifier_label.csv  # Label mapping (OPEN, CLOSE, POINTER, OK)
│       ├── keypoint_classifier.py         # TFLite inference wrapper class
│       └── keypoint.csv                   # Training data (landmark coordinates per gesture)
└── requirements.txt
```

## Quick start

```bash
uv venv
uv pip install -r requirements.txt

python script.py
```

Requires a webcam connected and accessible as device 0 (`cv.VideoCapture(0)`). The system starts automatically — no keyboard interaction needed.

## Dependencies

| Package | Role |
|---|---|
| `opencv-python` | Camera capture, image preprocessing |
| `mediapipe` | Face detection, face mesh (468 landmarks), hand landmark detection |
| `numpy` | Array operations |
| `tensorflow` / `tflite-runtime` | Running the pre-trained `.tflite` gesture classifier |

> **Note on TFLite**: on x86/x64 systems, `tensorflow` is used (includes TFLite). On ARM (Raspberry Pi, Jetson), install `tflite-runtime` instead for a lighter footprint — the `requirements.txt` handles this automatically via platform markers.

## Retraining the gesture classifier

The training data is in `model/keypoint_classifier/keypoint.csv` (one row per sample: 42 normalized landmark coordinates + label index). Edit `keypoint_classifier_label.csv` to add new gesture classes, collect new samples by running the capture script, and retrain the model using `keypoint_classifier.py`.
