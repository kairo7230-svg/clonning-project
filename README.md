# Specific Gesture - Clone Arc Trigger System

Train a custom hand gesture using your webcam, and watch an arc of clones appear the moment you perform it, complete with a smoke burst effect.

## Overview

`specificgesture.py` is a real-time hand-gesture recognition and visual effects system built with MediaPipe, OpenCV, and scikit-learn. It lets you:

1. Record your own custom gesture (label: clone) and a neutral hand (label: idle)
2. Train a K-Nearest Neighbours classifier on the fly
3. Trigger a smoke burst and arc clone formation whenever you perform your gesture live

## Requirements

### System Requirements
- Python 3.8 to 3.11 (MediaPipe does not fully support 3.12+)
- A working webcam
- Windows 10/11, macOS, or Linux

### Model Files

You need two MediaPipe model files in the same directory as `specificgesture.py`:

| File | Download |
|------|----------|
| hand_landmarker.task | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task |
| deeplab_v3.tflite | https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite |

## Installation

```
git clone https://github.com/kairo7230-svg/specificgesture-clonning.git
cd specificgesture-clonning
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download the model files into the same directory before running.

## Usage

```
python specificgesture.py
```

### Workflow

| Step | Key | Action |
|------|-----|--------|
| 1 | 1 | Toggle recording your clone gesture, hold for about 3 seconds then press 1 again to stop |
| 2 | 2 | Toggle recording a neutral idle hand, relax your hand for about 2 seconds then press 2 again to stop |
| 3 | F | Train the KNN classifier on collected samples |
| 4 | V | Save the trained model to gesture_model.pkl |
| 5 | perform gesture | The smoke burst and clone arc will appear |

### Keyboard Controls

| Key | Function |
|-----|----------|
| 1 | Toggle recording - clone gesture label |
| 2 | Toggle recording - idle gesture label |
| F | Train the classifier |
| V | Save model to disk |
| L | Load model from disk |
| C | Clear active clones and smoke |
| X | Wipe all training data and the saved model |
| Q | Quit |

After stopping a recording session the system will auto-train if enough samples are already available.

## Configuration

Open `specificgesture.py` and edit the CONFIG section near the top:

```python
TRIGGER_LABEL       = "clone"
PREDICT_CONFIDENCE  = 0.45
KNN_K               = 5
GESTURE_HOLD_FRAMES = 3
GESTURE_RESET_FRAMES= 30
SMOKE_DELAY_FRAMES  = 45
```

## Project Structure

```
specificgesture-clonning/
├── specificgesture.py
├── requirements.txt
├── README.md
├── hand_landmarker.task
├── deeplab_v3.tflite
└── gesture_model.pkl
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| FileNotFoundError: hand_landmarker.task | Download the model files listed above |
| Webcam not opening | Make sure no other app is using the camera, or try changing VideoCapture(0) to VideoCapture(1) |
| Import error: mediapipe | Run pip install mediapipe==0.10.14 |
| Gesture not triggering | Lower PREDICT_CONFIDENCE or record more samples, make sure you have both clone and idle labels |
| Clones look wrong or segmentation missing | Segmentation may still be warming up, wait a moment after the smoke clears |

## License

MIT
