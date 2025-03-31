# Hand Clap Detection using MediaPipe

This project detects **hand clapping gestures** using **MediaPipe** with two different methods:
1. **Hand-based detection** using hand landmarks (from `hand.py`)
2. **Pose-based detection** using wrist and shoulder landmarks (from `pose.py`)

It is mainly built for analyzing abnormal repetitive movements such as hand flapping or clapping in video datasets, especially relevant to behavior analysis (e.g., in ASD research).

---

## Project Structure

```plaintext
.
├── hand.py              # Detects claps based on hand landmarks
├── pose.py              # Detects claps based on body pose landmarks
├── functions.py         # Utility functions for landmark processing, vector calculation, and drawing
└── README.md            # Project documentation
```

---

## Dependencies

Install dependencies:

```bash
conda create -n clap-detection python=3.9
conda activate clap-detection
conda install -c conda-forge mediapipe opencv numpy
```

---

## How to Use

### 1. Hand-Based Detection

```python
from hand import handclap_detection

# Run detection (use 0 for webcam or provide path to a video file)
handclap_detection("your_video.mp4")
```

### 2. Pose-Based Detection

```python
from pose import pose_estimation

# Run detection (use 0 for webcam or provide path to a video file)
pose_estimation("your_video.mp4")
```

---

## Detection Logic Summary

### Hand-Based Detection

- Uses MediaPipe Hands model.
- Tracks:
  - Distance between left and right wrists
  - Change in distance over time
  - Normal vectors based on finger orientation
- Two-stage condition:
  - **Stage 1:** Hands come close (below threshold ratio)
  - **Stage 2:** Hands separate quickly within time threshold
- Both conditions must be met in sequence to count a clap.

### Pose-Based Detection

- Uses MediaPipe Pose model.
- Tracks:
  - Normalized 2D distance between both wrists
  - Normalization based on shoulder width
- Similar two-stage logic based on wrist ratio.
