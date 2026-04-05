# Industrial Inspection System (YOLOv8 + PatchCore)

A hybrid computer vision inspection system built with YOLOv8 and PatchCore, designed to detect both known defects (cracks, leaks) and unknown anomalies (e.g., rust) in industrial components.

# Features
* Detect known defects using YOLOv8 (object detection)
* Identify unknown defects using PatchCore (anomaly detection)
* Hybrid inference pipeline (YOLO → PatchCore fallback)
* Handles noisy and incomplete annotations
* Generates anomaly heatmaps for visual interpretability
* REST API for image-based inference using FastAPI
* Modular pipeline supporting multiple models
* Works on diverse metal surfaces (nuts, screws, mesh, etc.)

# System Design Overview
This system is designed to handle both **labeled and unlabeled** defect scenarios, which is common in real-world industrial settings.

### Flow
1. Input image is received via API
2. YOLOv8 model detects known defects
3. If defects are detected → return bounding boxes
4. If no defects detected → PatchCore evaluates anomaly score
5. Heatmap + anomaly score returned for localization

## Key Design Decisions
* Hybrid approach ensures robustness when labeled data is limited
* Anomaly detection handles unseen defect types (e.g., rust)
* Heatmaps improve interpretability for inspection systems
* API-based design allows easy integration into larger systems
* Model decoupling allows swapping detection/anomaly models independently

# Tech Stack
* Backend: FastAPI
* ML Framework: PyTorch
* Object Detection: YOLOv8 (Ultralytics)
* Anomaly Detection: PatchCore (Anomalib)
* Computer Vision: OpenCV
* Data Processing: NumPy

# API Endpoints
### Inspect Image
* POST /inspect
* Input: Image file
* Output:
  * Detection results (if YOLO finds defects)
  * OR anomaly score + heatmap (if PatchCore is used)
 
# Model Pipeline
### YOLOv8 (Supervised Detection)
* Trained to detect known defect classes (e.g., cracks, leaks)
* Outputs bounding boxes and class labels

### PatchCore (Unsupervised Anomaly Detection)
* Trained only on normal images
* Learns feature distribution of defect-free surfaces
* Flags deviations as anomalies
* Produces:
  * anomaly score
  * pixel-level heatmap
 
# Hybrid Logic
```
Input Image
   ↓
YOLO Detection
   ↓
[Defect Found?]
   ├── YES → Return bounding boxes
   └── NO  → Run PatchCore
                 ↓
           Return anomaly score + heatmap
```

# Performance Considerations
* GPU acceleration used for faster inference
* Image resizing and preprocessing for consistent input
* Lightweight API design for quick response handling
* Designed for near real-time usage (webcam pipeline ready)

# Data Challenges & Handling
* Dealt with noisy and inconsistent annotations in detection dataset
* Removed irrelevant classes (e.g., rust, defect overlaps)
* Used anomaly detection to compensate for missing labeled data
* Trained PatchCore on clean “good” samples only

# Future Improvements
* Real-time webcam endpoint in API
* Batch inference endpoint
* Model quantization for faster inference
* Multi-surface generalization (pipes, rods, sheets)
* Logging & monitoring for production use
* Optional frontend dashboard for visualization

# Author
### Anshul Manapure
* GitHub: https://github.com/AnshulManapure
* LinkedIn: https://www.linkedin.com/in/anshul-manapure-6a51a7179/
