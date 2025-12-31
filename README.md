# Part 1: Glove Compliance Detection Pipeline

## Project Overview
This repository contains a computer vision solution to detect `gloved_hand` and `bare_hand` classes for safety compliance monitoring.

## Technical Implementation
*   **Model:** YOLOv8 Nano (`yolov8n`). Selected for optimal real-time performance on edge hardware.
*   **Data Source:** Roboflow Universe (Gloves & Bare Hands).
*   **Preprocessing:** 
    *   Applied Transfer Learning on ~800 images.
    *   Utilized Mosaic augmentation to improve robustness.
    *   Implemented a **Label Normalization Layer** in the inference script to handle inconsistencies in the source dataset (e.g., mapping `gloverotation` to `gloved_hand`).

## Usage
**Run the detection script:**
```bash
python detection_script.py \
  --input data/images \
  --output results/ \
  --weights best.pt \
  --confidence 0.45
