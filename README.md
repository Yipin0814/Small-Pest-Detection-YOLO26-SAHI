# Small Pest Detection Using YOLO26n (P2 Head) and SAHI Sliced Inference

A high-resolution Computer Vision pipeline built to detect and count tiny agricultural pests across ultra-high-resolution (`5184 × 3456`) yellow sticky trap images.

## Overview
Standard object detectors shrink input images down to small sizes (such as `640 × 640`), which causes tiny insects on large sticky trap photos to blur or disappear completely.

This project tackles small-object detection from both the model side and the inference side. On the model side, **YOLO26n** is upgraded with a **P2 detection head** to preserve fine visual details of tiny targets and trained using the **MuSGD** optimizer. On the inference side, **SAHI (Slicing Aided Hyper Inference)** slices each large image into overlapping smaller tiles, analyzes them individually, and merges the detections back together without losing detail.

## Key Features
* **High-Resolution Target:** Designed for `5184 × 3456` yellow sticky trap images containing dense, tiny targets.
* **Multi-Class Pest Detection:** Detects and classifies three pest categories (`MR`, `NC`, and `WF`).
* **P2 Small-Object Detection Head:** Adds a higher-resolution P2 detection layer to `YOLO26n` so tiny insects are spotted before fine details are lost in deeper layers.
* **Lightweight Model & Optimizer:** Trains the **YOLO26n-P2** architecture using the **MuSGD** optimizer for fast, efficient convergence.
* **Sliced Inference with SAHI:** Splits full-resolution images into overlapping patches during inference and merges overlapping bounding boxes cleanly, greatly improving recall on tiny objects compared to standard full-image inference.

## System Pipeline
1. **Input:** Ultra-high-resolution (`5184 × 3456`) yellow sticky trap images.
2. **Model Architecture & Training:** Modify `YOLO26n` with a **P2 detection head** and train on labeled pest crops using the `MuSGD` optimizer.
3. **Image Slicing (SAHI):** Divide each full test image into smaller overlapping tiles at inference time.
4. **Detection & Merging:** Run `YOLO26n-P2` on every tile and combine overlapping detections back onto the full-resolution image.
5. **Output:** Annotated trap images with bounding boxes, class labels (`MR`, `NC`, `WF`), and automated pest counts.

## Tech Stack
* **Language:** Python
* **Deep Learning & Detection:** PyTorch, YOLO26n (P2 Detection Head, MuSGD Optimizer), SAHI
* **Image Processing & Data:** OpenCV, NumPy, Pandas
* **Environment:** Google Colab / Kaggle Notebooks

## Sample Results
<img width="1464" height="684" alt="image" src="https://github.com/user-attachments/assets/3c11d72b-adb1-4364-b90b-1db5f82f9429" />

## Author
* **Wong Yi Pin** — Artificial Intelligence Undergraduate, Xiamen University Malaysia (XMUM)
