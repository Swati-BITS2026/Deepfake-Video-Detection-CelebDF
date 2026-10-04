# Deepfake Video Detection on Celeb-DF v2

### Seven-Model Comparison with Low-Resolution and 64-Frame Evaluation
**BITS Pilani Capstone Project (AI/ML)**

## Project Overview
This project builds and evaluates an end-to-end deepfake detection pipeline that inputs a video file and returns a **REAL** or **FAKE** decision alongside a classification probability score. Based on an identity-disjoint dataset split of the Celeb-DF v2 dataset, the framework fine-tunes a **ConvNeXt-Tiny** architecture and introduces an automated frame-enhancement engine for low-resolution robustness testing.

## Core Performance Metrics
* **Holdout ROC-AUC:** 0.9999
* **Video Accuracy:** 99.51%
* **Optimized Decision Threshold:** 0.234913
* **Low-Resolution 64-Frame Accuracy:** 100% on tested authentic scenarios.

## Software Environment
The code is built using Python 3.12, PyTorch 2.10.0 (CUDA 12.8), ONNX Runtime GPU (1.26.0), and InsightFace (RetinaFace backbone).
