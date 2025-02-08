# Medical Image Segmentation with U-Net

A deep learning pipeline for medical image segmentation using the U-Net architecture. Designed for organ and lesion segmentation in CT/MRI scans with custom loss functions, data augmentation, and comprehensive evaluation.

## Overview

This project implements a complete medical image segmentation pipeline with a configurable U-Net architecture. It supports multiple loss functions (Dice, Focal, Tversky), extensive data augmentation, and provides detailed evaluation metrics including Dice coefficient, IoU, and Hausdorff distance.

## Architecture

```
cv-medical-image-segmentation/
├── src/
│   ├── data_loader.py      # Medical image dataset with tf.data pipeline
│   ├── preprocessing.py    # CLAHE, z-score, CT windowing
│   ├── augmentation.py     # Albumentations-based augmentation
│   ├── unet.py             # Configurable U-Net architecture
│   ├── losses.py           # Dice, BCE-Dice, Focal, Tversky losses
│   ├── train.py            # Training with callbacks and checkpointing
│   ├── metrics.py          # Dice, IoU, Hausdorff, precision, recall
│   ├── inference.py        # Prediction and post-processing pipeline
│   └── visualization.py    # Overlays, training curves, sample grids
├── config/config.yaml
├── tests/test_unet.py
└── main.py
```

## U-Net Architecture

```
Input → [Conv-BN-ReLU × 2 → MaxPool] × 4 → Bottleneck → [UpConv-Skip-Conv × 2] × 4 → Sigmoid
```

## Loss Functions

| Loss | Description | Best For |
|------|-------------|----------|
| Dice | Overlap-based | Balanced segmentation |
| BCE + Dice | Combined | General purpose |
| Focal | Hard example mining | Class imbalance |
| Tversky | Weighted FP/FN control | Precision/recall trade-off |

## Installation

```bash
git clone https://github.com/mouachiqab/cv-medical-image-segmentation.git
cd cv-medical-image-segmentation
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config config/config.yaml --mode train
python main.py --config config/config.yaml --mode predict
```

## Technologies

- Python 3.9+, TensorFlow/Keras
- OpenCV, albumentations
- nibabel (NIfTI format)
- scikit-image, scipy










