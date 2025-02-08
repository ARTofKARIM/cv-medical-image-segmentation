"""Segmentation evaluation metrics."""

import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_bin)
    return (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin) + 1e-8)


def iou_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_bin)
    union = np.sum(y_true) + np.sum(y_pred_bin) - intersection
    return intersection / (union + 1e-8)


def precision_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    tp = np.sum(y_true * y_pred_bin)
    fp = np.sum((1 - y_true) * y_pred_bin)
    return tp / (tp + fp + 1e-8)


def recall_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    tp = np.sum(y_true * y_pred_bin)
    fn = np.sum(y_true * (1 - y_pred_bin))
    return tp / (tp + fn + 1e-8)


def hausdorff_distance(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    points_true = np.argwhere(y_true > 0.5)
    points_pred = np.argwhere(y_pred_bin > 0.5)
    if len(points_true) == 0 or len(points_pred) == 0:
        return float("inf")
    d1 = directed_hausdorff(points_true, points_pred)[0]
    d2 = directed_hausdorff(points_pred, points_true)[0]
    return max(d1, d2)


def evaluate_segmentation(y_true, y_pred):
    return {
        "dice": dice_score(y_true, y_pred),
        "iou": iou_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "hausdorff": hausdorff_distance(y_true, y_pred),
    }
