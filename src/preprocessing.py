"""Image preprocessing for medical imaging."""

import numpy as np
import cv2


class ImagePreprocessor:
    """Preprocessing operations for medical images."""

    @staticmethod
    def normalize_zscore(image):
        mean = np.mean(image)
        std = np.std(image) + 1e-8
        return (image - mean) / std

    @staticmethod
    def normalize_minmax(image):
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val < 1e-8:
            return np.zeros_like(image)
        return (image - min_val) / (max_val - min_val)

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        return enhanced.astype(np.float32) / 255.0

    @staticmethod
    def window_ct(image, window_center=40, window_width=400):
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        windowed = np.clip(image, min_val, max_val)
        return (windowed - min_val) / (max_val - min_val)

    @staticmethod
    def resize(image, target_size, interpolation=cv2.INTER_LINEAR):
        return cv2.resize(image, target_size, interpolation=interpolation)

    def preprocess_pipeline(self, image, method="minmax", apply_clahe_flag=True):
        if method == "zscore":
            image = self.normalize_zscore(image)
        else:
            image = self.normalize_minmax(image)
        if apply_clahe_flag and image.ndim == 2:
            image = self.apply_clahe((image * 255).astype(np.uint8))
        return image
