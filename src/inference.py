"""Inference pipeline for segmentation predictions."""

import numpy as np
import cv2
import tensorflow as tf
from src.preprocessing import ImagePreprocessor


class InferencePipeline:
    """Runs inference on new medical images."""

    def __init__(self, model_path, image_size=(256, 256)):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.image_size = image_size
        self.preprocessor = ImagePreprocessor()

    def predict_single(self, image_path, threshold=0.5):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_size = img.shape[:2]
        img_resized = cv2.resize(img, self.image_size)
        img_norm = self.preprocessor.normalize_minmax(img_resized.astype(np.float32))
        img_input = np.expand_dims(np.expand_dims(img_norm, -1), 0)

        pred = self.model.predict(img_input, verbose=0)[0, :, :, 0]
        pred_binary = (pred > threshold).astype(np.uint8)
        pred_resized = cv2.resize(pred_binary, (original_size[1], original_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
        return pred_resized, pred

    def predict_batch(self, image_paths, threshold=0.5):
        results = []
        for path in image_paths:
            mask, prob = self.predict_single(path, threshold)
            results.append({"path": path, "mask": mask, "probability_map": prob})
        return results

    @staticmethod
    def post_process(mask, min_area=100):
        import cv2 as cv
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv.contourArea(cnt) >= min_area:
                cv.drawContours(clean_mask, [cnt], -1, 1, -1)
        return clean_mask
