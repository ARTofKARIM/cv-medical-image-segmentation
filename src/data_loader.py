"""Data loading utilities for medical image segmentation."""

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split


class MedicalImageDataset:
    """Handles loading and batching of medical image/mask pairs."""

    def __init__(self, image_dir, mask_dir, image_size=(256, 256), val_split=0.2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.val_split = val_split
        self.image_paths = []
        self.mask_paths = []

    def discover_files(self):
        """Find matching image-mask pairs."""
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        images = sorted([f for f in os.listdir(self.image_dir)
                        if os.path.splitext(f)[1].lower() in valid_ext])
        for img_name in images:
            mask_name = img_name  # Assumes same naming convention
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.image_paths.append(os.path.join(self.image_dir, img_name))
                self.mask_paths.append(mask_path)
        print(f"Found {len(self.image_paths)} image-mask pairs")
        return self

    def load_image(self, path):
        """Load and preprocess a single image."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=-1)

    def load_mask(self, path):
        """Load and preprocess a segmentation mask."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {path}")
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        return np.expand_dims(mask, axis=-1)

    def split(self, random_state=42):
        """Split into training and validation sets."""
        train_img, val_img, train_mask, val_mask = train_test_split(
            self.image_paths, self.mask_paths,
            test_size=self.val_split, random_state=random_state,
        )
        print(f"Train: {len(train_img)} | Val: {len(val_img)}")
        return train_img, val_img, train_mask, val_mask

    def create_tf_dataset(self, image_paths, mask_paths, batch_size=8, shuffle=True):
        """Create a tf.data.Dataset pipeline."""
        def generator():
            for img_p, mask_p in zip(image_paths, mask_paths):
                yield self.load_image(img_p), self.load_mask(mask_p)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.image_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(*self.image_size, 1), dtype=tf.float32),
            ),
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
