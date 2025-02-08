"""Data augmentation for medical image segmentation."""

import numpy as np
import albumentations as A


class DataAugmentor:
    """Applies consistent augmentations to image-mask pairs."""

    def __init__(self, config=None):
        config = config or {}
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=config.get("rotation_limit", 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config.get("brightness_limit", 0.2),
                contrast_limit=0.2, p=0.3,
            ),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
        ]
        if config.get("elastic_transform", True):
            transforms.append(A.ElasticTransform(alpha=120, sigma=12, p=0.3))
        transforms.append(A.GridDistortion(p=0.2))
        self.transform = A.Compose(transforms)

    def augment(self, image, mask):
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        augmented = self.transform(image=image, mask=mask)
        aug_image = np.expand_dims(augmented["image"], axis=-1)
        aug_mask = np.expand_dims(augmented["mask"], axis=-1)
        return aug_image, aug_mask

    def augment_batch(self, images, masks):
        aug_images, aug_masks = [], []
        for img, msk in zip(images, masks):
            a_img, a_msk = self.augment(img, msk)
            aug_images.append(a_img)
            aug_masks.append(a_msk)
        return np.array(aug_images), np.array(aug_masks)
