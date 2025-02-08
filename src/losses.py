"""Custom loss functions for segmentation tasks."""

import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
    return -K.mean(alpha_t * K.pow(1 - pt, gamma) * K.log(pt))


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1.0):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    tp = K.sum(y_true_flat * y_pred_flat)
    fp = K.sum((1 - y_true_flat) * y_pred_flat)
    fn = K.sum(y_true_flat * (1 - y_pred_flat))
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - tversky


LOSS_REGISTRY = {
    "dice": dice_loss,
    "bce_dice": bce_dice_loss,
    "focal": focal_loss,
    "tversky": tversky_loss,
    "bce": tf.keras.losses.binary_crossentropy,
}


def get_loss(name):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]
