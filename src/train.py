"""Training loop for medical image segmentation."""

import os
import tensorflow as tf
from tensorflow.keras import callbacks
from src.unet import UNet
from src.losses import get_loss, dice_coefficient


class Trainer:
    """Handles model training with callbacks and monitoring."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None

    def build_model(self):
        model_cfg = self.config["model"]
        unet = UNet(
            input_size=(*self.config["data"]["image_size"], 1),
            encoder_filters=model_cfg["encoder_filters"],
            bottleneck_filters=model_cfg["bottleneck_filters"],
            dropout=model_cfg["dropout"],
        )
        self.model = unet.build()
        loss_fn = get_loss(self.config["training"]["loss"])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["training"]["learning_rate"]),
            loss=loss_fn,
            metrics=[dice_coefficient, "accuracy"],
        )
        return self.model

    def get_callbacks(self):
        cb = [
            callbacks.EarlyStopping(
                monitor="val_dice_coefficient", mode="max",
                patience=self.config["training"]["early_stopping_patience"],
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=self.config["training"]["reduce_lr_patience"],
                min_lr=1e-7,
            ),
            callbacks.ModelCheckpoint(
                "models/best_model.keras", monitor="val_dice_coefficient",
                mode="max", save_best_only=True,
            ),
        ]
        return cb

    def train(self, train_dataset, val_dataset):
        if self.model is None:
            self.build_model()
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config["training"]["epochs"],
            callbacks=self.get_callbacks(),
            verbose=1,
        )
        return self.history

    def save_model(self, path="models/final_model.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
