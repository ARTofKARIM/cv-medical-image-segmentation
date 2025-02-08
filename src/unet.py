"""U-Net architecture for medical image segmentation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class UNet:
    """Configurable U-Net model for binary segmentation."""

    def __init__(self, input_size=(256, 256, 1), encoder_filters=None,
                 bottleneck_filters=1024, dropout=0.3):
        self.input_size = input_size
        self.encoder_filters = encoder_filters or [64, 128, 256, 512]
        self.bottleneck_filters = bottleneck_filters
        self.dropout = dropout
        self.model = None

    def _conv_block(self, x, filters, name_prefix):
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
        return x

    def _encoder_block(self, x, filters, name_prefix):
        skip = self._conv_block(x, filters, name_prefix)
        pool = layers.MaxPooling2D(2, name=f"{name_prefix}_pool")(skip)
        pool = layers.Dropout(self.dropout, name=f"{name_prefix}_drop")(pool)
        return skip, pool

    def _decoder_block(self, x, skip, filters, name_prefix):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                                    name=f"{name_prefix}_upconv")(x)
        x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip])
        x = layers.Dropout(self.dropout, name=f"{name_prefix}_drop")(x)
        x = self._conv_block(x, filters, name_prefix)
        return x

    def build(self):
        inputs = keras.Input(shape=self.input_size, name="input_image")
        skips = []
        x = inputs

        # Encoder
        for i, filters in enumerate(self.encoder_filters):
            skip, x = self._encoder_block(x, filters, f"enc{i+1}")
            skips.append(skip)

        # Bottleneck
        x = self._conv_block(x, self.bottleneck_filters, "bottleneck")

        # Decoder
        for i, filters in enumerate(reversed(self.encoder_filters)):
            x = self._decoder_block(x, skips[-(i+1)], filters, f"dec{i+1}")

        # Output
        outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(x)
        self.model = keras.Model(inputs, outputs, name="UNet")
        print(f"U-Net built: {self.model.count_params():,} parameters")
        return self.model

    def summary(self):
        if self.model:
            self.model.summary()
