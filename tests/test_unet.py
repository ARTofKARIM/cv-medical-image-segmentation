"""Unit tests for U-Net model."""

import unittest
import numpy as np
import tensorflow as tf
from src.unet import UNet
from src.losses import dice_loss, bce_dice_loss, dice_coefficient


class TestUNet(unittest.TestCase):

    def test_output_shape(self):
        unet = UNet(input_size=(128, 128, 1), encoder_filters=[32, 64])
        model = unet.build()
        dummy = np.random.randn(2, 128, 128, 1).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        self.assertEqual(output.shape, (2, 128, 128, 1))

    def test_output_range(self):
        unet = UNet(input_size=(64, 64, 1), encoder_filters=[16, 32])
        model = unet.build()
        dummy = np.random.randn(1, 64, 64, 1).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))


class TestLosses(unittest.TestCase):

    def test_dice_perfect(self):
        y = tf.constant([[1.0, 1.0, 0.0, 0.0]])
        score = dice_coefficient(y, y).numpy()
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_dice_loss_range(self):
        y_true = tf.constant([[1.0, 0.0, 1.0]])
        y_pred = tf.constant([[0.8, 0.2, 0.7]])
        loss = dice_loss(y_true, y_pred).numpy()
        self.assertGreater(loss, 0)
        self.assertLess(loss, 1)


if __name__ == "__main__":
    unittest.main()
