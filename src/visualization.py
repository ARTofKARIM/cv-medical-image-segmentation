"""Visualization tools for segmentation results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SegmentationVisualizer:
    """Generates plots for medical image segmentation."""

    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def plot_prediction_overlay(self, image, mask_true, mask_pred, save_path=None):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(image.squeeze(), cmap="gray")
        axes[0].set_title("Input Image")
        axes[1].imshow(mask_true.squeeze(), cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(mask_pred.squeeze(), cmap="gray")
        axes[2].set_title("Prediction")
        axes[3].imshow(image.squeeze(), cmap="gray")
        axes[3].imshow(mask_pred.squeeze(), cmap="jet", alpha=0.4)
        axes[3].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_training_curves(self, history, save=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history["loss"], label="Train")
        if "val_loss" in history.history:
            ax1.plot(history.history["val_loss"], label="Val")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        if "dice_coefficient" in history.history:
            ax2.plot(history.history["dice_coefficient"], label="Train")
            if "val_dice_coefficient" in history.history:
                ax2.plot(history.history["val_dice_coefficient"], label="Val")
        ax2.set_title("Dice Coefficient")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        if save:
            fig.savefig(f"{self.output_dir}training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_sample_grid(self, images, masks_true, masks_pred, n=4, save=True):
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        for i in range(min(n, len(images))):
            axes[i, 0].imshow(images[i].squeeze(), cmap="gray")
            axes[i, 1].imshow(masks_true[i].squeeze(), cmap="gray")
            axes[i, 2].imshow(masks_pred[i].squeeze(), cmap="gray")
            if i == 0:
                axes[i, 0].set_title("Image")
                axes[i, 1].set_title("Ground Truth")
                axes[i, 2].set_title("Prediction")
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")
        if save:
            fig.savefig(f"{self.output_dir}sample_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
