"""Main script for medical image segmentation pipeline."""

import argparse
import yaml
from src.data_loader import MedicalImageDataset
from src.train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Medical Image Segmentation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = MedicalImageDataset(
        image_dir=config["data"]["image_dir"],
        mask_dir=config["data"]["mask_dir"],
        image_size=tuple(config["data"]["image_size"]),
        val_split=config["data"]["val_split"],
    )
    dataset.discover_files()
    train_imgs, val_imgs, train_masks, val_masks = dataset.split()

    train_ds = dataset.create_tf_dataset(train_imgs, train_masks, config["data"]["batch_size"])
    val_ds = dataset.create_tf_dataset(val_imgs, val_masks, config["data"]["batch_size"], shuffle=False)

    if args.mode == "train":
        trainer = Trainer(config)
        trainer.build_model()
        trainer.train(train_ds, val_ds)
        trainer.save_model()
        print("Training complete.")


if __name__ == "__main__":
    main()
