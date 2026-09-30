"""Command-line entry point for baseline and ablation training."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

from ..datasets.kolektor import KolektorSDD2
from ..datasets.transforms import Compose, RandomHorizontalFlip, Resize
from ..models.resnet34_unet import ResNet34UNet
from .trainer import Trainer


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
	parser.add_argument("--data-root", type=Path, default=None)
	parser.add_argument("--experiment", default="E0_baseline")
	parser.add_argument("--epochs", type=int, default=None)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	config = yaml.safe_load(args.config.read_text())
	set_seed(config["project"]["seed"])
	experiment = config["experiments"][args.experiment]
	data_root = args.data_root or Path(config["paths"]["data_root"])
	image_size = config["data"]["image_size"]
	train_dataset_with_augmentation = KolektorSDD2(
		data_root,
		"train",
		image_size=image_size,
		transform=Compose([Resize(image_size), RandomHorizontalFlip()]),
	)
	train_dataset_without_augmentation = KolektorSDD2(
		data_root,
		"train",
		image_size=image_size,
		transform=Compose([Resize(image_size)]),
	)
	train_size = int(0.8 * len(train_dataset_with_augmentation))
	validation_size = len(train_dataset_with_augmentation) - train_size
	split_generator = torch.Generator().manual_seed(config["project"]["seed"])
	train_indices, validation_indices = random_split(
		range(len(train_dataset_with_augmentation)),
		[train_size, validation_size],
		generator=split_generator,
	)
	train_dataset = Subset(train_dataset_with_augmentation, train_indices.indices)
	validation_dataset = Subset(
		train_dataset_without_augmentation,
		validation_indices.indices,
	)
	train_loader = DataLoader(
		train_dataset,
		batch_size=config["data"]["batch_size"],
		shuffle=True,
		num_workers=config["data"]["num_workers"],
	)
	validation_loader = DataLoader(
		validation_dataset,
		batch_size=config["data"]["batch_size"],
		shuffle=False,
		num_workers=config["data"]["num_workers"],
	)
	device_name = config["training"]["device"]
	device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
	model = ResNet34UNet(
		multi_scale=experiment["multi_scale"],
		attention=experiment["attention"],
	)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=config["training"]["learning_rate"],
		weight_decay=config["training"]["weight_decay"],
	)
	trainer = Trainer(
		model,
		optimizer,
		device,
		use_boundary=experiment["boundary_loss"],
		boundary_weight=config["loss"]["boundary_weight"],
	)
	history = trainer.fit(
		train_loader,
		validation_loader,
		epochs=args.epochs or config["training"]["epochs"],
	)
	print(f"TRAINING COMPLETE: {args.experiment}, epochs={len(history)}, device={device}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
