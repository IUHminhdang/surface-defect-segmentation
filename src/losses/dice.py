"""Dice loss for binary segmentation."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def dice_loss(logits: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
	probabilities = logits.sigmoid().flatten(1)
	targets = targets.float().flatten(1)
	intersection = (probabilities * targets).sum(dim=1)
	denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
	dice = (2.0 * intersection + smooth) / (denominator + smooth)
	return 1.0 - dice.mean()


def binary_dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
	return dice_loss(logits, targets)
