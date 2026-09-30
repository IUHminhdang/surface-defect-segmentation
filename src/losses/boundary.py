"""Differentiable boundary-aware loss for binary masks."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def _boundary_map(values: Tensor, kernel_size: int = 3) -> Tensor:
	padding = kernel_size // 2
	dilated = F.max_pool2d(values, kernel_size, stride=1, padding=padding)
	eroded = -F.max_pool2d(-values, kernel_size, stride=1, padding=padding)
	return (dilated - eroded).clamp(0.0, 1.0)


def boundary_loss(logits: Tensor, targets: Tensor) -> Tensor:
	probabilities = logits.sigmoid()
	predicted_boundary = _boundary_map(probabilities)
	target_boundary = (_boundary_map(targets.float()) > 0).float()
	return F.binary_cross_entropy(predicted_boundary.clamp(1e-6, 1.0 - 1e-6), target_boundary)
