"""Training loop shared by E0-E4 experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from ..evaluation.metrics import segmentation_metrics
from ..losses.combined import combined_loss


class Trainer:
	def __init__(
		self,
		model: nn.Module,
		optimizer: torch.optim.Optimizer,
		device: torch.device,
		use_boundary: bool = False,
		boundary_weight: float = 0.1,
	) -> None:
		self.model = model.to(device)
		self.optimizer = optimizer
		self.device = device
		self.use_boundary = use_boundary
		self.boundary_weight = boundary_weight

	def run_epoch(self, loader: Iterable[dict[str, object]], training: bool) -> dict[str, float]:
		self.model.train(training)
		total_loss = 0.0
		metric_totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
		batches = 0

		for batch in loader:
			images = batch["image"].to(self.device)
			masks = batch["mask"].to(self.device)
			with torch.set_grad_enabled(training):
				logits = self.model(images)
				loss = combined_loss(
					logits,
					masks,
					use_boundary=self.use_boundary,
					boundary_weight=self.boundary_weight,
				)
				if training:
					self.optimizer.zero_grad(set_to_none=True)
					loss.backward()
					self.optimizer.step()

			total_loss += loss.detach().item()
			batch_metrics = segmentation_metrics(logits.detach(), masks)
			for name in metric_totals:
				metric_totals[name] += batch_metrics[name]
			batches += 1

		if batches == 0:
			raise ValueError("Cannot run an epoch with an empty dataloader")
		return {
			"loss": total_loss / batches,
			**{name: value / batches for name, value in metric_totals.items()},
		}

	def fit(
		self,
		train_loader: Iterable[dict[str, object]],
		validation_loader: Iterable[dict[str, object]],
		epochs: int,
	) -> list[dict[str, float]]:
		history: list[dict[str, float]] = []
		for epoch in range(1, epochs + 1):
			train_metrics = self.run_epoch(train_loader, training=True)
			validation_metrics = self.run_epoch(validation_loader, training=False)
			history.append(
				{
					"epoch": float(epoch),
					**{f"train_{name}": value for name, value in train_metrics.items()},
					**{f"val_{name}": value for name, value in validation_metrics.items()},
				}
			)
		return history
