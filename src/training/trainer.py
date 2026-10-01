"""Training loop shared by E0-E4 experiments."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

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
		scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
		use_boundary: bool = False,
		boundary_weight: float = 0.1,
		early_stopping_patience: int = 0,
		min_delta: float = 0.0,
	) -> None:
		self.model = model.to(device)
		self.optimizer = optimizer
		self.device = device
		self.scheduler = scheduler
		self.use_boundary = use_boundary
		self.boundary_weight = boundary_weight
		self.early_stopping_patience = early_stopping_patience
		self.min_delta = min_delta
		self.best_state: dict[str, object] | None = None

	def run_epoch(self, loader: Iterable[dict[str, object]], training: bool) -> dict[str, float]:
		self.model.train(training)
		total_loss = 0.0
		metric_totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
		defect_batches = 0
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
			if batch_metrics["defect_count"] > 0:
				metric_totals["dice"] += batch_metrics["dice"]
				metric_totals["iou"] += batch_metrics["iou"]
				defect_batches += 1
			metric_totals["precision"] += batch_metrics["precision"]
			metric_totals["recall"] += batch_metrics["recall"]
			batches += 1

		if batches == 0:
			raise ValueError("Cannot run an epoch with an empty dataloader")
		if defect_batches == 0:
			raise ValueError("An epoch contains no defective samples; cannot compute Dice/IoU")
		return {
			"loss": total_loss / batches,
			"dice": metric_totals["dice"] / defect_batches,
			"iou": metric_totals["iou"] / defect_batches,
			"precision": metric_totals["precision"] / batches,
			"recall": metric_totals["recall"] / batches,
		}

	def fit(
		self,
		train_loader: Iterable[dict[str, object]],
		validation_loader: Iterable[dict[str, object]],
		epochs: int,
		initial_history: list[dict[str, float]] | None = None,
	) -> list[dict[str, float]]:
		history = list(initial_history or [])
		best_dice = max((row["val_dice"] for row in history), default=-float("inf"))
		stale_epochs = 0
		for epoch in range(len(history) + 1, epochs + 1):
			train_metrics = self.run_epoch(train_loader, training=True)
			validation_metrics = self.run_epoch(validation_loader, training=False)
			if self.scheduler is not None:
				self.scheduler.step()
			history.append(
				{
					"epoch": float(epoch),
					"learning_rate": self.optimizer.param_groups[0]["lr"],
					**{f"train_{name}": value for name, value in train_metrics.items()},
					**{f"val_{name}": value for name, value in validation_metrics.items()},
				}
			)
			if validation_metrics["dice"] > best_dice + self.min_delta:
				best_dice = validation_metrics["dice"]
				stale_epochs = 0
				self.best_state = {
					"model_state_dict": deepcopy(self.model.state_dict()),
					"optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
					"scheduler_state_dict": deepcopy(self.scheduler.state_dict()) if self.scheduler else None,
					"best_epoch": epoch,
				}
			else:
				stale_epochs += 1
			if self.early_stopping_patience and stale_epochs >= self.early_stopping_patience:
				break
		return history
