
"""Training loop shared by E0-E4 experiments."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from tqdm.auto import tqdm

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

    def run_epoch(
        self,
        loader: Iterable[dict[str, object]],
        training: bool,
        epoch: int,
        total_epochs: int,
    ) -> dict[str, float]:
        self.model.train(training)

        total_loss = 0.0
        metric_totals = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
        defect_batches = 0
        batches = 0

        phase = "Train" if training else "Val"
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{total_epochs} [{phase}]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
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

            batch_metrics = segmentation_metrics(
                logits.detach(),
                masks,
            )

            if batch_metrics["defect_count"] > 0:
                metric_totals["dice"] += batch_metrics["dice"]
                metric_totals["iou"] += batch_metrics["iou"]
                defect_batches += 1

            metric_totals["precision"] += batch_metrics["precision"]
            metric_totals["recall"] += batch_metrics["recall"]
            batches += 1

            # Chỉ bổ sung hiển thị loss hiện tại, không đổi phép tính.
            pbar.set_postfix(loss=f"{loss.detach().item():.4f}")

        if batches == 0:
            raise ValueError("Cannot run an epoch with an empty dataloader")

        if defect_batches == 0:
            raise ValueError(
                "An epoch contains no defective samples; "
                "cannot compute Dice/IoU"
            )

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
        best_dice = max(
            (row["val_dice"] for row in history),
            default=-float("inf"),
        )
        stale_epochs = 0

        for epoch in range(len(history) + 1, epochs + 1):
            train_metrics = self.run_epoch(
                train_loader,
                training=True,
                epoch=epoch,
                total_epochs=epochs,
            )

            validation_metrics = self.run_epoch(
                validation_loader,
                training=False,
                epoch=epoch,
                total_epochs=epochs,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            history.append(
                {
                    "epoch": float(epoch),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{
                        f"train_{name}": value
                        for name, value in train_metrics.items()
                    },
                    **{
                        f"val_{name}": value
                        for name, value in validation_metrics.items()
                    },
                }
            )

            improved = (
                validation_metrics["dice"] > best_dice + self.min_delta
            )

            if improved:
                best_dice = validation_metrics["dice"]
                stale_epochs = 0

                self.best_state = {
                    "model_state_dict": deepcopy(
                        self.model.state_dict()
                    ),
                    "optimizer_state_dict": deepcopy(
                        self.optimizer.state_dict()
                    ),
                    "scheduler_state_dict": (
                        deepcopy(self.scheduler.state_dict())
                        if self.scheduler is not None
                        else None
                    ),
                    "best_epoch": epoch,
                }
            else:
                stale_epochs += 1

            # Tóm tắt kết quả sau mỗi epoch.
            best_marker = " [BEST]" if improved else ""
            print(
                f"\nEpoch {epoch}/{epochs}{best_marker}\n"
                f"  Train: Loss={train_metrics['loss']:.4f} | "
                f"Dice={train_metrics['dice']:.4f} | "
                f"IoU={train_metrics['iou']:.4f} | "
                f"Precision={train_metrics['precision']:.4f} | "
                f"Recall={train_metrics['recall']:.4f}\n"
                f"  Val:   Loss={validation_metrics['loss']:.4f} | "
                f"Dice={validation_metrics['dice']:.4f} | "
                f"IoU={validation_metrics['iou']:.4f} | "
                f"Precision={validation_metrics['precision']:.4f} | "
                f"Recall={validation_metrics['recall']:.4f}\n"
                f"  LR={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Early stopping={stale_epochs}/"
                f"{self.early_stopping_patience}",
                flush=True,
            )

            if (
                self.early_stopping_patience
                and stale_epochs >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best validation Dice: {best_dice:.4f}",
                    flush=True,
                )
                break

        return history