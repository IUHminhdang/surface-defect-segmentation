"""Persist training history, checkpoints, and curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn


def save_training_artifacts(
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	history: list[dict[str, float]],
	output_dir: Path,
	config: dict[str, Any],
	scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
	best_state: dict[str, object] | None = None,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "history.json").write_text(json.dumps(history, indent=2))

	fieldnames = list(history[0]) if history else []
	with (output_dir / "history.csv").open("w", newline="") as history_file:
		writer = csv.DictWriter(history_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(history)

	best = max(history, key=lambda row: row["val_dice"]) if history else None
	state = best_state or {
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict() if scheduler else None,
		"best_epoch": best["epoch"] if best else None,
	}
	torch.save(
		{
			**state,
			"history": history,
			"config": config,
		},
		output_dir / "best.pt",
	)

	for metric in ("loss", "dice", "iou"):
		figure, axis = plt.subplots(figsize=(8, 5))
		epochs = [row["epoch"] for row in history]
		axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="train")
		axis.plot(epochs, [row[f"val_{metric}"] for row in history], label="validation")
		axis.set_xlabel("Epoch")
		axis.set_ylabel(metric.upper())
		axis.set_title(f"Train vs validation {metric.upper()}")
		axis.legend()
		axis.grid(alpha=0.25)
		figure.tight_layout()
		figure.savefig(output_dir / f"{metric}_curve.png", dpi=160)
		plt.close(figure)
