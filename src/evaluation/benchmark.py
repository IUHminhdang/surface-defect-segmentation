"""Evaluation utilities for official KolektorSDD2 test predictions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

from .boundary_metrics import boundary_metrics


def _sample_pixel_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
	true_positive = np.logical_and(prediction, target).sum()
	false_positive = np.logical_and(prediction, ~target).sum()
	false_negative = np.logical_and(~prediction, target).sum()
	epsilon = 1e-7
	return {
		"dice": float((2 * true_positive + epsilon) / (2 * true_positive + false_positive + false_negative + epsilon)),
		"iou": float((true_positive + epsilon) / (true_positive + false_positive + false_negative + epsilon)),
		"precision": float((true_positive + epsilon) / (true_positive + false_positive + epsilon)),
		"recall": float((true_positive + epsilon) / (true_positive + false_negative + epsilon)),
	}


def evaluate_model(
	model: nn.Module,
	loader: Iterable[dict[str, Any]],
	device: torch.device,
	small_max_area: int,
	medium_max_area: int,
	boundary_tolerance: int = 2,
	pixel_threshold: float = 0.5,
	image_defect_threshold: int = 0,
) -> dict[str, float]:
	"""Evaluate pixel, boundary, image, and defect-size metrics."""
	model.eval()
	overall: list[dict[str, float]] = []
	boundary_values: list[dict[str, float]] = []
	groups: dict[str, list[dict[str, float]]] = defaultdict(list)
	predicted_images: list[bool] = []
	expected_images: list[bool] = []

	with torch.inference_mode():
		for batch in loader:
			images = batch["image"].to(device)
			targets = batch["mask"].to(device)
			probabilities = model(images).sigmoid().cpu().numpy()[:, 0]
			expected = targets.cpu().numpy()[:, 0] >= 0.5
			predictions = probabilities >= pixel_threshold

			for prediction, target in zip(predictions, expected):
				pixel = _sample_pixel_metrics(prediction, target)
				boundary = boundary_metrics(
					prediction,
					target,
					tolerance=boundary_tolerance,
				)
				overall.append(pixel)
				boundary_values.append(boundary)
				area = int(target.sum())
				if area == 0:
					group = "normal"
				elif area <= small_max_area:
					group = "small"
				elif area <= medium_max_area:
					group = "medium"
				else:
					group = "large"
				if group != "normal":
					groups[group].append(pixel)
				predicted_images.append(int(prediction.sum()) > image_defect_threshold)
				expected_images.append(area > 0)

	if not overall:
		raise ValueError("Cannot evaluate an empty dataloader")

	result: dict[str, float] = {}
	for name in ("dice", "iou", "precision", "recall"):
		result[name] = float(np.mean([metrics[name] for metrics in overall]))
	for name in ("boundary_precision", "boundary_recall", "boundary_f1"):
		result[name] = float(np.mean([metrics[name] for metrics in boundary_values]))
	for group, values in groups.items():
		for name in ("dice", "iou", "precision", "recall"):
			result[f"{group}_{name}"] = float(np.mean([metrics[name] for metrics in values]))

	predicted_images_array = np.asarray(predicted_images, dtype=bool)
	expected_images_array = np.asarray(expected_images, dtype=bool)
	true_positive = np.logical_and(predicted_images_array, expected_images_array).sum()
	false_positive = np.logical_and(predicted_images_array, ~expected_images_array).sum()
	false_negative = np.logical_and(~predicted_images_array, expected_images_array).sum()
	normal_count = (~expected_images_array).sum()
	epsilon = 1e-7
	result.update(
		{
			"image_precision": float((true_positive + epsilon) / (true_positive + false_positive + epsilon)),
			"image_recall": float((true_positive + epsilon) / (true_positive + false_negative + epsilon)),
			"image_f1": float((2 * true_positive + epsilon) / (2 * true_positive + false_positive + false_negative + epsilon)),
			"normal_false_positive_rate": float((false_positive + epsilon) / (normal_count + epsilon)),
		}
	)
	return result
