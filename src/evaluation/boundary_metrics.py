"""Boundary-level metrics with configurable pixel tolerance."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from skimage.morphology import binary_dilation, binary_erosion, footprint_rectangle


def _boundary(mask: np.ndarray) -> np.ndarray:
	return np.logical_xor(mask, binary_erosion(mask))


def boundary_metrics(
	predictions: np.ndarray,
	targets: np.ndarray,
	tolerance: int = 2,
	epsilon: float = 1e-7,
) -> Mapping[str, float]:
	predictions = np.asarray(predictions, dtype=bool)
	targets = np.asarray(targets, dtype=bool)
	if predictions.ndim == 2:
		predictions = predictions[None, ...]
		targets = targets[None, ...]

	footprint = footprint_rectangle((2 * tolerance + 1, 2 * tolerance + 1))
	precision_values: list[float] = []
	recall_values: list[float] = []
	for predicted, expected in zip(predictions, targets):
		predicted_boundary = _boundary(predicted)
		expected_boundary = _boundary(expected)
		predicted_match = np.logical_and(predicted_boundary, binary_dilation(expected_boundary, footprint)).sum()
		expected_match = np.logical_and(expected_boundary, binary_dilation(predicted_boundary, footprint)).sum()
		precision_values.append(float((predicted_match + epsilon) / (predicted_boundary.sum() + epsilon)))
		recall_values.append(float((expected_match + epsilon) / (expected_boundary.sum() + epsilon)))

	precision = float(np.mean(precision_values))
	recall = float(np.mean(recall_values))
	f1 = (2 * precision * recall + epsilon) / (precision + recall + epsilon)
	return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": float(f1)}
