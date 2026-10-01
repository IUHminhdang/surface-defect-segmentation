"""Pixel- and image-level segmentation metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor


def _binary_masks(logits: Tensor, targets: Tensor, threshold: float) -> tuple[np.ndarray, np.ndarray]:
	probabilities = logits.detach().sigmoid().cpu().numpy()
	predictions = probabilities >= threshold
	expected = targets.detach().cpu().numpy() >= 0.5
	return predictions.squeeze(1), expected.squeeze(1)


def segmentation_metrics(
	logits: Tensor,
	targets: Tensor,
	threshold: float = 0.5,
	epsilon: float = 1e-7,
) -> Mapping[str, float]:
	predictions, expected = _binary_masks(logits, targets, threshold)
	positive_samples = expected.reshape(expected.shape[0], -1).sum(axis=1) > 0
	defect_count = int(positive_samples.sum())
	metric_predictions = predictions[positive_samples] if defect_count else predictions
	metric_expected = expected[positive_samples] if defect_count else expected
	true_positive = np.logical_and(metric_predictions, metric_expected).sum()
	false_positive = np.logical_and(metric_predictions, ~metric_expected).sum()
	false_negative = np.logical_and(~metric_predictions, metric_expected).sum()
	dice = (2 * true_positive + epsilon) / (2 * true_positive + false_positive + false_negative + epsilon)
	iou = (true_positive + epsilon) / (true_positive + false_positive + false_negative + epsilon)
	all_true_positive = np.logical_and(predictions, expected).sum()
	all_false_positive = np.logical_and(predictions, ~expected).sum()
	all_false_negative = np.logical_and(~predictions, expected).sum()
	precision = (all_true_positive + epsilon) / (all_true_positive + all_false_positive + epsilon)
	recall = (all_true_positive + epsilon) / (all_true_positive + all_false_negative + epsilon)
	return {
		"dice": float(dice) if defect_count else 0.0,
		"iou": float(iou) if defect_count else 0.0,
		"precision": float(precision),
		"recall": float(recall),
		"defect_count": float(defect_count),
	}

def image_level_metrics(
	logits: Tensor,
	targets: Tensor,
	pixel_threshold: float = 0.5,
	defect_area_threshold: int = 0,
	epsilon: float = 1e-7,
) -> Mapping[str, float]:
	predictions, expected = _binary_masks(logits, targets, pixel_threshold)
	predicted_defect = predictions.reshape(predictions.shape[0], -1).sum(axis=1) > defect_area_threshold
	expected_defect = expected.reshape(expected.shape[0], -1).sum(axis=1) > 0
	true_positive = np.logical_and(predicted_defect, expected_defect).sum()
	false_positive = np.logical_and(predicted_defect, ~expected_defect).sum()
	false_negative = np.logical_and(~predicted_defect, expected_defect).sum()
	normal_count = (~expected_defect).sum()
	return {
		"image_precision": float((true_positive + epsilon) / (true_positive + false_positive + epsilon)),
		"image_recall": float((true_positive + epsilon) / (true_positive + false_negative + epsilon)),
		"image_f1": float((2 * true_positive + epsilon) / (2 * true_positive + false_positive + false_negative + epsilon)),
		"normal_false_positive_rate": float((false_positive + epsilon) / (normal_count + epsilon)),
	}


def defect_size_bucket(area: int, small_max: int, medium_max: int) -> str:
	if area <= small_max:
		return "small"
	if area <= medium_max:
		return "medium"
	return "large"
