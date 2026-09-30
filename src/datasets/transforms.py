"""Paired image-mask transforms for surface-defect segmentation."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from PIL import Image


Pair = tuple[Image.Image, Image.Image]


class Compose:
	def __init__(self, transforms: Iterable[Callable[[Image.Image, Image.Image], Pair]]) -> None:
		self.transforms = list(transforms)

	def __call__(self, image: Image.Image, mask: Image.Image) -> Pair:
		for transform in self.transforms:
			image, mask = transform(image, mask)
		return image, mask


class Resize:
	def __init__(self, size: int | tuple[int, int]) -> None:
		self.size = (size, size) if isinstance(size, int) else size

	def __call__(self, image: Image.Image, mask: Image.Image) -> Pair:
		width, height = self.size[1], self.size[0]
		return (
			image.resize((width, height), Image.Resampling.BILINEAR),
			mask.resize((width, height), Image.Resampling.NEAREST),
		)


class RandomHorizontalFlip:
	def __init__(self, probability: float = 0.5) -> None:
		self.probability = probability

	def __call__(self, image: Image.Image, mask: Image.Image) -> Pair:
		if random.random() >= self.probability:
			return image, mask
		return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(
			Image.Transpose.FLIP_LEFT_RIGHT
		)


class RandomVerticalFlip:
	def __init__(self, probability: float = 0.5) -> None:
		self.probability = probability

	def __call__(self, image: Image.Image, mask: Image.Image) -> Pair:
		if random.random() >= self.probability:
			return image, mask
		return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(
			Image.Transpose.FLIP_TOP_BOTTOM
		)
