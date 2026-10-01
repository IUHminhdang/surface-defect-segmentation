"""KolektorSDD2 dataset loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO
import base64
import json
import zlib

from .transforms import Pair


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
	image_path: Path
	mask_path: Path | None
	annotation_path: Path | None
	image_id: str


def _image_files(directory: Path) -> list[Path]:
	return sorted(
		path
		for path in directory.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	)


def _find_by_stem(directory: Path) -> dict[str, Path]:
	return {path.stem: path for path in _image_files(directory)}


def _find_datasetninja_annotations(directory: Path) -> dict[str, Path]:
	return {
		Path(path.name.removesuffix(".json")).stem: path
		for path in directory.glob("*.json")
	}


def _decode_datasetninja_mask(annotation_path: Path, image_size: tuple[int, int]) -> Image.Image:
	annotation = json.loads(annotation_path.read_text())
	mask = Image.new("L", image_size, 0)
	for object_data in annotation.get("objects", []):
		bitmap = object_data.get("bitmap")
		if not bitmap:
			continue
		compressed = base64.b64decode(bitmap["data"])
		bitmap_bytes = zlib.decompress(compressed)
		object_mask = Image.open(BytesIO(bitmap_bytes)).convert("L")
		origin = tuple(bitmap.get("origin", (0, 0)))
		mask.paste(object_mask, origin, object_mask)
	return mask


class KolektorSDD2(Dataset[dict[str, object]]):
	"""Load one official KolektorSDD2 split.

	The expected layout is ``<root>/<split>/images`` and
	``<root>/<split>/ground_truth``. Missing masks are treated as normal-image
	samples with an all-zero target.
	"""

	def __init__(
		self,
		root: str | Path,
		split: str,
		image_dir_name: str = "images",
		mask_dir_name: str = "ground_truth",
		image_size: int | tuple[int, int] | None = None,
		transform: Callable[[Image.Image, Image.Image], Pair] | None = None,
	) -> None:
		self.root = Path(root)
		split_root = self.root / split
		image_directory = split_root / image_dir_name
		mask_directory = split_root / mask_dir_name

		self.datasetninja = False
		if image_directory.is_dir() and mask_directory.is_dir():
			masks = _find_by_stem(mask_directory)
			image_paths = _image_files(image_directory)
			self.samples = [
				Sample(image_path, masks.get(image_path.stem), None, image_path.stem)
				for image_path in image_paths
			]
		else:
			image_directory = split_root / "img"
			annotation_directory = split_root / "ann"
			if not image_directory.is_dir() or not annotation_directory.is_dir():
				raise FileNotFoundError(
					f"Expected either {split_root / image_dir_name} + "
					f"{split_root / mask_dir_name}, or {image_directory} + {annotation_directory}"
				)
			annotations = _find_datasetninja_annotations(annotation_directory)
			self.datasetninja = True
			self.samples = [
				Sample(image_path, None, annotations.get(image_path.stem), image_path.stem)
				for image_path in _image_files(image_directory)
			]
		if not self.samples:
			raise ValueError(f"No images found in {image_directory}")

		self.image_size = image_size
		self.transform = transform

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> dict[str, object]:
		sample = self.samples[index]
		image = Image.open(sample.image_path).convert("RGB")
		if self.datasetninja and sample.annotation_path is not None:
			mask = _decode_datasetninja_mask(sample.annotation_path, image.size)
		elif sample.mask_path is not None:
			mask = Image.open(sample.mask_path).convert("L")
		else:
			mask = Image.new("L", image.size)

		if self.transform is not None:
			image, mask = self.transform(image, mask)
		elif self.image_size is not None:
			size = (
				(self.image_size, self.image_size)
				if isinstance(self.image_size, int)
				else self.image_size
			)
			image = image.resize(size[::-1], Image.Resampling.BILINEAR)
			mask = mask.resize(size[::-1], Image.Resampling.NEAREST)

		image_array = np.asarray(image, dtype=np.float32) / 255.0
		mask_array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
		image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
		mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

		return {
			"image": image_tensor,
			"mask": mask_tensor,
			"image_id": sample.image_id,
			"is_defective": bool(mask_tensor.any()),
		}
