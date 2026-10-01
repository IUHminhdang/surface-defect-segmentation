"""Validate the on-disk dataset contract before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_DIRECTORY_NAMES = ("ground_truth", "masks", "mask", "labels", "annotations")


def find_files(directory: Path) -> list[Path]:
	return sorted(
		path
		for path in directory.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	)


def verify_dataset(data_root: Path, split: str | None = None) -> tuple[int, int, int]:
	if not data_root.exists():
		raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
	if not data_root.is_dir():
		raise NotADirectoryError(f"Dataset root is not a directory: {data_root}")

	split_roots = [data_root / split] if split else [
		candidate for candidate in (data_root / "train", data_root / "test")
		if candidate.is_dir()
	]
	if not split_roots:
		split_roots = [data_root]

	image_count = annotated_count = 0
	for split_root in split_roots:
		image_directory = split_root / "images"
		annotation_directory = split_root / "ann"
		if not image_directory.is_dir() and (split_root / "img").is_dir():
			image_directory = split_root / "img"
		images = find_files(image_directory) if image_directory.is_dir() else []

		if annotation_directory.is_dir():
			annotated = {
				Path(path.name.removesuffix(".json")).stem
				for path in annotation_directory.glob("*.json")
				if json.loads(path.read_text()).get("objects", [])
			}
			annotated_count += sum(path.stem in annotated for path in images)
		else:
			mask_directories = [
				directory
				for name in MASK_DIRECTORY_NAMES
				for directory in split_root.rglob(name)
				if directory.is_dir()
			]
			masks_by_stem = {
				path.stem: path
				for directory in mask_directories
				for path in find_files(directory)
			}
			annotated_count += sum(path.stem in masks_by_stem for path in images)
		image_count += len(images)

	if not image_count:
		raise ValueError(f"No image files found below: {data_root}")
	if not annotated_count:
		raise ValueError("No non-empty mask annotations found")
	return image_count, annotated_count, image_count - annotated_count


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--data-root",
		type=Path,
		default=Path("data/raw"),
		help="Root directory containing the dataset (default: data/raw)",
	)
	parser.add_argument(
		"--split",
		choices=("train", "test"),
		help="Validate only one official split; otherwise validate the root.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		image_count, annotated_count, normal_count = verify_dataset(
			args.data_root, args.split
		)
	except (FileNotFoundError, NotADirectoryError, ValueError) as error:
		print(f"DATASET INVALID: {error}")
		return 1

	print(
		"DATASET OK: "
		f"{image_count} images, {annotated_count} annotated, "
		f"{normal_count} normal"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
