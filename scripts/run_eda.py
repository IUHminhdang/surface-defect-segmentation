"""Run EDA for the raw KolektorSDD2 image/annotation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.kolektor import KolektorSDD2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/eda"))
    return parser.parse_args()


def collect_split(dataset: KolektorSDD2, split: str) -> tuple[list[dict[str, object]], list[tuple[str, np.ndarray, np.ndarray]]]:
    records: list[dict[str, object]] = []
    examples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, sample in enumerate(dataset.samples):
        try:
            item = dataset[index]
            image = np.asarray(Image.open(sample.image_path).convert("RGB"))
            mask = item["mask"].numpy()[0] > 0.5
            components = label(mask, connectivity=2)
            areas = sorted((region.area for region in regionprops(components)), reverse=True)
            height, width = mask.shape
            has_defect = bool(mask.any())
            records.append(
                {
                    "split": split,
                    "image_id": sample.image_id,
                    "filename": sample.image_path.name,
                    "width": width,
                    "height": height,
                    "channels": image.shape[2],
                    "aspect_ratio": round(width / height, 6),
                    "has_defect": has_defect,
                    "mask_area_px": int(mask.sum()),
                    "relative_defect_area_pct": round(float(mask.mean() * 100), 6),
                    "num_components": len(areas),
                    "largest_component_area_px": int(areas[0]) if areas else 0,
                    "annotation_present": sample.annotation_path is not None
                    or sample.mask_path is not None,
                    "integrity_status": "OK",
                }
            )
            if has_defect and len(examples) < 4:
                examples.append((sample.image_id, image, mask))
        except Exception as error:
            records.append(
                {
                    "split": split,
                    "image_id": sample.image_id,
                    "filename": sample.image_path.name,
                    "width": 0,
                    "height": 0,
                    "channels": 0,
                    "aspect_ratio": 0.0,
                    "has_defect": False,
                    "mask_area_px": 0,
                    "relative_defect_area_pct": 0.0,
                    "num_components": 0,
                    "largest_component_area_px": 0,
                    "annotation_present": False,
                    "integrity_status": f"ERROR: {type(error).__name__}: {error}",
                }
            )
    return records, examples


def choose_size_thresholds(frame: pd.DataFrame) -> dict[str, int | str]:
    train_areas = frame.loc[
        (frame["split"] == "train") & frame["has_defect"], "mask_area_px"
    ].to_numpy()
    if len(train_areas) < 3:
        raise ValueError("At least three defective train masks are required for size thresholds")
    small_max, medium_max = np.quantile(train_areas, [1 / 3, 2 / 3]).round().astype(int)
    return {
        "method": "train defective-mask area tertiles",
        "small_max_area_px": int(small_max),
        "medium_max_area_px": int(medium_max),
    }


def add_size_category(frame: pd.DataFrame, thresholds: dict[str, int | str]) -> pd.DataFrame:
    small_max = int(thresholds["small_max_area_px"])
    medium_max = int(thresholds["medium_max_area_px"])

    def classify(area: int) -> str:
        if area == 0:
            return "normal"
        if area <= small_max:
            return "small"
        if area <= medium_max:
            return "medium"
        return "large"

    result = frame.copy()
    result["defect_size"] = result["mask_area_px"].map(classify)
    return result


def save_figures(frame: pd.DataFrame, examples: list[tuple[str, np.ndarray, np.ndarray]], output_dir: Path) -> None:
    positive_counts = frame.groupby(["split", "has_defect"]).size().unstack(fill_value=0)
    positive_counts.rename(columns={False: "normal", True: "defective"}).plot(kind="bar")
    plt.title("Normal and defective images by split")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(output_dir / "positive_negative_by_split.png", dpi=160)
    plt.close()

    defective = frame[frame["has_defect"]]
    plt.hist(defective["mask_area_px"], bins=40)
    plt.yscale("log")
    plt.xlabel("Defect mask area (pixels)")
    plt.ylabel("Count (log scale)")
    plt.title("Defect mask area distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "defect_area_distribution.png", dpi=160)
    plt.close()

    plt.scatter(frame["width"], frame["height"], c=frame["has_defect"].astype(int), alpha=0.35)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image dimensions")
    plt.tight_layout()
    plt.savefig(output_dir / "image_dimensions.png", dpi=160)
    plt.close()

    if examples:
        figure, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
        axes = np.atleast_2d(axes)
        for row, (image_id, image, mask) in enumerate(examples):
            axes[row, 0].imshow(image)
            axes[row, 1].imshow(mask, cmap="gray")
            axes[row, 2].imshow(image)
            axes[row, 2].imshow(mask, cmap="Reds", alpha=0.4)
            axes[row, 0].set_title(image_id)
            axes[row, 1].set_title("Ground truth mask")
            axes[row, 2].set_title("Overlay")
            for axis in axes[row]:
                axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / "defect_examples.png", dpi=160)
        plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, object]] = []
    all_examples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for split in ("train", "test"):
        dataset = KolektorSDD2(args.data_root, split)
        records, examples = collect_split(dataset, split)
        all_records.extend(records)
        all_examples.extend(examples)

    frame = pd.DataFrame(all_records)
    thresholds = choose_size_thresholds(frame)
    frame = add_size_category(frame, thresholds)
    frame.to_csv(args.output_dir / "dataset_statistics.csv", index=False)
    (args.output_dir / "size_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    save_figures(frame, all_examples, args.output_dir)

    print(f"EDA rows: {len(frame)}")
    print(frame.groupby(["split", "has_defect"]).size())
    print("Size thresholds:", thresholds)
    print("Integrity:", frame["integrity_status"].value_counts().to_dict())
    print(f"EDA outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())