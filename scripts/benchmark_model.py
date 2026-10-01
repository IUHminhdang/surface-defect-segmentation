"""Run one trained experiment on the official KolektorSDD2 test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.kolektor import KolektorSDD2
from src.datasets.transforms import Compose, Resize
from src.evaluation.benchmark import evaluate_model
from src.models.resnet34_unet import ResNet34UNet


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-root", type=Path, required=True)
	parser.add_argument("--experiment", default="E0_baseline")
	parser.add_argument("--checkpoint", type=Path, default=None)
	parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
	parser.add_argument("--thresholds", type=Path, default=Path("results/eda/size_thresholds.json"))
	parser.add_argument("--output", type=Path, default=None)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	config = yaml.safe_load(args.config.read_text())
	experiment_config = config["experiments"][args.experiment]
	checkpoint_path = args.checkpoint or Path(config["paths"]["experiments_root"]) / args.experiment / "best.pt"
	thresholds = json.loads(args.thresholds.read_text())
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = KolektorSDD2(
		args.data_root,
		"test",
		image_size=config["data"]["image_size"],
		transform=Compose([Resize(config["data"]["image_size"])]),
	)
	loader = DataLoader(
		dataset,
		batch_size=config["data"]["batch_size"],
		shuffle=False,
		num_workers=config["data"]["num_workers"],
	)
	model = ResNet34UNet(
		multi_scale=experiment_config["multi_scale"],
		attention=experiment_config["attention"],
	)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	metrics = evaluate_model(
		model,
		loader,
		device,
		small_max_area=int(thresholds["small_max_area_px"]),
		medium_max_area=int(thresholds["medium_max_area_px"]),
		boundary_tolerance=config["evaluation"]["boundary_tolerance"],
		image_defect_threshold=int(config["evaluation"]["image_defect_threshold"]),
	)
	output_path = args.output or Path(config["paths"]["results_root"]) / f"metrics_{args.experiment}.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(metrics, indent=2))
	print(json.dumps(metrics, indent=2))
	print(f"METRICS SAVED: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
