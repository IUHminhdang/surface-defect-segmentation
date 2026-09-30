# Surface Defect Segmentation

Project for segmenting surface defects with a reproducible local and Kaggle workflow.

## Project status

The repository is being built in small, validated phases:

1. **Setup:** dependency list, paths, dataset verification, and reproducibility.
2. **Data:** Kolektor dataset loading, masks, transforms, and train/validation/test splits.
3. **Baseline:** ResNet34-UNet, Dice/BCE losses, training loop, checkpoints, and metrics.
4. **Experiments:** multiscale, attention, boundary loss, and full model variants.
5. **Evaluation:** benchmark tables, visualizations, ablations, and final Kaggle run.

Do not commit files ignored by `.gitignore`, especially raw/processed datasets,
checkpoints, logs, and generated predictions.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Kaggle workflow

1. Push the code repository without the dataset.
2. Add the dataset to the Kaggle notebook or attach it as a Kaggle Dataset.
3. Clone or pull the repository into `/kaggle/working`.
4. Run the dataset verification script before training.
5. Run one small baseline smoke test before a full experiment.

The active paths and training defaults live in `configs/base.yaml`.

## Training

After the dataset is available under `data/raw`, verify it first:

```powershell
python scripts/verify_dataset.py --data-root data/raw
```

Run one experiment as a Python module so package imports work in both local and
Kaggle environments:

```powershell
python -m src.training.train --experiment E0_baseline --epochs 1
```

The available experiment names are `E0_baseline`, `E1_multiscale`,
`E2_attention`, `E3_boundary`, and `E4_full`.
