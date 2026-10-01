# Dataset

## Dataset Name

**KolektorSDD2** (Kolektor Surface Defect Detection Dataset 2)

## Task Type

Industrial surface defect segmentation (semantic segmentation)

## Input Data

- **Format**: RGB images
- **Typical Resolution**: 256×256 pixels (after preprocessing)
- **Color Space**: RGB (3 channels)

## Labels

- **Format**: Binary masks
- **Annotation Style**: White pixels = defect, Black pixels = background
- **Class Distribution**:
  - Background (normal surface): ~89.3%
  - Defect: ~10.7%

## Dataset Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Total Images | 3,335 | 100% |
| Defective Images | 356 | 10.7% |
| Normal Images | 2,979 | 89.3% |
| Train Set | 2,331 | 69.9% |
| Test Set | 1,004 | 30.1% |

## Important Characteristics

### 1. Tiny Defects
Defects occupy only a small portion of the image:
- Average defect area: < 1% of image area
- Many defects are < 10 pixels in size
- Very challenging for standard segmentation models

### 2. Low Contrast
Defects often have similar intensity to background textures:
- Visual similarity between defect and normal surface
- Requires attention mechanisms to distinguish features

### 3. Class Imbalance
Severe imbalance between defect and background:
- Background pixels dominate (~90%)
- Defect pixels are rare (~10%)
- Requires careful handling in loss functions

### 4. Small Sample Size
Limited number of defective images:
- Only 356 defective images for training
- May require data augmentation

## Directory Structure

```
kolektorsdd2-DatasetNinja/
├── train/
│   ├── img/           # Training images
│   └── ann/           # Training annotations
├── test/
│   ├── img/           # Test images
│   └── ann/           # Test annotations
├── meta.json          # Dataset metadata
├── README.md          # Dataset documentation
└── LICENSE.md         # License information
```

## Preprocessing Pipeline

### 1. Resize
```python
# All images resized to 256×256
Image (original size) → 256×256×3
```

### 2. Normalize
```python
# Normalize to [0, 1] range
pixel_value / 255.0
```

### 3. Channel Order
```python
# Convert from HWC to CHW format
[H, W, C] → [C, H, W]
```

### 4. Tensor Conversion
```python
# Convert to PyTorch tensor
uint8 array → float32 tensor
```

## Data Augmentation

To address class imbalance and small sample size:

### 1. Random Horizontal Flip
- Probability: 0.5
- Helps model learn from different perspectives

### 2. Random Rotation
- Range: ±15 degrees
- Helps with rotation-invariant features

### 3. Random Crop
- Crop size: 224×224 (with padding to 256×256)
- Helps with small defect localization

### 4. Color Jittering (optional)
- Brightness, contrast adjustments
- Helps with low-contrast defects

## Annotation Format

Annotations are stored as JSON files:
```json
{
  "filename": "image_name.png",
  "objects": [
    {
      "class_id": 1,
      "polygon": [[x1, y1], [x2, y2], ...],
      "area": 1234
    }
  ]
}
```

## Image-Annotation Pairing

Each image has a corresponding annotation file:
- Image: `image_name.png`
- Annotation: `image_name.png.json`

The annotation defines the polygon/bounding box of each defect, which is converted to a binary mask.

## Class Imbalance Handling

### 1. Loss Weighting
- Use weighted binary cross-entropy
- Give higher weight to defect class

### 2. Focal Loss (optional)
- Focus training on hard examples
- Reduce influence of easy background pixels

### 3. Sampling Strategies
- Oversample defective images
- Use balanced mini-batches

## Defect Size Classification

Based on EDA results, defects are classified into:

| Category | Size Range | Percentage |
|----------|------------|------------|
| Small | < 50 pixels | ~40% |
| Medium | 50-200 pixels | ~35% |
| Large | > 200 pixels | ~25% |

This classification is used for:
- Defect-size analysis in evaluation
- Understanding model performance on different defect sizes
- Ablation studies

## Quality Assurance

### 1. Image Mask Integrity
- Verify all images have corresponding masks
- Check for corrupt image files
- Validate mask boundaries

### 2. Annotation Accuracy
- Review sample annotations
- Ensure correct class assignment
- Verify polygon completeness

### 3. Split Consistency
- Official train/test split maintained
- No data leakage between splits
- Consistent indexing

## Data Access

The dataset is stored locally at:
```
data/kolektorsdd2-DatasetNinja/
```

Training and evaluation scripts automatically handle data loading and preprocessing.
