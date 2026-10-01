# Project Overview

## Problem Statement

In industrial quality inspection, surface defects on products are typically very small, have low contrast, and can closely resemble normal surface textures. Detecting these defects accurately at the pixel level is crucial for automated quality control systems.

## Task Definition

This project focuses on **semantic segmentation** for industrial surface defect detection:

- **Input**: RGB images of industrial surfaces
- **Output**: Pixel-level binary masks (defect vs. background)
- **Task**: Given an image $I \in \mathbb{R}^{H \times W \times 3}$, predict a mask $M \in \{0,1\}^{H \times W}$ where:
  - $0$ = background (normal surface)
  - $1$ = defect

## Research Questions

### RQ1: Multi-Scale Features
Does multi-scale feature fusion improve segmentation of small defects?

**Problem**: When encoder repeatedly downsamples images:
```
Image (256×256)
  ↓
1/2 (128×128)
  ↓
1/4 (64×64)
  ↓
1/8 (32×32)
  ↓
1/16 (16×16)
  ↓
1/32 (8×8)
```
Tiny defects can lose critical information during downsampling.

### RQ2: Attention Mechanism
Does attention gates improve feature selection and reduce false positives?

**Problem**: Skip connections transmit a lot of information from encoder to decoder, including irrelevant background/texture features.

### RQ3: Boundary-Aware Loss
Does boundary-aware loss improve boundary quality?

**Problem**: Defects can be very small and have difficult-to-identify boundaries.

## Proposed Solution

We propose a **U-Net based architecture** with three key improvements:

1. **Multi-Scale Feature Fusion**: Extract features at multiple scales to preserve small defect information
2. **Attention Gates**: Filter skip connections to focus on relevant features
3. **Boundary-Aware Loss**: $L = L_{BCE} + L_{Dice} + \lambda L_{Boundary}$

## Baseline Model

**U-Net with ResNet34 Encoder**

Standard U-Net architecture with ResNet34 as the encoder backbone.

## Experiment Setup

We conduct **5 core experiments**:

| Experiment | Multi-Scale | Attention | Boundary Loss | Purpose |
|------------|-------------|-----------|---------------|---------|
| **E0** | ❌ | ❌ | ❌ | Baseline |
| **E1** | ✅ | ❌ | ❌ | Evaluate Multi-scale |
| **E2** | ❌ | ✅ | ❌ | Evaluate Attention |
| **E3** | ❌ | ❌ | ✅ | Evaluate Boundary Loss |
| **E4** | ✅ | ✅ | ✅ | Full Model |

## Evaluation Metrics

### Pixel-Level Metrics
- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Measures segmentation quality
- **Precision**: Among predicted defect pixels, how many are truly defective?
- **Recall**: Among true defect pixels, how many did we find?

### Boundary-Level Metrics
- **Boundary Precision**: Quality of predicted boundaries
- **Boundary Recall**: Coverage of true boundaries
- **Boundary F1**: Harmonic mean of boundary precision and recall

### Image-Level Metrics
- **Image Precision**: Proportion of predicted defective images that are truly defective
- **Image Recall**: Proportion of truly defective images that are detected
- **Normal FP Rate**: False positive rate on normal (non-defective) images

## Dataset

**KolektorSDD2** - Industrial surface defect segmentation dataset
- 3,335 total images (356 defective, 2,979 normal)
- Official split: 2,331 train + 1,004 test
- Pixel-level annotations
- Defects are very small and have low contrast
