#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: run_eda_and_pipeline.py
Thực thi toàn bộ quy trình:
1. Tự động phát hiện và quét toàn diện Magnetic-tile-defect-datasets.-master (1,344 ảnh thực tế)
2. Quét KolektorSDD2 (3,337 ảnh thực tế) và ghép cặp toàn bộ ảnh - mask
3. Dataset Audit & Mask Integrity Check
4. Image Hashing Deduplication nâng cấp:
   - Thuật toán dHash 256-bit (16x16)
   - Lọc trùng nội bộ từng dataset (Intra-dataset)
   - Bảo toàn tối đa mẫu khuyết tật và đa dạng ảnh sạch
5. Defect Analysis & Size Categorization (Small/Medium/Large theo chuẩn COCO)
6. Stratified Split 70% Train - 15% Val - 15% Test
7. Xuất file dataset_statistics.csv
8. Sinh 7 biểu đồ EDA Figures chuẩn khoa học lưu vào eda_figures/
9. Tạo thư mục Combined_Surface_Defect_YOLO (kèm Train Augmentation và data.yaml)
"""

import sys
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import module dataset
from dataset import (
    DatasetAuditor, ImageHasher, DefectExtractor,
    DatasetSplitter, TrainAugmentor, SampleRecord
)

PROJECT_DIR = Path(__file__).resolve().parent
KOLEKTOR_DIR = PROJECT_DIR / "KolektorSDD2"
OUTPUT_YOLO_DIR = PROJECT_DIR / "Combined_Surface_Defect_YOLO"
FIGURES_DIR = PROJECT_DIR / "eda_figures"
CSV_PATH = PROJECT_DIR / "dataset_statistics.csv"


def generate_eda_figures(df: pd.DataFrame, samples: list, figures_dir: Path):
    """
    Sinh 7 biểu đồ phân tích EDA chuẩn khoa học và lưu vào eda_figures/
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Biểu đồ Positive vs Negative Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pos_neg_counts = df['has_defect'].value_counts()
    labels = ['Negative (Clean)', 'Positive (Defective)']
    vals = [pos_neg_counts.get(False, 0), pos_neg_counts.get(True, 0)]
    
    bars = axes[0].bar(labels, vals, color=['#3b82f6', '#ef4444'], width=0.5, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval:,} ({yval/len(df)*100:.1f}%)",
                     ha='center', va='bottom', fontweight='bold')
    axes[0].set_title("Số lượng mẫu Positive vs Negative", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Số lượng ảnh")

    axes[1].pie(vals, labels=labels, autopct='%1.1f%%', startangle=140,
                colors=['#3b82f6', '#ef4444'], explode=(0, 0.08),
                wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
    axes[1].set_title("Tỷ lệ phân phối khuyết tật", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "01_positive_negative_distribution.png", dpi=300)
    plt.close()

    # 2. Phân bố nguồn dataset
    plt.figure(figsize=(8, 5))
    source_counts = df['dataset_source'].value_counts()
    bars = plt.bar(source_counts.index, source_counts.values, color='#0284c7', width=0.45, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval:,} ảnh ({yval/len(df)*100:.1f}%)",
                 ha='center', va='bottom', fontweight='bold')
    plt.title("Phân bố dữ liệu theo nguồn (Dataset Sources)", fontsize=13, fontweight='bold')
    plt.ylabel("Số lượng ảnh")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_dataset_sources_breakdown.png", dpi=300)
    plt.close()

    # 3. Kích thước ảnh và Tỷ lệ khung hình (Image Sizes & Aspect Ratio)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scatter = axes[0].scatter(df['width'], df['height'], c=df['aspect_ratio'],
                              cmap='viridis', alpha=0.6, edgecolors='none', s=40)
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Aspect Ratio (W/H)')
    axes[0].set_title("Phân bố kích thước ảnh (Width vs Height)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Chiều rộng (Pixels)")
    axes[0].set_ylabel("Chiều cao (Pixels)")

    axes[1].hist(df['aspect_ratio'], bins=25, color='#8b5cf6', edgecolor='black', alpha=0.8)
    axes[1].set_title("Histogram tỷ lệ khung hình (Aspect Ratio)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Tỷ lệ W / H")
    axes[1].set_ylabel("Tần suất")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_image_dimensions_aspect_ratio.png", dpi=300)
    plt.close()

    # 4. Phân loại kích thước khuyết tật (Small / Medium / Large theo COCO)
    all_defect_categories = []
    all_defect_areas = []
    for s in samples:
        for d in s.defects:
            all_defect_categories.append(d.category)
            all_defect_areas.append(d.area_pixels)

    if all_defect_categories:
        plt.figure(figsize=(9, 5))
        cat_series = pd.Series(all_defect_categories).value_counts()[['Small', 'Medium', 'Large']].dropna()
        bar_colors = ['#10b981', '#f59e0b', '#dc2626']
        bars = plt.bar(cat_series.index, cat_series.values, color=bar_colors, width=0.45, edgecolor='black')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10,
                     f"{yval:,} ({yval/len(all_defect_categories)*100:.1f}%)",
                     ha='center', va='bottom', fontweight='bold')
        plt.title("Phân loại quy mô khuyết tật theo chuẩn COCO\n(Small: <32², Medium: 32²-96², Large: >96²)",
                  fontsize=12, fontweight='bold')
        plt.ylabel("Số lượng hộp khuyết tật")
        plt.tight_layout()
        plt.savefig(figures_dir / "04_defect_size_categories.png", dpi=300)
        plt.close()

    # 5. Phân phối diện tích khuyết tật (Defect Area Distribution)
    if all_defect_areas:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(all_defect_areas, bins=40, color='#0ea5e9', edgecolor='black', alpha=0.8)
        axes[0].set_yscale('log')
        axes[0].set_title("Phân phối diện tích khuyết tật (Log scale)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Diện tích (Pixels²)")
        axes[0].set_ylabel("Số lượng (Log scale)")

        axes[1].boxplot(all_defect_areas, vert=False, patch_artist=True,
                        boxprops=dict(facecolor='#a7f3d0', color='black'),
                        medianprops=dict(color='red', linewidth=2))
        axes[1].set_title("Boxplot diện tích khuyết tật", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Diện tích (Pixels²)")
        plt.tight_layout()
        plt.savefig(figures_dir / "05_defect_area_distribution.png", dpi=300)
        plt.close()

    # 6. Trực quan hóa mẫu ảnh kèm Mask & YOLO Bounding Box
    pos_samples = [s for s in samples if s.has_defect and s.mask_path and s.mask_path.exists()]
    if pos_samples:
        random.seed(42)
        selected_samples = random.sample(pos_samples, min(4, len(pos_samples)))
        fig, axes = plt.subplots(len(selected_samples), 3, figsize=(12, 3.5 * len(selected_samples)))
        if len(selected_samples) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, sample in enumerate(selected_samples):
            try:
                img = Image.open(sample.image_path).convert('RGB')
                axes[row_idx, 0].imshow(img)
                axes[row_idx, 0].set_title(f"Ảnh gốc: {sample.sample_id}", fontsize=10)
                axes[row_idx, 0].axis('off')

                mask = Image.open(sample.mask_path).convert('L')
                axes[row_idx, 1].imshow(mask, cmap='gray')
                axes[row_idx, 1].set_title("Ground Truth Mask", fontsize=10)
                axes[row_idx, 1].axis('off')

                axes[row_idx, 2].imshow(img)
                for d in sample.defects:
                    rect = patches.Rectangle(
                        (d.x_min, d.y_min), d.x_max - d.x_min, d.y_max - d.y_min,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    axes[row_idx, 2].add_patch(rect)
                    axes[row_idx, 2].text(
                        d.x_min, max(0, d.y_min - 4), f"{d.category} ({d.area_pixels}px)",
                        color='white', backgroundcolor='red', fontsize=8, fontweight='bold'
                    )
                axes[row_idx, 2].set_title(f"YOLO BBoxes ({len(sample.defects)} lỗi)", fontsize=10)
                axes[row_idx, 2].axis('off')
            except Exception:
                pass

        plt.tight_layout()
        plt.savefig(figures_dir / "06_sample_defect_visualizations.png", dpi=300)
        plt.close()

    # 7. Phân phối Train / Val / Test Split
    split_df = df.groupby(['split', 'has_defect']).size().unstack(fill_value=0)
    split_df.columns = ['Clean (Neg)', 'Defect (Pos)']
    split_df.plot(kind='bar', stacked=True, color=['#3b82f6', '#ef4444'], figsize=(8, 5), edgecolor='black')
    plt.title("Phân bổ mẫu trên 3 tập Train (70%) - Val (15%) - Test (15%)", fontsize=12, fontweight='bold')
    plt.xlabel("Tập dữ liệu")
    plt.ylabel("Số lượng ảnh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "07_train_val_test_split.png", dpi=300)
    plt.close()

    print(f"[OK] Đã xuất 7 biểu đồ phân tích thành công vào: {figures_dir}")


def export_to_yolo(samples: list, output_dir: Path, apply_train_aug: bool = True):
    """
    Xuất tập dữ liệu ra chuẩn YOLO Object Detection:
    - Tự động làm sạch thư mục cũ nếu đã tồn tại để tránh trộn lẫn file.
    - images/{train, val, test}
    - labels/{train, val, test}
    - Train Augmentation (chỉ áp dụng cho tập train)
    - data.yaml
    """
    if output_dir.exists():
        print(f"\n---> Đang làm sạch thư mục kết quả cũ: {output_dir.name}...")
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        (output_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    print(f"\n---> Đang xuất dữ liệu ra định dạng YOLO ({output_dir})...")
    counts = {"train": 0, "val": 0, "test": 0}

    for s in samples:
        if not s.image_path.exists():
            continue

        sp = s.split
        img_dest = output_dir / "images" / sp / f"{s.sample_id}.png"
        lbl_dest = output_dir / "labels" / sp / f"{s.sample_id}.txt"

        try:
            if not img_dest.exists():
                with Image.open(s.image_path) as im:
                    im.convert('RGB').save(img_dest)
            counts[sp] += 1

            with open(lbl_dest, "w", encoding="utf-8") as f:
                for d in s.defects:
                    f.write(f"0 {d.x_center_norm:.6f} {d.y_center_norm:.6f} {d.width_norm:.6f} {d.height_norm:.6f}\n")

            # Tăng cường dữ liệu (CHỈ DÀNH CHO TẬP TRAIN)
            if apply_train_aug and sp == "train" and s.has_defect:
                with Image.open(s.image_path) as orig_img:
                    aug_results = TrainAugmentor.augment_sample(orig_img, s.defects)
                    for aug_idx, (aug_img, aug_boxes) in enumerate(aug_results, 1):
                        aug_id = f"{s.sample_id}_aug{aug_idx}"
                        aug_img_path = output_dir / "images" / "train" / f"{aug_id}.png"
                        aug_lbl_path = output_dir / "labels" / "train" / f"{aug_id}.txt"
                        aug_img.save(aug_img_path)
                        counts["train"] += 1

                        with open(aug_lbl_path, "w", encoding="utf-8") as af:
                            for ab in aug_boxes:
                                af.write(f"0 {ab.x_center_norm:.6f} {ab.y_center_norm:.6f} {ab.width_norm:.6f} {ab.height_norm:.6f}\n")
        except Exception as e:
            print(f"[Cảnh báo] Bỏ qua file {s.image_path.name} do lỗi: {e}")

    yaml_path = output_dir / "data.yaml"
    yaml_content = f"""# Cấu hình Dataset kết hợp (KolektorSDD2 + Magnetic-tile-defect-datasets.-master)
path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

# Số lớp
nc: 1

# Tên nhãn khuyết tật
names:
  0: defect
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"  [OK] Hoàn tất xuất YOLO: Train: {counts['train']} ảnh (đã kèm Augment), Val: {counts['val']} ảnh, Test: {counts['test']} ảnh")
    print(f"  [OK] Đã tạo file: {yaml_path}")


def main():
    print("=" * 70)
    print(" KẾT HỢP KOLEKTORSDD2 & MAGNETIC TILE SANG CHUẨN YOLO DETECTION")
    print("=" * 70)

    # 1. Quét KolektorSDD2
    print("\n[Bước 1/6] Quét dữ liệu KolektorSDD2...")
    kolektor_samples = DatasetAuditor.scan_kolektor(KOLEKTOR_DIR)
    print(f"  -> Quét được: {len(kolektor_samples):,} ảnh từ KolektorSDD2.")

    # 2. Quét Magnetic-tile-defect-datasets.-master
    print("\n[Bước 2/6] Quét dữ liệu Magnetic-tile-defect-datasets.-master...")
    mt_dir = DatasetAuditor.locate_magnetic_tile_dir(PROJECT_DIR)
    mt_samples = []
    if mt_dir:
        print(f"  -> Tìm thấy thư mục: {mt_dir.name}")
        mt_samples = DatasetAuditor.scan_magnetic_tile(mt_dir)
        print(f"  -> Quét được: {len(mt_samples):,} ảnh từ Magnetic Tile.")
    else:
        print(f"  -> [CẢNH BÁO] Không tìm thấy thư mục hoặc file ZIP Magnetic-tile-defect-datasets.-master!")
        print(f"     Hãy chắc chắn thư mục hoặc file ZIP nằm trong: {PROJECT_DIR}")

    all_samples = kolektor_samples + mt_samples
    print(f"\n---> Tổng số ảnh thu thập ban đầu từ cả 2 bộ dữ liệu: {len(all_samples):,} ảnh")
    print(f"     * KolektorSDD2: {len(kolektor_samples):,} ảnh")
    print(f"     * MagneticTile: {len(mt_samples):,} ảnh")

    # 3. Lọc trùng lặp thông minh bằng dHash 256-bit (Lọc nội bộ từng dataset)
    print("\n[Bước 3/6] Lọc trùng lặp thông minh (dHash 256-bit, Intra-dataset)...")
    unique_samples, dup_stats = ImageHasher.deduplicate(all_samples, pos_threshold=2, neg_threshold=4)
    total_dups = sum(dup_stats.values())
    print(f"  -> Thống kê lọc trùng:")
    for src, cnt in dup_stats.items():
        print(f"     + Nguồn {src}: loại bỏ {cnt:,} ảnh trùng lặp.")
    print(f"  -> Tổng số ảnh trùng lặp loại bỏ : {total_dups:,}")
    print(f"  -> Tổng số ảnh sạch giữ lại      : {len(unique_samples):,}")

    # 4. Phân chia Stratified Split (70% Train - 15% Val - 15% Test)
    print("\n[Bước 4/6] Phân chia dữ liệu phân tầng 70% Train - 15% Val - 15% Test...")
    DatasetSplitter.split(unique_samples, train_r=0.7, val_r=0.15, test_r=0.15, seed=42)

    # 5. Xuất file dataset_statistics.csv
    print("\n[Bước 5/6] Xuất bảng thống kê dataset_statistics.csv...")
    records_data = []
    for s in unique_samples:
        total_area = sum(d.area_pixels for d in s.defects)
        rel_area = sum(d.relative_area for d in s.defects)
        categories = list(set(d.category for d in s.defects))
        cat_str = ", ".join(categories) if categories else "None"

        records_data.append({
            "sample_id": s.sample_id,
            "dataset_source": s.dataset_source,
            "filename": s.image_path.name,
            "width": s.width,
            "height": s.height,
            "channels": s.channels,
            "aspect_ratio": s.aspect_ratio,
            "has_defect": s.has_defect,
            "num_defects": s.num_defects,
            "total_defect_area_px": total_area,
            "relative_defect_area_pct": round(rel_area, 4),
            "defect_size_categories": cat_str,
            "split": s.split,
            "integrity_status": s.integrity_status
        })

    df = pd.DataFrame(records_data)
    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"  [OK] Đã lưu bảng thống kê chi tiết ({len(df):,} dòng) tại: {CSV_PATH}")

    # 6. Sinh 7 biểu đồ phân tích EDA
    print("\n[Bước 6/6] Sinh toàn bộ 7 biểu đồ EDA lưu vào eda_figures/...")
    generate_eda_figures(df, unique_samples, FIGURES_DIR)

    # 7. Xuất thư mục YOLO Combined
    export_to_yolo(unique_samples, OUTPUT_YOLO_DIR, apply_train_aug=True)

    print("\n" + "=" * 70)
    print("        HOÀN TẤT KẾT HỢP VÀ TIỀN XỬ LÝ 2 BỘ DỮ LIỆU!")
    print("=" * 70)
    print(f"Thư mục kết quả YOLO: {OUTPUT_YOLO_DIR}")
    print(f"Bảng thống kê CSV   : {CSV_PATH}")
    print(f"Thư mục biểu đồ EDA : {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
