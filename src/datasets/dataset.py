#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: dataset.py
Dự án: Phân loại & Phát hiện khuyết tật bề mặt kim loại (Combined_Surface_Defect_YOLO)
Phân hệ: Người 1 — Dataset & EDA

Hỗ trợ kết hợp:
- KolektorSDD2 (2,332 train + 1,004 test = 3,336 ảnh thực tế)
- Magnetic-tile-defect-datasets.-master (CASIA: MT_Blowhole, MT_Break, MT_Crack, MT_Fray, MT_Uneven, MT_Free = 1,344 ảnh thực tế)

TỐI ƯU HÓA QUAN TRỌNG:
1. Nhận diện chuẩn xác 1,344 ảnh của Magnetic Tile (bỏ qua file mask .png và ảnh minh họa root).
2. Nâng cấp dHash lên 256-bit (16x16) và lọc trùng NỘI BỘ TỪNG DATASET (Intra-dataset Deduplication).
   Không để ảnh KolektorSDD2 loại bỏ nhầm ảnh Magnetic Tile.
3. Ngưỡng phân tầng thông minh (Adaptive Threshold):
   - Mẫu có khuyết tật (Positive): Ngưỡng cực kỳ chặt (Hamming <= 2 trên 256 bits) để BẢO TOÀN TỐI ĐA mẫu lỗi.
   - Mẫu sạch (Negative): Ngưỡng Hamming <= 4 trên 256 bits, tránh ngộ nhận các phôi gốm/thép phẳng là trùng lặp.
"""

import os
import shutil
import random
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass


# ==============================================================================
# 1. ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU (DATA STRUCTURES)
# ==============================================================================

@dataclass
class DefectBoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    area_pixels: int
    relative_area: float       # Tỷ lệ diện tích khuyết tật so với toàn bộ ảnh (%)
    category: str              # 'Small', 'Medium', 'Large'
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float
    class_id: int = 0          # 0: defect


@dataclass
class SampleRecord:
    sample_id: str
    dataset_source: str        # 'KolektorSDD2' hoặc 'MagneticTile'
    image_path: Path
    mask_path: Optional[Path]
    width: int
    height: int
    channels: int
    aspect_ratio: float
    has_defect: bool
    num_defects: int
    defects: List[DefectBoundingBox] = field(default_factory=list)
    image_hash: Optional[np.ndarray] = None
    split: str = "train"       # 'train', 'val', 'test'
    integrity_status: str = "OK"


# ==============================================================================
# 2. IMAGE HASHING & DEDUPLICATION (NÂNG CẤP 256-BIT & LỌC NỘI BỘ NGUỒN)
# ==============================================================================

class ImageHasher:
    """
    Nâng cấp thuật toán Difference Hash (dHash) lên 256-bit (16x16) và lọc trùng nội bộ.
    Khắc phục triệt để lỗi làm mất ảnh do bề mặt kim loại trơn phẳng.
    """
    @staticmethod
    def compute_dhash(img: Image.Image, hash_size: int = 16) -> np.ndarray:
        # Resize về (hash_size + 1, hash_size) -> 17x16 -> 256 bits
        resized = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(resized, dtype=np.int16)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return diff.flatten()

    @classmethod
    def deduplicate(cls, samples: List[SampleRecord],
                    pos_threshold: int = 2,
                    neg_threshold: int = 4) -> Tuple[List[SampleRecord], Dict[str, int]]:
        """
        Lọc trùng thông minh theo từng nguồn dataset (Intra-dataset):
        - Không để ảnh KolektorSDD2 xóa nhầm ảnh Magnetic Tile.
        - Mẫu có khuyết tật (Positive): Ngưỡng cực kỳ chặt (Hamming <= 2) để bảo vệ 100% mẫu lỗi.
        - Mẫu sạch (Negative): Ngưỡng Hamming <= 4 trên 256-bit, tránh xóa nhầm ảnh sạch của các phôi khác nhau.
        """
        unique_samples: List[SampleRecord] = []
        dup_stats: Dict[str, int] = {}

        # 1. Tính hash cho toàn bộ mẫu trước
        for sample in samples:
            if sample.image_hash is None:
                try:
                    with Image.open(sample.image_path) as img:
                        sample.image_hash = cls.compute_dhash(img, hash_size=16)
                except Exception:
                    pass

        # 2. Lọc riêng cho từng nguồn dữ liệu (KolektorSDD2 và MagneticTile)
        sources = sorted(list(set(s.dataset_source for s in samples)))
        for src in sources:
            src_samples = [s for s in samples if s.dataset_source == src]
            src_unique: List[SampleRecord] = []
            src_dups = 0

            # Phân tách positive và negative trong từng nguồn
            pos_samples = [s for s in src_samples if s.has_defect]
            neg_samples = [s for s in src_samples if not s.has_defect]

            # Lọc positive: ngưỡng rất chặt (pos_threshold = 2)
            pos_unique_hashes = []
            for s in pos_samples:
                if s.image_hash is None:
                    src_unique.append(s)
                    continue
                if not pos_unique_hashes:
                    pos_unique_hashes.append(s.image_hash)
                    src_unique.append(s)
                    continue
                u_stack = np.array(pos_unique_hashes)
                dists = np.count_nonzero(u_stack != s.image_hash, axis=1)
                if np.any(dists <= pos_threshold):
                    src_dups += 1
                else:
                    pos_unique_hashes.append(s.image_hash)
                    src_unique.append(s)

            # Lọc negative: ngưỡng neg_threshold = 4 trên 256 bits
            neg_unique_hashes = []
            for s in neg_samples:
                if s.image_hash is None:
                    src_unique.append(s)
                    continue
                if not neg_unique_hashes:
                    neg_unique_hashes.append(s.image_hash)
                    src_unique.append(s)
                    continue
                u_stack = np.array(neg_unique_hashes)
                dists = np.count_nonzero(u_stack != s.image_hash, axis=1)
                if np.any(dists <= neg_threshold):
                    src_dups += 1
                else:
                    neg_unique_hashes.append(s.image_hash)
                    src_unique.append(s)

            unique_samples.extend(src_unique)
            dup_stats[src] = src_dups

        return unique_samples, dup_stats


# ==============================================================================
# 3. DEFECT EXTRACTOR & DEFINITION (SMALL / MEDIUM / LARGE THEO COCO)
# ==============================================================================

class DefectExtractor:
    SMALL_THRESH = 32 * 32     # 1,024 px²
    LARGE_THRESH = 96 * 96     # 9,216 px²

    @classmethod
    def classify_defect_size(cls, area_pixels: int) -> str:
        if area_pixels < cls.SMALL_THRESH:
            return "Small"
        elif area_pixels <= cls.LARGE_THRESH:
            return "Medium"
        else:
            return "Large"

    @classmethod
    def extract_bboxes_from_mask(cls, mask_np: np.ndarray, img_w: int, img_h: int) -> List[DefectBoundingBox]:
        binary = (mask_np > 127).astype(np.uint8)
        if np.sum(binary) == 0:
            return []

        bboxes = []
        img_area = float(img_w * img_h)

        if HAS_CV2:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 2 and bh < 2:
                    continue
                area_px = int(cv2.contourArea(cnt))
                if area_px == 0:
                    area_px = bw * bh
                rel_area = (area_px / img_area) * 100.0
                cat = cls.classify_defect_size(area_px)

                xc_norm = min(max((x + bw / 2.0) / img_w, 0.0), 1.0)
                yc_norm = min(max((y + bh / 2.0) / img_h, 0.0), 1.0)
                w_norm = min(max(bw / float(img_w), 0.0), 1.0)
                h_norm = min(max(bh / float(img_h), 0.0), 1.0)

                bboxes.append(DefectBoundingBox(
                    x_min=x, y_min=y, x_max=x + bw, y_max=y + bh,
                    area_pixels=area_px, relative_area=rel_area,
                    category=cat, x_center_norm=xc_norm, y_center_norm=yc_norm,
                    width_norm=w_norm, height_norm=h_norm, class_id=0
                ))
        else:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)
            slices = ndimage.find_objects(labeled)
            for sl in slices:
                ys, xs = sl
                x, y = xs.start, ys.start
                bw = xs.stop - xs.start
                bh = ys.stop - ys.start
                if bw < 2 and bh < 2:
                    continue
                area_px = int(np.sum(binary[ys, xs]))
                rel_area = (area_px / img_area) * 100.0
                cat = cls.classify_defect_size(area_px)

                xc_norm = min(max((x + bw / 2.0) / img_w, 0.0), 1.0)
                yc_norm = min(max((y + bh / 2.0) / img_h, 0.0), 1.0)
                w_norm = min(max(bw / float(img_w), 0.0), 1.0)
                h_norm = min(max(bh / float(img_h), 0.0), 1.0)

                bboxes.append(DefectBoundingBox(
                    x_min=x, y_min=y, x_max=x + bw, y_max=y + bh,
                    area_pixels=area_px, relative_area=rel_area,
                    category=cat, x_center_norm=xc_norm, y_center_norm=yc_norm,
                    width_norm=w_norm, height_norm=h_norm, class_id=0
                ))

        return bboxes


# ==============================================================================
# 4. DATASET AUDITOR (QUÉT VÀ KIỂM TRA TÍNH TOÀN VẸN CẢ 2 BỘ DỮ LIỆU)
# ==============================================================================

class DatasetAuditor:
    @staticmethod
    def audit_sample(img_path: Path, mask_path: Optional[Path], dataset_name: str, sample_id: str) -> SampleRecord:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                channels = len(mode) if mode in ['RGB', 'RGBA'] else 1
                aspect_ratio = round(w / float(h), 4) if h > 0 else 1.0
        except Exception as e:
            return SampleRecord(
                sample_id=sample_id, dataset_source=dataset_name,
                image_path=img_path, mask_path=mask_path,
                width=0, height=0, channels=0, aspect_ratio=0.0,
                has_defect=False, num_defects=0, integrity_status=f"CORRUPT_IMAGE: {e}"
            )

        if mask_path is None or not mask_path.exists():
            return SampleRecord(
                sample_id=sample_id, dataset_source=dataset_name,
                image_path=img_path, mask_path=None,
                width=w, height=h, channels=channels, aspect_ratio=aspect_ratio,
                has_defect=False, num_defects=0, integrity_status="OK (Clean/Negative)"
            )

        try:
            with Image.open(mask_path) as m_img:
                mw, mh = m_img.size
                if mw != w or mh != h:
                    status = f"DIM_MISMATCH (Img:{w}x{h} vs Mask:{mw}x{mh})"
                else:
                    status = "OK"
                mask_np = np.array(m_img.convert('L'))
        except Exception as e:
            return SampleRecord(
                sample_id=sample_id, dataset_source=dataset_name,
                image_path=img_path, mask_path=mask_path,
                width=w, height=h, channels=channels, aspect_ratio=aspect_ratio,
                has_defect=False, num_defects=0, integrity_status=f"CORRUPT_MASK: {e}"
            )

        bboxes = DefectExtractor.extract_bboxes_from_mask(mask_np, w, h)
        has_defect = len(bboxes) > 0

        return SampleRecord(
            sample_id=sample_id, dataset_source=dataset_name,
            image_path=img_path, mask_path=mask_path,
            width=w, height=h, channels=channels, aspect_ratio=aspect_ratio,
            has_defect=has_defect, num_defects=len(bboxes),
            defects=bboxes, integrity_status=status
        )

    @classmethod
    def scan_kolektor(cls, kolektor_dir: Path) -> List[SampleRecord]:
        records = []
        if not kolektor_dir.exists():
            return records

        for split_folder in ["train", "test"]:
            dir_path = kolektor_dir / split_folder
            if not dir_path.exists():
                continue
            for img_file in sorted(dir_path.glob("*.png")):
                if img_file.stem.endswith("_GT"):
                    continue
                mask_file = dir_path / f"{img_file.stem}_GT.png"
                record = cls.audit_sample(
                    img_path=img_file,
                    mask_path=mask_file if mask_file.exists() else None,
                    dataset_name="KolektorSDD2",
                    sample_id=f"ksdd2_{img_file.stem}"
                )
                records.append(record)
        return records

    @classmethod
    def locate_magnetic_tile_dir(cls, base_dir: Path) -> Optional[Path]:
        candidates = [
            base_dir / "Magnetic-tile-defect-datasets.-master",
            base_dir / "Magnetic-tile-defect-datasets-master",
            base_dir / "Magnetic-tile-defect-datasets.",
            base_dir / "Magnetic_Tile",
            base_dir / "Magnetic-Tile-Defect",
            base_dir / "MT_Defect"
        ]
        for p in candidates:
            if p.exists() and p.is_dir():
                return p

        for p in base_dir.iterdir():
            if p.is_dir() and "magnetic" in p.name.lower():
                return p

        # Tự động giải nén ZIP nếu có
        for z in [
            base_dir / "Magnetic-tile-defect-datasets.-master.zip",
            base_dir / "Magnetic-tile-defect-datasets-master.zip"
        ]:
            if z.exists() and z.is_file():
                try:
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(base_dir)
                    for p in candidates:
                        if p.exists() and p.is_dir():
                            return p
                except Exception:
                    pass

        return None

    @classmethod
    def scan_magnetic_tile(cls, mt_dir: Path) -> List[SampleRecord]:
        """
        Quét chuẩn đệ quy cho Magnetic-tile-defect-datasets.-master:
        - Bỏ qua các file ảnh minh họa ở root (dataset.jpg, dataset.png, sample.jpg,...)
        - Nhận diện 5 thư mục lỗi: MT_Blowhole, MT_Break, MT_Crack, MT_Fray, MT_Uneven
        - Nhận diện thư mục sạch: MT_Free
        - Chuẩn xác lấy file .jpg làm ảnh chụp và file .png làm mask tương ứng
        - Tổng số ảnh thực tế chính xác: 1,344 ảnh
        """
        records = []
        if not mt_dir.exists():
            return records

        print(f"---> Đang quét toàn diện thư mục Magnetic Tile: {mt_dir.name}...")
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        all_files = sorted(list(mt_dir.rglob("*")))
        for item in all_files:
            if not item.is_file() or item.suffix.lower() not in image_extensions:
                continue

            # Bỏ qua ảnh minh họa root
            if item.name.lower() in ["dataset.jpg", "dataset.png", "sample.jpg", "readme.jpg"]:
                continue

            name_lower = item.name.lower()
            # Bỏ qua nếu là file mask có hậu tố _GT hoặc nằm trong thư mục mask
            if "_gt" in name_lower or "groundtruth" in str(item.parent).lower() or "mask" in str(item.parent).lower():
                continue

            parent_parts = [p.lower() for p in item.parts]
            is_free = any(k in parent_parts for k in ["free", "mt_free", "normal", "good"])

            # Xác định tên lớp khuyết tật
            cat_name = "defect"
            for part in item.parts:
                if part.startswith("MT_"):
                    cat_name = part.replace("MT_", "").lower()
                    break
            if is_free:
                cat_name = "free"

            # Tìm mask tương ứng
            mask_file = None
            if not is_free:
                parent = item.parent
                stem = item.stem
                c1 = parent / f"{stem}.png"
                if c1.exists() and c1 != item:
                    mask_file = c1
                else:
                    c2 = parent / f"{stem}_GT.png"
                    if c2.exists():
                        mask_file = c2
                    else:
                        for sub in ["GroundTruth", "groundtruth", "Mask", "masks", "GT"]:
                            c3 = parent / sub / f"{stem}.png"
                            if c3.exists():
                                mask_file = c3
                                break
                            c4 = parent.parent / sub / f"{stem}.png"
                            if c4.exists():
                                mask_file = c4
                                break

            # Nếu item là file .png và có tồn tại .jpg cùng tên, thì .png chính là mask
            if item.suffix.lower() == ".png" and (item.parent / f"{item.stem}.jpg").exists():
                continue

            rec = cls.audit_sample(
                img_path=item,
                mask_path=mask_file,
                dataset_name="MagneticTile",
                sample_id=f"mt_{cat_name}_{item.stem}"
            )
            records.append(rec)

        print(f"  [OK] Đã quét thành công {len(records):,} ảnh thực tế từ Magnetic Tile.")
        return records


# ==============================================================================
# 5. STRATIFIED SPLITTER & AUGMENTATION
# ==============================================================================

class DatasetSplitter:
    @staticmethod
    def split(samples: List[SampleRecord], train_r: float = 0.7, val_r: float = 0.15,
              test_r: float = 0.15, seed: int = 42) -> None:
        random.seed(seed)
        groups: Dict[Tuple[str, bool], List[SampleRecord]] = {}
        for s in samples:
            key = (s.dataset_source, s.has_defect)
            groups.setdefault(key, []).append(s)

        for key, grp in groups.items():
            random.shuffle(grp)
            n_total = len(grp)
            n_train = int(n_total * train_r)
            n_val = int(n_total * val_r)

            for i, s in enumerate(grp):
                if i < n_train:
                    s.split = "train"
                elif i < n_train + n_val:
                    s.split = "val"
                else:
                    s.split = "test"


class TrainAugmentor:
    @staticmethod
    def augment_sample(img: Image.Image, bboxes: List[DefectBoundingBox]) -> List[Tuple[Image.Image, List[DefectBoundingBox]]]:
        augmented = []
        w, h = img.size

        # 1. Lật ngang (Horizontal Flip)
        flipped_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped_boxes = []
        for b in bboxes:
            new_xc = 1.0 - b.x_center_norm
            flipped_boxes.append(DefectBoundingBox(
                x_min=w - b.x_max, y_min=b.y_min, x_max=w - b.x_min, y_max=b.y_max,
                area_pixels=b.area_pixels, relative_area=b.relative_area, category=b.category,
                x_center_norm=new_xc, y_center_norm=b.y_center_norm,
                width_norm=b.width_norm, height_norm=b.height_norm, class_id=b.class_id
            ))
        augmented.append((flipped_img, flipped_boxes))

        # 2. Lật dọc (Vertical Flip)
        v_flipped_img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        v_flipped_boxes = []
        for b in bboxes:
            new_yc = 1.0 - b.y_center_norm
            v_flipped_boxes.append(DefectBoundingBox(
                x_min=b.x_min, y_min=h - b.y_max, x_max=b.x_max, y_max=h - b.y_min,
                area_pixels=b.area_pixels, relative_area=b.relative_area, category=b.category,
                x_center_norm=b.x_center_norm, y_center_norm=new_yc,
                width_norm=b.width_norm, height_norm=b.height_norm, class_id=b.class_id
            ))
        augmented.append((v_flipped_img, v_flipped_boxes))

        return augmented


# ==============================================================================
# 6. PYTORCH DATASET ADAPTER
# ==============================================================================

if HAS_TORCH:
    class SurfaceDefectDataset(Dataset):
        def __init__(self, samples: List[SampleRecord], split: str = "train", transform=None):
            self.samples = [s for s in samples if s.split == split]
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            img = Image.open(sample.image_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            target = {
                "boxes": torch.tensor([[b.x_min, b.y_min, b.x_max, b.y_max] for b in sample.defects], dtype=torch.float32),
                "labels": torch.zeros((len(sample.defects),), dtype=torch.int64),
                "sample_id": sample.sample_id,
                "has_defect": sample.has_defect
            }
            return img, target
