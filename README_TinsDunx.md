# Dự Án: Phân Loại & Phát Hiện Khuyết Tật Bề Mặt Kim Loại (Combined_Surface_Defect_YOLO)

## Phân Hệ: Người 1 — Dataset & EDA

Repository này chứa toàn bộ mã nguồn xử lý dữ liệu, kiểm toán chất lượng (*Dataset Audit*), lọc trùng lặp (*Deduplication*), phân tích dữ liệu khám phá (*EDA*) và tái cấu trúc dữ liệu theo chuẩn **YOLO Object Detection** (cho YOLOv11 / DAC-YOLO).

---

## 1. Danh Mục Deliverables Bàn Giao

| Tệp tin / Thư mục | Mô tả chức năng |
| :--- | :--- |
| **`dataset.py`** | Module mã nguồn chính: chứa các class kiểm toán (`DatasetAuditor`), thuật toán lọc trùng lặp dHash 256-bit (`ImageHasher`), trích xuất khuyết tật theo chuẩn COCO (`DefectExtractor`), phân tầng Train/Val/Test (`DatasetSplitter`) và tăng cường ảnh (`TrainAugmentor`). |
| **`01_eda.ipynb`** | Jupyter Notebook trình bày toàn bộ 11 bước phân tích dữ liệu trực quan: phân phối nhãn, kích thước ảnh, kiểm tra tính toàn vẹn mask, phân loại Small/Medium/Large và trực quan hóa mẫu ảnh. |
| **`dataset_statistics.csv`** | Bảng số liệu thống kê chi tiết từng ảnh (W, H, Aspect Ratio, số lượng lỗi, diện tích pixel, % diện tích, phân loại quy mô, tập train/val/test). |
| **`eda_figures/`** | Thư mục chứa 7 biểu đồ phân tích trực quan chuẩn khoa học (300 DPI) để đưa vào báo cáo và slide thuyết trình. |
| **`run_eda_and_pipeline.py`** | Script tự động thực thi toàn bộ pipeline từ dữ liệu thô sang thư mục chuẩn YOLO. |

---

## 2. Thông Tin Bộ Dữ Liệu Kết Hợp (Combined Dataset)

Bộ dữ liệu được hợp nhất từ 2 nguồn công nghiệp thực tế:
1. **KolektorSDD2**: Bề mặt đồng tử động cơ điện (3,337 ảnh).
2. **Magnetic Tile Defect**: Bề mặt gốm nam châm vĩnh cửu (1,344 ảnh gồm các lỗi Blowhole, Break, Crack, Fray, Uneven và bề mặt sạch Free).

### Quy chuẩn xử lý:
* **Gán nhãn đồng nhất:** Quy về 1 lớp duy nhất: `0: defect`.
* **Lọc trùng lặp thông minh:** Áp dụng Difference Hash 256-bit lọc nội bộ từng nguồn dữ liệu, bảo toàn $100\%$ các mẫu khuyết tật hiếm.
* **Phân chia dữ liệu:** **70% Train - 15% Validation - 15% Test** (Stratified Split cân bằng tỷ lệ ảnh có lỗi và ảnh sạch).
* **Augmentation:** Tăng cường dữ liệu (lật ngang, lật dọc) **chỉ cho tập Train**, giữ nguyên tập Val và Test để đánh giá khách quan.

---

## 3. Hướng Dẫn Tái Tạo Pipeline

### Yêu cầu môi trường:
```bash
pip install numpy pandas pillow matplotlib
# Khuyến nghị cài thêm nếu có:
pip install opencv-python torch torchvision
```

### Chạy script tạo dữ liệu và biểu đồ:
```bash
python run_eda_and_pipeline.py
```

> **Lưu ý về Dữ liệu gốc:**  
> Toàn bộ thư mục ảnh thô và thư mục ảnh sau khi xuất (`Combined_Surface_Defect_YOLO/`) có dung lượng lớn nên được lưu trữ trên **Google Drive nhóm** (xem link đính kèm trong nhóm Zalo/Teams), không commit trực tiếp lên Git để tránh vượt hạn mức dung lượng của GitHub.
