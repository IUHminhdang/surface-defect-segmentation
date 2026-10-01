# 08. Hướng Dẫn Cài Đặt và Thực Thi (Setup & Execution Guide)

Tài liệu này cung cấp hướng dẫn từng bước từ việc chuẩn bị môi trường, cài đặt các gói phụ thuộc, kiểm tra tính hợp lệ của dữ liệu, chạy phân tích EDA cho đến việc khởi chạy các phiên huấn luyện mô hình trên máy cục bộ (Windows/Linux) và môi trường đám mây Kaggle.

---

## 1. Yêu cầu Tiên quyết về Môi trường (Prerequisites)

* **Hệ điều hành**: Windows 10/11 (PowerShell) hoặc Linux (Ubuntu 20.04/22.04).
* **Python**: Phiên bản $3.10$ đến $3.11$ (đã kiểm định tương thích hoàn toàn trên Python 3.11).
* **Phần cứng**:
  * Tối thiểu: 8GB RAM, CPU 4 nhân (chạy được ở chế độ kiểm thử hoặc smoke test).
  * Khuyến nghị huấn luyện: NVIDIA GPU có tối thiểu 8GB VRAM (ví dụ RTX 3060, RTX 4060, hoặc GPU T4/P100 trên Kaggle/Google Colab) hỗ trợ CUDA 11.8 hoặc 12.x.
* **Dung lượng ổ cứng trống**: Khoảng 5GB (dành cho mã nguồn, dataset KolektorSDD2 và các checkpoint mô hình).

---

## 2. Các bước Cài đặt Môi trường Cục bộ (Local Environment Setup)

### Bước 1: Mở Terminal tại thư mục gốc của dự án
Đảm bảo bạn đang đứng ở thư mục gốc chứa tệp `requirements.txt`:
```powershell
# Kiểm tra thư mục làm việc hiện tại trên Windows PowerShell
Get-Location
```

### Bước 2: Tạo và Kích hoạt Môi trường ảo (Virtual Environment)
* **Trên Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(Nếu gặp lỗi kích hoạt script trên PowerShell, chạy trước lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

* **Trên Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 3: Nâng cấp pip và Cài đặt Thư viện
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

* **Ghi chú về PyTorch có hỗ trợ GPU CUDA**:
  Nếu máy tính của bạn có card đồ họa rời NVIDIA nhưng lệnh `pip install -r requirements.txt` chỉ cài phiên bản PyTorch CPU, hãy cài đặt bổ sung PyTorch CUDA từ trang chủ:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

---

## 3. Chuẩn bị và Kiểm tra Tập Dữ liệu (Dataset Verification)

Dữ liệu KolektorSDD2 đã được đặt sẵn tại thư mục:
`data/kolektorsdd2-DatasetNinja/`

### Chạy script kiểm tra tính toàn vẹn của dữ liệu:
```powershell
python scripts/verify_dataset.py --data-root data/kolektorsdd2-DatasetNinja
```

* **Kết quả kỳ vọng trên màn hình**:
  ```
  DATASET OK: 3335 images, 356 annotated, 2979 normal
  ```
  *(Nếu thiếu file hoặc sai đường dẫn, script sẽ in thông báo `DATASET INVALID: ...` và kết thúc với mã lỗi 1).*

---

## 4. Chạy Phân Tích Thăm Dò Dữ Liệu Tự Động (Run EDA)

Để tạo ra bảng thống kê chi tiết diện tích khuyết tật, xác định các ngưỡng phân vị kích thước Small/Medium/Large và xuất các biểu đồ trực quan vào `results/eda/`, hãy chạy:

```powershell
python scripts/run_eda.py --data-root data/kolektorsdd2-DatasetNinja --output-dir results/eda
```

* **Các sản phẩm đầu ra sẽ được lưu tại `results/eda/`**:
  * `dataset_statistics.csv`: Bảng số liệu chi tiết từng ảnh.
  * `size_thresholds.json`: Ngưỡng phân vị (Small $\le 1392$ px, Medium $\le 3675$ px).
  * `defect_area_distribution.png`: Biểu đồ phân phối diện tích vết nứt.
  * `positive_negative_by_split.png`: Tỷ lệ ảnh lỗi và ảnh sạch theo từng tập.
  * `defect_examples.png`: Mẫu trực quan ảnh lỗi và mặt nạ.

---

## 5. Hướng dẫn Huấn Luyện Mô Hình (Model Training)

> **Nguyên tắc quan trọng**: Luôn chạy script huấn luyện dưới dạng module Python bằng cờ `-m` để đảm bảo hệ thống nhận diện đúng các gói nội bộ trong `src`.

### 5.1. Chạy thử nghiệm nhanh (Smoke Test - 1 Epoch)
Trước khi chạy phiên huấn luyện đầy đủ kéo dài nhiều giờ, hãy chạy thử 1 epoch để đảm bảo GPU, CUDA, DataLoader và hàm Loss không bị xung đột bộ nhớ:
```powershell
python -m src.training.train --experiment E0_baseline --epochs 1 --output-dir experiments/E0_baseline_smoke
```

### 5.2. Chạy Huấn luyện Chính thức (Full Training - 50 Epochs)

#### Chạy Baseline E0 (ResNet34-UNet tiêu chuẩn):
```powershell
python -m src.training.train --experiment E0_baseline --epochs 50
```

#### Chạy Thực nghiệm E1 (Tích hợp Multi-Scale Feature Fusion):
```powershell
python -m src.training.train --experiment E1_multiscale --epochs 50
```

#### Chạy Thực nghiệm E2 (Tích hợp Attention Gates):
```powershell
python -m src.training.train --experiment E2_attention --epochs 50
```

#### Chạy Thực nghiệm E3 (Tích hợp Boundary-Aware Loss):
```powershell
python -m src.training.train --experiment E3_boundary --epochs 50
```

#### Chạy Thực nghiệm E4 (Mô hình Toàn diện Full Model):
```powershell
python -m src.training.train --experiment E4_full --epochs 50
```

---

## 6. Quy trình Làm việc trên Nền tảng Đám mây Kaggle (Kaggle Workflow)

Nếu máy tính cá nhân không có GPU hoặc dung lượng VRAM dưới 6GB, hãy sử dụng GPU miễn phí trên Kaggle:

1. **Đẩy mã nguồn lên GitHub / Kaggle**:
   * Chỉ đẩy mã nguồn, **tuyệt đối không đẩy thư mục `data/`** lên Git (đã được cấu hình trong `.gitignore`).
2. **Nạp Dataset trên Kaggle**:
   * Tải tập dữ liệu KolektorSDD2 lên Kaggle Dataset (hoặc gắn dataset có sẵn trên cộng đồng Kaggle).
3. **Mở một Kaggle Notebook mới**:
   * Thiết lập **Accelerator: GPU T4 x2** hoặc **GPU P100**.
   * Bật **Internet: ON**.
4. **Clone mã nguồn vào thư mục `/kaggle/working`**:
   ```bash
   cd /kaggle/working
   git clone https://github.com/<your-username>/surface-defect-segmentation.git
   cd surface-defect-segmentation
   pip install -r requirements.txt
   ```
5. **Chạy huấn luyện trên Kaggle**:
   Chỉ định cờ `--data-root` trỏ tới đường dẫn dataset trong Kaggle Input (ví dụ: `/kaggle/input/kolektorsdd2-datasetninja`):
   ```bash
   python -m src.training.train --experiment E0_baseline --data-root /kaggle/input/kolektorsdd2-datasetninja --epochs 50
   
   ```
Lưu ý đổi đường dẫn sao cho phù hợp với đường dẫn thật.




6. **Lưu trữ kết quả**:
   Tải thư mục `experiments/` về máy sau khi phiên chạy trên Kaggle kết thúc.

---

## 7. Xử lý Sự cố Thường Gặp (Troubleshooting)

### Vấn đề 1: Tràn bộ nhớ GPU (CUDA Out Of Memory - OOM)
* **Triệu chứng**: `torch.cuda.OutOfMemoryError: CUDA out of memory.`
* **Nguyên nhân**: Kích thước batch quá lớn so với dung lượng VRAM card đồ họa.
* **Cách khắc phục**: Mở file `configs/base.yaml`, giảm `batch_size: 8` xuống `batch_size: 4` hoặc `batch_size: 2`.

### Vấn đề 2: Sai đường dẫn dữ liệu mặc định (`FileNotFoundError`)
* **Triệu chứng**: `DATASET INVALID: Dataset root does not exist: data/raw`
* **Nguyên nhân**: Script `scripts/verify_dataset.py` có giá trị mặc định là `data/raw`, trong khi thư mục thực tế trong repo là `data/kolektorsdd2-DatasetNinja`.
* **Cách khắc phục**: Luôn truyền tham số rõ ràng:
  `--data-root data/kolektorsdd2-DatasetNinja`.

### Vấn đề 3: Lỗi tiến trình nạp dữ liệu song song trên Windows (`RuntimeError: DataLoader worker ...`)
* **Triệu chứng**: Đứng hình hoặc báo lỗi liên quan đến `multiprocessing` khi nạp batch.
* **Nguyên nhân**: Trên Windows, cơ chế `fork` không được hỗ trợ như Linux, có thể gây xung đột khi chia luồng `num_workers`.
* **Cách khắc phục**: Mở `configs/base.yaml`, chỉnh tạm thời `num_workers: 0` (chạy trên tiến trình chính) để kiểm tra.
Mình sẽ kiểm tra đúng ba điểm: Dice loss/metric hiện tại có đang tính cả ảnh normal không, Trainer có scheduler/early stopping chưa, và checkpoint có đủ optimizer/scheduler state chưa. Sau đó mình sẽ sửa một lát cắt nhỏ, chạy lại smoke test, rồi hướng dẫn quy trình Kaggle 50–100 epochs theo cấu hình đã cố định.



## Việc cần làm trước khi lên Kaggle

Các checkpoint E0–E4 smoke trước đó được tạo trước khi sửa Dice, nên cần chạy lại smoke E0:

```powershell
python -m src.training.train `
  --data-root data/kolektorsdd2-DatasetNinja `
  --experiment E0_baseline `
  --epochs 1 `
  --output-dir experiments/E0_corrected_smoke
```

Kiểm tra thư mục phải có:

```text
experiments/E0_corrected_smoke/
├── best.pt
├── history.json
├── history.csv
├── loss_curve.png
├── dice_curve.png
└── iou_curve.png
```

## Có nên train 50–100 epochs trên Kaggle không?

Nên bắt đầu **50 epochs**, chưa cần 100.

Cấu hình hiện tại:

```yaml
epochs: 50
early_stopping:
  patience: 10
  min_delta: 0.0001
scheduler: cosine_annealing
```

Nếu validation Dice không cải thiện trong 10 epochs, training sẽ dừng sớm. Vì vậy 50 epochs là mức phù hợp để chạy E0–E4 trước.

## Lệnh Kaggle

Sau khi push code mới:

```python
%cd /kaggle/working/<repository>
!git pull
```

Kiểm tra dataset:

```python
DATA_ROOT = "/kaggle/input/<ten-dataset-raw>"
!python scripts/verify_dataset.py --data-root "$DATA_ROOT"
```

Chạy E0:

```python
!python -m src.training.train \
    --data-root "$DATA_ROOT" \
    --experiment E0_baseline \
    --epochs 50 \
    --output-dir experiments/E0_baseline
```

Chạy E1–E4:

```python
!python -m src.training.train --data-root "$DATA_ROOT" --experiment E1_multiscale --epochs 50 --output-dir experiments/E1_multiscale
!python -m src.training.train --data-root "$DATA_ROOT" --experiment E2_attention --epochs 50 --output-dir experiments/E2_attention
!python -m src.training.train --data-root "$DATA_ROOT" --experiment E3_boundary --epochs 50 --output-dir experiments/E3_boundary
!python -m src.training.train --data-root "$DATA_ROOT" --experiment E4_full --epochs 50 --output-dir experiments/E4_full
```

Sau mỗi model, benchmark official test:

```python
!python scripts/benchmark_model.py \
    --data-root "$DATA_ROOT" \
    --experiment E0_baseline \
    --checkpoint experiments/E0_baseline/best.pt
```

Đổi `E0_baseline` tương ứng cho E1–E4.

Kết quả cuối sẽ nằm trong:

```text
results/metrics_E0_baseline.json
results/metrics_E1_multiscale.json
results/metrics_E2_attention.json
results/metrics_E3_boundary.json
results/metrics_E4_full.json
```

`--resume` dùng khi training bị gián đoạn:

```python
!python -m src.training.train \
    --data-root "$DATA_ROOT" \
    --experiment E4_full \
    --epochs 50 \
    --resume experiments/E4_full/best.pt \
    --output-dir experiments/E4_full
```

Nên dùng cùng tổng số epochs khi resume, ví dụ checkpoint đang thuộc kế hoạch 50 epochs thì resume với `--epochs 50`.

Made changes.