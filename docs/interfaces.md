# Giao Tiếp Kỹ Thuật Giữa Các Tầng Module (Module Interfaces & Contracts)

Tài liệu này định nghĩa các hợp đồng giao tiếp (Data Contracts & API Interfaces) giữa các tầng kiến trúc trong hệ thống, đảm bảo tính đóng gói và độc lập khi các thành viên cùng phát triển song song.

---

## 1. Giao tiếp giữa Dataset và DataLoader (`src.datasets` $\rightarrow$ `src.training`)

Mỗi khi `DataLoader` gọi hàm `__getitem__(index)` trên đối tượng `KolektorSDD2`, đối tượng trả về bắt buộc phải tuân thủ cấu trúc từ điển (Dictionary Contract) sau:

```python
{
    # Tensor ảnh 3 kênh màu RGB, kiểu float32, khoảng giá trị [0.0, 1.0]
    "image": torch.Tensor,        # Shape: [3, 256, 256]

    # Tensor mặt nạ nhị phân 1 kênh, kiểu float32, giá trị {0.0, 1.0}
    "mask": torch.Tensor,         # Shape: [1, 256, 256]

    # Chuỗi định danh duy nhất của mẫu ảnh (stem tệp tin)
    "image_id": str,              # Ví dụ: "10001"

    # Cờ nhị phân biểu thị mẫu có chứa khuyết tật hay không
    "is_defective": bool          # True nếu mask có pixel > 0, ngược lại False
}
```

Khi được gom cụm qua `DataLoader` thành mini-batch, cấu trúc dữ liệu trở thành:
```python
{
    "image": torch.Tensor,        # Shape: [B, 3, 256, 256], dtype: torch.float32
    "mask": torch.Tensor,         # Shape: [B, 1, 256, 256], dtype: torch.float32
    "image_id": list[str],        # Danh sách B chuỗi định danh
    "is_defective": torch.Tensor  # Shape: [B], dtype: torch.bool
}
```

---

## 2. Giao tiếp giữa DataLoader và Mô hình (`src.training` $\rightarrow$ `src.models`)

### 2.1. Đầu vào của Mô hình (`ResNet34UNet.forward`)
* **Kiểu dữ liệu**: `torch.Tensor`
* **Kích thước**: `[B, 3, H, W]` (thông thường là `[B, 3, 256, 256]`)
* **Khoảng giá trị**: `[0.0, 1.0]`, kiểu `torch.float32`

### 2.2. Đầu ra của Mô hình
* **Kiểu dữ liệu**: `torch.Tensor`
* **Kích thước**: `[B, 1, H, W]`
* **Bản chất**: **Logits thô** (chưa đi qua hàm Sigmoid).
* **Quy ước**:
  * Các hàm mất mát (`BCEWithLogitsLoss`, `dice_loss`, `boundary_loss`) sẽ tự động áp dụng `sigmoid()` hoặc xử lý số học ổn định nội bộ.
  * Khi xuất mặt nạ dự đoán nhị phân cho người dùng hoặc tính độ đo, áp dụng:
    $$\text{Binary Mask} = (\text{logits.sigmoid()} \ge 0.5).\text{float}()$$

---

## 3. Giao tiếp giữa Mô hình và Hàm Mất Mát (`src.models` $\rightarrow$ `src.losses`)

Hàm `combined_loss` tại `src/losses/combined.py`:

```python
def combined_loss(
    logits: torch.Tensor,          # Shape: [B, 1, H, W] (Logits thô từ model)
    targets: torch.Tensor,         # Shape: [B, 1, H, W] (Mặt nạ nhãn 0 hoặc 1)
    use_boundary: bool = False,    # Bật/tắt Boundary Loss (dành cho E3, E4)
    boundary_weight: float = 0.1   # Trọng số lambda của Boundary Loss
) -> torch.Tensor                  # Scalar tensor có thể gọi .backward()
```

---

## 4. Giao tiếp giữa Mô hình và Module Độ Đo (`src.models` $\rightarrow$ `src.evaluation`)

### 4.1. Hàm đo cấp độ điểm ảnh (`segmentation_metrics`)
```python
def segmentation_metrics(
    logits: torch.Tensor,          # Shape: [B, 1, H, W]
    targets: torch.Tensor,         # Shape: [B, 1, H, W]
    threshold: float = 0.5,        # Ngưỡng phân lớp nhị phân
    epsilon: float = 1e-7          # Hằng số chống chia cho 0
) -> Mapping[str, float]:
    # Trả về dict: {"dice": float, "iou": float, "precision": float, "recall": float}
```

### 4.2. Hàm đo cấp độ ranh giới (`boundary_metrics`)
```python
def boundary_metrics(
    predictions: np.ndarray,       # Mảng nhị phân boolean [B, H, W] hoặc [H, W]
    targets: np.ndarray,           # Mảng nhị phân boolean [B, H, W] hoặc [H, W]
    tolerance: int = 2,            # Dung sai sai số viền (pixel)
    epsilon: float = 1e-7
) -> Mapping[str, float]:
    # Trả về dict: {"boundary_precision": float, "boundary_recall": float, "boundary_f1": float}
```

---

## 5. Giao tiếp giữa Trainer và Bộ Ghi Nhật Ký (`src.training` $\rightarrow$ `src.training.logger`)

Hàm `save_training_artifacts` lưu trữ toàn bộ trạng thái sau phiên huấn luyện:

```python
def save_training_artifacts(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, float]], # Danh sách metrics từng epoch
    output_dir: Path,               # Thư mục đích (ví dụ experiments/E0_baseline)
    config: dict[str, Any]          # Toàn bộ cấu hình YAML được sử dụng
) -> None
```
