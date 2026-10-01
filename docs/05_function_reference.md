# 05. Cẩm Nang Tra Cứu Hàm và Lớp Chi Tiết (Function Reference)

Tài liệu này cung cấp đặc tả kỹ thuật chi tiết cho từng lớp (class), hàm (function) và phương thức (method) trong toàn bộ kho mã nguồn `src/` và `scripts/`.

---

## 1. Module `src.datasets`

### 1.1. Tệp `src/datasets/kolektor.py`

#### `Sample` (DataClass)
* **Định nghĩa**: `@dataclass(frozen=True) class Sample`
* **Mục đích**: Lưu trữ thông tin định danh và đường dẫn tệp của một mẫu dữ liệu không thể thay đổi (immutable).
* **Các trường dữ liệu**:
  * `image_path: Path`: Đường dẫn tuyệt đối hoặc tương đối tới tệp ảnh.
  * `mask_path: Path | None`: Đường dẫn tệp mặt nạ dạng ảnh PNG nếu có (chuẩn Kolektor gốc).
  * `annotation_path: Path | None`: Đường dẫn tệp nhãn JSON nếu dùng định dạng DatasetNinja.
  * `image_id: str`: Tên định danh mẫu (stem của tệp ảnh, ví dụ `"10000"`).

#### `_decode_datasetninja_mask(annotation_path: Path, image_size: tuple[int, int]) -> Image.Image`
* **Mục đích**: Giải mã mặt nạ nhị phân được mã hóa và nén trong tệp JSON của DatasetNinja.
* **Tham số**:
  * `annotation_path`: Đường dẫn tệp `.json`.
  * `image_size`: Kích thước gốc của ảnh `(width, height)`.
* **Trả về**: Đối tượng `PIL.Image` chế độ `"L"` (Grayscale 8-bit), trong đó pixel lỗi có giá trị $>0$, nền có giá trị $0$.
* **Logic nội bộ**:
  1. Đọc tệp JSON, duyệt qua danh sách `objects`.
  2. Lấy đối tượng `bitmap` (gồm `data` và tọa độ dán `origin`).
  3. Gọi `base64.b64decode` để giải mã chuỗi base64 thành mảng byte nén.
  4. Gọi `zlib.decompress` để giải nén byte bitmap gốc.
  5. Đọc byte vào bộ nhớ đệm `BytesIO` và mở qua `PIL.Image.open().convert("L")`.
  6. Dán (paste) vùng khuyết tật lên mặt nạ nền đen toàn phần tại vị trí `origin`.

#### `KolektorSDD2(Dataset[dict[str, object]])`
* **Mục đích**: Lớp nạp dữ liệu chuẩn PyTorch cho tập KolektorSDD2.
* **Phương thức khởi tạo `__init__(root, split, image_dir_name="images", mask_dir_name="ground_truth", image_size=None, transform=None)`**:
  * **Tham số**:
    * `root: str | Path`: Thư mục gốc chứa tập dữ liệu.
    * `split: str`: Tên phân chia (`"train"` hoặc `"test"`).
    * `image_size: int | tuple[int, int] | None`: Kích thước ảnh mục tiêu để resize (nếu không dùng transform tùy biến).
    * `transform: Callable[[Image.Image, Image.Image], Pair] | None`: Chuỗi các hàm biến đổi đồng thời cả ảnh và mặt nạ.
  * **Hành vi**: Tự động phát hiện cấu trúc chuẩn (`images/` + `ground_truth/`) hoặc cấu trúc DatasetNinja (`img/` + `ann/`). Khởi tạo danh sách `self.samples`.
* **Phương thức `__len__() -> int`**:
  * Trả về tổng số lượng mẫu trong split.
* **Phương thức `__getitem__(index: int) -> dict[str, object]`**:
  * **Trả về**: Một dictionary gồm 4 phần tử:
    * `"image"`: `torch.Tensor` kích thước `[3, H, W]`, kiểu `float32`, giá trị chuẩn hóa trong khoảng $[0.0, 1.0]$.
    * `"mask"`: `torch.Tensor` kích thước `[1, H, W]`, kiểu `float32`, giá trị nhị phân $\{0.0, 1.0\}$.
    * `"image_id"`: Chuỗi chuỗi định danh ảnh (ví dụ `"10001"`).
    * `"is_defective"`: `bool` (`True` nếu có ít nhất 1 pixel lỗi, `False` nếu sạch).

---

### 1.2. Tệp `src/datasets/transforms.py`

* Kiểu dữ liệu quy ước: `Pair = tuple[Image.Image, Image.Image]`
* **`Compose(transforms: Iterable[Callable[[Image.Image, Image.Image], Pair]])`**:
  * Nhận danh sách các phép biến đổi, áp dụng tuần tự lên cặp `(image, mask)`.
* **`Resize(size: int | tuple[int, int])`**:
  * Thực hiện resize ảnh bằng phép nội suy song tuyến tính `BILINEAR`, mặt nạ bằng phép nội suy láng giềng gần nhất `NEAREST` (để bảo toàn tính chất nhị phân không sinh giá trị mờ trung gian).
* **`RandomHorizontalFlip(probability: float = 0.5)`**:
  * Lật ảnh và mặt nạ theo trục ngang (trái - phải) với xác suất chỉ định.
* **`RandomVerticalFlip(probability: float = 0.5)`**:
  * Lật ảnh và mặt nạ theo trục dọc (trên - dưới) với xác suất chỉ định.

---

## 2. Module `src.models`

### 2.1. Tệp `src/models/resnet_encoder.py`

#### `ResNet34Encoder(nn.Module)`
* **Mục đích**: Trích xuất đặc trưng phân cấp từ ảnh đầu vào bằng xương sống ResNet-34.
* **Phương thức khởi tạo `__init__(pretrained: bool = False)`**:
  * Sử dụng `torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)`.
  * Tách thành: `stem` (conv1 + bn1 + relu), `pool` (maxpool), `layer1`, `layer2`, `layer3`, `layer4`.
* **Phương thức `forward(inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]`**:
  * **Tham số**: `inputs` kích thước `[B, 3, H, W]`.
  * **Trả về tuple 5 đặc trưng phân cấp**:
    * `stem`: kích thước `[B, 64, H/2, W/2]` (stride 2)
    * `feature1`: kích thước `[B, 64, H/4, W/4]` (stride 4)
    * `feature2`: kích thước `[B, 128, H/8, W/8]` (stride 8)
    * `feature3`: kích thước `[B, 256, H/16, W/16]` (stride 16)
    * `feature4`: kích thước `[B, 512, H/32, W/32]` (stride 32 - bottleneck)

---

### 2.2. Tệp `src/models/multiscale.py`

#### `MultiScaleFeatureFusion(nn.Module)`
* **Mục đích**: Tổng hợp ngữ cảnh đa trường nhìn tại tầng bottleneck sâu nhất để bảo tồn thông tin các vết nứt siêu nhỏ.
* **Phương thức khởi tạo `__init__(channels: int = 512)`**:
  * Chia số kênh thành 4 nhánh bằng nhau (`branch_channels = channels // 4 = 128`).
  * `branches`:
    * Nhánh 1: Conv $1 \times 1$
    * Nhánh 2: Conv $3 \times 3$, padding=1, dilation=1
    * Nhánh 3: Conv $3 \times 3$, padding=2, dilation=2
    * Nhánh 4: Conv $3 \times 3$, padding=4, dilation=4
  * `project`: Conv $1 \times 1$ từ 512 về 512 kênh kèm `BatchNorm2d` và `ReLU`.
* **Phương thức `forward(inputs: Tensor) -> Tensor`**:
  * Nối kênh đầu ra của 4 nhánh: `fused = torch.cat(..., dim=1)`
  * Chiếu và cộng kết nối tắt phần dư: `return self.project(fused) + inputs`

---

### 2.3. Tệp `src/models/attention_gate.py`

#### `AttentionGate(nn.Module)`
* **Mục đích**: Sử dụng tín hiệu điều khiển từ Decoder để lọc bỏ các đặc trưng nhiễu vân nền từ Encoder trước khi ghép kênh.
* **Phương thức khởi tạo `__init__(gate_channels: int, skip_channels: int, inter_channels: int)`**:
  * `self.gate`: Conv $1 \times 1$ + BatchNorm chuyển `gate_channels` về `inter_channels`.
  * `self.skip`: Conv $1 \times 1$ + BatchNorm chuyển `skip_channels` về `inter_channels`.
  * `self.score`: ReLU $\rightarrow$ Conv $1 \times 1$ về 1 kênh $\rightarrow$ Sigmoid $\rightarrow$ hệ số $\alpha \in [0, 1]$.
* **Phương thức `forward(gate: Tensor, skip: Tensor) -> Tensor`**:
  * `attention = self.score(self.gate(gate) + self.skip(skip))`
  * `return skip * attention` (nhân phần tử với đặc trưng skip)

---

### 2.4. Tệp `src/models/decoder.py`

* **`ConvBlock(in_channels: int, out_channels: int)`**:
  * Khối tích chập kép chuẩn U-Net: `[Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU]`.
* **`UpBlock(in_channels: int, skip_channels: int, out_channels: int, attention: bool = False)`**:
  * **Hành vi**:
    1. Phóng to `inputs` bằng nội suy song tuyến tính `F.interpolate` khớp với kích thước không gian của `skip`.
    2. Nếu `attention=True`, đưa `inputs` và `skip` qua `AttentionGate` để lọc `skip`.
    3. Ghép kênh `torch.cat((inputs, skip), dim=1)`.
    4. Đưa qua `ConvBlock` giảm về `out_channels`.
* **`UNetDecoder(attention: bool = False)`**:
  * Chứa 4 tầng nâng mẫu:
    * `up3`: từ 512 (feature4) + 256 (feature3) $\rightarrow$ 256 kênh
    * `up2`: từ 256 + 128 (feature2) $\rightarrow$ 128 kênh
    * `up1`: từ 128 + 64 (feature1) $\rightarrow$ 64 kênh
    * `up0`: từ 64 + 64 (stem) $\rightarrow$ 64 kênh
    * `head`: Conv $1 \times 1$ từ 64 kênh về 1 kênh logit nhị phân (sau khi interpolate $\times 2$ về kích thước ảnh gốc).

---

### 2.5. Tệp `src/models/resnet34_unet.py`

#### `ResNet34UNet(nn.Module)`
* **Mục đích**: Mô hình tổng thể có thể cấu hình linh hoạt cho cả 5 thực nghiệm E0–E4.
* **Phương thức khởi tạo `__init__(pretrained=False, multi_scale=False, attention=False)`**:
  * `self.encoder = ResNet34Encoder(pretrained=pretrained)`
  * `self.multi_scale = MultiScaleFeatureFusion(512) if multi_scale else nn.Identity()`
  * `self.decoder = UNetDecoder(attention=attention)`
* **Phương thức `forward(inputs: Tensor) -> Tensor`**:
  * Trích xuất danh sách 5 đặc trưng từ encoder.
  * Đưa đặc trưng tầng sâu nhất `features[-1]` qua `self.multi_scale`.
  * Truyền cả 5 đặc trưng vào `self.decoder(*features)`.
  * Trả về logits thô kích thước `[B, 1, H, W]`.

---

## 3. Module `src.losses`

### 3.1. Tệp `src/losses/dice.py`

#### `dice_loss(logits: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor`
* **Công thức toán học**:
  $$\text{Dice} = \frac{2 \sum (p \cdot y) + s}{\sum p + \sum y + s}, \quad L_{\text{Dice}} = 1 - \text{Dice}$$
* **Hành vi**: Áp dụng hàm `sigmoid()` lên logits để thu được xác suất $p \in (0, 1)$, trải phẳng không gian không gian thành vector 1D, tính chỉ số trên từng mẫu trong batch rồi lấy trung bình.

---

### 3.2. Tệp `src/losses/boundary.py`

#### `_boundary_map(values: Tensor, kernel_size: int = 3) -> Tensor`
* **Mục đích**: Tính toán bản đồ ranh giới vi phân thông qua phép dãn và phép co bằng `max_pool2d`.
* **Công thức**:
  $$\text{dilated} = \text{max\_pool2d}(x, k=3, s=1, p=1)$$
  $$\text{eroded} = -\text{max\_pool2d}(-x, k=3, s=1, p=1)$$
  $$\text{boundary} = \text{clamp}(\text{dilated} - \text{eroded}, 0.0, 1.0)$$

#### `boundary_loss(logits: Tensor, targets: Tensor) -> Tensor`
* **Mục đích**: Tính hàm mất mát ranh giới nhị phân (Binary Cross-Entropy) giữa ranh giới dự đoán và ranh giới nhãn ground-truth:
  $$L_{\text{Boundary}} = \text{BCE}(\text{boundary}(p), \text{boundary}(y))$$

---

### 3.3. Tệp `src/losses/combined.py`

#### `combined_loss(logits, targets, use_boundary=False, boundary_weight=0.1) -> Tensor`
* **Công thức**:
  $$L = L_{\text{BCE}}(logits, targets) + L_{\text{Dice}}(logits, targets) + (\lambda L_{\text{Boundary}} \text{ nếu } use\_boundary)$$
* Kết hợp cân bằng giữa phạt lỗi pixel toàn cục, phạt mất cân bằng diện tích khuyết tật và ép sắc nét đường ranh giới.

---

## 4. Module `src.training`

### 4.1. Tệp `src/training/trainer.py`

#### `Trainer`
* **Khởi tạo `__init__(model, optimizer, device, use_boundary=False, boundary_weight=0.1)`**
* **`run_epoch(loader: Iterable[dict[str, object]], training: bool) -> dict[str, float]`**:
  * Chạy qua toàn bộ các batch trong loader.
  * Nếu `training=True`: bật `torch.set_grad_enabled(True)`, tính backward, gọi `optimizer.step()`, xóa gradient bằng `zero_grad(set_to_none=True)`.
  * Nếu `training=False`: tắt gradient để tiết kiệm bộ nhớ VRAM.
  * Tích lũy và trả về giá trị trung bình của `loss`, `dice`, `iou`, `precision`, `recall`.
* **`fit(train_loader, validation_loader, epochs: int) -> list[dict[str, float]]`**:
  * Điều phối chạy tuần tự từ epoch $1$ đến `epochs`.
  * Thu thập lịch sử train/validation metrics thành danh sách `history`.

---

### 4.2. Tệp `src/training/logger.py`

#### `save_training_artifacts(model, optimizer, history, output_dir: Path, config: dict) -> None`
* **Hành vi**:
  1. Tạo thư mục `output_dir` (ví dụ `experiments/E0_baseline/`).
  2. Ghi nhật ký huấn luyện ra `history.json` và `history.csv`.
  3. Tìm epoch có `val_dice` cao nhất, đóng gói và lưu trọng số tối ưu thành `best.pt`.
  4. Tự động vẽ và xuất 3 biểu đồ độ phân giải cao (160 DPI): `loss_curve.png`, `dice_curve.png`, `iou_curve.png`.

---

### 4.3. Tệp `src/training/train.py`

#### `main() -> int`
* **Hành vi**: Điểm nhập thực thi dòng lệnh. Đọc file cấu hình, thiết lập `set_seed(42)`, khởi tạo dataset có tăng cường (cho train) và không tăng cường (cho validation), chia tỷ lệ 80/20 ngẫu nhiên có kiểm soát hạt giống, khởi tạo mô hình, tối ưu hóa và kích hoạt `Trainer.fit()`.

---

## 5. Module `src.evaluation`

### 5.1. Tệp `src/evaluation/metrics.py`

* **`segmentation_metrics(logits, targets, threshold=0.5, epsilon=1e-7)`**:
  * Chuyển logits sang xác suất nhị phân, tính các chỉ số pixel-level:
    $$\text{Dice} = \frac{2TP + \epsilon}{2TP + FP + FN + \epsilon}, \quad \text{IoU} = \frac{TP + \epsilon}{TP + FP + FN + \epsilon}$$
    $$\text{Precision} = \frac{TP + \epsilon}{TP + FP + \epsilon}, \quad \text{Recall} = \frac{TP + \epsilon}{TP + FN + \epsilon}$$
* **`image_level_metrics(logits, targets, pixel_threshold=0.5, defect_area_threshold=0)`**:
  * Đánh giá ở cấp độ toàn bộ sản phẩm: Một ảnh được phân loại là khuyết tật nếu tổng diện tích pixel lỗi dự đoán vượt qua ngưỡng `defect_area_threshold`.
  * Tính: `image_precision`, `image_recall`, `image_f1`, và đặc biệt là **Tỷ lệ báo động giả trên ảnh bình thường**:
    $$\text{Normal FPR} = \frac{\text{Số ảnh bình thường bị đoán nhầm là có lỗi}}{\text{Tổng số ảnh bình thường}}$$
* **`defect_size_bucket(area: int, small_max: int, medium_max: int) -> str`**:
  * Phân loại kích thước vết nứt thành `"small"` ($area \le small\_max$), `"medium"` ($small\_max < area \le medium\_max$) hoặc `"large"`.

---

### 5.2. Tệp `src/evaluation/boundary_metrics.py`

#### `boundary_metrics(predictions, targets, tolerance=2, epsilon=1e-7)`
* **Mục đích**: Đo chất lượng đường viền khuyết tật cho phép một khoảng dung sai pixel (nhằm tránh phạt quá khắt khe khi đường viền chỉ lệch 1-2 pixel).
* **Logic nội bộ**:
  1. Trích xuất đường viền dự đoán và đường viền nhãn bằng phép XOR giữa mặt nạ và phép co: `_boundary(M) = M ^ binary_erosion(M)`.
  2. Tạo phần tử cấu trúc hình chữ nhật kích thước $(2 \times tolerance + 1)$.
  3. Dãn nở đường viền nhãn bằng phần tử cấu trúc trên (`binary_dilation`). Một pixel viền dự đoán được xem là đúng nếu nó rơi vào vùng dãn nở của viền nhãn.
  4. Tính `boundary_precision`, `boundary_recall` và chỉ số F1 điều hòa `boundary_f1`.
