# 10. Nợ Kỹ Thuật và Đề Xuất Cải Tiến (Technical Debt & Improvements)

Tài liệu này ghi nhận khách quan các vấn đề về chất lượng mã nguồn, các điểm nghẽn kỹ thuật và các rủi ro kiến trúc được phát hiện trong quá trình kiểm toán mã nguồn (Code Audit), kèm theo các đề xuất giải pháp có thể thực hiện trong tương lai mà không làm ảnh hưởng đến mã nguồn gốc hiện tại.

---

## 1. Bảng Tổng Hợp Nợ Kỹ Thuật Được Xác Nhận (Confirmed Technical Debts)

| Mã | Hạng mục nợ kỹ thuật | Vị trí tập tin (File Path) | Bằng chứng kiểm định thực tế | Mức độ nghiêm trọng | Tác động |
|:---:|:---|:---|:---|:---:|:---|
| **TD-01** | Các tệp mã nguồn rỗng (0-byte stubs) chưa triển khai | `src/evaluation/benchmark.py`<br>`src/evaluation/visualization.py`<br>`scripts/benchmark_model.py` | Cả 3 file đều có dung lượng `Length = 0` byte trên đĩa. | 🔴 Cao | Người dùng không thể chạy đánh giá tổng hợp trên tập Test hoặc xuất các bảng Benchmark FLOPs/Latency nếu không viết thêm code. |
| **TD-02** | Các sổ tay Jupyter rỗng (0-byte notebooks) | `notebooks/01_eda.ipynb`<br>`notebooks/02_visualization.ipynb`<br>`notebooks/03_analysis.ipynb` | Cả 3 file notebook đều có dung lượng `Length = 0` byte, chỉ có file `04_eda_kolektorsdd2.ipynb` là có nội dung. | 🟡 Trung bình | Gây nhầm lẫn cho người mới khi mở sổ tay số 01 nhưng thấy trang trắng. |
| **TD-03** | Méo tỷ lệ khung hình do Resize cưỡng bức (Aspect Ratio Distortion) | `src/datasets/transforms.py`<br>`src/training/train.py` | Ảnh gốc có kích thước trung bình $230 \times 640$ px (tỷ lệ $0.36$), nhưng hàm `Resize(256)` nén chiều cao gấp gần 2.5 lần so với chiều rộng để thành hình vuông $256 \times 256$. | 🟡 Trung bình | Làm biến dạng hình thái học của các vết nứt hẹp dài, có thể làm giảm khả năng nhận diện hình thái thực tế. |
| **TD-04** | Sai lệch giá trị mặc định của thư mục dữ liệu | `scripts/verify_dataset.py` (dòng 77)<br>`configs/base.yaml` (dòng 7) | `verify_dataset.py` để mặc định `default=Path("data/raw")`, trong khi cấu trúc thực tế trong repo là `data/kolektorsdd2-DatasetNinja`. | 🟢 Thấp | Nếu người dùng gõ `python scripts/verify_dataset.py` mà không truyền `--data-root`, lệnh sẽ văng lỗi `FileNotFoundError`. |
| **TD-05** | Thiếu bộ kiểm thử tự động (Unit Tests / Integration Tests) | Thư mục gốc | Toàn bộ repo không có thư mục `tests/` và không có tệp `pytest`. | 🟡 Trung bình | Khó phát hiện sớm các lỗi gãy vỡ (regression) khi lập trình viên tinh chỉnh lại các phép toán tensor trong `models/` hoặc `losses/`. |

---

## 2. Phân Tích Chi Tiết Các Rủi Ro Kỹ Thuật

### 2.1. Rủi ro méo hình học do phép Resize (TD-03)
* **Thực trạng**: Các tấm phôi kim loại trong KolektorSDD2 là các thanh chữ nhật thon dài (~$230 \times 640$). Khi đưa vào hàm `image.resize((256, 256))`, trục dọc bị nén khoảng $60\%$, trong khi trục ngang được phóng to nhẹ.
* **Hậu quả**: Các vết nứt tròn biến thành hình elip dẹt; các đường nứt dọc bị co ngắn lại.
* **Đề xuất cải tiến (Không sửa code hiện tại, dành cho Phase sau)**:
  * Phương án A: Áp dụng **Letterbox Padding** (giữ nguyên tỷ lệ khung hình $0.36$, phóng tỷ lệ và chèn dải đen xung quanh để thành hình vuông).
  * Phương án B: Áp dụng **Cắt vá cửa sổ trượt (Sliding Window Tiling / Patching)** thành các ô $256 \times 256$ chồng lấn nhau.

### 2.2. Sự vắng mặt của Module Benchmark (TD-01)
* **Thực trạng**: Yêu cầu tại `request.txt` (Mục 17 & 18) đòi hỏi xuất ra Bảng 1 đến Bảng 5 và đo thông số FLOPs, Latency, FPS. Tuy nhiên file `scripts/benchmark_model.py` và `src/evaluation/benchmark.py` mới chỉ được tạo file rỗng.
* **Hậu quả**: Dự án chưa có cơ chế tự động hóa một nút bấm để kiểm tra toàn bộ tập Test sau khi huấn luyện xong.
* **Kế hoạch khắc phục**: Đây là nhiệm vụ ưu tiên **P0** đã được đưa vào lộ trình hành động trong tài liệu `docs/07_project_progress.md`.

---

## 3. Danh Mục Đề Xuất Nâng Cấp Hệ Thống (Improvement Proposals)

| Mã ĐX | Tên giải pháp cải tiến | Lợi ích mang lại | Độ phức tạp triển khai | Phụ thuộc |
|:---:|:---|:---|:---:|:---|
| **IMP-01** | Bổ sung hàm đo thời gian suy luận GPU (`torch.cuda.Event`) và FLOPs | Hoàn thiện yêu cầu đo lường chi phí tính toán (Table 5) trong `request.txt`. | Dễ (1-2 giờ) | Viết vào `scripts/benchmark_model.py` |
| **IMP-02** | Xây dựng pipeline trực quan hóa 5 ca điển hình | Phục vụ trực tiếp cho báo cáo nghiệm thu và slide thuyết trình. | Dễ (2 giờ) | Viết vào `src/evaluation/visualization.py` |
| **IMP-03** | Viết bộ kiểm thử hồi quy cơ bản (`tests/test_pipeline.py`) | Đảm bảo kích thước đầu ra của tất cả các model (E0-E4) luôn đúng `[B, 1, 256, 256]`. | Dễ (2 giờ) | Cài đặt `pytest` |
| **IMP-04** | Chuyển đổi định dạng mô hình sang ONNX / TensorRT | Tăng tốc độ suy luận gấp 3-5 lần, mở đường cho việc triển khai vào camera công nghiệp thực tế. | Trung bình (1 ngày) | Thư viện `onnx`, `onnxruntime` |
