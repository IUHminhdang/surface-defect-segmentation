# 11. Nhật Ký Kiểm Toán Mã Nguồn và Ghi Chú Phân Tích (Audit Log & Notes)

Tài liệu này ghi lại toàn bộ tiến trình kiểm toán kỹ thuật độc lập được thực hiện trên kho mã nguồn, tổng hợp các điểm sai lệch giữa tài liệu cũ và mã nguồn thực tế, và đưa ra các câu hỏi mở cần nhóm phát triển làm rõ.

---

## 1. Nhật Ký Tiến Trình Kiểm Toán (Audit Trail)

* **Thời gian thực hiện kiểm toán**: Tháng 10/2026.
* **Môi trường phân tích**: Hệ điều hành Windows, Python 3.11, PyTorch 2.x.
* **Phương pháp kiểm định**: 
  * Kiểm toán tĩnh (Static Code Analysis): Rà soát từng dòng mã nguồn trong `src/`, `scripts/`, `configs/`.
  * Phân tích cấu trúc dữ liệu thực địa: Kiểm tra 3,335 tệp ảnh và 3,335 tệp nhãn JSON trong `data/kolektorsdd2-DatasetNinja/`.
  * Đối chiếu chéo (Cross-verification): So sánh các tuyên bố trong `docs/01_project_overview.md`, `docs/02_dataset.md`, `README.md`, `request.txt` với mã nguồn và dữ liệu thực tế.
  * Kiểm tra các sản phẩm chạy thử nghiệm thực tế: Rà soát nhật ký `experiments/E0_baseline/history.json` và số liệu EDA trong `results/eda/`.

---

## 2. Bảng Đối Soát Sai Lệch Giữa Tài Liệu Cũ và Mã Nguồn Thực Tế (Discrepancy Log)

Trong quá trình phân tích, kiểm toán viên đã phát hiện các điểm mâu thuẫn kỹ thuật sau:

| STT | Vấn đề / Khái niệm | Tuyên bố trong tài liệu cũ | Hiện trạng thực tế trong Code & Dữ liệu | Phân tích & Khuyến nghị |
|:---:|:---|:---|:---|:---|
| **D-01** | **Ngưỡng diện tích khuyết tật (Defect Size Thresholds)** | `docs/02_dataset.md` (dòng 163-166) ghi:<br>- Small: $<50$ pixels<br>- Medium: $50-200$ pixels<br>- Large: $>200$ pixels | Tệp `results/eda/size_thresholds.json` do script `scripts/run_eda.py` tính toán thực tế ghi:<br>- Small: $\le 1,392$ px<br>- Medium: $\le 3,675$ px<br>- Large: $>3,675$ px | Ngưỡng cũ trong `docs/02_dataset.md` là con số ước lượng giả định ban đầu. Ngưỡng trong `size_thresholds.json` được tính khoa học theo phân vị tam phân vị (tertiles 33% và 66%) trên các mặt nạ thật của tập Train. Cần thống nhất dùng ngưỡng thực nghiệm $1,392$ và $3,675$ px. |
| **D-02** | **Đường dẫn thư mục dữ liệu mặc định** | `README.md` (dòng 39) và `scripts/verify_dataset.py` (dòng 77) trỏ tới: `data/raw` | `configs/base.yaml` (dòng 7) và thư mục vật lý trên đĩa là: `data/kolektorsdd2-DatasetNinja` | Tài liệu cũ dùng tên quy ước `data/raw`. Trên thực tế, dữ liệu tải về mang cấu trúc đóng gói DatasetNinja. Đã cập nhật hướng dẫn trong `docs/08_setup_and_execution.md` để người dùng không gặp lỗi `FileNotFoundError`. |
| **D-03** | **Định dạng lưu trữ nhãn mặt nạ** | `docs/02_dataset.md` (dòng 123) mô tả nhãn là các đa giác polygon: `{"polygon": [[x1, y1], ...]}` | Tệp JSON thực tế lưu bitmap nhị phân nén dạng: `{"bitmap": {"origin": [x,y], "data": "base64_zlib_string"}}` | Lớp `KolektorSDD2` tại `src/datasets/kolektor.py` đã viết riêng hàm `_decode_datasetninja_mask` để giải nén zlib. Cần cập nhật lại mô tả nhãn trong tài liệu cho đúng thực tế. |
| **D-04** | **Số lượng Epochs đã huấn luyện của Baseline E0** | `configs/base.yaml` đặt `epochs: 50` | Thư mục `experiments/E0_baseline/history.json` chỉ có duy nhất 1 bản ghi của `epoch: 1.0` | Đây mới chỉ là lần chạy kiểm thử khói (smoke test) kiểm tra luồng chạy, chưa phải kết quả hội tụ 50 epochs hoàn chỉnh. Không được nhầm lẫn số liệu này là kết quả nghiệm thu cuối cùng. |
| **D-05** | **Các file chức năng 0-byte** | `README.md` và `request.txt` nhắc tới script benchmark | `src/evaluation/benchmark.py`, `src/evaluation/visualization.py`, `scripts/benchmark_model.py` và 3 file `.ipynb` có kích thước 0 bytes | Đã đưa vào danh mục Nợ kỹ thuật (TD-01, TD-02) và phân loại ưu tiên hành động P0. |

---

## 3. Danh Mục Các Tệp Tin Đã Được Tạo Mới / Cập Nhật

Trong đợt công tác này, các tệp tin tài liệu Markdown sau đã được biên soạn mới và chuẩn hóa trong thư mục `docs/`:

1. `docs/README.md`: Trang chủ tài liệu kỹ thuật, sơ đồ định vị và lộ trình đọc khuyến nghị.
2. `docs/01_project_overview.md`: Tổng quan đề tài, bài toán 3 rào cản, 4 câu hỏi nghiên cứu RQ, 5 thực nghiệm E0-E4, biểu đồ kiến trúc PlantUML.
3. `docs/02_repository_structure.md`: Bản đồ cây thư mục chi tiết, phân loại tệp tin và hướng dẫn định vị nghiệp vụ.
4. `docs/03_system_architecture.md`: Thiết kế kiến trúc phân lớp, sơ đồ khối PlantUML, biểu đồ tuần tự luồng huấn luyện, chi tiết toán học của Multi-Scale, Attention Gate và vi phân Boundary Loss.
5. `docs/04_database_and_data_model.md`: Mô hình dữ liệu hướng tệp, sơ đồ thực thể PlantUML, đặc tả JSON DatasetNinja zlib, YAML config, log CSV/JSON và Checkpoint.
6. `docs/05_function_reference.md`: Cẩm nang tra cứu chi tiết toàn bộ các lớp, hàm, tham số, giá trị trả về và logic giải thuật trong `src/` và `scripts/`.
7. `docs/06_data_pipeline.md`: Quy trình xử lý dữ liệu từ đầu đến cuối, chiến lược chia tập 80/20 có seed, các biện pháp chống rò rỉ dữ liệu.
8. `docs/07_project_progress.md`: Ma trận đối soát yêu cầu kỹ thuật (RTM), ma trận hiện trạng module, checklist Done, biểu đồ phân rã công việc WBS & lộ trình hành động ưu tiên P0-P3 bằng PlantUML.
9. `docs/08_setup_and_execution.md`: Hướng dẫn chi tiết cài đặt môi trường cục bộ (Windows PowerShell) và nền tảng Kaggle GPU, các lệnh chạy mẫu và xử lý sự cố.
10. `docs/09_glossary_and_learning_guide.md`: Bảng tra cứu thuật ngữ chuyên ngành (Semantic Segmentation, Dice, Boundary F1, Dilated Conv, Attention Gate) và lộ trình tự học.
11. `docs/10_technical_debt_and_improvements.md`: Bảng tổng hợp nợ kỹ thuật xác nhận (file 0-byte, méo tỷ lệ khung hình), phân tích rủi ro và giải pháp cải tiến.
12. `docs/11_change_log_and_analysis_notes.md`: Nhật ký kiểm toán, đối soát sai lệch và các câu hỏi cần xác nhận.
13. `docs/interfaces.md`: Bổ sung đặc tả giao tiếp chuẩn giữa các tầng module trong hệ thống.

---

## 4. Các Vùng Chưa Kiểm Tra (Uninspected Areas) và Lý Do

1. **Hiệu năng suy luận trên GPU thực tế**:
   * *Lý do*: Lệnh kiểm toán mã nguồn tuân thủ nguyên tắc không can thiệp, không kích hoạt các tiến trình huấn luyện nặng trên máy người dùng trong quá trình viết tài liệu.
2. **Nội dung nhị phân của Checkpoint `experiments/E0_baseline/best.pt`**:
   * *Lý do*: File nhị phân dạng nén PyTorch. Đã kiểm tra tính hợp lệ gián tiếp thông qua tệp nhật ký văn bản `history.json` và đồ thị `loss_curve.png`.

---

## 5. Các Câu Hỏi Mở Cần Nhóm Phát Triển Xác Nhận (Questions for Confirmation)

1. **Về việc xử lý tỷ lệ khung hình**: Nhóm có kế hoạch chuyển từ phép ép khung `Resize(256)` sang `Letterbox Padding` hoặc chia lát ảnh (Sliding Window Tiling) không, hay vẫn giữ nguyên phép co giãn vuông cho toàn bộ các thực nghiệm E0-E4?
2. **Về 3 sổ tay Jupyter rỗng**: Có nên di chuyển nội dung phân tích từ `04_eda_kolektorsdd2.ipynb` sang `01_eda.ipynb`, đồng thời bổ sung mã cho `02_visualization.ipynb` và `03_analysis.ipynb` hay không?
3. **Về môi trường chạy 50 epochs**: Nhóm dự kiến chạy toàn bộ 5 thực nghiệm (mỗi thực nghiệm 50 epochs) trên máy trạm cục bộ hay sẽ sử dụng GPU Kaggle / Google Colab?
