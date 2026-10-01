# Hệ Thống Tài Liệu Kỹ Thuật: Surface Defect Segmentation

> **Dự án**: Phân vùng khuyết tật bề mặt nhỏ trên tập dữ liệu KolektorSDD2 bằng Multi-Scale Features, Attention Gates và Boundary-Aware Loss.  
> **Phiên bản tài liệu**: 1.0 (Audit & Reference Documentation)  
> **Ngày phân tích**: Tháng 10/2026  
> **Trạng thái kho mã nguồn**: Đang trong giai đoạn phát triển (Baseline E0 đã kiểm thử chạy 1 epoch; các module kiến trúc E1-E4, Loss, Metrics đã hoàn thành code; module Benchmark và Visualization đang ở dạng stub 0-byte).

---

## 1. Giới thiệu tổng quan

Kho mã nguồn này triển khai một hệ thống Deep Learning hoàn chỉnh cho bài toán **Semantic Segmentation (Phân vùng ngữ nghĩa)** khuyết tật trên bề mặt sản phẩm công nghiệp, cụ thể là tập dữ liệu chuẩn **KolektorSDD2**.

Hệ thống được thiết kế theo kiến trúc module hóa cao, hỗ trợ cả môi trường huấn luyện cục bộ (Local máy trạm Windows/Linux) và môi trường điện toán đám mây Kaggle GPU.

```
                           +---------------------------+
                           |  KolektorSDD2 Dataset     |
                           |  (3,335 ảnh công nghiệp)  |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |   Data Pipeline           |
                           |   (Decode, Resize, Aug)   |
                           +-------------+-------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       Mô hình ResNet34-UNet Cải Tiến                             |
|                                                                                 |
|  [ResNet34 Encoder] ---> [Multi-Scale Fusion] ---> [Attention Gates]           |
|         |                                                  |                    |
|         +------------ (Skip Connections) ------------------+                    |
|                                                            v                    |
|                                                   [UNet Decoder]                |
|                                                            |                    |
|                                                            v                    |
|                                                   [Defect Mask Output]          |
+---------------------------------------------------------------------------------+
                                         |
                                         v
                           +---------------------------+
                           |  Combined Loss Function   |
                           |  (BCE + Dice + Boundary)  |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |  Đánh giá đa chiều        |
                           |  - Pixel-level (Dice/IoU) |
                           |  - Boundary F1            |
                           |  - Defect Size (S/M/L)    |
                           |  - Image-level / FP Rate  |
                           +---------------------------+
```

---

## 2. Bản đồ hệ thống tài liệu (Documentation Map)

Hệ thống tài liệu được chia thành 12 chuyên đề độc lập nhưng liên kết chặt chẽ với nhau:

| Thứ tự | Tài liệu | Nội dung chính | Đối tượng độc giả |
|:---:|:---|:---|:---|
| **01** | [01_project_overview.md](01_project_overview.md) | Mục tiêu, bài toán, 4 câu hỏi nghiên cứu (RQ1–RQ4), 5 thực nghiệm (E0–E4), công nghệ cốt lõi. | Mọi đối tượng, quản lý dự án, kỹ sư mới |
| **02** | [02_repository_structure.md](02_repository_structure.md) | Cấu trúc cây thư mục chi tiết, phân loại file/folder, vị trí entry point, config, checkpoint. | Kỹ sư phần mềm, người mới tiếp cận repo |
| **03** | [03_system_architecture.md](03_system_architecture.md) | Thiết kế kiến trúc tổng thể, sơ đồ luồng dữ liệu, biểu đồ tuần tự, PlantUML component diagram. | Kiến trúc sư phần mềm, AI Engineer |
| **04** | [04_database_and_data_model.md](04_database_and_data_model.md) | Mô hình lưu trữ dữ liệu dạng tệp (DatasetNinja zlib, PyTorch Tensor, YAML config, log CSV/JSON). | Kỹ sư dữ liệu, lập trình viên Backend/ML |
| **05** | [05_function_reference.md](05_function_reference.md) | Cẩm nang chi tiết từng class, hàm, tham số, giá trị trả về, logic nội bộ và tác dụng phụ (side-effects). | Lập trình viên trực tiếp phát triển & debug |
| **06** | [06_data_pipeline.md](06_data_pipeline.md) | Chi tiết luồng tiền xử lý ảnh, giải mã mask nhị phân, augmentation, chống rò rỉ dữ liệu (data leakage). | ML Engineer phụ trách dữ liệu |
| **07** | [07_project_progress.md](07_project_progress.md) | Ma trận đối soát yêu cầu (RTM), ma trận trạng thái thực tế, checklist Done, lộ trình P0–P3. | Trưởng nhóm, người theo dõi tiến độ |
| **08** | [08_setup_and_execution.md](08_setup_and_execution.md) | Hướng dẫn cài đặt môi trường, chạy kiểm tra dataset, chạy EDA, huấn luyện E0–E4, xử lý sự cố. | Người vận hành, người chạy thực nghiệm |
| **09** | [09_glossary_and_learning_guide.md](09_glossary_and_learning_guide.md) | Giải thích thuật ngữ chuyên sâu (Dice, Boundary F1, Dilation, Dilated Conv, Attention Gate) & lộ trình học. | Người mới bắt đầu học Deep Learning / CV |
| **10** | [10_technical_debt_and_improvements.md](10_technical_debt_and_improvements.md) | Phân tích nợ kỹ thuật, mã nguồn rỗng (0-byte stubs), rủi ro méo tỷ lệ ảnh và đề xuất tối ưu. | Kỹ sư kiểm định chất lượng (QA/Auditor) |
| **11** | [11_change_log_and_analysis_notes.md](11_change_log_and_analysis_notes.md) | Nhật ký phân tích, danh sách sai lệch giữa tài liệu cũ và code thực tế, các vùng cần làm rõ. | Kiểm toán viên mã nguồn, nhóm phát triển |
| **--** | [interfaces.md](interfaces.md) | Định nghĩa hợp đồng giao tiếp giữa các tầng (Dataset -> Model -> Loss -> Metrics -> Logger). | Nhà phát triển module |
| **--** | [02_dataset.md](02_dataset.md) | Bản đặc tả dữ liệu gốc (đã được đối chiếu và hiệu chỉnh thông số thực tế). | Kỹ sư dữ liệu |

---

## 3. Lộ trình đọc khuyến nghị cho người mới bắt đầu (Recommended Reading Path)

Nếu bạn là người mới tiếp cận dự án, hãy tuân thủ trình tự sau để nắm bắt toàn diện hệ thống:

```
[Bước 1] 01_project_overview.md ────> Hiểu bài toán và mục tiêu nghiên cứu
                │
                v
[Bước 2] 02_repository_structure.md ─> Định vị các thư mục và tập tin
                │
                v
[Bước 3] 08_setup_and_execution.md ──> Cài đặt môi trường & chạy verify dataset
                │
                v
[Bước 4] 06_data_pipeline.md ────────> Nắm được cách nạp ảnh và giải mã mask
                │
                v
[Bước 5] 03_system_architecture.md ──> Hiểu sâu cơ chế Multi-Scale & Attention
                │
                v
[Bước 6] 05_function_reference.md ───> Tra cứu chi tiết code khi thực hiện chỉnh sửa
                │
                v
[Bước 7] 07_project_progress.md ─────> Nắm rõ các tác vụ cần hoàn thiện tiếp theo (P0/P1)
```

---

## 4. Tóm tắt nhanh trạng thái hiện tại của dự án

| Hạng mục | Trạng thái | Ghi chú từ cuộc kiểm định thực tế |
|:---|:---:|:---|
| **Dữ liệu & EDA** | `[COMPLETED & VERIFIED]` | Tập dữ liệu KolektorSDD2 đã sẵn sàng. Script EDA `scripts/run_eda.py` đã tạo thống kê và biểu đồ tại `results/eda/`. Ngưỡng khuyết tật nhỏ/vừa/lớn được xác định khoa học theo phân vị 33% và 66% (1392 px và 3675 px). |
| **Kiến trúc mô hình** | `[COMPLETED & VERIFIED]` | Đã hoàn thành 5 module: `ResNet34Encoder`, `MultiScaleFeatureFusion`, `AttentionGate`, `UNetDecoder`, `ResNet34UNet`. Hỗ trợ chuyển đổi cờ `multi_scale` và `attention`. |
| **Hàm mất mát** | `[COMPLETED & VERIFIED]` | Đã có Dice Loss, Differentiable Morphological Boundary Loss, Combined Loss ($L_{BCE} + L_{Dice} + \lambda L_{Boundary}$). |
| **Vòng lặp huấn luyện** | `[COMPLETED & VERIFIED]` | `Trainer`, `train.py`, `logger.py` đã hoàn chỉnh. Đã kiểm thử chạy thử nghiệm thành công 1 epoch cho `E0_baseline`. |
| **Chỉ số đánh giá** | `[COMPLETED & VERIFIED]` | Có đầy đủ hàm tính Dice, IoU, Precision, Recall, Boundary Precision/Recall/F1, Image-level Precision/Recall/F1 và Normal FPR. |
| **Hạ tầng thực nghiệm đầy đủ (E0-E4 50 epochs)** | `[PARTIALLY IMPLEMENTED]` | Mới chỉ chạy khói (smoke test) 1 epoch cho E0; chưa huấn luyện đủ 50 epochs cho 5 thực nghiệm chính. |
| **Module Benchmark & Trực quan hóa** | `[NOT IMPLEMENTED]` | Các file `src/evaluation/benchmark.py`, `src/evaluation/visualization.py`, `scripts/benchmark_model.py` hiện là file rỗng (0 bytes). |
| **Notebooks phân tích** | `[PARTIALLY IMPLEMENTED]` | File `notebooks/04_eda_kolektorsdd2.ipynb` đầy đủ; tuy nhiên `01_eda.ipynb`, `02_visualization.ipynb`, `03_analysis.ipynb` hiện đang là file rỗng (0 bytes). |
