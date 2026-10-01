# 09. Thuật Ngữ Kỹ Thuật và Hướng Dẫn Tự Học (Glossary & Learning Guide)

Tài liệu này được biên soạn dành riêng cho các bạn sinh viên, kỹ sư mới tiếp cận lĩnh vực Thị giác máy tính (Computer Vision) và Phân vùng ngữ nghĩa (Semantic Segmentation), giải thích tường tận các thuật ngữ học thuật và xây dựng lộ trình tiếp thu kiến thức từng bước.

---

## 1. Bảng Tra Cứu Thuật Ngữ Chuyên Sâu (Technical Glossary)

### 1.1. Các Khái niệm Thị giác máy tính & Phân vùng (Segmentation Concepts)

* **Semantic Segmentation (Phân vùng ngữ nghĩa)**:
  * *Định nghĩa*: Tác vụ phân loại từng pixel riêng lẻ trong ảnh vào một lớp cụ thể. Trong bài toán này, mỗi pixel chỉ nhận một trong 2 nhãn: $0$ (bề mặt bình thường) hoặc $1$ (khuyết tật).
  * *So sánh*: Khác với Object Detection (chỉ vẽ hộp chữ nhật Bounding Box bao quanh vật thể), Semantic Segmentation tìm chính xác đường bao thực tế của vết nứt.
* **Ground Truth (Nhãn thực tế)**:
  * Mặt nạ chuẩn do các chuyên gia kiểm định chất lượng gán nhãn thủ công, dùng làm tiêu chuẩn vàng để so sánh với dự đoán của mô hình.
* **Skip Connection (Kết nối tắt)**:
  * Đường nối trực tiếp đặc trưng từ các tầng đầu của Encoder sang các tầng tương ứng của Decoder trong mạng U-Net. Giúp Decoder lấy lại các chi tiết không gian sắc nét đã bị mất trong quá trình giảm mẫu (Downsampling).
* **Bottleneck (Cổ chai)**:
  * Tầng sâu nhất của mạng nơ-ron (trong bài này là `feature4` của ResNet34), nơi độ phân giải không gian bị thu nhỏ tối đa ($8 \times 8$) nhưng chứa lượng kênh ngữ cảnh cao nhất (512 kênh).

---

### 1.2. Các Thành phần Kiến trúc Nâng cao trong Dự án

* **Atrous / Dilated Convolution (Tích chập giãn nở)**:
  * *Ý nghĩa*: Một phép tích chập chèn các khoảng trống (lỗ hổng - "holes") giữa các trọng số của bộ lọc (kernel).
  * *Tác dụng trong dự án*: Giúp tăng kích thước trường thụ cảm (Receptive Field) để mô hình nhìn bao quát ngữ cảnh xung quanh mà không cần giảm kích thước ảnh hoặc tăng thêm số lượng tham số. Khối `MultiScaleFeatureFusion` dùng các tỷ lệ giãn nở $1, 2, 4$.
* **Attention Gate (Cổng chú ý)**:
  * *Ý nghĩa*: Cơ chế tự động gán trọng số ưu tiên $\alpha \in [0, 1]$ cho từng vị trí không gian.
  * *Tác dụng trong dự án*: Giúp mô hình tập trung vào vùng nghi ngờ có vết nứt và chủ động làm mờ (triệt tiêu) các tín hiệu vân bề mặt kim loại không liên quan từ Encoder truyền sang.
* **Morphological Dilation & Erosion (Phép Dãn & Co hình thái học)**:
  * *Ý nghĩa*: Các phép toán hình học xử lý ảnh kinh điển. Phép dãn làm nở rộng đường viền; phép co làm gọt bớt đường viền.
  * *Tác dụng trong dự án*: Bằng cách lấy $\text{Dilation}(M) - \text{Erosion}(M)$, ta thu được đường viền mảnh (boundary map) một cách hoàn toàn khả vi thông qua lớp `max_pool2d`.

---

### 1.3. Các Chỉ số Đánh giá Hiệu năng (Evaluation Metrics)

* **Dice Coefficient (F1-Score cấp độ Pixel)**:
  * Đo lường mức độ trùng khớp giữa vùng dự đoán $P$ và vùng nhãn $Y$:
    $$\text{Dice} = \frac{2 |P \cap Y|}{|P| + |Y|}$$
  * Đặc biệt hữu hiệu khi dữ liệu bị mất cân bằng lớp trầm trọng (như vết nứt chỉ chiếm $<1\%$ ảnh).
* **IoU (Intersection over Union / Jaccard Index)**:
  * Tỷ lệ giữa diện tích phần giao và diện tích phần hợp:
    $$\text{IoU} = \frac{|P \cap Y|}{|P \cup Y|} = \frac{\text{Dice}}{2 - \text{Dice}}$$
* **Boundary F1 ($F1_B$)**:
  * Chỉ số F1 được tính riêng biệt trên các pixel thuộc đường viền biên giới của khuyết tật, cho phép sai số vị trí trong khoảng dung sai $2$ pixel. Phản ánh trực tiếp độ sắc nét và tính chuẩn xác của ranh giới vết nứt.
* **Normal False Positive Rate (Tỷ lệ báo động giả trên ảnh sạch)**:
  * $$FPR_{\text{normal}} = \frac{\text{Số ảnh bình thường bị mô hình đoán nhầm là có lỗi}}{\text{Tổng số ảnh bình thường}}$$
  * Trong nhà máy, nếu chỉ số này quá cao, dây chuyền sẽ liên tục dừng máy sai hoặc loại bỏ sản phẩm đạt tiêu chuẩn, gây tổn thất kinh tế lớn.

---

## 2. Lộ trình Tự học cho Người Mới Bắt Đầu (Beginner Learning Path)

Để làm chủ mã nguồn này từ con số 0, hãy chia quá trình học thành 4 chặng:

```
[Chặng 1: Nền tảng Dữ liệu]
    │  - Hiểu bài toán KolektorSDD2: docs/01_project_overview.md
    │  - Khảo sát dữ liệu thực tế: kết quả trong results/eda/
    │  - Đọc code nạp dữ liệu: src/datasets/kolektor.py
    ▼
[Chặng 2: Kiến trúc Mạng Nơ-ron]
    │  - Tìm hiểu mô hình U-Net kinh điển
    │  - Đọc cách cài đặt ResNet34UNet: src/models/resnet34_unet.py
    │  - Tìm hiểu cơ chế Attention Gate: src/models/attention_gate.py
    ▼
[Chặng 3: Tối ưu hóa & Hàm mất mát]
    │  - Hiểu vì sao chỉ dùng BCE sẽ thất bại khi mất cân bằng lớp
    │  - Đọc code Soft Dice: src/losses/dice.py
    │  - Đọc code vi phân Boundary Loss: src/losses/boundary.py
    ▼
[Chặng 4: Vận hành Thực nghiệm]
    │  - Chạy thử nghiệm 1 epoch: docs/08_setup_and_execution.md
    │  - Đọc đồ thị hội tụ: experiments/E0_baseline/loss_curve.png
    │  - Thực hiện tiếp các nhiệm vụ ưu tiên P0: docs/07_project_progress.md
```

---

## 3. Các Ví dụ Minh Họa Thực Tế trong Dự Án

### 3.1. Tại sao hàm mất mát Binary Cross-Entropy (BCE) đơn thuần lại không hiệu quả?
Trong tập dữ liệu KolektorSDD2, có tới $89.3\%$ tổng số ảnh là ảnh hoàn toàn sạch lỗi. Ngay cả trong $10.7\%$ ảnh có lỗi, diện tích vết nứt trung bình chỉ chiếm khoảng $0.5\%$ đến $1\%$ diện tích bức ảnh.  
Nếu một mô hình chỉ học cách dự đoán **tất cả mọi pixel đều bằng 0 (Background)**:
* Độ chính xác cấp pixel (Pixel Accuracy) vẫn đạt tới $99.5\%$!
* Hàm mất mát BCE sẽ giảm xuống rất thấp.
* Tuy nhiên, mô hình hoàn toàn vô dụng vì không phát hiện được bất kỳ lỗi nào ($\text{Recall} = 0$, $\text{Dice} = 0$).  
👉 **Giải pháp**: Phải kết hợp **Dice Loss** (tối ưu hóa diện tích giao nhau, phạt nặng nếu bỏ sót vùng lỗi) cùng với **Boundary Loss** (ép mô hình học đúng ranh giới).

### 3.2. Cơ chế bù trừ dung sai trong Boundary Metrics
Đường viền của các vết nứt trong môi trường công nghiệp thường có độ dày chỉ từ 1 đến 3 pixel. Do đó, việc người gán nhãn chấm viền lệch 1 pixel là sai số tự nhiên không thể tránh khỏi giữa các chuyên gia.  
Nếu tính Boundary F1 theo kiểu so khớp pixel cứng nhắc (Exact Match), chỉ số sẽ rất thấp dù mô hình dự đoán gần như hoàn hảo.  
Hàm `boundary_metrics` trong `src/evaluation/boundary_metrics.py` áp dụng kỹ thuật dãn nở đường viền nhãn với cửa sổ $(2 \times tolerance + 1) = 5 \times 5$ (với $tolerance = 2$). Bất kỳ pixel viền dự đoán nào rơi vào vùng lân cận 2 pixel của viền thực tế đều được công nhận là phát hiện ranh giới chính xác.
