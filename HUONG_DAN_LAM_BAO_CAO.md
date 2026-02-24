# HƯỚNG DẪN LÀM BÁO CÁO DỰ ÁN
# PCB Defect Detection System với YOLOv8
## Phát hiện và khoanh vùng lỗi trên mạch PCB
## (Tập trung vào TRIỂN KHAI và ĐÓNG GÓP của Nhóm)

---

## 📋 CẤU TRÚC BÁO CÁO MỚI (Nghiêng về Implementation)

### **Trang bìa + Mục lục**
### **I. GIỚI THIỆU VÀ MỤC TIÊU** (2 trang)
### **II. TỔNG QUAN YOLOv8 VÀ DATASET** (2 trang) - *Ngắn gọn, chỉ nêu cái nhóm sử dụng*
### **III. THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG** (5-6 trang) - *⭐ PHẦN QUAN TRỌNG NHẤT*
### **IV. QUÁ TRÌNH TRAINING VÀ FINE-TUNING** (3-4 trang) - *Nhóm đã làm gì*
### **V. TESTING VÀ ĐÁNH GIÁ** (3-4 trang) - *Kết quả nhóm đạt được*
### **VI. KẾT LUẬN VÀ ĐÓNG GÓP** (2 trang)
### **VII. TÀI LIỆU THAM KHẢO**
### **PHỤ LỤC**

**Tổng số trang:** 17-20 trang

---

## 📝 NỘI DUNG CHI TIẾT TỪNG PHẦN

---

## **I. GIỚI THIỆU VÀ MỤC TIÊU** (2 trang)

> **Tóm tắt:** Trình bày lý do thực hiện dự án, mục tiêu kỹ thuật và triển khai, phạm vi công việc, và phân công nhiệm vụ trong nhóm.

### 1.1. Đặt vấn đề

> Giải thích vấn đề thực tế cần giải quyết, tại sao cần tự động hóa việc phát hiện lỗi trên mạch PCB.

**Nội dung:**
- Bài toán phát hiện lỗi trên mạch PCB trong thực tế sản xuất
- Tại sao cần tự động hóa (đảm bảo chất lượng, giảm sai sót, tăng năng suất)
- 6 loại lỗi phổ biến: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
- Thách thức khi triển khai thực tế

**Ví dụ viết (góc độ thực tế):**
```
Trong quá trình sản xuất mạch in (PCB), việc kiểm tra chất lượng thủ công 
tốn nhiều thời gian, dễ bỏ sót lỗi, đặc biệt với các lỗi nhỏ như mouse_bite 
hay spur. Nhóm chúng em xây dựng hệ thống tự động phát hiện và khoanh vùng 
6 loại lỗi phổ biến trên PCB.

Với YOLOv8, hệ thống có khả năng phát hiện lỗi real-time, đánh giá mức độ 
nghiêm trọng (CRITICAL/HIGH/MEDIUM/LOW), và tự động đưa ra kết quả kiểm tra 
chất lượng (QC) PASS/FAIL.
```

### 1.2. Mục tiêu của nhóm

> Nêu rõ 3 nhóm mục tiêu: kỹ thuật (độ chính xác, tốc độ), triển khai (code quality), và học tập (kiến thức thu được).

**Liệt kê rõ ràng những gì NHÓM MUỐN LÀM:**

✅ **Mục tiêu kỹ thuật:**
- Phát hiện và khoanh vùng 6 loại lỗi PCB
- Đánh giá mức độ nghiêm trọng của từng lỗi
- Đạt độ chính xác cao (mAP@0.5 > 90%)
- Tốc độ real-time (>25 FPS)

✅ **Mục tiêu triển khai:**
- Code module hóa, dễ bảo trì và mở rộng
- Tự động kiểm tra chất lượng QC (PASS/FAIL)
- Tạo báo cáo QC chi tiết dạng CSV
- Hỗ trợ cả batch processing và real-time detection

✅ **Mục tiêu học tập:**
- Nắm vững quy trình training deep learning model
- Hiểu cách deploy model vào ứng dụng thực tế
- Làm việc nhóm và quản lý project

### 1.3. Phạm vi dự án

> Xác định rõ công cụ, dữ liệu, ngôn ngữ lập trình và các sản phẩm đầu ra của dự án.

**Nêu rõ:**
- **Công cụ sử dụng:** YOLOv8 (Ultralytics)
- **Dataset:** PCB Defect Dataset với 6 loại lỗi (từ Roboflow/Kaggle)
- **Ngôn ngữ:** Python 3.10+
- **Sản phẩm:** Module code + Scripts + QC Reports + Documentation

### 1.4. Phân công công việc nhóm (Nếu có)

**Ví dụ:**
```
[Bảng 1.1] Phân công công việc

| Thành viên | Công việc chính                           |
|------------|-------------------------------------------|
| Thành viên A | Dataset preparation, Training            |
| Thành viên B | Code module development, Testing         |
| Thành viên C | Webcam implementation, Documentation     |
| Toàn nhóm   | Testing, Debugging, Report writing       |
```

*(Nếu làm cá nhân, bỏ qua phần này hoặc viết "Dự án thực hiện bởi...")*

### 1.5. Bố cục báo cáo

Tóm tắt nội dung các phần tiếp theo (ngắn gọn).

---

## **II. TỔNG QUAN YOLOv8 VÀ DATASET** (2 trang) - *Ngắn gọn*

> **Tóm tắt:** Giới thiệu ngắn gọn YOLOv8 là gì, tại sao nhóm chọn model này, và tổng quan về dataset sử dụng.

> **Lưu ý:** Phần này KHÔNG cần viết dài dòng về lý thuyết. Chỉ giới thiệu 
> ngắn gọn YOLOv8 là gì và dataset nhóm sử dụng thế nào.

### 2.1. Giới thiệu YOLOv8

> Mô tả YOLOv8 một cách ngắn gọn, nhấn mạnh tại sao nhóm lựa chọn model này thay vì các model khác.

**Viết ngắn gọn (0.5 trang):**

```
YOLOv8 là phiên bản mới nhất của YOLO (You Only Look Once), được phát 
triển bởi Ultralytics vào năm 2023. Đây là một trong những model Object 
Detection tiên tiến nhất hiện nay, nổi bật với:

- Tốc độ nhanh: Phù hợp cho real-time applications
- Độ chính xác cao: State-of-the-art trên nhiều benchmarks
- Dễ sử dụng: API đơn giản, documentation đầy đủ
- Nhiều variants: n/s/m/l/x cho các nhu cầu khác nhau

Nhóm chọn YOLOv8 vì những lý do sau:
- ✅ Open-source và active development
- ✅ Có pretrained weights (COCO dataset)
- ✅ Hỗ trợ đầy đủ cho training custom dataset
- ✅ Export sang nhiều format (ONNX, TFLite...)
```

**Sơ đồ đơn giản:**
```
[Hình 2.1] Kiến trúc YOLOv8 (High-level)

Input Image → [Backbone] → [Neck] → [Head] → Outputs
            (Features)   (Fusion)  (Detect)  (Boxes+Classes)
```

### 2.2. Dataset - PCB Defect Detection

> Trình bày nguồn dataset, số lượng ảnh, cách chia train/val/test, 6 loại lỗi PCB, và đánh giá chất lượng dataset.

**2.2.1. Nguồn và thống kê:**

```
[Bảng 2.1] Thông tin Dataset

| Thông tin        | Chi tiết                                |
|------------------|-----------------------------------------|
| Nguồn            | Roboflow / PKU-Market-PCB               |
| License          | CC BY 4.0                               |
| Tổng số ảnh      | [Số ảnh trong dataset]                  |
| Training         | [Số ảnh train]                          |
| Validation       | [Số ảnh val]                            |
| Test             | [Số ảnh test]                           |
| Số classes       | 6 loại lỗi PCB                          |
| Format           | YOLO (TXT annotations)                  |
| Image size       | Đa dạng (resize về 640x640 khi train)   |
```

**2.2.2. 6 Loại lỗi PCB:**

```
[Bảng 2.2] Danh sách Classes - Các loại lỗi PCB

| ID | Loại lỗi          | Mô tả                          | Mức độ    |
|----|--------------------|---------------------------------|-----------|
| 0  | missing_hole       | Lỗ khoan bị thiếu trên PCB     | HIGH      |
| 1  | mouse_bite         | Khuyết tật ở cạnh mạch         | MEDIUM    |
| 2  | open_circuit       | Mạch hở - đường mạch bị đứt    | CRITICAL  |
| 3  | short              | Ngắn mạch - 2 mạch nối nhầm    | CRITICAL  |
| 4  | spur               | Gai đồng thừa từ đường mạch    | MEDIUM    |
| 5  | spurious_copper    | Đồng thừa không mong muốn      | LOW       |
```

**2.2.3. Phân loại mức độ nghiêm trọng:**

```
- CRITICAL: open_circuit, short → Lỗi gây hỏng mạch, cần loại bỏ ngay
- HIGH: missing_hole → Ảnh hưởng lắp ráp linh kiện
- MEDIUM: mouse_bite, spur → Có thể ảnh hưởng chất lượng
- LOW: spurious_copper → Lỗi nhẹ
```

**2.2.4. Chất lượng dataset:**

**Nhóm đã kiểm tra:**
- ✅ Labels: Annotations chính xác, bounding boxes khít với defects
- ✅ Balance: Phân bố các loại lỗi
- ✅ Quality: Chất lượng ảnh PCB đa dạng

---

## **III. THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG** (5-6 trang) ⭐

> **Tóm tắt:** Mô tả chi tiết kiến trúc hệ thống, thiết kế code module, implementation details, challenges gặp phải và cách giải quyết.

> **Đây là phần QUAN TRỌNG NHẤT** - Viết chi tiết những gì nhóm đã làm!

### 3.1. Tổng quan kiến trúc hệ thống

> Trình bày sơ đồ tổng thể hệ thống nhóm xây dựng, từ dataset đến training, testing và deployment.

**3.1.1. Sơ đồ tổng quát:**

```
[Hình 3.1] Kiến trúc hệ thống do nhóm xây dựng

┌─────────────────────────────────────────────────────────────┐
│                    HỆ THỐNG NHÓM XÂY DỰNG                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐       ┌──────────────────┐       ┌─────────────┐
│   Dataset   │       │   TRAINING       │       │   Trained   │
│  (Roboflow) │  ───► │   - Data Aug     │  ───► │    Model    │
│             │       │   - Fine-tuning  │       │   (best.pt) │
└─────────────┘       └──────────────────┘       └─────────────┘
                                                         │
                          ┌──────────────────────────────┴────┐
                          │                                   │
                          ▼                                   ▼
              ┌──────────────────────┐         ┌──────────────────────┐
              │   TESTING MODULE     │         │   DEPLOYMENT MODULE  │
              │   - Batch test       │         │   - Webcam stream    │
              │   - Metrics eval     │         │   - Real-time UI     │
              │   - Visualization    │         │   - Interactive      │
              └──────────────────────┘         └──────────────────────┘
```

**3.1.2. Stack công nghệ:**

```
[Bảng 3.1] Technology Stack

| Layer            | Công nghệ/Tool                          |
|------------------|-----------------------------------------|
| Deep Learning    | PyTorch, YOLOv8 (Ultralytics)           |
| Computer Vision  | OpenCV, Pillow                          |
| Data Processing  | NumPy, Pandas                           |
| Visualization    | Matplotlib                              |
| Development      | Python 3.10, Git, GitHub                |
| Hardware         | [GPU/CPU cụ thể bạn dùng]              |
```

### 3.2. Thiết kế Module Code

> Giải thích chi tiết cấu trúc module code, design principles áp dụng, và lý do thiết kế như vậy.

> **Đây là ĐÓNG GÓP CHÍNH của nhóm** - Code architecture

**3.2.1. Cấu trúc module:**

```
[Hình 3.2] Code Architecture do nhóm thiết kế

Final-Deep-Learning-main/
│
├── defect_detector.py              ◄─── CORE MODULE
│   ├── Class: DefectDetector
│   │     ├── __init__()         # Khởi tạo model
│   │     ├── train()            # Training logic
│   │     ├── predict()          # Inference / Phát hiện lỗi
│   │     ├── validate()         # Validation
│   │     ├── analyze_defects()  # Phân tích chi tiết lỗi + mức độ nghiêm trọng
│   │     ├── visualize()        # Visualization với QC status
│   │     └── generate_report()  # Tạo báo cáo QC (CSV)
│   │
│   └── Class: WebcamDefectDetector
│         ├── __init__()         # Load model
│         └── run()              # Real-time defect detection
│
├── train_detector.py                ◄─── TRAINING SCRIPT
│   └── CLI để train với args
│
├── test_detector.py                 ◄─── TESTING SCRIPT
│   └── CLI để test + tạo QC report
│
└── webcam_detector.py               ◄─── WEBCAM SCRIPT
    └── CLI để chạy webcam detection
```

**3.2.2. Design Principles:**

**Nhóm áp dụng các nguyên tắc:**

1. **Modularity (Module hóa):**
   - Core logic tách riêng trong `DefectDetector` class
   - Scripts chỉ là wrapper đơn giản
   - Dễ maintain và extend

2. **Reusability (Tái sử dụng):**
   - Một class `DefectDetector` cho cả train/test/predict/analyze
   - Không duplicate code
   - DRY principle

3. **User-friendly:**
   - CLI scripts với argparse
   - Clear documentation
   - Helpful error messages

4. **Flexibility:**
   - Support nhiều YOLOv8 variants (n/s/m/l/x)
   - Customizable hyperparameters
   - Easy to export different formats

**3.2.3. Chi tiết DefectDetector class:**

```python
class DefectDetector:
    """
    ĐÓNG GÓP CHÍNH: PCB Defect Detection Engine
    
    Nhóm thiết kế class này để:
    - Phát hiện 6 loại lỗi PCB
    - Đánh giá mức độ nghiêm trọng (CRITICAL/HIGH/MEDIUM/LOW)
    - Tự động kiểm tra QC (PASS/FAIL)
    - Tạo báo cáo QC chi tiết
    """
    
    def __init__(self, model_type='n', pretrained=True):
        """Khởi tạo model với pretrained weights"""
        pass
    
    def train(self, data_yaml, epochs, batch, ...):
        """Training pipeline cho defect detection"""
        pass
    
    def analyze_defects(self, image_path, conf):
        """
        Phân tích chi tiết lỗi trên ảnh PCB:
        - Đếm số lỗi theo từng loại
        - Đánh giá severity
        - Kết luận QC PASS/FAIL
        """
        pass
    
    def generate_report(self, image_dir, conf, save_path):
        """Tạo báo cáo QC cho batch ảnh PCB"""
        pass
```

**Giải thích tại sao thiết kế như vậy:**
```
Thay vì gọi trực tiếp YOLOv8 API, nhóm wrap lại trong 
DefectDetector class với các lợi ích:

1. Interface đơn giản hơn:
   analysis = detector.analyze_defects(image_path)  # Phân tích lỗi
   report = detector.generate_report(image_dir)     # Tạo báo cáo

2. Thêm domain-specific logic:
   - Phân loại mức độ nghiêm trọng cho từng loại lỗi
   - Tự động đánh giá QC PASS/FAIL
   - Tạo báo cáo CSV chi tiết
   - Màu sắc cố định cho từng loại lỗi

3. Maintains state:
   - Defect types, severity levels
   - Color mapping, descriptions
   - Model config
```

### 3.3. Implementation Details

> Mô tả chi tiết cách nhóm implement 3 scripts chính: training, testing, và webcam detection, kèm features đặc biệt.

**3.3.1. Training Script (train_detector.py):**

> Trình bày các features nhóm thêm vào training script: CLI arguments, device handling, auto validation, plot generation.

**Những gì nhóm implement:**

```python
# Nhóm thiết kế CLI với argparse để dễ sử dụng
parser.add_argument('--model', choices=['n','s','m','l','x'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=16)
# ... và nhiều args khác

# Nhóm thêm device handling thông minh
device = args.device
if device.lower() != 'cpu':
    try:
        device = int(device)  # Convert '0' → 0
    except ValueError:
        device = 'cpu'  # Fallback

# Nhóm tự động generate training analysis
plot_training_results(results_dir)
```

**Các tính năng đặc biệt nhóm thêm vào:**
- ✅ Tự động validate sau khi train
- ✅ Generate training plots
- ✅ Print summary rõ ràng
- ✅ Handle errors gracefully
- ✅ Support resume training

**3.3.2. Testing Script (test_detector.py):**

> Giải thích batch testing, visualization options, và metrics reporting mà nhóm đã implement.

**Nhóm implement các features:**

```
1. Batch Testing:
   - Test trên toàn bộ folder images
   - Tự động count detections
   - Phân tích class distribution

2. Visualization:
   - Option để visualize predictions
   - Save kết quả ra file
   - Matplotlib-based plots

3. Metrics Reporting:
   - In ra số lượng detections
   - Class distribution per image
   - Clear summary sau khi test
```

**3.3.3. Webcam Script (webcam_detector.py):**

> Trình bày tính năng real-time detection qua webcam với performance monitoring, interactive controls, và visual enhancements.

**Đây là tính năng DEMO THỰC TẾ nhóm xây dựng:**

**Features nhóm implement:**

1. **Real-time Performance Monitoring:**
   ```python
   # Display FPS, Detection count, Confidence threshold
   info_text = [
       f"FPS: {current_fps:.1f}",
       f"Detections: {detection_count}",
       f"Conf: {self.conf_threshold:.2f}"
   ]
   ```

2. **Interactive Controls:**
   ```
   Nhóm thiết kế keyboard controls:
   - 'q': Quit
   - 's': Save current frame
   - 'p': Pause/Resume
   - '+/-': Adjust confidence threshold
   ```

3. **Visual Enhancements:**
   - Colored bounding boxes per class
   - Labels với confidence scores
   - Info overlay
   - Frame counter

**Challenges nhóm gặp và giải quyết:**

```
[Bảng 3.2] Challenges trong Implementation

| Vấn đề                    | Giải pháp của nhóm              |
|---------------------------|---------------------------------|
| FPS thấp khi dùng CPU     | Optimize inference, reduce size |
| Webcam lag                | Async processing, frame skip    |
| Bounding box vẽ không đẹp | Custom draw với OpenCV          |
| Hotkeys không hoạt động   | Use cv.waitKey() đúng cách      |
```

### 3.4. Documentation và Code Quality

> Nhấn mạnh các nỗ lực của nhóm trong việc viết docstrings, README, comments để đảm bảo code quality.

**Nhóm chú trọng vào:**

1. **Docstrings đầy đủ:**
   ```python
   def train(self, data_yaml, epochs, ...):
       """
       Train the component detector
       
       Args:
           data_yaml: Path to data.yaml
           epochs: Number of epochs
           ...
       
       Returns:
           Training results
       """
   ```

2. **README.md chi tiết:**
   - Installation instructions
   - Usage examples
   - Troubleshooting guide

3. **Comments trong code:**
   - Giải thích logic phức tạp
   - Note các edge cases
   - TODO cho future improvements

**3.5. Testing và Debugging Process:**

> Mô tả development workflow và các công cụ nhóm sử dụng để test, debug và optimize code.

**Quy trình nhóm thực hiện:**

```
[Hình 3.3] Development Workflow

1. Code → 2. Unit Test → 3. Integration → 4. Debug → 5. Refactor
   ↑                                                          |
   └──────────────────────────────────────────────────────────┘
```

**Các công cụ sử dụng:**
- Git cho version control
- GitHub cho collaboration
- Print debugging
- PyTorch profiler (nếu cần optimize)

---

## **IV. KẾT QUẢ THỰC NGHIỆM** (3-4 trang)

> **Tóm tắt:** Trình bày kết quả training (loss curves, metrics evolution), validation (confusion matrix, PR curves), testing (metrics trên test set), và real-time performance.

### 4.1. Kết quả Training

> Phân tích quá trình training qua 100 epochs: loss curves giảm như thế nào, metrics evolution, có dấu hiệu overfitting không.

**4.1.1. Training curves:**

**Mô tả:**
```
Quá trình training được thực hiện trong 100 epochs. Hình 4.1 cho thấy 
sự hội tụ của các loss functions theo thời gian.
```

**Chèn hình:**
```
[Hình 4.1] Training Loss Curves
(Chèn file: runs/detect/.../results.png)

Nhận xét:
- Box Loss giảm từ 2.04 → 1.11 (giảm 45.6%)
- Class Loss giảm từ 2.64 → 0.54 (giảm 79.5%)
- DFL Loss giảm ổn định
- Không có dấu hiệu overfitting
```

**4.1.2. Metrics evolution:**

```
[Bảng 4.1] Evolution của Metrics qua Epochs

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| 1     | 0.788     | 0.702  | 0.773   | 0.391        |
| 10    | 0.888     | 0.875  | 0.907   | 0.535        |
| 25    | 0.920     | 0.933  | 0.951   | 0.604        |
| 50    | 0.931     | 0.944  | 0.962   | 0.640        |
| 75    | 0.935     | 0.948  | 0.966   | 0.663        |
| 100   | 0.936     | 0.943  | 0.964   | 0.672        |
```

**Nhận xét:**
```
- Precision đạt 93.6%: Model có độ tin cậy cao khi phát hiện
- Recall đạt 94.3%: Model hiếm khi bỏ sót linh kiện
- mAP@0.5 đạt 96.4%: Kết quả xuất sắc cho ứng dụng thực tế
- Model hội tụ tốt sau epoch 50
```

### 4.2. Kết quả Validation

> Phân tích kết quả validation qua confusion matrix, PR curves, F1-confidence curves để đánh giá chất lượng model.

**4.2.1. Confusion Matrix:**

```
[Hình 4.2] Confusion Matrix (Normalized)
(Chèn file: runs/detect/.../confusion_matrix_normalized.png)

Phân tích:
- Các class chính có độ chính xác cao (> 95%)
- Nhầm lẫn chủ yếu giữa Capacitor và Ceramic Capacitor
- Điều này hợp lý vì 2 loại này có hình dạng tương tự
```

**4.2.2. Precision-Recall Curve:**

```
[Hình 4.3] Precision-Recall Curves
(Chèn file: runs/detect/.../BoxPR_curve.png)

Nhận xét:
- Hầu hết classes có đường cong gần góc trên-phải (lý tưởng)
- mAP@0.5 = 0.964 (rất cao)
```

**4.2.3. F1-Confidence Curve:**

```
[Hình 4.4] F1-Confidence Curve
(Chèn file: runs/detect/.../BoxF1_curve.png)

Nhận xét:
- F1 score đạt cao nhất ở confidence threshold ~0.4
- Tại conf=0.25 (mặc định): F1 vẫn rất cao
```

### 4.3. Kết quả Test

> Báo cáo metrics chi tiết trên test set (367 ảnh chưa từng thấy), kết quả theo từng class, inference time.

**4.3.1. Metrics trên Test Set:**

```
[Bảng 4.2] Kết quả trên Test Set (367 ảnh)

| Metric           | Giá trị  | Đánh giá        |
|------------------|----------|-----------------|
| Precision        | 93.6%    | Rất tốt         |
| Recall           | 94.3%    | Rất tốt         |
| mAP@0.5          | 96.4%    | Xuất sắc        |
| mAP@0.5:0.95     | 67.2%    | Tốt             |
| Inference Time   | ~8ms     | Real-time       |
```

**4.3.2. Kết quả theo từng class:**

```
[Bảng 4.3] Performance từng Class

| Defect Type        | Precision | Recall | mAP@0.5 | Severity |
|--------------------|-----------|--------|---------|----------|
| missing_hole       | [value]   | [value]| [value] | HIGH     |
| mouse_bite         | [value]   | [value]| [value] | MEDIUM   |
| open_circuit       | [value]   | [value]| [value] | CRITICAL |
| short              | [value]   | [value]| [value] | CRITICAL |
| spur               | [value]   | [value]| [value] | MEDIUM   |
| spurious_copper    | [value]   | [value]| [value] | LOW      |
```

### 4.4. Kết quả Visualization

> Trình bày các ví dụ detection thành công, phân bố labels, để minh họa trực quan chất lượng model.

**4.4.1. Ví dụ Detection thành công:**

```
[Hình 4.5] Ví dụ Detection trên Test Images
(Chèn file: runs/detect/.../val_batch0_pred.jpg)

Mô tả:
- Model phát hiện chính xác tất cả linh kiện
- Bounding boxes khít với objects
- Confidence scores cao (> 0.8)
```

**4.4.2. Label Distribution:**

```
[Hình 4.6] Phân bố Labels trong Dataset
(Chèn file: runs/detect/.../labels.jpg)

Nhận xét:
- Dataset có sự cân bằng tốt giữa các classes
- Kích thước objects đa dạng
```

### 4.5. Real-time Performance

> Đo đạc performance của webcam detection: FPS trên GPU/CPU, latency, resolution, để chứng minh khả năng real-time.

**4.5.1. Webcam Detection:**

```
[Bảng 4.4] Performance Real-time

| Metric              | Giá trị      |
|---------------------|--------------|
| FPS (GPU)           | ~120 FPS     |
| FPS (CPU)           | ~25 FPS      |
| Latency             | ~8ms         |
| Resolution          | 640x480      |
| Confidence Threshold| 0.5          |
```

**Nhận xét:**
```
- YOLOv8n đủ nhanh cho real-time trên cả GPU và CPU
- FPS ổn định, không bị lag
- Có thể điều chỉnh confidence threshold real-time
```

### 4.6. So sánh với các Model khác

> So sánh YOLOv8n với các variants khác (s, m) về mAP, số parameters, tốc độ để justify lựa chọn của nhóm.

```
[Bảng 4.5] So sánh YOLOv8 variants

| Model    | mAP@0.5 | Params | Speed (ms) | Use Case        |
|----------|---------|--------|------------|-----------------|
| YOLOv8n  | 96.4%   | 3.2M   | 8          | ✅ Real-time    |
| YOLOv8s  | 97.2%   | 11.2M  | 15         | Balanced        |
| YOLOv8m  | 97.8%   | 25.9M  | 28         | High accuracy   |
```

**Kết luận:**
```
- YOLOv8n được chọn vì cân bằng giữa tốc độ và độ chính xác
- mAP chênh lệch không nhiều so với variants lớn hơn
- Phù hợp cho ứng dụng real-time
```

---

## **V. ĐÁNH GIÁ VÀ KẾT LUẬN** (2-3 trang)

> **Tóm tắt:** Đánh giá ưu nhược điểm, tổng kết những gì đã đạt được, hướng phát triển, đóng góp khoa học/thực tiễn, và bài học kinh nghiệm.

### 5.1. Đánh giá chung

> Phân tích ưu/nhược điểm của hệ thống dựa trên kết quả thực nghiệm, nhấn mạnh điểm mạnh và hạn chế cần cải thiện.

**5.1.1. Ưu điểm:**

✅ **Độ chính xác cao:**
- mAP@0.5 = 96.4% - vượt mục tiêu đề ra (> 90%)
- Precision và Recall đều > 93%

✅ **Tốc độ Real-time:**
- FPS ~120 trên GPU, ~25 trên CPU
- Latency thấp (~8ms)

✅ **Khả năng tổng quát hóa:**
- Model hoạt động tốt trên test set chưa từng thấy
- Không có dấu hiệu overfitting

✅ **Dễ triển khai:**
- Code module hóa rõ ràng
- Hỗ trợ cả batch processing và real-time
- Có thể export sang các format khác (ONNX, TFLite)

**5.1.2. Nhược điểm:**

⚠️ **Một số lỗi nhỏ khó phát hiện:**
- Lỗi spur và spurious_copper đôi khi có kích thước rất nhỏ
- Cần resolution cao hơn để detect tốt hơn

⚠️ **Nhầm lẫn giữa một số loại lỗi:**
- spur và spurious_copper có thể bị nhầm
- Cần thêm dữ liệu để phân biệt

⚠️ **Chưa tối ưu cho edge devices:**
- Model vẫn còn nặng cho embedded systems
- Cần quantization để triển khai trên dây chuyền sản xuất

### 5.2. Kết luận

> Tổng kết những gì nhóm đã hoàn thành, kiến thức học được, sản phẩm tạo ra, và tính ứng dụng thực tế.

**5.2.1. Những gì đã đạt được:**

1. ✅ **Hoàn thành mục tiêu đề ra:**
   - Xây dựng thành công hệ thống phát hiện lỗi PCB
   - Phát hiện được 6 loại lỗi phổ biến
   - Tự động đánh giá QC PASS/FAIL
   - Triển khai được real-time detection

2. ✅ **Kiến thức thu được:**
   - Hiểu sâu về Object Detection
   - Nắm vững kiến trúc YOLOv8
   - Kinh nghiệm training deep learning model
   - Kỹ năng triển khai ứng dụng kiểm tra chất lượng

3. ✅ **Sản phẩm:**
   - Code hoàn chỉnh, module hóa tốt
   - Hệ thống QC tự động
   - Báo cáo QC chi tiết dạng CSV
   - Demo real-time hoạt động ổn định

**5.2.2. Tính ứng dụng thực tế:**

📌 **Kiểm tra chất lượng PCB (QC):**
- Tự động phát hiện 6 loại lỗi trên mạch PCB
- Phân loại mức độ nghiêm trọng (CRITICAL/HIGH/MEDIUM/LOW)
- Đưa ra kết luận PASS/FAIL tự động
- Tạo báo cáo QC chi tiết

📌 **Dây chuyền sản xuất:**
- Kiểm tra PCB real-time trên dây chuyền
- Giảm tỷ lệ PCB lỗi đến tay khách hàng
- Tăng năng suất so với kiểm tra thủ công

📌 **Nghiên cứu:**
- Dataset và model cho nghiên cứu PCB defect detection
- Baseline cho các nghiên cứu tiếp theo

### 5.3. Hướng phát triển

> Đề xuất các hướng cải thiện model, mở rộng chức năng, và nâng cao trải nghiệm người dùng trong tương lai.

**5.3.1. Cải thiện model:**

🔧 **Tăng dataset:**
- Thu thập thêm 5000-10000 ảnh
- Đa dạng góc chụp, điều kiện ánh sáng
- Thêm ảnh từ nhiều loại bo mạch khác nhau

🔧 **Fine-tuning:**
- Thử YOLOv8s, YOLOv8m để tăng độ chính xác
- Tối ưu hyperparameters
- Thử các augmentation strategies khác

🔧 **Giải quyết class confusion:**
- Tăng dữ liệu phân biệt spur vs spurious_copper
- Thêm augmentation cho lỗi nhỏ

**5.3.2. Mở rộng chức năng:**

🚀 **Thêm loại lỗi:**
- Mở rộng thêm lỗi hàn (cold solder, excess solder)
- Lỗi linh kiện (missing component, wrong component)
- Lỗi alignment (misalignment, tombstoning)

🚀 **Tích hợp thêm:**
- Kết nối với database quản lý
- Export báo cáo tự động
- API REST cho ứng dụng web/mobile

🚀 **Triển khai edge:**
- Quantization để giảm model size
- Deploy lên Raspberry Pi, Jetson Nano
- Mobile app (iOS/Android)

**5.3.3. Cải thiện UX:**

💡 **GUI application:**
- Desktop app với giao diện đẹp
- Drag-and-drop ảnh
- Hiển thị kết quả trực quan

💡 **Web interface:**
- Upload ảnh qua web
- Real-time detection qua browser
- Cloud deployment

💡 **Batch processing:**
- Xử lý hàng loạt ảnh
- Progress tracking
- Export kết quả sang Excel/CSV

### 5.4. Đóng góp của đề tài

> Nêu rõ đóng góp về mặt khoa học (methodology, pipeline) và thực tiễn (tool sử dụng được, open-source).

**5.4.1. Đóng góp về mặt khoa học:**
- Áp dụng thành công YOLOv8 cho bài toán phát hiện lỗi PCB
- Xây dựng hệ thống phân loại mức độ nghiêm trọng lỗi
- Pipeline hoàn chỉnh từ detection → analysis → QC report

**5.4.2. Đóng góp về mặt thực tiễn:**
- Tool kiểm tra chất lượng PCB tự động
- Hệ thống QC PASS/FAIL với báo cáo chi tiết
- Open-source code để cộng đồng sử dụng

### 5.5. Bài học kinh nghiệm

> Chia sẻ những bài học về kỹ thuật (data quality, augmentation...) và quy trình (workflow, documentation...).

**5.5.1. Về kỹ thuật:**
- Data quality quan trọng hơn model complexity
- Data augmentation giúp model tổng quát hóa tốt hơn
- Early stopping tránh overfitting hiệu quả
- Module hóa code giúp dễ maintain và mở rộng

**5.5.2. Về quá trình thực hiện:**
- Nên bắt đầu với baseline đơn giản trước
- Theo dõi metrics liên tục trong quá trình training
- Thử nghiệm nhiều confidence threshold để chọn tối ưu
- Documentation ngay từ đầu giúp tiết kiệm thời gian

### 5.6. Lời kết

```
Đề tài "PCB Defect Detection System với YOLOv8" đã hoàn thành 
các mục tiêu đặt ra. Hệ thống có khả năng phát hiện 6 loại lỗi 
phổ biến trên mạch PCB, đánh giá mức độ nghiêm trọng, và tự động 
đưa ra kết quả kiểm tra chất lượng (QC) PASS/FAIL.

Đây là một bước tiến trong việc ứng dụng Deep Learning vào kiểm tra 
chất lượng PCB. Với những cải tiến trong tương lai, hệ thống có thể 
được triển khai trên dây chuyền sản xuất thực tế, góp phần nâng cao 
chất lượng sản phẩm và giảm chi phí kiểm tra.
```

---

## **VI. TÀI LIỆU THAM KHẢO**

> **Tóm tắt:** Liệt kê đầy đủ các papers, documentation, dataset sources, và online resources đã tham khảo trong quá trình thực hiện.

### Sắp xếp theo thứ tự ABC:

**Papers:**

[1] Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.

[2] Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. GitHub repository. https://github.com/ultralytics/ultralytics

[3] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016.

[4] Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015.

**Documentation:**

[5] Ultralytics YOLOv8 Documentation. https://docs.ultralytics.com/

[6] PyTorch Documentation. https://pytorch.org/docs/

[7] OpenCV Documentation. https://docs.opencv.org/

**Dataset:**

[8] PKU-Market-PCB Dataset. Open Lab on Human Robot Interaction, Peking University.

[9] PCB Defect Detection Dataset. Roboflow Universe. https://universe.roboflow.com/

**Online Resources:**

[10] Papers With Code - Object Detection. https://paperswithcode.com/task/object-detection

[10] Towards Data Science - YOLO Family. https://towardsdatascience.com/

---

## **PHỤ LỤC**

### Phụ lục A: Source Code chính

**A.1. DefectDetector class (defect_detector.py):**
```python
# Chèn code của DefectDetector class (hoặc link GitHub)
# Bao gồm: detect, analyze_defects, generate_report, visualize
```

**A.2. Training script (train_detector.py):**
```python
# Chèn code training script
```

### Phụ lục B: Cấu hình chi tiết

**B.1. data.yaml:**
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6
names: ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
```

**B.2. args.yaml (training arguments):**
```yaml
# Chèn nội dung file args.yaml từ runs/detect/.../
```

### Phụ lục C: Kết quả chi tiết

**C.1. Training logs:**
```
Epoch 1/100: loss=6.01, precision=0.788, recall=0.702
Epoch 10/100: loss=4.20, precision=0.888, recall=0.875
...
Epoch 100/100: loss=2.64, precision=0.936, recall=0.943
```

**C.2. results.csv đầy đủ:**
```
[Chèn file results.csv hoặc link]
```

### Phụ lục D: Hình ảnh minh họa

**D.1. Training samples:**
```
[Hình D.1] train_batch0.jpg
[Hình D.2] train_batch1.jpg
[Hình D.3] train_batch2.jpg
```

**D.2. Validation results:**
```
[Hình D.4] val_batch0_labels.jpg (Ground Truth)
[Hình D.5] val_batch0_pred.jpg (Predictions)
```

### Phụ lục E: Hướng dẫn sử dụng

**E.1. Installation:**
```bash
# Clone repository
git clone https://github.com/TrKhacQuang89/Final-Deep-Learning.git
cd Final-Deep-Learning

# Install dependencies
pip install -r requirements.txt
```

**E.2. Quick Start:**
```bash
# Training
python train_detector.py --model n --epochs 100

# Testing
python test_detector.py --weights best.pt --source test/images

# Webcam
python webcam_detector.py --weights best.pt
```

---

## 📌 TIPS QUAN TRỌNG KHI VIẾT BÁO CÁO

### ✅ Format chung:
- **Font:** Times New Roman, size 13 (nội dung), 14-16 (tiêu đề)
- **Line spacing:** 1.5
- **Margin:** Left 3cm, Right 2cm, Top/Bottom 2cm
- **Số trang:** Đánh số từ trang Giới thiệu

### ✅ Hình ảnh và Bảng:
- **Đánh số:** [Hình 2.1], [Bảng 3.2]
- **Caption:** Bên dưới hình, bên trên bảng
- **Chất lượng:** HD, không bị vỡ
- **Căn giữa:** Center align

### ✅ Trích dẫn:
- **Trong text:** [1], [2], [3]
- **Cuối câu:** ...như đã đề cập [5].
- **Nhiều nguồn:** ...theo các nghiên cứu [1, 3, 7].

### ✅ Ngôn ngữ:
- **Formal:** Không dùng ngôn ngữ thân mật
- **Khách quan:** "Kết quả cho thấy..." thay vì "Tôi thấy..."
- **Rõ ràng:** Tránh mơ hồ, dùng số liệu cụ thể

### ✅ Logic:
- Mỗi đoạn có 1 ý chính
- Có câu topic sentence mở đầu
- Liên kết các đoạn bằng từ nối (Tuy nhiên, Do đó, Ngoài ra...)

### ✅ Số liệu:
- **Chính xác:** 96.4% không phải ~96%
- **Đơn vị:** Ghi rõ (ms, FPS, MB, %)
- **So sánh:** Luôn có baseline hoặc reference

---

## 🎯 CHECKLIST HOÀN THÀNH BÁO CÁO

### Trước khi nộp, kiểm tra:

- [ ] Trang bìa đầy đủ thông tin
- [ ] Mục lục có đánh số trang đúng
- [ ] Tất cả hình ảnh có caption và đánh số
- [ ] Tất cả bảng có tiêu đề và đánh số
- [ ] Tài liệu tham khảo đầy đủ và đúng format
- [ ] Không có lỗi chính tả
- [ ] Số liệu khớp với kết quả thực tế
- [ ] Code trong phụ lục chạy được
- [ ] File PDF không bị lỗi font
- [ ] Kích thước file hợp lý (< 50MB)

---

**Chúc bạn hoàn thành báo cáo xuất sắc! 🎓**
