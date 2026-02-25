# PCB Defect Detection Project - Final Project

## Giới thiệu

Dự án này xây dựng hệ thống **phát hiện và khoanh vùng lỗi trên mạch PCB** (Printed Circuit Board) sử dụng YOLOv8. Hệ thống có khả năng phát hiện 6 loại lỗi phổ biến trên PCB, đánh giá mức độ nghiêm trọng, và đưa ra kết quả kiểm tra chất lượng (QC) tự động.

### Mục tiêu

**PCB Defect Detection (Object Detection):**
- Input: Ảnh mạch PCB có thể chứa nhiều lỗi
- Output: Bounding boxes + loại lỗi + mức độ nghiêm trọng + kết quả QC (PASS/FAIL)
- Model: YOLOv8 (Ultralytics)
- Bonus: Real-time detection từ webcam

### Kiến trúc YOLOv8
```
Input Image → Backbone (CSPDarknet) → Neck (PANet) → Detection Head → Output (BBoxes + Classes)
```

## Các loại lỗi PCB phát hiện

| # | Loại lỗi | Mô tả | Mức độ |
|---|----------|--------|--------|
| 1 | **missing_hole** | Lỗ khoan bị thiếu trên PCB | 🔴 HIGH |
| 2 | **mouse_bite** | Vết cắn chuột - khuyết tật ở cạnh mạch | 🟡 MEDIUM |
| 3 | **open_circuit** | Mạch hở - đường mạch bị đứt | 🔴 CRITICAL |
| 4 | **short** | Ngắn mạch - 2 đường mạch bị nối nhầm | 🔴 CRITICAL |
| 5 | **spur** | Gai đồng thừa nhô ra từ đường mạch | 🟡 MEDIUM |
| 6 | **spurious_copper** | Đồng thừa không mong muốn trên PCB | 🟢 LOW |

### Phân loại mức độ nghiêm trọng

- **CRITICAL**: `open_circuit`, `short` - Lỗi gây hỏng mạch hoàn toàn, cần loại bỏ ngay
- **HIGH**: `missing_hole` - Lỗi ảnh hưởng đến lắp ráp linh kiện
- **MEDIUM**: `mouse_bite`, `spur` - Lỗi có thể ảnh hưởng đến chất lượng
- **LOW**: `spurious_copper` - Lỗi nhẹ, có thể chấp nhận trong một số trường hợp

## Dataset

Dataset PCB Defect Detection gồm 6 classes lỗi, được annotate theo format YOLOv8.

**Cấu trúc:**
```
Final-Deep-Learning-main/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** Nếu có GPU NVIDIA, cài đặt PyTorch với CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Sử dụng

### 1. Training

Train model phát hiện lỗi PCB với YOLOv8 nano (nhanh nhất):

```bash
python train_detector.py --model n --epochs 100 --batch 16
```

Train với YOLOv8 small (cân bằng tốc độ/độ chính xác):

```bash
python train_detector.py --model s --epochs 100 --batch 16
```

Train với YOLOv8 medium (độ chính xác cao hơn):

```bash
python train_detector.py --model m --epochs 150 --batch 8
```

**Các tham số quan trọng:**
- `--model`: Kích thước model (n/s/m/l/x)
- `--epochs`: Số epochs
- `--batch`: Batch size
- `--imgsz`: Kích thước ảnh input (default: 640)
- `--device`: Device (0 cho GPU, cpu cho CPU)
- `--patience`: Early stopping patience
- `--lr0`: Learning rate ban đầu

**Ví dụ training đầy đủ:**

```bash
python train_detector.py \
    --model s \
    --epochs 150 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --patience 50 \
    --lr0 0.01 \
    --save-period 10
```

### 2. Testing

Test model trên test set:

```bash
python test_detector.py \
    --weights runs/detect/pcb_defect_detector/weights/best.pt \
    --source test/images \
    --conf 0.25 \
    --save
```

Test trên một ảnh cụ thể với visualization:

```bash
python test_detector.py \
    --weights runs/detect/pcb_defect_detector/weights/best.pt \
    --source test/images/sample.jpg \
    --conf 0.25 \
    --visualize
```

Tạo báo cáo QC cho toàn bộ test set:

```bash
python test_detector.py \
    --weights runs/detect/pcb_defect_detector/weights/best.pt \
    --source test/images \
    --conf 0.25 \
    --report \
    --save
```

### 3. Real-time Webcam Detection

Chạy phát hiện lỗi PCB real-time từ webcam:

```bash
python webcam_detector.py \
    --weights runs/detect/pcb_defect_detector/weights/best.pt \
    --camera 0 \
    --conf 0.25
```

**Controls trong webcam mode:**
- `q`: Thoát
- `s`: Lưu frame hiện tại
- `p`: Pause/Resume
- `+`: Tăng confidence threshold
- `-`: Giảm confidence threshold

### 4. Sử dụng trong Python Code

```python
from defect_detector import DefectDetector, WebcamDefectDetector

# Training
detector = DefectDetector(model_type='n', pretrained=True)
detector.train(
    data_yaml='data.yaml',
    epochs=100,
    batch=16,
    device='0'
)

# Inference - Phát hiện lỗi
detector.load_weights('runs/detect/pcb_defect_detector/weights/best.pt')
results = detector.predict('test/images/sample.jpg', conf=0.25)

# Phân tích chi tiết lỗi
analysis = detector.analyze_defects('test/images/sample.jpg', conf=0.25)
print(f"Total defects: {analysis['total_defects']}")
print(f"QC Result: {'PASS' if analysis['is_pass'] else 'FAIL'}")
print(f"Defects: {analysis['defect_counts']}")

# Tạo báo cáo QC cho nhiều ảnh
df = detector.generate_report('test/images/', conf=0.25, save_path='qc_report.csv')

# Webcam
webcam = WebcamDefectDetector(
    model_path='runs/detect/pcb_defect_detector/weights/best.pt',
    conf_threshold=0.25
)
webcam.run(camera_id=0)
```

## Cấu trúc Project

```
Final-Deep-Learning-main/
├── defect_detector.py          # Core defect detection module
├── train_detector.py           # Training script
├── test_detector.py            # Testing script
├── webcam_detector.py          # Real-time webcam detection
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── data.yaml                   # Dataset configuration (6 defect classes)
├── train/                      # Training data
├── valid/                      # Validation data
├── test/                       # Test data
└── runs/                       # Training results (auto-generated)
    └── detect/
        └── pcb_defect_detector/
            ├── weights/
            │   ├── best.pt     # Best model weights
            │   └── last.pt     # Last epoch weights
            ├── results.csv     # Training metrics
            └── *.png           # Training plots
```

## Tính năng nổi bật

### 1. Phát hiện 6 loại lỗi PCB
Sử dụng YOLOv8 để detect và phân loại 6 loại khuyết tật phổ biến nhất trên mạch PCB.

### 2. Đánh giá mức độ nghiêm trọng
Mỗi lỗi được gán mức độ nghiêm trọng (CRITICAL/HIGH/MEDIUM/LOW) giúp ưu tiên xử lý.

### 3. Kiểm tra chất lượng tự động (QC)
- **PASS**: Không phát hiện lỗi → PCB đạt tiêu chuẩn
- **FAIL**: Phát hiện lỗi → PCB cần kiểm tra lại

### 4. Tạo báo cáo QC
Tự động tạo báo cáo CSV chi tiết cho batch ảnh PCB, bao gồm:
- Số lượng lỗi theo từng loại
- Mức độ nghiêm trọng
- Kết quả PASS/FAIL

### 5. Real-time Detection
Phát hiện lỗi PCB real-time qua webcam với hiển thị:
- FPS
- Số lượng lỗi
- Trạng thái QC (PASS/FAIL)
- Mức độ nghiêm trọng

## Kết quả Training

Sau khi training, kết quả sẽ được lưu trong `runs/detect/pcb_defect_detector/`:

- **weights/best.pt**: Model tốt nhất (theo validation mAP)
- **weights/last.pt**: Model ở epoch cuối cùng
- **results.csv**: Metrics theo từng epoch
- **confusion_matrix.png**: Confusion matrix
- **results.png**: Training curves (loss, mAP, precision, recall)
- **training_analysis.png**: Phân tích chi tiết (custom plot)

## Kết quả Test (Evaluation trên Test Set)

> **Model:** YOLOv8s (small) — 11.1M parameters, 28.4 GFLOPs  
> **Test set:** 70 ảnh, 301 instances lỗi  
> **Confidence threshold:** 0.25 | **IoU threshold:** 0.45 | **Image size:** 640×640  
> **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU  
> **Tốc độ:** 1.8ms tiền xử lý, 14.3ms suy luận, 2.8ms hậu xử lý/ảnh  

### Kết quả tổng quan

| Metric | Giá trị |
|--------|---------|
| **Precision** | **94.3%** |
| **Recall** | **90.1%** |
| **mAP@0.5** | **93.2%** |
| **mAP@0.5:0.95** | **51.1%** |

### Kết quả chi tiết theo từng loại lỗi

| Loại lỗi | Instances | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|-----------|-----------|-----------|--------|--------|-------------|
| **missing_hole** | 75 | 98.3% | 98.7% | 98.3% | 63.0% |
| **mouse_bite** | 52 | 88.6% | 89.9% | 91.2% | 47.1% |
| **open_circuit** | 37 | 97.0% | 88.8% | 96.2% | 55.9% |
| **short** | 39 | 97.4% | 97.0% | 98.5% | 54.5% |
| **spur** | 30 | 90.6% | 76.7% | 82.2% | 37.2% |
| **spurious_copper** | 68 | 93.7% | 89.7% | 93.0% | 48.7% |

### Giải thích các chỉ số đánh giá

#### 1. **Precision (Độ chính xác) — 94.3%**
Precision đo tỉ lệ các dự đoán đúng trong tổng số dự đoán mà model đưa ra. Nói cách khác, khi model nói "đây là lỗi", thì **94.3% trường hợp là đúng**. Precision cao nghĩa là model ít đưa ra cảnh báo sai (false positive).

> **Công thức:** `Precision = TP / (TP + FP)`  
> Trong đó: TP = True Positive (dự đoán đúng), FP = False Positive (dự đoán sai — báo lỗi nhưng thực tế không có lỗi)

#### 2. **Recall (Độ phủ) — 90.1%**
Recall đo tỉ lệ các lỗi thực tế mà model phát hiện được. Với Recall 90.1%, model phát hiện được **90.1% tổng số lỗi** có trong ảnh. Recall cao nghĩa là model ít bỏ sót lỗi (false negative).

> **Công thức:** `Recall = TP / (TP + FN)`  
> Trong đó: FN = False Negative (bỏ sót — có lỗi nhưng model không phát hiện)

#### 3. **mAP@0.5 (Mean Average Precision tại IoU 0.5) — 93.2%**
Đây là chỉ số quan trọng nhất trong object detection. mAP@0.5 đánh giá khả năng phát hiện lỗi khi yêu cầu bounding box dự đoán trùng ít nhất **50%** với bounding box thực tế (IoU ≥ 0.5). Giá trị này là **trung bình AP của tất cả 6 loại lỗi**.

> **IoU (Intersection over Union):** Tỉ lệ diện tích giao nhau giữa bounding box dự đoán và ground truth.

#### 4. **mAP@0.5:0.95 (Mean Average Precision trung bình) — 51.1%**
Chỉ số này **khắt khe hơn** mAP@0.5 rất nhiều. Nó tính trung bình AP tại các ngưỡng IoU từ 0.5 đến 0.95 (bước nhảy 0.05). Nghĩa là model phải khoanh vùng lỗi **rất chính xác** (trùng tới 95% diện tích) mới được tính đúng ở các ngưỡng cao. Đây là metric chuẩn của cuộc thi COCO.

### Đánh giá chất lượng model

#### ✅ Đánh giá tổng quan: **TỐT — Đạt yêu cầu ứng dụng thực tế**

| Chỉ số | Giá trị | Đánh giá |
|--------|---------|----------|
| Precision 94.3% | 🟢 **Rất tốt** | Model rất ít đưa ra cảnh báo sai, đáng tin cậy |
| Recall 90.1% | 🟢 **Tốt** | Phát hiện được hầu hết các lỗi, chỉ bỏ sót ~10% |
| mAP@0.5 93.2% | 🟢 **Rất tốt** | Khả năng phát hiện + định vị lỗi rất chính xác |
| mAP@0.5:0.95 51.1% | 🟡 **Trung bình** | Bounding box chưa thật sự khít với lỗi ở ngưỡng cao |

#### Phân tích chi tiết:

1. **Các lỗi phát hiện tốt nhất:**
   - `missing_hole` (AP@0.5: 98.3%) và `short` (AP@0.5: 98.5%): Gần như phát hiện hoàn hảo. Đây là các lỗi có hình dạng rõ ràng, dễ nhận diện.

2. **Lỗi cần cải thiện:**
   - `spur` (AP@0.5: 82.2%, Recall: 76.7%): Đây là loại lỗi khó nhất vì gai đồng thường rất nhỏ, dễ bị bỏ sót (~23% bị miss). Cần thêm data hoặc augmentation cho loại lỗi này.

3. **So sánh với tiêu chuẩn ngành:**
   - mAP@0.5 > 90% được coi là **rất tốt** cho bài toán object detection trong công nghiệp.
   - mAP@0.5:0.95 ở mức 51% là **bình thường** — chỉ số này luôn thấp hơn nhiều so với mAP@0.5 do yêu cầu khắt khe.
   - Precision > 94% đảm bảo hệ thống **không gây nhiều phiền toái** bằng cảnh báo sai trong sản xuất.

4. **Kết luận:**
   - Model **đủ tốt** để triển khai vào hệ thống kiểm tra chất lượng PCB tự động.
   - Tốc độ inference ~14.3ms/ảnh (~70 FPS) cho phép ứng dụng **real-time** qua webcam.
   - Để cải thiện thêm, có thể: tăng data cho `spur`, sử dụng model lớn hơn (YOLOv8m/l), hoặc fine-tune augmentation.

## Giải thích Metrics

- **mAP@0.5**: Mean Average Precision tại IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision trung bình từ IoU 0.5 đến 0.95
- **Precision**: Tỉ lệ detections đúng trong tất cả detections
- **Recall**: Tỉ lệ lỗi được detect trong tất cả ground truth
- **Box Loss**: Loss cho bounding box regression
- **Class Loss**: Loss cho classification
- **DFL Loss**: Distribution Focal Loss

## So sánh Model Sizes

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| YOLOv8n | 3.2M params | Fastest | Lowest | Real-time, embedded |
| YOLOv8s | 11.2M params | Fast | Medium | Balanced |
| YOLOv8m | 25.9M params | Medium | High | Accuracy priority |
| YOLOv8l | 43.7M params | Slow | Higher | High accuracy |
| YOLOv8x | 68.2M params | Slowest | Highest | Best accuracy |

**Khuyến nghị:**
- **Real-time webcam**: YOLOv8n hoặc YOLOv8s
- **Cân bằng**: YOLOv8s hoặc YOLOv8m
- **Độ chính xác cao**: YOLOv8m hoặc YOLOv8l

## Troubleshooting

### 1. CUDA Out of Memory

Giảm batch size:
```bash
python train_detector.py --model n --batch 8
```

Hoặc giảm image size:
```bash
python train_detector.py --model n --imgsz 416
```

### 2. Webcam không hoạt động

Thử camera ID khác:
```bash
python webcam_detector.py --weights best.pt --camera 1
```

Kiểm tra OpenCV:
```python
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())
```

### 3. Training quá chậm

- Sử dụng GPU: `--device 0`
- Giảm workers: `--workers 4`
- Sử dụng model nhỏ hơn: `--model n`

## Tips để cải thiện Performance

1. **Data Augmentation**: Điều chỉnh augmentation parameters
   ```bash
   python train_detector.py --model s --mosaic 1.0 --mixup 0.1 --fliplr 0.5
   ```

2. **Learning Rate**: Thử learning rate khác
   ```bash
   python train_detector.py --model s --lr0 0.001 --lrf 0.01
   ```

3. **Image Size**: Tăng image size (nếu có GPU mạnh)
   ```bash
   python train_detector.py --model s --imgsz 800
   ```

4. **Epochs**: Train lâu hơn với early stopping
   ```bash
   python train_detector.py --model s --epochs 300 --patience 100
   ```

## Tài liệu tham khảo

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [PCB Defect Detection Dataset](https://universe.roboflow.com/) (Roboflow)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [PKU-Market-PCB Dataset](https://robotics.pkusz.edu.cn/resources/dataset/)

## License

Dataset: CC BY 4.0
Code: MIT License

## Tác giả

Dự án cuối kỳ - Phát hiện và khoanh vùng lỗi trên mạch PCB sử dụng YOLOv8
