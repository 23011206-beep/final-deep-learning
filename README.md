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

## Metrics Đánh giá

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
