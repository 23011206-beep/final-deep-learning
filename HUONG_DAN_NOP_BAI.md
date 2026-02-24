# HƯỚNG DẪN NỘP BÀI - PCB DEFECT DETECTION PROJECT

## 📦 DANH SÁCH FILES CẦN NỘP

### ✅ 1. FILES CODE BẮT BUỘC (6 files)

#### **File Python chính:**
1. **`defect_detector.py`** (~25KB)
   - Module core chứa class DefectDetector và WebcamDefectDetector
   - **MỤC ĐÍCH:** Chứa toàn bộ logic phát hiện lỗi PCB, phân tích mức độ nghiêm trọng, tạo báo cáo QC

2. **`train_detector.py`** (~8KB)
   - Script huấn luyện model từ command line
   - **MỤC ĐÍCH:** Cho phép train model phát hiện lỗi PCB với các tham số tùy chỉnh

3. **`test_detector.py`** (~6KB)
   - Script kiểm thử model trên test set
   - **MỤC ĐÍCH:** Đánh giá độ chính xác của model, tạo báo cáo QC

4. **`webcam_detector.py`** (~2KB)
   - Script chạy real-time detection từ webcam
   - **MỤC ĐÍCH:** Demo phát hiện lỗi PCB real-time

#### **File cấu hình:**
5. **`data.yaml`** (~400 bytes)
   - Cấu hình dataset (đường dẫn, 6 classes lỗi PCB)
   - **MỤC ĐÍCH:** YOLO cần file này để biết dataset ở đâu

6. **`requirements.txt`** (~1.6KB)
   - Danh sách thư viện cần cài đặt
   - **MỤC ĐÍCH:** Giúp thầy cài đặt dependencies dễ dàng

### ⚠️ 2. FILES TÀI LIỆU (2 files - KHUYẾN NGHỊ)

7. **`README.md`** (~10KB)
   - Hướng dẫn sử dụng chi tiết
   - **MỤC ĐÍCH:** Giúp thầy hiểu và chạy được dự án

8. **`HUONG_DAN_NOP_BAI.md`** (file này)
   - Hướng dẫn chi tiết cho thầy giáo

### 📊 3. DỮ LIỆU (Folders - BẮT BUỘC)

**Cấu trúc thư mục dataset:**
```
Final-Deep-Learning-main/
├── train/          (Thư mục chứa ảnh training + labels)
│   ├── images/
│   └── labels/
├── valid/          (Thư mục chứa ảnh validation + labels)
│   ├── images/
│   └── labels/
└── test/           (Thư mục chứa ảnh test + labels)
    ├── images/
    └── labels/
```

**LƯU Ý:** Ba thư mục này là DATASET, bắt buộc phải có để train và test.

### 🏆 4. MODEL ĐÃ TRAIN (Optional - nhưng NÊN NỘP)

**Nếu muốn demo luôn mà không cần train lại:**

```
runs/detect/pcb_defect_detector/
├── weights/
│   └── best.pt                    (File model đã train)
├── results.csv                    (Kết quả training theo epoch)
├── confusion_matrix.png           (Ma trận nhầm lẫn)
├── results.png                    (Biểu đồ training)
└── [các file khác...]
```

---

## 📋 CẤU TRÚC THỦ MỤC ĐẦY ĐỦ ĐỂ NỘP

```
Final-Deep-Learning-main/                   👈 Thư mục gốc (nén thành ZIP để nộp)
│
├── 📄 FILES CODE
│   ├── defect_detector.py                  ✅ BẮT BUỘC
│   ├── train_detector.py                   ✅ BẮT BUỘC
│   ├── test_detector.py                    ✅ BẮT BUỘC
│   ├── webcam_detector.py                  ✅ BẮT BUỘC
│   ├── requirements.txt                    ✅ BẮT BUỘC
│   └── data.yaml                           ✅ BẮT BUỘC
│
├── 📖 TÀI LIỆU
│   ├── README.md                           ⚠️ KHUYẾN NGHỊ
│   └── HUONG_DAN_NOP_BAI.md                ⚠️ File này
│
├── 📊 DATASET
│   ├── train/                              ✅ BẮT BUỘC (images + labels)
│   ├── valid/                              ✅ BẮT BUỘC (images + labels)
│   └── test/                               ✅ BẮT BUỘC (images + labels)
│
└── 🏆 KẾT QUẢ TRAINING (Optional)
    └── runs/detect/pcb_defect_detector/
        ├── weights/
        │   └── best.pt                     ⚠️ Model đã train
        ├── results.csv                     ⚠️ Kết quả training
        ├── confusion_matrix.png            ⚠️ Confusion matrix
        └── results.png                     ⚠️ Training curves
```

---

## 🚀 HƯỚNG DẪN CHO THẦY GIÁO CHẠY DỰ ÁN

### Bước 1: Cài đặt môi trường (Lần đầu tiên)

```powershell
# Di chuyển vào thư mục dự án
cd Final-Deep-Learning-main

# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv

# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

**LƯU Ý:** Nếu có GPU NVIDIA và muốn train nhanh hơn:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Bước 2: OPTION A - Sử dụng model đã train (NHANH)

**Nếu em nộp kèm file `best.pt` trong folder `runs/`, thầy có thể chạy luôn:**

#### 2A.1. Test model trên test set
```powershell
python test_detector.py --weights runs/detect/pcb_defect_detector/weights/best.pt --source test/images --save --project runs/detect --name demo_test
```

**Kết quả:** Ảnh với bounding boxes khoanh vùng lỗi sẽ được lưu trong `runs/detect/demo_test/`

#### 2A.2. Tạo báo cáo QC
```powershell
python test_detector.py --weights runs/detect/pcb_defect_detector/weights/best.pt --source test/images --report
```

**Kết quả:** File `qc_report.csv` chứa kết quả PASS/FAIL cho từng ảnh PCB

#### 2A.3. Chạy Real-time Webcam Detection (DEMO TRỰC QUAN)
```powershell
python webcam_detector.py --weights runs/detect/pcb_defect_detector/weights/best.pt --conf 0.5
```

**Thao tác trong webcam:**
- Nhấn `q` để thoát
- Nhấn `s` để lưu ảnh frame hiện tại
- Nhấn `+` hoặc `-` để điều chỉnh confidence threshold

---

### Bước 3: OPTION B - Train lại từ đầu (MẤT THỜI GIAN)

#### 3.1. Training
```powershell
python train_detector.py --model n --epochs 100 --batch 16 --project runs/detect --name my_training
```

#### 3.2. Testing sau khi train
```powershell
python test_detector.py --weights runs/detect/my_training/weights/best.pt --source test/images --save
```

#### 3.3. Webcam Detection
```powershell
python webcam_detector.py --weights runs/detect/my_training/weights/best.pt
```

---

## 📊 ĐÁNH GIÁ KẾT QUẢ MODEL

### Metrics quan trọng

| Metric | Ý nghĩa |
|--------|---------|
| **Precision** | Khi model báo "phát hiện lỗi", thì bao nhiêu % là đúng |
| **Recall** | Model tìm được bao nhiêu % tổng số lỗi có trong ảnh |
| **mAP@0.5** | Độ chính xác trung bình (ngưỡng IoU=0.5) |
| **mAP@0.5:0.95** | Độ chính xác trung bình (ngưỡng khắt khe) |

### Xem kết quả training chi tiết

1. **File `results.csv`**: Chứa metrics theo từng epoch
2. **File `confusion_matrix.png`**: Ma trận nhầm lẫn giữa các loại lỗi
3. **File `results.png`**: Biểu đồ Loss và Metrics qua các epochs

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. File paths trong `data.yaml`
File `data.yaml` hiện tại dùng đường dẫn tương đối:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

### 2. Dependencies
**Thư viện quan trọng nhất:**
- `ultralytics==8.4.14` (YOLOv8)
- `torch` (PyTorch - tự động cài kèm ultralytics)
- `opencv-python` (xử lý webcam)

**Nếu thầy gặp lỗi cài đặt:**
```powershell
pip install ultralytics opencv-python matplotlib numpy pandas pyyaml
```

### 3. GPU vs CPU
- **Có GPU:** Training ~2-3 giờ (100 epochs)
- **Không GPU:** Training ~8-12 giờ (hoặc hơn)

**Để train trên CPU:**
```powershell
python train_detector.py --model n --epochs 100 --batch 8 --device cpu
```

### 4. Webcam
- Cần có webcam để chạy `webcam_detector.py`
- Nếu không có webcam, có thể bỏ qua phần này
- Thay vào đó test trên ảnh tĩnh với `test_detector.py`

---

## 🎓 TÓM TẮT

**Em đã làm gì:**
1. ✅ Xây dựng hệ thống phát hiện và khoanh vùng lỗi trên mạch PCB
2. ✅ Sử dụng YOLOv8 để detect 6 loại lỗi PCB
3. ✅ Phân loại mức độ nghiêm trọng (CRITICAL/HIGH/MEDIUM/LOW)
4. ✅ Tự động đánh giá QC (PASS/FAIL)
5. ✅ Tạo báo cáo QC chi tiết dạng CSV
6. ✅ Xây dựng Real-time Webcam Detection
7. ✅ Viết đầy đủ documentation và testing scripts

**Thầy có thể:**
1. ✅ Cài đặt dependencies bằng 1 lệnh
2. ✅ Train model bằng 1 lệnh
3. ✅ Test model bằng 1 lệnh
4. ✅ Tạo báo cáo QC bằng 1 lệnh
5. ✅ Chạy webcam detection bằng 1 lệnh
6. ✅ Đọc tài liệu đầy đủ trong README.md

---

**Ngày tạo:** 2026-02-22
**Dự án:** PCB Defect Detection - Final Project
**Dataset:** 6 loại lỗi PCB (missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper)
**Model:** YOLOv8 (Ultralytics)
