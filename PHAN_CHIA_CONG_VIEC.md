# 📋 PHÂN CHIA CÔNG VIỆC LÀM BÁO CÁO - NHÓM 3 NGƯỜI
# PCB Defect Detection System với YOLOv8

---

## 👥 TỔNG QUAN PHÂN CÔNG

| Thành viên | Vai trò chính | Phần báo cáo phụ trách |
|------------|---------------|------------------------|
| **Thành viên A** | Training & Kết quả thực nghiệm | Phần II + Phần IV |
| **Thành viên B** | Thiết kế & Triển khai hệ thống | Phần III (⭐ Phần chính) |
| **Thành viên C** | Giới thiệu, Đánh giá & Tổng hợp | Phần I + Phần V + Phần VI + PHỤ LỤC |

> **Lưu ý:** Phần **Trang bìa + Mục lục** và **Phần VII (Tài liệu tham khảo)** cả 3 người cùng làm.

---

## 🔵 THÀNH VIÊN A — Training & Kết quả thực nghiệm

### 📌 Phần phụ trách trong báo cáo:

#### **Phần II. TỔNG QUAN YOLOv8 VÀ DATASET** (~2 trang)

| Mục | Nội dung cần viết | File/Nguồn tham khảo |
|-----|-------------------|----------------------|
| 2.1. Giới thiệu YOLOv8 | Mô tả ngắn gọn YOLOv8, tại sao chọn model này | Ultralytics docs, papers |
| 2.2.1. Nguồn và thống kê dataset | Bảng thông tin dataset (số ảnh, chia train/val/test) | `data.yaml`, thư mục `train/`, `valid/`, `test/` |
| 2.2.2. 6 loại lỗi PCB | Bảng mô tả 6 classes + mức độ nghiêm trọng | `defect_detector.py` (dòng 49-68) |
| 2.2.3. Phân loại mức độ nghiêm trọng | CRITICAL/HIGH/MEDIUM/LOW | `defect_detector.py` (DEFECT_SEVERITY) |
| 2.2.4. Chất lượng dataset | Đánh giá labels, balance, quality | Kiểm tra dataset thực tế |

#### **Phần IV. KẾT QUẢ THỰC NGHIỆM** (~3-4 trang)

| Mục | Nội dung cần viết | File/Nguồn tham khảo |
|-----|-------------------|----------------------|
| 4.1. Kết quả Training | Training curves, loss giảm thế nào | `runs/detect/.../results.png`, `results.csv` |
| 4.1.2. Metrics evolution | Bảng Precision/Recall/mAP qua các epochs | `runs/detect/.../results.csv` |
| 4.2.1. Confusion Matrix | Phân tích confusion matrix | `runs/detect/.../confusion_matrix_normalized.png` |
| 4.2.2. PR Curve | Nhận xét đường PR | `runs/detect/.../BoxPR_curve.png` |
| 4.2.3. F1-Confidence Curve | Phân tích F1 theo confidence | `runs/detect/.../BoxF1_curve.png` |
| 4.3. Kết quả Test | Metrics trên test set | Chạy `test_detector.py` |
| 4.4. Visualization | Ví dụ ảnh detection, label distribution | `runs/detect/.../val_batch0_pred.jpg`, `labels.jpg` |
| 4.5. Real-time Performance | FPS, latency trên GPU/CPU | Chạy `webcam_detector.py` đo thực tế |
| 4.6. So sánh Model | Bảng so sánh YOLOv8 n/s/m | Tra cứu thêm + kết quả thực tế |

### ✅ Checklist cho Thành viên A:
- [ ] Đếm chính xác số ảnh train/val/test trong thư mục
- [ ] Chụp/export hình `results.png`, confusion matrix, PR curve, F1 curve
- [ ] Điền đầy đủ bảng metrics (KHÔNG dùng `[value]`, phải là số thực)
- [ ] Chạy test trên test set và ghi lại kết quả
- [ ] Đo FPS thực tế trên máy của nhóm
- [ ] Viết nhận xét phân tích cho mỗi hình/bảng
- [ ] Tổng cộng: **~5-6 trang**

---

## 🟢 THÀNH VIÊN B — Thiết kế & Triển khai hệ thống

### 📌 Phần phụ trách trong báo cáo:

#### **Phần III. THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG** (~5-6 trang) ⭐ PHẦN QUAN TRỌNG NHẤT

| Mục | Nội dung cần viết | File/Nguồn tham khảo |
|-----|-------------------|----------------------|
| 3.1.1. Sơ đồ tổng quát | Vẽ kiến trúc hệ thống (Dataset → Training → Model → Testing/Deployment) | Tổng hợp từ cấu trúc project |
| 3.1.2. Stack công nghệ | Bảng Technology Stack | `requirements.txt` |
| 3.2.1. Cấu trúc module | Sơ đồ cây code, giải thích role từng file | Cấu trúc thư mục project |
| 3.2.2. Design Principles | Modularity, Reusability, User-friendly, Flexibility | Phân tích code `defect_detector.py` |
| 3.2.3. Chi tiết DefectDetector | Giải thích class chính, tại sao wrap YOLOv8 API | `defect_detector.py` |
| 3.3.1. Training Script | Features nhóm thêm vào (CLI, device handling, auto validate) | `train_detector.py` |
| 3.3.2. Testing Script | Batch testing, visualization, metrics reporting | `test_detector.py` |
| 3.3.3. Webcam Script | Real-time features, interactive controls, visual enhancements | `webcam_detector.py` |
| 3.4. Documentation & Code Quality | Docstrings, README, comments | `README.md`, docstrings trong code |
| 3.5. Testing & Debugging | Development workflow, tools sử dụng | Mô tả quy trình làm việc thực tế |

### 📂 Các file cần đọc kỹ:

```
defect_detector.py          ← File chính (1191 dòng) - Đọc kỹ class DefectDetector
                               + class WebcamDefectDetector + class TrackedDetection
train_detector.py           ← Training script (đọc CLI args, features)
test_detector.py            ← Testing script (đọc features)
webcam_detector.py          ← Webcam script (đọc controls, features)
collect_webcam_data.py      ← Data collection script
simulate_webcam.py          ← Webcam simulation script
convert_dataset.py          ← Dataset conversion
download_dataset.py         ← Dataset download
```

### ✅ Checklist cho Thành viên B:
- [ ] Vẽ sơ đồ kiến trúc hệ thống (dùng draw.io hoặc Word shapes)
- [ ] Vẽ sơ đồ cấu trúc code (cây thư mục)
- [ ] Giải thích rõ DefectDetector class: các method, input/output
- [ ] Liệt kê features từng script (training, testing, webcam)
- [ ] Tạo bảng Challenges & Solutions
- [ ] Chụp ảnh minh họa code (nếu cần chèn vào báo cáo)
- [ ] Tổng cộng: **~5-6 trang**

---

## 🟠 THÀNH VIÊN C — Giới thiệu, Đánh giá & Tổng hợp

### 📌 Phần phụ trách trong báo cáo:

#### **Phần I. GIỚI THIỆU VÀ MỤC TIÊU** (~2 trang)

| Mục | Nội dung cần viết | Ghi chú |
|-----|-------------------|---------|
| 1.1. Đặt vấn đề | Tại sao cần phát hiện lỗi PCB tự động | Viết theo góc độ thực tế sản xuất |
| 1.2. Mục tiêu | 3 nhóm mục tiêu: kỹ thuật, triển khai, học tập | Liệt kê rõ ràng |
| 1.3. Phạm vi | Công cụ, dataset, ngôn ngữ, sản phẩm | Dựa trên project thực tế |
| 1.4. Phân công nhóm | Bảng phân công 3 người | Dựa trên file này |
| 1.5. Bố cục báo cáo | Tóm tắt nội dung từng phần | Viết sau khi các phần khác xong |

#### **Phần V. ĐÁNH GIÁ VÀ KẾT LUẬN** (~2-3 trang)

| Mục | Nội dung cần viết | Ghi chú |
|-----|-------------------|---------|
| 5.1. Đánh giá chung | Ưu điểm + Nhược điểm | Dựa trên kết quả Phần IV |
| 5.2. Kết luận | Tổng kết, kiến thức thu được, sản phẩm | Tổng hợp toàn bộ |
| 5.3. Hướng phát triển | Cải thiện model, mở rộng chức năng, UX | Đề xuất thực tế |
| 5.4. Đóng góp đề tài | Đóng góp khoa học + thực tiễn | Nhấn mạnh giá trị |
| 5.5. Bài học kinh nghiệm | Kỹ thuật + quy trình | Chia sẻ thực tế |
| 5.6. Lời kết | Kết luận cuối cùng | ~1 đoạn |

#### **Phần VI. TÀI LIỆU THAM KHẢO**

| Nội dung | Ghi chú |
|----------|---------|
| Papers (YOLO, Faster R-CNN...) | Tham khảo file hướng dẫn |
| Documentation (Ultralytics, PyTorch, OpenCV) | Links chính thức |
| Dataset sources (Roboflow, PKU) | Nguồn dataset |
| Online Resources | Các trang web tham khảo |

#### **PHỤ LỤC**

| Mục | Nội dung |
|-----|----------|
| Phụ lục A | Source code chính (copy từ `defect_detector.py`) |
| Phụ lục B | data.yaml + args.yaml |
| Phụ lục C | Training logs, results.csv |
| Phụ lục D | Hình minh họa (train_batch, val_batch) |
| Phụ lục E | Hướng dẫn sử dụng (installation, quick start) |

### ✅ Checklist cho Thành viên C:
- [ ] Viết phần đặt vấn đề hấp dẫn, thuyết phục
- [ ] Liệt kê mục tiêu rõ ràng, đo lường được
- [ ] Tạo bảng phân công công việc nhóm
- [ ] Viết đánh giá dựa trên SỐ LIỆU THỰC (từ Thành viên A)
- [ ] Đề xuất hướng phát triển hợp lý
- [ ] Thu thập tài liệu tham khảo đầy đủ (≥10 references)
- [ ] Làm phụ lục đầy đủ
- [ ] **Tổng hợp + format toàn bộ báo cáo cuối cùng**
- [ ] Tổng cộng: **~6-8 trang** (bao gồm phụ lục)

---

## 📊 BẢNG TỔNG HỢP KHỐI LƯỢNG CÔNG VIỆC

| Thành viên | Phần báo cáo | Số trang ước tính | Deadline gợi ý |
|------------|-------------|-------------------|-----------------|
| **A** | II (2 trang) + IV (3-4 trang) | **5-6 trang** | Hoàn thành trước B, C 2 ngày |
| **B** | III (5-6 trang) | **5-6 trang** | Cùng deadline với A |
| **C** | I (2 trang) + V (2-3 trang) + VI + Phụ lục | **6-8 trang** | Sau A, B 1-2 ngày (cần kết quả) |

### ⏰ Timeline gợi ý:

```
Ngày 1-2:  Cả 3 người đọc kỹ file HUONG_DAN_LAM_BAO_CAO.md
           + Đọc code để hiểu project

Ngày 3-5:  Thành viên A: Chạy test, thu thập kết quả, chụp hình
           Thành viên B: Đọc code, vẽ sơ đồ, viết mô tả
           Thành viên C: Viết phần Giới thiệu (I)

Ngày 6-8:  Thành viên A: Viết phần II + IV (điền số liệu thực)
           Thành viên B: Viết phần III
           Thành viên C: Viết phần V (chờ kết quả từ A)

Ngày 9-10: Thành viên C: Viết phần VI + Phụ lục
           Cả 3 người: Review chéo, sửa lỗi

Ngày 11:   Thành viên C: Tổng hợp, format, tạo mục lục
           Cả 3 người: Kiểm tra lần cuối

Ngày 12:   NỘP BÁO CÁO
```

---

## 🔗 CÔNG VIỆC CHUNG (Cả 3 người)

### 1. Trang bìa
- Tên đề tài, tên trường, tên nhóm, tên GVHD, ngày nộp
- **Ai làm:** Thành viên C (chịu trách nhiệm format)

### 2. Mục lục
- Tự động generate từ Word/Google Docs
- **Ai làm:** Thành viên C (sau khi tổng hợp)

### 3. Review chéo
```
Thành viên A → Review phần của B
Thành viên B → Review phần của C
Thành viên C → Review phần của A
```

### 4. Format cuối cùng
- Font: Times New Roman, size 13 (nội dung), 14-16 (tiêu đề)
- Line spacing: 1.5
- Margin: Left 3cm, Right 2cm, Top/Bottom 2cm
- Hình ảnh: Đánh số [Hình X.Y], caption bên dưới
- Bảng: Đánh số [Bảng X.Y], tiêu đề bên trên
- **Ai chịu trách nhiệm:** Thành viên C

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 🚫 Tránh:
- ❌ Copy nguyên văn từ file hướng dẫn → Phải viết lại bằng lời của mình
- ❌ Để `[value]`, `[Số ảnh]` → Phải điền số liệu thực tế
- ❌ Mỗi người viết xong rồi ghép → Phải review chéo và thống nhất văn phong
- ❌ Chèn code quá dài vào phần chính → Đưa code dài vào Phụ lục

### ✅ Nên:
- ✅ Đọc kỹ `HUONG_DAN_LAM_BAO_CAO.md` trước khi viết
- ✅ Chạy thực tế các script để có số liệu thật
- ✅ Viết nhận xét/phân tích cho MỌI bảng và hình
- ✅ Thống nhất cách trình bày (font, style, thuật ngữ)
- ✅ Backup thường xuyên
- ✅ Giao tiếp khi gặp khó khăn

---

## 📁 FILE CẦN ĐỌC TRƯỚC KHI VIẾT

| File | Ai cần đọc | Mục đích |
|------|-----------|----------|
| `HUONG_DAN_LAM_BAO_CAO.md` | **Cả 3 người** | Hiểu cấu trúc báo cáo |
| `defect_detector.py` | **B** (kỹ) + A, C (overview) | Hiểu code chính |
| `train_detector.py` | **A** + B | Hiểu training pipeline |
| `test_detector.py` | **A** + B | Hiểu testing pipeline |
| `webcam_detector.py` | **B** + A | Hiểu webcam features |
| `collect_webcam_data.py` | **B** | Hiểu data collection |
| `simulate_webcam.py` | **B** | Hiểu data simulation |
| `data.yaml` | **A** | Cấu hình dataset |
| `README.md` | **C** + B | Tổng quan project |
| `requirements.txt` | **B** | Technology stack |
| `runs/detect/...` | **A** | Kết quả training |

---

## 🎯 BẢNG PHÂN CÔNG TÓM TẮT (In ra dán lên bàn)

```
╔══════════════════════════════════════════════════════════════╗
║           PHÂN CÔNG CÔNG VIỆC - BÁO CÁO NHÓM              ║
╠══════════════╦═══════════════════════╦═══════════════════════╣
║ THÀNH VIÊN A ║    THÀNH VIÊN B       ║    THÀNH VIÊN C       ║
╠══════════════╬═══════════════════════╬═══════════════════════╣
║ II. YOLOv8   ║ III. Thiết kế &      ║ I. Giới thiệu        ║
║    & Dataset ║     Triển khai ⭐     ║ V. Đánh giá &        ║
║ IV. Kết quả  ║     (5-6 trang)      ║    Kết luận           ║
║    thực      ║                       ║ VI. Tài liệu TK      ║
║    nghiệm   ║ Đọc kỹ code:         ║ PHỤ LỤC              ║
║              ║ - defect_detector.py  ║                       ║
║ Chạy test,   ║ - train_detector.py   ║ Tổng hợp +           ║
║ thu thập     ║ - test_detector.py    ║ Format toàn bộ       ║
║ số liệu     ║ - webcam_detector.py  ║ báo cáo              ║
╠══════════════╬═══════════════════════╬═══════════════════════╣
║  ~5-6 trang  ║     ~5-6 trang        ║    ~6-8 trang         ║
╚══════════════╩═══════════════════════╩═══════════════════════╝
```

---

**💡 Mẹo:** Tạo 1 Google Drive chung để cả 3 cùng edit + theo dõi tiến độ!

**Chúc nhóm hoàn thành báo cáo tốt! 🎓✨**
