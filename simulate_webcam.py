"""
Mô phỏng Webcam - Data Degradation cho PCB Defect Detection
=============================================================
Lấy ảnh PCB scan chất lượng cao, "làm xấu" để giả lập chất lượng webcam.

Kỹ thuật áp dụng:
- GaussNoise: Nhiễu hạt muỗi (cảm biến webcam rẻ tiền)
- MotionBlur: Mờ khi di chuyển PCB trước ống kính
- RandomBrightnessContrast: Thay đổi ánh sáng phòng
- GaussianBlur: Webcam không nét bằng máy scan
- ImageCompression: Giảm chất lượng nén JPEG
- ColorJitter: Sai lệch màu sắc webcam

Usage:
    python simulate_webcam.py                          # Chạy mặc định (3 biến thể/ảnh)
    python simulate_webcam.py --variants 5             # 5 biến thể mỗi ảnh
    python simulate_webcam.py --preview                # Xem trước trước khi tạo
    python simulate_webcam.py --input train/images --output train/images  # Ghi thẳng vào train
"""

import argparse
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import random


def create_webcam_transform(severity="medium"):
    """
    Tạo bộ augmentation giả lập chất lượng webcam
    
    Args:
        severity: Mức độ "xấu" - "light", "medium", "heavy"
    
    Returns:
        albumentations.Compose transform
    """
    if severity == "light":
        return A.Compose([
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=1.0),
            ], p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.6
            ),
            A.ImageCompression(quality_range=(75, 95), p=0.3),
        ])
    
    elif severity == "medium":
        return A.Compose([
            A.OneOf([
                A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            ], p=0.8),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.Defocus(radius=(2, 4), p=1.0),
            ], p=0.6),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.15, 0.15),
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=0.4
            ),
            A.ImageCompression(quality_range=(60, 90), p=0.4),
        ])
    
    else:  # heavy
        return A.Compose([
            A.OneOf([
                A.GaussNoise(std_range=(0.08, 0.2), p=1.0),
                A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.2, 0.5), p=1.0),
            ], p=0.9),
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 11), p=1.0),
                A.MotionBlur(blur_limit=(5, 11), p=1.0),
                A.Defocus(radius=(3, 6), p=1.0),
                A.ZoomBlur(max_factor=(1.02, 1.08), p=1.0),
            ], p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.2, 0.2),
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=25,
                val_shift_limit=25,
                p=0.5
            ),
            A.ImageCompression(quality_range=(40, 80), p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        ])


def simulate_webcam_images(
    input_dir,
    output_dir,
    label_input_dir=None,
    label_output_dir=None,
    variants_per_image=3,
    severities=None,
    preview=False
):
    """
    Tạo các phiên bản "webcam" từ ảnh gốc chất lượng cao
    
    Args:
        input_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục lưu ảnh đã "làm xấu"
        label_input_dir: Thư mục chứa nhãn YOLO gốc
        label_output_dir: Thư mục lưu nhãn YOLO (copy nguyên)
        variants_per_image: Số biến thể cho mỗi ảnh gốc
        severities: Danh sách mức độ ["light", "medium", "heavy"]
        preview: Nếu True, chỉ hiển thị preview không lưu
    """
    if severities is None:
        severities = ["light", "medium", "heavy"]
    
    os.makedirs(output_dir, exist_ok=True)
    if label_output_dir:
        os.makedirs(label_output_dir, exist_ok=True)
    
    # Thu thập ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).glob(f"*{ext}"))
        image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        print(f"Error: Không tìm thấy ảnh nào trong {input_dir}")
        return
    
    # Tạo transforms cho mỗi severity
    transforms = {sev: create_webcam_transform(sev) for sev in severities}
    
    print("\n" + "="*70)
    print("MÔ PHỎNG WEBCAM - DATA DEGRADATION")
    print("="*70)
    print(f"Ảnh gốc: {len(image_paths)} ảnh trong {input_dir}")
    print(f"Số biến thể/ảnh: {variants_per_image}")
    print(f"Mức độ: {', '.join(severities)}")
    print(f"Tổng ảnh sẽ tạo: {len(image_paths) * variants_per_image}")
    print(f"Output: {output_dir}")
    if label_input_dir:
        print(f"Labels: {label_input_dir} → {label_output_dir}")
    print("="*70 + "\n")
    
    if preview:
        # Chỉ hiển thị preview cho 3 ảnh đầu tiên
        print("CHẾ ĐỘ PREVIEW - Hiển thị 3 ảnh mẫu")
        for img_path in image_paths[:3]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Resize cho dễ xem
            h, w = img.shape[:2]
            scale = min(400 / w, 400 / h)
            display_w, display_h = int(w * scale), int(h * scale)
            
            original = cv2.resize(img, (display_w, display_h))
            
            # Tạo 1 biến thể cho mỗi severity
            row_images = [original]
            titles = ["Original"]
            
            for sev in severities:
                transform = transforms[sev]
                augmented = transform(image=img)["image"]
                augmented_resized = cv2.resize(augmented, (display_w, display_h))
                row_images.append(augmented_resized)
                titles.append(sev.capitalize())
            
            # Ghép ảnh ngang
            combined = np.hstack(row_images)
            
            # Thêm tiêu đề
            y_offset = 20
            x_offset = 0
            for title in titles:
                cv2.putText(combined, title, (x_offset + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                x_offset += display_w
            
            cv2.imshow(f"Preview: {img_path.name}", combined)
            print(f"  Hiển thị: {img_path.name} - Nhấn phím bất kỳ để tiếp...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("\nXong preview! Chạy lại không có --preview để tạo ảnh thực.")
        return
    
    # Tạo ảnh augmented
    total_created = 0
    
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Không đọc được: {img_path.name}")
            continue
        
        # Tìm file label tương ứng
        label_path = None
        if label_input_dir:
            label_filename = img_path.stem + ".txt"
            label_path = Path(label_input_dir) / label_filename
            if not label_path.exists():
                label_path = None
        
        for v in range(variants_per_image):
            # Chọn severity ngẫu nhiên
            severity = random.choice(severities)
            transform = transforms[severity]
            
            # Áp dụng augmentation
            augmented = transform(image=img)["image"]
            
            # Tạo tên file mới
            new_name = f"{img_path.stem}_webcam_{severity}_{v}{img_path.suffix}"
            output_path = os.path.join(output_dir, new_name)
            
            # Lưu ảnh
            cv2.imwrite(output_path, augmented)
            total_created += 1
            
            # Copy label (vì chỉ thay đổi pixel, không thay đổi vị trí bounding box)
            if label_path and label_output_dir:
                new_label_name = f"{img_path.stem}_webcam_{severity}_{v}.txt"
                label_output_path = os.path.join(label_output_dir, new_label_name)
                shutil.copy2(str(label_path), label_output_path)
        
        if (idx + 1) % 50 == 0 or idx == len(image_paths) - 1:
            print(f"  [{idx+1}/{len(image_paths)}] Đã xử lý {idx+1} ảnh → {total_created} biến thể")
    
    print("\n" + "="*70)
    print("HOÀN THÀNH!")
    print("="*70)
    print(f"  Ảnh gốc: {len(image_paths)}")
    print(f"  Ảnh webcam tạo mới: {total_created}")
    print(f"  Tổng ảnh train: {len(image_paths) + total_created}")
    print(f"  Lưu tại: {output_dir}")
    
    if label_output_dir:
        print(f"  Labels: {label_output_dir}")
    
    print(f"\n📌 BƯỚC TIẾP THEO:")
    print(f"  Train lại mô hình với dữ liệu mới:")
    print(f"  python train_detector.py --model s --epochs 100 --device 0 --name pcb_defect_v2")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Mô phỏng Webcam - Làm xấu ảnh PCB chất lượng cao',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  Xem trước hiệu ứng:
    python simulate_webcam.py --preview

  Tạo 3 biến thể/ảnh, ghi thẳng vào thư mục train:
    python simulate_webcam.py --variants 3

  Tạo 5 biến thể/ảnh với mức độ nặng:
    python simulate_webcam.py --variants 5 --severity heavy

  Ghi ra thư mục riêng (không ghi đè):
    python simulate_webcam.py --output webcam_data/augmented --label-output webcam_data/labels
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='train/images',
        help='Thư mục chứa ảnh gốc (mặc định: train/images)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='train/images',
        help='Thư mục lưu ảnh webcam (mặc định: train/images - ghi cùng thư mục)'
    )
    parser.add_argument(
        '--label-input',
        type=str,
        default='train/labels',
        help='Thư mục chứa nhãn YOLO gốc'
    )
    parser.add_argument(
        '--label-output',
        type=str,
        default='train/labels',
        help='Thư mục lưu nhãn YOLO cho ảnh webcam'
    )
    parser.add_argument(
        '--variants',
        type=int,
        default=3,
        help='Số biến thể webcam cho mỗi ảnh gốc (mặc định: 3)'
    )
    parser.add_argument(
        '--severity',
        type=str,
        default='all',
        choices=['light', 'medium', 'heavy', 'all'],
        help='Mức độ "xấu" của ảnh (mặc định: all = tất cả mức độ)'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Xem trước hiệu ứng trên 3 ảnh mẫu (không tạo file)'
    )
    
    args = parser.parse_args()
    
    # Xác định severity
    if args.severity == 'all':
        severities = ['light', 'medium', 'heavy']
    else:
        severities = [args.severity]
    
    simulate_webcam_images(
        input_dir=args.input,
        output_dir=args.output,
        label_input_dir=args.label_input,
        label_output_dir=args.label_output,
        variants_per_image=args.variants,
        severities=severities,
        preview=args.preview
    )


if __name__ == "__main__":
    main()
