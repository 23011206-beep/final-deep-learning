"""
Download PCB Defect Detection Dataset
======================================
Script tải dataset PCB defect từ Roboflow

Hướng dẫn:
1. Đăng ký tài khoản Roboflow (miễn phí): https://app.roboflow.com/
2. Lấy API Key tại: https://app.roboflow.com/settings/api
3. Chạy script này:
   python download_dataset.py --api-key YOUR_API_KEY

Hoặc dùng Cách 2 (không cần API key):
   python download_dataset.py --method manual
"""

import argparse
import os
import sys
from pathlib import Path
import shutil


def download_from_roboflow(api_key: str):
    """
    Cách 1: Tải dataset từ Roboflow (Cần API Key)
    
    Dataset: PCB Defect Detection - 693 images
    Classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Cài đặt roboflow package trước...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow
    
    print("\n" + "="*70)
    print("DOWNLOADING PCB DEFECT DATASET FROM ROBOFLOW")
    print("="*70)
    
    rf = Roboflow(api_key=api_key)
    
    # Thử nhiều dataset phổ biến
    datasets = [
        ("biancapcbdefects", "pcb-defects-detection-yolov8", 1),
        ("pcbdataset", "pcb-defect-detection-mspgp", 1),
        ("rahul-cqtjf", "pcb-defects-dataset", 1),
    ]
    
    dataset_downloaded = False
    for workspace, project_name, version in datasets:
        try:
            print(f"\nThử tải từ: {workspace}/{project_name} v{version}...")
            project = rf.workspace(workspace).project(project_name)
            version_obj = project.version(version)
            dataset = version_obj.download("yolov8", location="./dataset_temp")
            dataset_downloaded = True
            print(f"✓ Tải thành công từ {workspace}/{project_name}!")
            break
        except Exception as e:
            print(f"  ✗ Không tải được: {e}")
            continue
    
    if not dataset_downloaded:
        print("\n❌ Không tải được từ Roboflow.")
        print("Hãy thử Cách 2: python download_dataset.py --method manual")
        return False
    
    # Di chuyển data vào đúng thư mục
    organize_dataset("./dataset_temp")
    return True


def download_from_kaggle():
    """
    Cách 2: Tải dataset từ Kaggle
    
    Cần cài đặt kaggle CLI và có kaggle.json API token
    """
    try:
        import kaggle
    except ImportError:
        print("Cài đặt kaggle package trước...")
        os.system(f"{sys.executable} -m pip install kaggle")
    
    print("\n" + "="*70)
    print("DOWNLOADING PCB DEFECT DATASET FROM KAGGLE")
    print("="*70)
    
    try:
        os.system("kaggle datasets download -d akhatova/pcb-defects -p ./dataset_temp --unzip")
        organize_dataset("./dataset_temp")
        return True
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def organize_dataset(temp_dir: str):
    """
    Sắp xếp dataset vào đúng cấu trúc thư mục
    
    Cấu trúc cần:
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    """
    print("\n" + "="*70)
    print("ORGANIZING DATASET")
    print("="*70)
    
    temp_path = Path(temp_dir)
    base_dir = Path(".")
    
    # Tìm thư mục train/valid/test trong temp
    source_dirs = {}
    for split in ['train', 'valid', 'test', 'val']:
        for candidate in temp_path.rglob(split):
            if candidate.is_dir():
                source_dirs[split] = candidate
                break
    
    # Rename 'val' to 'valid' if needed
    if 'val' in source_dirs and 'valid' not in source_dirs:
        source_dirs['valid'] = source_dirs.pop('val')
    
    for split_name in ['train', 'valid', 'test']:
        if split_name not in source_dirs:
            print(f"  ⚠️ Không tìm thấy thư mục {split_name} trong dataset")
            continue
        
        src = source_dirs[split_name]
        dst = base_dir / split_name
        
        # Tạo thư mục đích
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copy images
        img_src = src / "images"
        lbl_src = src / "labels"
        
        if img_src.exists():
            img_count = 0
            for img_file in img_src.iterdir():
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    shutil.copy2(str(img_file), str(dst / "images" / img_file.name))
                    img_count += 1
            print(f"  ✓ {split_name}/images: {img_count} files")
        
        if lbl_src.exists():
            lbl_count = 0
            for lbl_file in lbl_src.iterdir():
                if lbl_file.suffix.lower() == '.txt':
                    shutil.copy2(str(lbl_file), str(dst / "labels" / lbl_file.name))
                    lbl_count += 1
            print(f"  ✓ {split_name}/labels: {lbl_count} files")
    
    # Cleanup temp
    try:
        shutil.rmtree(temp_dir)
        print(f"\n  ✓ Đã xóa thư mục tạm: {temp_dir}")
    except Exception:
        pass
    
    print("\n" + "="*70)
    print("✅ DATASET ORGANIZED SUCCESSFULLY!")
    print("="*70)
    
    # Verify
    verify_dataset()


def verify_dataset():
    """Kiểm tra dataset đã đúng cấu trúc chưa"""
    print("\n📊 DATASET VERIFICATION:")
    print("-" * 40)
    
    base = Path(".")
    total_images = 0
    
    for split in ['train', 'valid', 'test']:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        
        if img_dir.exists():
            imgs = list(img_dir.glob("*.[jJ][pP][gG]")) + \
                   list(img_dir.glob("*.[jJ][pP][eE][gG]")) + \
                   list(img_dir.glob("*.[pP][nN][gG]"))
            lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
            
            print(f"  {split:>6}: {len(imgs):>4} images, {len(lbls):>4} labels", end="")
            if len(imgs) != len(lbls):
                print(f" ⚠️ MISMATCH!")
            else:
                print(f" ✅")
            total_images += len(imgs)
        else:
            print(f"  {split:>6}: ❌ NOT FOUND")
    
    print(f"\n  Total: {total_images} images")
    
    if total_images > 0:
        print("\n✅ Dataset sẵn sàng! Chạy training:")
        print("   python train_detector.py --model n --epochs 100 --batch 16")
    else:
        print("\n❌ Dataset chưa có. Hãy tải dataset theo hướng dẫn bên dưới.")


def print_manual_instructions():
    """In hướng dẫn tải thủ công"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           HƯỚNG DẪN TẢI DATASET PCB DEFECT DETECTION               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Có 3 cách để tải dataset:                                         ║
║                                                                    ║
║  ═══════════════════════════════════════════════════════════════     ║
║  CÁCH 1: TẢI TỪ ROBOFLOW (Khuyến nghị - Dễ nhất)                  ║
║  ═══════════════════════════════════════════════════════════════     ║
║                                                                    ║
║  Bước 1: Vào link sau:                                             ║
║    https://universe.roboflow.com/search?q=pcb+defect+yolov8        ║
║                                                                    ║
║  Bước 2: Chọn dataset "PCB Defects Detection" hoặc tương tự        ║
║                                                                    ║
║  Bước 3: Click "Download Dataset"                                  ║
║    - Format: YOLOv8                                                ║
║    - Chọn "download zip to computer"                               ║
║                                                                    ║
║  Bước 4: Giải nén vào thư mục project:                             ║
║    Final-Deep-Learning-main/                                       ║
║    ├── train/                                                      ║
║    │   ├── images/    (ảnh training)                                ║
║    │   └── labels/    (annotations)                                 ║
║    ├── valid/                                                      ║
║    │   ├── images/                                                 ║
║    │   └── labels/                                                 ║
║    └── test/                                                       ║
║        ├── images/                                                 ║
║        └── labels/                                                 ║
║                                                                    ║
║  ═══════════════════════════════════════════════════════════════     ║
║  CÁCH 2: TẢI TỪ ROBOFLOW BẰNG PYTHON (Tự động)                    ║
║  ═══════════════════════════════════════════════════════════════     ║
║                                                                    ║
║  Bước 1: Đăng ký tài khoản Roboflow (miễn phí)                    ║
║    https://app.roboflow.com/                                       ║
║                                                                    ║
║  Bước 2: Lấy API Key tại:                                         ║
║    https://app.roboflow.com/settings/api                           ║
║                                                                    ║
║  Bước 3: Chạy lệnh:                                               ║
║    python download_dataset.py --api-key YOUR_API_KEY               ║
║                                                                    ║
║  ═══════════════════════════════════════════════════════════════     ║
║  CÁCH 3: TẢI TỪ KAGGLE                                            ║
║  ═══════════════════════════════════════════════════════════════     ║
║                                                                    ║
║  Bước 1: Vào Kaggle và tải dataset:                                ║
║    https://www.kaggle.com/datasets/akhatova/pcb-defects             ║
║                                                                    ║
║  Bước 2: Giải nén vào thư mục train/valid/test                     ║
║                                                                    ║
║  LƯU Ý: Dataset Kaggle có thể cần chuyển format sang YOLO          ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(description='Download PCB Defect Dataset')
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Roboflow API Key'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='roboflow',
        choices=['roboflow', 'kaggle', 'manual', 'verify'],
        help='Download method (roboflow/kaggle/manual/verify)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PCB DEFECT DETECTION - DATASET DOWNLOADER")
    print("="*70)
    print("Dataset: PCB Defect Detection")
    print("Classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper")
    print("="*70)
    
    if args.method == 'verify':
        verify_dataset()
        return
    
    if args.method == 'manual':
        print_manual_instructions()
        return
    
    if args.method == 'roboflow':
        if args.api_key:
            success = download_from_roboflow(args.api_key)
        else:
            print("\n⚠️ Cần API Key để tải từ Roboflow!")
            print("  Lấy API Key tại: https://app.roboflow.com/settings/api")
            print("\n  Chạy lại: python download_dataset.py --api-key YOUR_API_KEY")
            print("\n  Hoặc xem hướng dẫn tải thủ công:")
            print("  python download_dataset.py --method manual")
            print_manual_instructions()
            return
    
    elif args.method == 'kaggle':
        success = download_from_kaggle()


if __name__ == "__main__":
    main()
