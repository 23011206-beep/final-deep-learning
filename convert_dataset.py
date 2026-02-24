"""
Chuyển đổi PCB Defect Dataset từ Pascal VOC (XML) sang YOLO format
và chia thành train/valid/test

Dataset source: Kaggle - akhatova/pcb-defects
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ====================== CẤU HÌNH ======================
DATASET_PATH = r"C:\Users\Admin\.cache\kagglehub\datasets\akhatova\pcb-defects\versions\1\PCB_DATASET"
OUTPUT_DIR = r"d:\hocsau\Final-Deep-Learning-main"

# Tỷ lệ chia dataset
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# Class mapping (tên class -> class_id cho YOLO)
CLASS_MAP = {
    'missing_hole': 0,
    'mouse_bite': 1,
    'open_circuit': 2,
    'short': 3,
    'spur': 4,
    'spurious_copper': 5,
}

# Tên thư mục trong dataset gốc -> tên class YOLO
FOLDER_TO_CLASS = {
    'Missing_hole': 'missing_hole',
    'Mouse_bite': 'mouse_bite',
    'Open_circuit': 'open_circuit',
    'Short': 'short',
    'Spur': 'spur',
    'Spurious_copper': 'spurious_copper',
}
# =======================================================


def parse_voc_xml(xml_path: str) -> list:
    """
    Parse Pascal VOC XML annotation file
    
    Returns: list of (class_name, xmin, ymin, xmax, ymax, img_width, img_height)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append((class_name, xmin, ymin, xmax, ymax, img_width, img_height))
    
    return objects


def voc_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    Chuyển đổi bounding box từ VOC (xmin, ymin, xmax, ymax) 
    sang YOLO format (x_center, y_center, width, height) - normalized
    """
    x_center = ((xmin + xmax) / 2.0) / img_width
    y_center = ((ymin + ymax) / 2.0) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    # Clamp values
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def convert_dataset():
    """Main conversion function"""
    
    print("="*70)
    print("PCB DEFECT DATASET CONVERTER")
    print("VOC XML → YOLO Format + Train/Valid/Test Split")
    print("="*70)
    
    dataset_path = Path(DATASET_PATH)
    output_dir = Path(OUTPUT_DIR)
    
    images_dir = dataset_path / "images"
    annotations_dir = dataset_path / "Annotations"
    
    # Thu thập tất cả cặp (image, annotation)
    all_samples = []
    
    print("\n📂 Scanning dataset...")
    
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        img_folder = images_dir / folder_name
        ann_folder = annotations_dir / folder_name
        
        if not img_folder.exists():
            print(f"  ⚠️ Folder not found: {img_folder}")
            continue
        
        img_files = sorted(list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png")))
        
        count = 0
        for img_file in img_files:
            xml_file = ann_folder / (img_file.stem + ".xml")
            
            if xml_file.exists():
                all_samples.append((img_file, xml_file, class_name))
                count += 1
            else:
                print(f"  ⚠️ No annotation for: {img_file.name}")
        
        print(f"  ✓ {folder_name}: {count} samples")
    
    print(f"\n📊 Total samples: {len(all_samples)}")
    
    # Shuffle và chia dataset
    random.seed(42)
    random.shuffle(all_samples)
    
    n_total = len(all_samples)
    n_train = int(n_total * TRAIN_RATIO)
    n_valid = int(n_total * VALID_RATIO)
    n_test = n_total - n_train - n_valid
    
    splits = {
        'train': all_samples[:n_train],
        'valid': all_samples[n_train:n_train + n_valid],
        'test': all_samples[n_train + n_valid:],
    }
    
    print(f"\n📐 Dataset split:")
    print(f"  Train: {len(splits['train'])} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Valid: {len(splits['valid'])} ({VALID_RATIO*100:.0f}%)")
    print(f"  Test:  {len(splits['test'])}  ({TEST_RATIO*100:.0f}%)")
    
    # Tạo thư mục output và chuyển đổi
    print(f"\n🔄 Converting and copying files...")
    
    stats = {'total_images': 0, 'total_annotations': 0, 'total_objects': 0}
    class_counts = {name: 0 for name in CLASS_MAP}
    
    for split_name, samples in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img_file, xml_file, default_class in samples:
            # Copy image
            dst_img = img_out / img_file.name
            shutil.copy2(str(img_file), str(dst_img))
            stats['total_images'] += 1
            
            # Parse XML and convert to YOLO
            objects = parse_voc_xml(str(xml_file))
            
            yolo_lines = []
            for class_name, xmin, ymin, xmax, ymax, w, h in objects:
                # Map class name to ID
                if class_name in CLASS_MAP:
                    class_id = CLASS_MAP[class_name]
                else:
                    print(f"  ⚠️ Unknown class: {class_name}")
                    continue
                
                # Convert to YOLO format
                x_c, y_c, bw, bh = voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
                yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
                
                stats['total_objects'] += 1
                class_counts[class_name] += 1
            
            # Save YOLO label file
            lbl_file = lbl_out / (img_file.stem + ".txt")
            with open(lbl_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            stats['total_annotations'] += 1
        
        print(f"  ✓ {split_name}: {len(samples)} images processed")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETED!")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total labels: {stats['total_annotations']}")
    print(f"  Total objects: {stats['total_objects']}")
    
    print(f"\n📋 Class distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        class_id = CLASS_MAP[class_name]
        print(f"  [{class_id}] {class_name}: {count} objects")
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"  ├── train/")
    print(f"  │   ├── images/ ({len(splits['train'])} files)")
    print(f"  │   └── labels/ ({len(splits['train'])} files)")
    print(f"  ├── valid/")
    print(f"  │   ├── images/ ({len(splits['valid'])} files)")
    print(f"  │   └── labels/ ({len(splits['valid'])} files)")
    print(f"  └── test/")
    print(f"      ├── images/ ({len(splits['test'])} files)")
    print(f"      └── labels/ ({len(splits['test'])} files)")
    
    print(f"\n🚀 Sẵn sàng training!")
    print(f"   python train_detector.py --model n --epochs 100 --batch 16")
    print("="*70)


if __name__ == "__main__":
    convert_dataset()
