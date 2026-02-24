"""
PCB Defect Detection Framework using YOLOv8
=============================================
Phát hiện và khoanh vùng lỗi trên mạch PCB (Printed Circuit Board)

Kiến trúc: YOLOv8 (Ultralytics)
- Backbone: CSPDarknet
- Neck: PANet (Path Aggregation Network)
- Head: Decoupled detection head

Các loại lỗi phát hiện:
- missing_hole: Lỗ bị thiếu trên PCB
- mouse_bite: Vết cắn chuột (khuyết tật cạnh)
- open_circuit: Mạch hở (đứt mạch)
- short: Ngắn mạch
- spur: Gai đồng thừa
- spurious_copper: Đồng thừa

Hỗ trợ:
- Training từ scratch hoặc transfer learning
- Real-time detection từ webcam
- Inference trên images/videos
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import pandas as pd


# Màu sắc cố định cho từng loại lỗi (BGR cho OpenCV, RGB cho matplotlib)
DEFECT_COLORS = {
    'missing_hole': (255, 0, 0),       # Đỏ
    'mouse_bite': (255, 165, 0),       # Cam
    'open_circuit': (0, 0, 255),       # Xanh dương
    'short': (255, 255, 0),            # Vàng
    'spur': (0, 255, 0),               # Xanh lá
    'spurious_copper': (255, 0, 255),  # Tím
}

# Mô tả chi tiết từng loại lỗi
DEFECT_DESCRIPTIONS = {
    'missing_hole': 'Lỗ khoan bị thiếu trên PCB',
    'mouse_bite': 'Vết cắn chuột - khuyết tật ở cạnh mạch',
    'open_circuit': 'Mạch hở - đường mạch bị đứt',
    'short': 'Ngắn mạch - 2 đường mạch bị nối nhầm',
    'spur': 'Gai đồng thừa nhô ra từ đường mạch',
    'spurious_copper': 'Đồng thừa không mong muốn trên PCB',
}

# Mức độ nghiêm trọng của lỗi
DEFECT_SEVERITY = {
    'missing_hole': 'HIGH',
    'mouse_bite': 'MEDIUM',
    'open_circuit': 'CRITICAL',
    'short': 'CRITICAL',
    'spur': 'MEDIUM',
    'spurious_copper': 'LOW',
}


class DefectDetector:
    """
    PCB Defect Detection Model using YOLOv8
    
    Phát hiện và khoanh vùng các lỗi trên mạch PCB sử dụng YOLOv8.
    
    Attributes:
        model_type: Loại model YOLOv8 ('n', 's', 'm', 'l', 'x')
        model: YOLO model instance
        class_names: List tên các classes (loại lỗi)
        colors: Dict màu sắc cho mỗi loại lỗi
    """
    
    def __init__(self, model_type: str = 'n', pretrained: bool = True):
        """
        Initialize PCB Defect Detector
        
        Args:
            model_type: YOLOv8 model size
                - 'n': nano (fastest, least accurate)
                - 's': small
                - 'm': medium
                - 'l': large
                - 'x': xlarge (slowest, most accurate)
            pretrained: Sử dụng pretrained weights (COCO) hay không
        """
        self.model_type = model_type
        
        # Load model
        if pretrained:
            model_name = f'yolov8{model_type}.pt'
            print(f"Loading pretrained YOLOv8{model_type} model...")
        else:
            model_name = f'yolov8{model_type}.yaml'
            print(f"Initializing YOLOv8{model_type} from scratch...")
            
        self.model = YOLO(model_name)
        
        # Class names (sẽ được update sau khi load data.yaml)
        self.class_names = []
        self.colors = {}
        
        print(f"✓ Model initialized: YOLOv8{model_type}")
    
    def load_data_config(self, data_yaml_path: str):
        """
        Load data configuration từ data.yaml
        
        Args:
            data_yaml_path: Path to data.yaml file
        """
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        self.class_names = data_config['names']
        self.num_classes = data_config['nc']
        
        # Sử dụng màu sắc cố định cho từng loại lỗi
        self._assign_colors()
        
        print(f"✓ Loaded {self.num_classes} defect types: {', '.join(self.class_names)}")
        
        return data_config
    
    def _assign_colors(self):
        """Gán màu sắc cố định cho mỗi loại lỗi"""
        for class_name in self.class_names:
            if class_name in DEFECT_COLORS:
                self.colors[class_name] = DEFECT_COLORS[class_name]
            else:
                # Fallback: random color
                np.random.seed(hash(class_name) % 2**32)
                self.colors[class_name] = tuple(map(int, np.random.randint(50, 255, 3)))
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device=0,
        project: str = 'runs/detect',
        name: str = 'pcb_defect_detector',
        patience: int = 50,
        save_period: int = 10,
        **kwargs
    ):
        """
        Train the PCB defect detector
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size (-1 for auto)
            device: Device to train on (int: 0, 1, etc. for GPU or str: 'cpu' for CPU)
            project: Project directory
            name: Experiment name
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            **kwargs: Additional training arguments
        
        Returns:
            Training results
        """
        print("\n" + "="*70)
        print("TRAINING PCB DEFECT DETECTOR")
        print("="*70)
        
        # Load data config
        self.load_data_config(data_yaml)
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'project': project,
            'name': name,
            'patience': patience,
            'save_period': save_period,
            'plots': True,
            'verbose': True,
            **kwargs
        }
        
        print(f"\nTraining Configuration:")
        print(f"  Model: YOLOv8{self.model_type}")
        print(f"  Defect Types: {self.num_classes}")
        print(f"  Classes: {', '.join(self.class_names)}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
        print(f"  Device: {device}")
        print(f"  Save to: {project}/{name}")
        print("\n" + "="*70 + "\n")
        
        # Train
        results = self.model.train(**train_args)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)
        
        return results
    
    def validate(self, data_yaml: str = None, **kwargs):
        """
        Validate the model
        
        Args:
            data_yaml: Path to data.yaml (optional if already trained)
            **kwargs: Additional validation arguments
        
        Returns:
            Validation results
        """
        if data_yaml:
            val_args = {'data': data_yaml, **kwargs}
        else:
            val_args = kwargs
            
        results = self.model.val(**val_args)
        
        return results
    
    def predict(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        save: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        **kwargs
    ):
        """
        Run inference - Phát hiện lỗi trên ảnh PCB
        
        Args:
            source: Image path, directory, video, or webcam (0)
            conf: Confidence threshold
            iou: NMS IoU threshold
            imgsz: Input image size
            save: Save results
            save_txt: Save results as .txt
            save_conf: Save confidence in .txt
            **kwargs: Additional prediction arguments
        
        Returns:
            Prediction results
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            **kwargs
        )
        
        return results
    
    def export(self, format: str = 'onnx', **kwargs):
        """
        Export model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'coreml', 'tflite', etc.)
            **kwargs: Additional export arguments
        
        Returns:
            Export path
        """
        export_path = self.model.export(format=format, **kwargs)
        print(f"✓ Model exported to: {export_path}")
        
        return export_path
    
    def load_weights(self, weights_path: str):
        """
        Load trained weights
        
        Args:
            weights_path: Path to weights file (.pt)
        """
        self.model = YOLO(weights_path)
        print(f"✓ Loaded weights from: {weights_path}")
    
    def analyze_defects(self, image_path: str, conf: float = 0.25) -> Dict:
        """
        Phân tích chi tiết lỗi trên ảnh PCB
        
        Args:
            image_path: Path to PCB image
            conf: Confidence threshold
        
        Returns:
            Dict chứa thông tin phân tích lỗi chi tiết
        """
        results = self.predict(image_path, conf=conf, save=False)
        
        analysis = {
            'image_path': image_path,
            'total_defects': 0,
            'defect_counts': {},
            'defects': [],
            'severity_summary': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'is_pass': True,  # Có pass QC không
        }
        
        for result in results:
            boxes = result.boxes
            analysis['total_defects'] = len(boxes)
            
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                conf_score = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Count defects per class
                analysis['defect_counts'][class_name] = analysis['defect_counts'].get(class_name, 0) + 1
                
                # Get severity
                severity = DEFECT_SEVERITY.get(class_name, 'UNKNOWN')
                analysis['severity_summary'][severity] = analysis['severity_summary'].get(severity, 0) + 1
                
                # Add defect detail
                analysis['defects'].append({
                    'type': class_name,
                    'confidence': conf_score,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'severity': severity,
                    'description': DEFECT_DESCRIPTIONS.get(class_name, ''),
                })
        
        # Nếu có lỗi CRITICAL hoặc HIGH → FAIL
        if analysis['severity_summary']['CRITICAL'] > 0 or analysis['severity_summary']['HIGH'] > 0:
            analysis['is_pass'] = False
        
        # Nếu có bất kỳ lỗi nào → FAIL
        if analysis['total_defects'] > 0:
            analysis['is_pass'] = False
        
        return analysis
    
    def visualize_predictions(
        self,
        image_path: str,
        conf: float = 0.25,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize defect predictions on a PCB image
        
        Args:
            image_path: Path to PCB image
            conf: Confidence threshold
            save_path: Path to save visualization
            show: Show the plot
        """
        # Run prediction
        results = self.predict(image_path, conf=conf, save=False)
        
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with 2 subplots
        fig, (ax_img, ax_info) = plt.subplots(1, 2, figsize=(18, 8),
                                                gridspec_kw={'width_ratios': [3, 1]})
        
        ax_img.imshow(img)
        ax_img.set_title('PCB Defect Detection', fontsize=14, fontweight='bold')
        
        # Statistics
        defect_counts = {}
        total_defects = 0
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        # Draw predictions
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_score = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                total_defects += 1
                defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
                severity = DEFECT_SEVERITY.get(class_name, 'UNKNOWN')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Draw rectangle
                width = x2 - x1
                height = y2 - y1
                
                color = np.array(self.colors.get(class_name, (255, 0, 0))) / 255.0
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=(color[0], color[1], color[2], 0.1)
                )
                ax_img.add_patch(rect)
                
                # Draw label with severity
                label = f"{class_name} [{severity}]: {conf_score:.2f}"
                ax_img.text(
                    x1, y1 - 5,
                    label,
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
                )
        
        ax_img.axis('off')
        
        # Info panel
        ax_info.axis('off')
        
        # QC Result
        is_pass = total_defects == 0
        qc_status = "✅ PASS" if is_pass else "❌ FAIL"
        qc_color = 'green' if is_pass else 'red'
        
        info_text = f"QC Result: {qc_status}\n"
        info_text += f"{'─' * 30}\n\n"
        info_text += f"Total Defects: {total_defects}\n\n"
        
        if total_defects > 0:
            info_text += "Defect Breakdown:\n"
            for defect, count in sorted(defect_counts.items()):
                severity = DEFECT_SEVERITY.get(defect, '?')
                info_text += f"  • {defect}: {count} [{severity}]\n"
            
            info_text += f"\nSeverity Summary:\n"
            for sev, count in severity_counts.items():
                if count > 0:
                    info_text += f"  • {sev}: {count}\n"
        else:
            info_text += "No defects detected.\nPCB passes quality check."
        
        ax_info.text(
            0.1, 0.95, info_text,
            transform=ax_info.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax_info.set_title('Analysis Report', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_report(
        self,
        image_dir: str,
        conf: float = 0.25,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Tạo báo cáo kiểm tra lỗi cho nhiều ảnh PCB
        
        Args:
            image_dir: Thư mục chứa ảnh PCB
            conf: Confidence threshold
            save_path: Path lưu báo cáo CSV
        
        Returns:
            DataFrame chứa báo cáo
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(image_dir).glob(f'*{ext}'))
            image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        reports = []
        
        print(f"\nAnalyzing {len(image_paths)} PCB images...")
        print("=" * 70)
        
        for i, img_path in enumerate(sorted(image_paths)):
            analysis = self.analyze_defects(str(img_path), conf=conf)
            
            report_row = {
                'image': img_path.name,
                'total_defects': analysis['total_defects'],
                'qc_result': 'PASS' if analysis['is_pass'] else 'FAIL',
            }
            
            # Add count per defect type
            for defect_name in self.class_names:
                report_row[defect_name] = analysis['defect_counts'].get(defect_name, 0)
            
            # Add severity counts
            for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                report_row[f'severity_{sev}'] = analysis['severity_summary'].get(sev, 0)
            
            reports.append(report_row)
            
            status = "PASS ✅" if analysis['is_pass'] else "FAIL ❌"
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name}: "
                  f"{analysis['total_defects']} defects → {status}")
        
        df = pd.DataFrame(reports)
        
        # Summary
        print("\n" + "=" * 70)
        print("QC SUMMARY")
        print("=" * 70)
        total = len(df)
        passed = len(df[df['qc_result'] == 'PASS'])
        failed = total - passed
        print(f"  Total PCBs: {total}")
        print(f"  Passed: {passed} ({100*passed/max(total,1):.1f}%)")
        print(f"  Failed: {failed} ({100*failed/max(total,1):.1f}%)")
        print(f"  Total defects found: {df['total_defects'].sum()}")
        print("=" * 70)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\n✓ Report saved to: {save_path}")
        
        return df


class WebcamDefectDetector:
    """
    Real-time PCB defect detection từ webcam
    
    Phát hiện lỗi PCB trong thời gian thực thông qua camera.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize webcam defect detector
        
        Args:
            model_path: Path to trained model (.pt file)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        """
        print(f"Loading PCB defect detection model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Get class names
        self.class_names = self.model.names
        
        # Assign colors
        self.colors = self._assign_colors()
        
        print("✓ Model loaded successfully!")
        print(f"✓ Defect types: {len(self.class_names)}")
        for class_id, class_name in self.class_names.items():
            severity = DEFECT_SEVERITY.get(class_name, '?')
            print(f"  - {class_name} [{severity}]")
    
    def _assign_colors(self):
        """Gán màu sắc cố định cho mỗi loại lỗi"""
        colors = {}
        for class_id, class_name in self.class_names.items():
            if class_name in DEFECT_COLORS:
                colors[class_id] = DEFECT_COLORS[class_name]
            else:
                np.random.seed(class_id * 42)
                colors[class_id] = tuple(map(int, np.random.randint(50, 255, 3)))
        return colors
    
    def run(
        self,
        camera_id: int = 0,
        window_name: str = "PCB Defect Detector",
        display_fps: bool = True
    ):
        """
        Run real-time defect detection từ webcam
        
        Args:
            camera_id: Camera ID (0 for default webcam)
            window_name: Window name
            display_fps: Display FPS on frame
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print("\n" + "="*70)
        print("REAL-TIME PCB DEFECT DETECTION")
        print("="*70)
        print(f"Camera ID: {camera_id}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"\nDefect Types:")
        for class_id, class_name in self.class_names.items():
            severity = DEFECT_SEVERITY.get(class_name, '?')
            desc = DEFECT_DESCRIPTIONS.get(class_name, '')
            print(f"  - {class_name} [{severity}]: {desc}")
        print(f"\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'p' to pause/resume")
        print("  - Press '+' to increase confidence threshold")
        print("  - Press '-' to decrease confidence threshold")
        print("="*70 + "\n")
        
        # FPS calculation
        fps_start_time = datetime.now()
        fps_frame_count = 0
        current_fps = 0
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    # Read frame
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("Error: Cannot read frame")
                        break
                    
                    frame_count += 1
                    
                    # Run detection
                    results = self.model.predict(
                        frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )
                    
                    # Draw detections
                    detection_count = 0
                    severity_in_frame = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
                    
                    for result in results:
                        boxes = result.boxes
                        
                        for box in boxes:
                            detection_count += 1
                            
                            # Get box info
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.class_names[class_id]
                            severity = DEFECT_SEVERITY.get(class_name, '?')
                            
                            # Track severity
                            if severity in severity_in_frame:
                                severity_in_frame[severity] += 1
                            
                            # Get color
                            color = self.colors[class_id]
                            
                            # Draw bounding box (thicker for critical defects)
                            thickness = 3 if severity in ['CRITICAL', 'HIGH'] else 2
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Prepare label with severity
                            label = f"{class_name} [{severity}]: {conf:.2f}"
                            
                            # Get label size
                            (label_w, label_h), baseline = cv2.getTextSize(
                                label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                1
                            )
                            
                            # Draw label background
                            cv2.rectangle(
                                frame,
                                (x1, y1 - label_h - baseline - 5),
                                (x1 + label_w, y1),
                                color,
                                -1
                            )
                            
                            # Draw label text
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )
                    
                    # Calculate FPS
                    fps_frame_count += 1
                    if fps_frame_count >= 10:
                        fps_end_time = datetime.now()
                        current_fps = fps_frame_count / (fps_end_time - fps_start_time).total_seconds()
                        fps_start_time = fps_end_time
                        fps_frame_count = 0
                    
                    # QC Status
                    if detection_count == 0:
                        qc_status = "PASS"
                        qc_color = (0, 255, 0)  # Green
                    elif severity_in_frame['CRITICAL'] > 0:
                        qc_status = "FAIL - CRITICAL"
                        qc_color = (0, 0, 255)  # Red
                    elif severity_in_frame['HIGH'] > 0:
                        qc_status = "FAIL - HIGH"
                        qc_color = (0, 100, 255)  # Orange
                    else:
                        qc_status = "FAIL - MINOR"
                        qc_color = (0, 255, 255)  # Yellow
                    
                    # Draw info overlay
                    if display_fps:
                        info_text = [
                            f"FPS: {current_fps:.1f}",
                            f"Defects: {detection_count}",
                            f"QC: {qc_status}",
                            f"Conf: {self.conf_threshold:.2f}",
                            f"Frame: {frame_count}"
                        ]
                        
                        y_offset = 30
                        for i, text in enumerate(info_text):
                            # Choose color based on content
                            text_color = qc_color if 'QC:' in text else (0, 255, 0)
                            
                            # Draw text background
                            (text_w, text_h), _ = cv2.getTextSize(
                                text,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                2
                            )
                            cv2.rectangle(
                                frame,
                                (5, y_offset - text_h - 5),
                                (15 + text_w, y_offset + 5),
                                (0, 0, 0),
                                -1
                            )
                            
                            # Draw text
                            cv2.putText(
                                frame,
                                text,
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                text_color,
                                2,
                                cv2.LINE_AA
                            )
                            y_offset += 30
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pcb_defect_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Frame saved: {filename}")
                elif key == ord('p'):
                    # Pause/Resume
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n{status}")
                elif key == ord('+') or key == ord('='):
                    # Increase confidence threshold
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"\nConfidence threshold: {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease confidence threshold
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"\nConfidence threshold: {self.conf_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*70)
            print(f"Total frames processed: {frame_count}")
            print(f"Average FPS: {current_fps:.1f}")
            print("Camera closed.")
            print("="*70 + "\n")


def plot_training_results(results_dir: str):
    """
    Plot training results từ CSV files
    
    Args:
        results_dir: Directory chứa results.csv
    """
    csv_path = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    # Read results
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PCB Defect Detection - Training Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o', markersize=3)
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss', marker='s', markersize=3)
    axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: mAP
    if 'metrics/mAP50(B)' in df.columns:
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', marker='o', markersize=3)
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', marker='s', markersize=3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Validation mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision & Recall
    if 'metrics/precision(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o', markersize=3)
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='s', markersize=3)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    if 'lr/pg0' in df.columns:
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR pg0', marker='o', markersize=3)
        axes[1, 1].plot(df['epoch'], df['lr/pg1'], label='LR pg1', marker='s', markersize=3)
        axes[1, 1].plot(df['epoch'], df['lr/pg2'], label='LR pg2', marker='^', markersize=3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'training_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"✓ Training analysis saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("PCB Defect Detector Module")
    print("=" * 70)
    print("Phát hiện và khoanh vùng lỗi trên mạch PCB")
    print()
    print("This module provides:")
    print("  - DefectDetector: Main defect detection model class")
    print("  - WebcamDefectDetector: Real-time webcam defect detection")
    print("  - plot_training_results: Visualize training metrics")
    print()
    print("Defect Types:")
    for defect, desc in DEFECT_DESCRIPTIONS.items():
        severity = DEFECT_SEVERITY[defect]
        print(f"  - {defect} [{severity}]: {desc}")
    print()
    print("Usage examples:")
    print("  1. Training: See train_detector.py")
    print("  2. Testing: See test_detector.py")
    print("  3. Webcam: See webcam_detector.py")
    print("=" * 70)
