"""
Real-time PCB Defect Detection from Webcam
============================================
Ứng dụng phát hiện lỗi PCB trong thời gian thực sử dụng trained YOLOv8 model

Các loại lỗi phát hiện:
- missing_hole: Lỗ bị thiếu
- mouse_bite: Vết cắn chuột
- open_circuit: Mạch hở
- short: Ngắn mạch
- spur: Gai đồng thừa
- spurious_copper: Đồng thừa

Usage:
    python webcam_detector.py --weights runs/detect/pcb_defect_detector/weights/best.pt
"""

import argparse
from defect_detector import WebcamDefectDetector


def main():
    parser = argparse.ArgumentParser(description='Real-time PCB Defect Detection')
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera ID (0 for default webcam)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='NMS IoU threshold'
    )
    parser.add_argument(
        '--window-name',
        type=str,
        default='PCB Defect Detector',
        help='Window name'
    )
    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Disable FPS display'
    )
    
    args = parser.parse_args()
    
    # Initialize webcam detector
    detector = WebcamDefectDetector(
        model_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run detection
    detector.run(
        camera_id=args.camera,
        window_name=args.window_name,
        display_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()
