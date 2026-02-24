"""
Test PCB Defect Detector on Images
====================================
Script để test trained model phát hiện lỗi trên ảnh PCB

Usage:
    python test_detector.py --weights runs/pcb_defect_detector/weights/best.pt --source test/images
"""

import argparse
from pathlib import Path
from defect_detector import DefectDetector, DEFECT_SEVERITY, DEFECT_DESCRIPTIONS
import os


def main():
    parser = argparse.ArgumentParser(description='Test PCB Defect Detector')
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to test images directory or single image'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data.yaml',
        help='Path to data.yaml file'
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
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save detection results'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results as .txt files'
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Save confidence scores in .txt files'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs',
        help='Project directory'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='test',
        help='Experiment name'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize predictions with matplotlib'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate QC report for all images'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device (0 for GPU, cpu for CPU)'
    )
    
    args = parser.parse_args()
    
    # Check if weights exist
    if not Path(args.weights).exists():
        print(f"Error: Weights file not found: {args.weights}")
        return
    
    # Check if source exists
    if not Path(args.source).exists():
        print(f"Error: Source not found: {args.source}")
        return
    
    print("\n" + "="*70)
    print("PCB DEFECT DETECTOR TESTING")
    print("="*70)
    print("Phát hiện và khoanh vùng lỗi trên mạch PCB")
    print(f"Weights: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")
    print("\nDefect Types:")
    for defect, desc in DEFECT_DESCRIPTIONS.items():
        severity = DEFECT_SEVERITY[defect]
        print(f"  - {defect} [{severity}]: {desc}")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = DefectDetector(model_type='n', pretrained=False)
    detector.load_weights(args.weights)
    
    # Load data config
    if Path(args.data).exists():
        detector.load_data_config(args.data)
    
    # Generate QC report if requested
    if args.report and Path(args.source).is_dir():
        print("\n" + "="*70)
        print("GENERATING QC REPORT")
        print("="*70)
        
        report_path = Path(args.project) / args.name / "qc_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = detector.generate_report(
            image_dir=args.source,
            conf=args.conf,
            save_path=str(report_path)
        )
        
        print(f"\n✓ QC Report saved to: {report_path}")
    
    # Run prediction
    print("\n" + "="*70)
    print("RUNNING DEFECT DETECTION...")
    print("="*70)
    
    results = detector.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        device=args.device
    )
    
    # Print results
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    
    total_defects = 0
    total_images = 0
    pass_count = 0
    fail_count = 0
    overall_defect_counts = {}
    
    for i, result in enumerate(results):
        total_images += 1
        image_defects = len(result.boxes)
        total_defects += image_defects
        
        print(f"\nImage {i+1}: {result.path}")
        print(f"  Defects found: {image_defects}")
        
        if image_defects == 0:
            pass_count += 1
            print(f"  QC Status: ✅ PASS")
        else:
            fail_count += 1
            print(f"  QC Status: ❌ FAIL")
            
            # Count defects per class
            class_counts = {}
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = detector.class_names[class_id]
                severity = DEFECT_SEVERITY.get(class_name, '?')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                overall_defect_counts[class_name] = overall_defect_counts.get(class_name, 0) + 1
            
            print("  Defect breakdown:")
            for class_name, count in sorted(class_counts.items()):
                severity = DEFECT_SEVERITY.get(class_name, '?')
                print(f"    - {class_name} [{severity}]: {count}")
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL QC SUMMARY")
    print("="*70)
    print(f"  Total images: {total_images}")
    print(f"  Passed: {pass_count} ({100*pass_count/max(total_images,1):.1f}%)")
    print(f"  Failed: {fail_count} ({100*fail_count/max(total_images,1):.1f}%)")
    print(f"  Total defects: {total_defects}")
    
    if overall_defect_counts:
        print(f"\n  Defect Distribution:")
        for class_name, count in sorted(overall_defect_counts.items(), key=lambda x: -x[1]):
            severity = DEFECT_SEVERITY.get(class_name, '?')
            print(f"    - {class_name} [{severity}]: {count}")
    
    # Visualize if requested
    if args.visualize and Path(args.source).is_file():
        print("\n" + "="*70)
        print("VISUALIZATION")
        print("="*70)
        
        save_path = Path(args.project) / args.name / "visualization.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        detector.visualize_predictions(
            image_path=args.source,
            conf=args.conf,
            save_path=str(save_path),
            show=False
        )
    
    print("\n" + "="*70)
    print("TESTING COMPLETED")
    print("="*70)
    if args.save:
        print(f"Results saved to: {Path(args.project) / args.name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
