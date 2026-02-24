"""
Training Script for PCB Defect Detector
========================================
Script để train YOLOv8 model cho phát hiện lỗi trên mạch PCB

Các loại lỗi phát hiện:
- missing_hole: Lỗ bị thiếu
- mouse_bite: Vết cắn chuột (khuyết tật cạnh)
- open_circuit: Mạch hở
- short: Ngắn mạch
- spur: Gai đồng thừa
- spurious_copper: Đồng thừa

Usage:
    python train_detector.py --model n --epochs 100 --batch 16
"""

import argparse
import os
from pathlib import Path
from defect_detector import DefectDetector, plot_training_results


def main():
    parser = argparse.ArgumentParser(description='Train PCB Defect Detector')
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights (COCO)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        default='data.yaml',
        help='Path to data.yaml file'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (-1 for auto)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to train on (0 for GPU, cpu for CPU)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,  # Set to 0 for Windows to avoid multiprocessing deadlocks
        help='Number of dataloader workers (0=no multiprocessing, safer on Windows)'
    )
    
    # Save arguments
    parser.add_argument(
        '--project',
        type=str,
        default='runs',
        help='Project directory'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='pcb_defect_detector',
        help='Experiment name'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    # Optimization arguments
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='Final learning rate (lr0 * lrf)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='Weight decay'
    )
    
    # Augmentation arguments
    parser.add_argument(
        '--hsv-h',
        type=float,
        default=0.015,
        help='HSV-Hue augmentation'
    )
    parser.add_argument(
        '--hsv-s',
        type=float,
        default=0.7,
        help='HSV-Saturation augmentation'
    )
    parser.add_argument(
        '--hsv-v',
        type=float,
        default=0.4,
        help='HSV-Value augmentation'
    )
    parser.add_argument(
        '--degrees',
        type=float,
        default=0.0,
        help='Rotation augmentation (degrees)'
    )
    parser.add_argument(
        '--translate',
        type=float,
        default=0.1,
        help='Translation augmentation'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Scale augmentation'
    )
    parser.add_argument(
        '--shear',
        type=float,
        default=0.0,
        help='Shear augmentation (degrees)'
    )
    parser.add_argument(
        '--perspective',
        type=float,
        default=0.0,
        help='Perspective augmentation'
    )
    parser.add_argument(
        '--flipud',
        type=float,
        default=0.0,
        help='Vertical flip probability'
    )
    parser.add_argument(
        '--fliplr',
        type=float,
        default=0.5,
        help='Horizontal flip probability'
    )
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='Mosaic augmentation probability'
    )
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        help='Mixup augmentation probability'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Convert device string to appropriate type
    # YOLOv8 expects integer for GPU (0, 1, etc.) or 'cpu' string
    device = args.device
    if device.lower() != 'cpu':
        try:
            device = int(device)  # Convert '0' to 0 for proper CUDA detection
        except ValueError:
            print(f"Warning: Invalid device '{device}', using CPU")
            device = 'cpu'
    
    # Convert data path to absolute path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return
    
    # Initialize detector
    print("\n" + "="*70)
    print("PCB DEFECT DETECTOR TRAINING")
    print("="*70)
    print("Phát hiện và khoanh vùng lỗi trên mạch PCB")
    print("Defect Types: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper")
    
    detector = DefectDetector(
        model_type=args.model,
        pretrained=args.pretrained
    )
    
    # Prepare training arguments
    train_kwargs = {
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'workers': args.workers,
    }
    
    if args.resume:
        train_kwargs['resume'] = args.resume
    
    # Train
    results = detector.train(
        data_yaml=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,  # Use converted device (int or 'cpu')
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        **train_kwargs
    )
    
    # Plot results
    results_dir = Path(args.project) / args.name
    if results_dir.exists():
        print("\n" + "="*70)
        print("GENERATING TRAINING ANALYSIS")
        print("="*70)
        plot_training_results(str(results_dir))
    
    # Validate
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    val_results = detector.validate(data_yaml=str(data_path))
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model: YOLOv8{args.model}")
    print(f"Task: PCB Defect Detection")
    print(f"Defect Types: 6 (missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper)")
    print(f"Epochs trained: {args.epochs}")
    print(f"Results saved to: {results_dir}")
    print(f"Best weights: {results_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {results_dir / 'weights' / 'last.pt'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Critical for Windows multiprocessing with PyTorch/CUDA
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()
