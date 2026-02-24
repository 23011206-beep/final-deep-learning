"""
Thu thập dữ liệu PCB từ Webcam để cải thiện mô hình
=====================================================
Script này giúp bạn:
1. Quay video mạch PCB từ webcam
2. Trích xuất frame từ video (3-5 fps)
3. Tổ chức ảnh để gán nhãn (label) và train lại

Quy trình:
    Bước 1: Quay video     → python collect_webcam_data.py --mode record
    Bước 2: Trích frame    → python collect_webcam_data.py --mode extract --video <path>
    Bước 3: Xem hướng dẫn  → python collect_webcam_data.py --mode guide

Lưu ý khi quay video:
    - Đặt mạch PCB thật dưới camera
    - Quay ở nhiều góc nghiêng nhỏ
    - Quay trong điều kiện ánh sáng ban ngày VÀ ban đêm
    - Mỗi video nên quay 30-60 giây
    - Quay cả PCB có lỗi và PCB không lỗi
"""

import argparse
import cv2
import os
import sys
from pathlib import Path
from datetime import datetime


def record_video(camera_id=0, output_dir="webcam_data/videos", fps=30):
    """
    Quay video mạch PCB từ webcam
    
    Controls:
        - Press 'r' to start/stop recording
        - Press 'q' to quit
        - Press 's' to take a screenshot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Không thể mở camera {camera_id}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n" + "="*70)
    print("THU THẬP DỮ LIỆU PCB TỪ WEBCAM")
    print("="*70)
    print(f"Camera: {camera_id}")
    print(f"Resolution: {width}x{height}")
    print(f"Output: {output_dir}")
    print()
    print("HƯỚNG DẪN QUAY VIDEO:")
    print("  1. Đặt mạch PCB thật dưới camera")
    print("  2. Nhấn 'r' để BẮT ĐẦU quay")
    print("  3. Di chuyển PCB nhẹ nhàng, thay đổi góc nghiêng")
    print("  4. Nhấn 'r' lần nữa để DỪNG quay")
    print("  5. Lặp lại với PCB khác hoặc điều kiện ánh sáng khác")
    print("  6. Nhấn 'q' để THOÁT")
    print()
    print("MẸO:")
    print("  - Quay cả PCB có lỗi và PCB bình thường")
    print("  - Thay đổi ánh sáng (bật/tắt đèn)")
    print("  - Xoay PCB nhẹ để có nhiều góc nhìn")
    print("  - Mỗi video nên dài 30-60 giây")
    print("="*70 + "\n")
    
    recording = False
    video_writer = None
    video_count = 0
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Không thể đọc frame")
                break
            
            # Hiển thị trạng thái
            display_frame = frame.copy()
            
            if recording:
                # Viền đỏ khi đang quay
                cv2.rectangle(display_frame, (0, 0), (width-1, height-1), (0, 0, 255), 4)
                status = "REC"
                status_color = (0, 0, 255)
                
                # Chấm tròn đỏ nhấp nháy
                if (cv2.getTickCount() // cv2.getTickFrequency()) % 2 == 0:
                    cv2.circle(display_frame, (30, 30), 12, (0, 0, 255), -1)
            else:
                status = "READY"
                status_color = (0, 255, 0)
            
            # Vẽ thông tin
            cv2.putText(display_frame, f"[{status}] Press 'r' to record, 'q' to quit",
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display_frame, f"Videos: {video_count} | Screenshots: {screenshot_count}",
                       (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("PCB Data Collection", display_frame)
            
            # Ghi video nếu đang recording
            if recording and video_writer is not None:
                video_writer.write(frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    # Bắt đầu quay
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = os.path.join(output_dir, f"pcb_video_{timestamp}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    recording = True
                    print(f"🔴 Bắt đầu quay: {video_path}")
                else:
                    # Dừng quay
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    video_count += 1
                    print(f"⏹️  Dừng quay. Tổng video: {video_count}")
            elif key == ord('s'):
                # Chụp ảnh
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(output_dir, f"pcb_screenshot_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                screenshot_count += 1
                print(f"📸 Đã chụp: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\nĐã dừng bởi người dùng")
    
    finally:
        if recording and video_writer:
            video_writer.release()
            video_count += 1
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Hoàn thành! Đã quay {video_count} video, {screenshot_count} ảnh")
        print(f"  Lưu tại: {output_dir}")
        print(f"\nBước tiếp theo: Trích xuất frame từ video:")
        print(f"  python collect_webcam_data.py --mode extract --video {output_dir}")


def extract_frames(video_source, output_dir="webcam_data/frames", target_fps=3):
    """
    Trích xuất frame từ video hoặc thư mục chứa video
    
    Args:
        video_source: Đường dẫn đến file video hoặc thư mục chứa video
        output_dir: Thư mục lưu các frame trích xuất
        target_fps: Số frame trích xuất mỗi giây (3-5 fps recommended)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Thu thập danh sách video
    video_paths = []
    source_path = Path(video_source)
    
    if source_path.is_file():
        video_paths = [source_path]
    elif source_path.is_dir():
        video_extensions = {'.avi', '.mp4', '.mkv', '.mov', '.wmv'}
        for ext in video_extensions:
            video_paths.extend(source_path.glob(f"*{ext}"))
            video_paths.extend(source_path.glob(f"*{ext.upper()}"))
    else:
        print(f"Error: Không tìm thấy: {video_source}")
        return
    
    if not video_paths:
        print(f"Error: Không tìm thấy video nào trong: {video_source}")
        return
    
    print("\n" + "="*70)
    print("TRÍCH XUẤT FRAME TỪ VIDEO")
    print("="*70)
    print(f"Số video: {len(video_paths)}")
    print(f"Target FPS: {target_fps}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    total_frames = 0
    
    for video_path in sorted(video_paths):
        print(f"\n📹 Đang xử lý: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ⚠️  Không thể mở video: {video_path}")
            continue
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_video_frames / video_fps if video_fps > 0 else 0
        
        # Tính interval giữa các frame cần lấy
        frame_interval = max(1, int(video_fps / target_fps))
        
        print(f"  Video FPS: {video_fps:.1f}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Total frames: {total_video_frames}")
        print(f"  Extracting every {frame_interval} frames (~{target_fps} fps)")
        
        video_frame_count = 0
        extracted_count = 0
        video_name = video_path.stem
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            video_frame_count += 1
            
            # Chỉ lấy frame theo interval
            if video_frame_count % frame_interval == 0:
                frame_filename = f"{video_name}_frame_{extracted_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
        
        cap.release()
        total_frames += extracted_count
        print(f"  ✓ Đã trích xuất: {extracted_count} frames")
    
    print("\n" + "="*70)
    print(f"TỔNG KẾT")
    print("="*70)
    print(f"  Tổng video xử lý: {len(video_paths)}")
    print(f"  Tổng frame trích xuất: {total_frames}")
    print(f"  Lưu tại: {output_dir}")
    print("="*70)
    
    print(f"\n📌 BƯỚC TIẾP THEO:")
    print(f"  1. Gán nhãn (label) cho các frame bằng Roboflow hoặc LabelImg")
    print(f"  2. Xem hướng dẫn chi tiết: python collect_webcam_data.py --mode guide")


def show_guide():
    """Hiển thị hướng dẫn đầy đủ quy trình thu thập và train lại"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           HƯỚNG DẪN THU THẬP DỮ LIỆU & TRAIN LẠI MÔ HÌNH          ║
╚══════════════════════════════════════════════════════════════════════╝

📋 TỔNG QUAN QUY TRÌNH:
   Quay video → Trích frame → Gán nhãn → Gộp dữ liệu → Train lại

══════════════════════════════════════════════════════════════════════
BƯỚC 1: QUAY VIDEO TỪ WEBCAM
══════════════════════════════════════════════════════════════════════
Lệnh:
    python collect_webcam_data.py --mode record

Lưu ý khi quay:
    ✅ Quay PCB có lỗi (đánh dấu loại lỗi gì)
    ✅ Quay PCB bình thường (KHÔNG có lỗi) - RẤT QUAN TRỌNG!
    ✅ Quay ở nhiều góc nghiêng nhỏ (5-15 độ)
    ✅ Quay trong ánh sáng ban ngày
    ✅ Quay trong ánh sáng ban đêm (đèn phòng)
    ✅ Thay đổi khoảng cách camera-PCB
    ✅ Mỗi video 30-60 giây
    ❌ KHÔNG di chuyển quá nhanh (tránh bị mờ)
    ❌ KHÔNG để PCB bị che khuất

══════════════════════════════════════════════════════════════════════
BƯỚC 2: TRÍCH XUẤT FRAME
══════════════════════════════════════════════════════════════════════
Lệnh:
    python collect_webcam_data.py --mode extract --video webcam_data/videos --fps 3

Kết quả: Các frame ảnh sẽ được lưu trong webcam_data/frames/

══════════════════════════════════════════════════════════════════════
BƯỚC 3: GÁN NHÃN (LABEL) CHO ẢNH
══════════════════════════════════════════════════════════════════════
Bạn cần gán nhãn (vẽ bounding box) cho từng lỗi trên ảnh.

Cách 1: Dùng Roboflow (Dễ nhất, Online)
    1. Vào https://roboflow.com → Tạo project mới
    2. Upload các frame ảnh lên
    3. Vẽ bounding box cho từng lỗi
    4. Chọn các class: missing_hole, mouse_bite, open_circuit, 
       short, spur, spurious_copper
    5. Export dưới dạng "YOLOv8" format
    6. Download và giải nén vào thư mục webcam_data/labeled/

Cách 2: Dùng LabelImg (Offline)
    1. pip install labelImg
    2. labelImg webcam_data/frames/
    3. Chọn format: YOLO
    4. Vẽ bounding box và chọn class cho từng lỗi
    5. Save labels

Quan trọng: Ảnh PCB KHÔNG có lỗi thì KHÔNG cần gán nhãn gì cả,
            chỉ cần để ảnh trong thư mục images và tạo file .txt 
            rỗng tương ứng trong thư mục labels.

══════════════════════════════════════════════════════════════════════
BƯỚC 4: GỘP DỮ LIỆU VÀ TRAIN LẠI
══════════════════════════════════════════════════════════════════════
Sau khi gán nhãn xong:

    1. Copy ảnh vào:     train/images/
    2. Copy labels vào:  train/labels/
    3. Train lại mô hình:

    python train_detector.py --model s --epochs 100 --device 0 --name pcb_defect_v2

    Hoặc tiếp tục train từ model cũ (transfer learning, nhanh hơn):

    python train_detector.py --model s --epochs 50 --device 0 --name pcb_defect_v2 --resume "runs\\detect\\runs\\pcb_defect_detector\\weights\\best.pt"

══════════════════════════════════════════════════════════════════════
MẸO NÂNG CAO
══════════════════════════════════════════════════════════════════════
    • Nên có ít nhất 50-100 ảnh từ webcam
    • Tỉ lệ ảnh có lỗi : ảnh không lỗi nên là 70:30
    • Nếu mô hình hay nhận nhầm ở vùng nào, 
      hãy quay thêm video ở vùng đó
    • Train lại với patience=30 để tránh overfitting
""")


def main():
    parser = argparse.ArgumentParser(
        description='Thu thập dữ liệu PCB từ Webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  Quay video:      python collect_webcam_data.py --mode record
  Trích frame:     python collect_webcam_data.py --mode extract --video webcam_data/videos
  Xem hướng dẫn:   python collect_webcam_data.py --mode guide
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['record', 'extract', 'guide'],
        help='Chế độ: record (quay video), extract (trích frame), guide (hướng dẫn)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera ID (mặc định: 0)'
    )
    parser.add_argument(
        '--video',
        type=str,
        default='webcam_data/videos',
        help='Đường dẫn video hoặc thư mục chứa video (cho mode extract)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=3,
        help='Số frame trích xuất mỗi giây (mặc định: 3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Thư mục lưu kết quả'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'record':
        output = args.output or 'webcam_data/videos'
        record_video(camera_id=args.camera, output_dir=output)
    
    elif args.mode == 'extract':
        output = args.output or 'webcam_data/frames'
        extract_frames(
            video_source=args.video,
            output_dir=output,
            target_fps=args.fps
        )
    
    elif args.mode == 'guide':
        show_guide()


if __name__ == "__main__":
    main()
