#!/usr/bin/env python3
"""
Debug script to test position bar detection on specific frames
"""

import cv2
import numpy as np
from src.position_bar_reader import PositionBarReader
import matplotlib.pyplot as plt

def debug_position_bar_detection(video_path: str, test_frames: list = None):
    """Test position bar detection on specific frames"""
    
    if test_frames is None:
        # Test frames: early race, mid race, late race
        test_frames = [25, 50, 100, 150]  # Frame numbers
    
    reader = PositionBarReader()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {fps} fps, {total_frames} total frames")
    print(f"Testing frames: {test_frames}")
    print()
    
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Could not read frame {frame_num}")
            continue
            
        print(f"=== FRAME {frame_num} (time: {frame_num/fps:.1f}s) ===")
        
        # Show frame dimensions and position bar region
        height, width = frame.shape[:2]
        y1 = int(height * reader.bar_y_percent_start)  # 85% down
        y2 = int(height * reader.bar_y_percent_end)    # 92% down  
        x1 = int(width * reader.bar_x_percent_start)   # 15% from left
        x2 = int(width * reader.bar_x_percent_end)     # 85% from right
        
        print(f"Frame size: {width}x{height}")
        print(f"Position bar region: {x1},{y1} to {x2},{y2} (width={x2-x1}, height={y2-y1})")
        
        # Extract position bar region
        bar_region = frame[y1:y2, x1:x2]
        
        if bar_region.size == 0:
            print("Position bar region is empty!")
            continue
            
        # Try to detect position bar
        snapshot = reader.read_position_bar(frame, frame_num, fps)
        
        if snapshot:
            print(f"SUCCESS: Found positions {snapshot.positions} (confidence: {snapshot.confidence:.2f})")
        else:
            print("FAILED: No valid position bar reading")
            
        # Save debug images
        cv2.imwrite(f"debug_frame_{frame_num}_full.jpg", frame)
        cv2.imwrite(f"debug_frame_{frame_num}_bar_region.jpg", bar_region)
        
        # Draw rectangle around position bar region on full frame
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f"debug_frame_{frame_num}_with_region.jpg", debug_frame)
        
        print(f"Saved debug images: debug_frame_{frame_num}_*.jpg")
        print()
    
    cap.release()
    print(f"Debug complete. Check debug_frame_*.jpg files in current directory.")

if __name__ == "__main__":
    debug_position_bar_detection("data/videos/race_194367.mp4")