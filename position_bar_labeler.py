#!/usr/bin/env python3
"""
Interactive Position Bar Labeling Tool
Allows frame-by-frame labeling of position bar for ground truth data collection
"""

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
from pathlib import Path
from loguru import logger

@dataclass
class PositionBarLabel:
    """Ground truth label for a position bar reading"""
    frame_num: int
    timestamp: float
    positions: List[int]  # Horse numbers in order (1st, 2nd, 3rd, etc.)
    bar_region: Tuple[int, int, int, int]  # x, y, w, h of position bar
    confidence: float  # User confidence in label (0-1)
    notes: str = ""

class PositionBarLabeler:
    """Interactive tool for labeling position bars in race videos"""
    
    def __init__(self, video_path: str, output_dir: str = "data/position_labels"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Labeling state
        self.current_frame_num = 0
        self.current_frame = None
        self.labels: List[PositionBarLabel] = []
        self.bar_region = None  # Selected position bar region
        self.selecting_region = False
        self.selection_start = None
        self.selection_end = None
        
        # Load existing labels if available
        self.labels_file = self.output_dir / f"{Path(video_path).stem}_labels.json"
        self.load_labels()
        
        logger.info(f"Initialized labeler for video: {video_path}")
        logger.info(f"Video info: {self.width}x{self.height}, {self.fps} fps, {self.total_frames} frames")
    
    def load_labels(self):
        """Load existing labels if available"""
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                data = json.load(f)
                self.labels = [PositionBarLabel(**label) for label in data['labels']]
                if 'default_region' in data and data['default_region']:
                    self.bar_region = tuple(data['default_region'])
                logger.info(f"Loaded {len(self.labels)} existing labels")
    
    def save_labels(self):
        """Save labels to JSON file"""
        data = {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'default_region': self.bar_region,
            'labels': [asdict(label) for label in self.labels]
        }
        
        with open(self.labels_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.labels)} labels to {self.labels_file}")
    
    def jump_to_frame(self, frame_num: int):
        """Jump to specific frame"""
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.current_frame_num = frame_num
            return True
        return False
    
    def get_display_frame(self):
        """Get frame with overlays for display"""
        if self.current_frame is None:
            return None
        
        display = self.current_frame.copy()
        
        # Draw position bar region if set
        if self.bar_region:
            x, y, w, h = self.bar_region
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, "Position Bar", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw selection in progress
        if self.selecting_region and self.selection_start and self.selection_end:
            cv2.rectangle(display, self.selection_start, self.selection_end, (255, 0, 0), 2)
        
        # Add frame info
        info_text = f"Frame {self.current_frame_num}/{self.total_frames} | Time: {self.current_frame_num/self.fps:.2f}s"
        cv2.putText(display, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Check if this frame has a label
        frame_label = self.get_frame_label(self.current_frame_num)
        if frame_label:
            label_text = f"Labeled: {frame_label.positions}"
            cv2.putText(display, label_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instructions
        instructions = [
            "Controls:",
            "SPACE: Play/Pause | <- ->: Frame step | [ ]: Jump 10 frames",
            "R: Select region | L: Label positions | S: Save | Q: Quit",
            "J: Jump to frame | E: Extract digits for CNN training"
        ]
        y_pos = self.height - 120
        for instruction in instructions:
            cv2.putText(display, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
        
        return display
    
    def get_frame_label(self, frame_num: int) -> Optional[PositionBarLabel]:
        """Get label for specific frame if exists"""
        for label in self.labels:
            if label.frame_num == frame_num:
                return label
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting_region = True
            self.selection_start = (x, y)
            self.selection_end = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting_region:
                self.selection_end = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selecting_region:
                self.selecting_region = False
                # Calculate region bounds
                x1 = min(self.selection_start[0], self.selection_end[0])
                y1 = min(self.selection_start[1], self.selection_end[1])
                x2 = max(self.selection_start[0], self.selection_end[0])
                y2 = max(self.selection_start[1], self.selection_end[1])
                
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.bar_region = (x1, y1, x2 - x1, y2 - y1)
                    logger.info(f"Position bar region set: {self.bar_region}")
    
    def label_current_frame(self):
        """Interactively label the current frame's position bar"""
        if self.current_frame is None:
            logger.warning("No frame loaded")
            return
        
        print("\n" + "="*50)
        print(f"LABELING FRAME {self.current_frame_num}")
        print("="*50)
        
        # Show current frame with region
        if self.bar_region:
            x, y, w, h = self.bar_region
            bar_img = self.current_frame[y:y+h, x:x+w]
            
            # Show enlarged version
            scale = 4
            enlarged = cv2.resize(bar_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Position Bar (Enlarged)", enlarged)
        
        print("\nEnter horse numbers in order from 1st to last place")
        print("Example: 2,7,5,4,6,3,8,1")
        print("Or press Enter to skip this frame")
        
        positions_str = input("Positions: ").strip()
        
        if positions_str:
            try:
                positions = [int(x.strip()) for x in positions_str.split(',')]
                
                confidence_str = input("Confidence (0-1, default 1.0): ").strip()
                confidence = float(confidence_str) if confidence_str else 1.0
                
                notes = input("Notes (optional): ").strip()
                
                # Create label
                label = PositionBarLabel(
                    frame_num=self.current_frame_num,
                    timestamp=self.current_frame_num / self.fps,
                    positions=positions,
                    bar_region=self.bar_region if self.bar_region else (0, 0, self.width, self.height),
                    confidence=confidence,
                    notes=notes
                )
                
                # Remove any existing label for this frame
                self.labels = [l for l in self.labels if l.frame_num != self.current_frame_num]
                self.labels.append(label)
                
                logger.info(f"Labeled frame {self.current_frame_num}: {positions}")
                print(f"✓ Frame labeled successfully!")
                
            except Exception as e:
                logger.error(f"Error parsing input: {e}")
                print(f"✗ Error: {e}")
    
    def extract_digit_samples(self):
        """Extract individual digit images for CNN training"""
        if self.current_frame is None or not self.bar_region:
            logger.warning("Need frame and region to extract digits")
            return
        
        x, y, w, h = self.bar_region
        bar_img = self.current_frame[y:y+h, x:x+w]
        
        # Get label for this frame
        frame_label = self.get_frame_label(self.current_frame_num)
        if not frame_label:
            print("This frame needs to be labeled first!")
            return
        
        # Create output directory
        samples_dir = Path("data/cnn_training_samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Divide bar into segments based on number of positions
        num_positions = len(frame_label.positions)
        segment_width = w // num_positions
        
        for i, horse_num in enumerate(frame_label.positions):
            # Extract segment
            x_start = i * segment_width
            x_end = (i + 1) * segment_width if i < num_positions - 1 else w
            
            digit_img = bar_img[:, x_start:x_end]
            
            # Save to appropriate digit folder
            digit_dir = samples_dir / str(horse_num)
            digit_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            filename = f"frame_{self.current_frame_num}_pos_{i+1}.png"
            filepath = digit_dir / filename
            
            cv2.imwrite(str(filepath), digit_img)
            logger.info(f"Saved digit sample: {filepath}")
        
        print(f"✓ Extracted {num_positions} digit samples")
    
    def run(self):
        """Run the interactive labeling interface"""
        cv2.namedWindow("Position Bar Labeler", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Position Bar Labeler", self.mouse_callback)
        
        # Start with first frame
        self.jump_to_frame(0)
        
        playing = False
        frame_skip = 1
        
        while True:
            if playing:
                # Advance frame
                self.current_frame_num += frame_skip
                if self.current_frame_num >= self.total_frames:
                    self.current_frame_num = self.total_frames - 1
                    playing = False
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
                    ret, self.current_frame = self.cap.read()
                    if not ret:
                        playing = False
            
            # Display current frame
            display = self.get_display_frame()
            if display is not None:
                cv2.imshow("Position Bar Labeler", display)
            
            # Handle keyboard input
            key = cv2.waitKey(30 if playing else 1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - play/pause
                playing = not playing
            elif key == 81 or key == 2:  # Left arrow
                self.jump_to_frame(self.current_frame_num - 1)
            elif key == 83 or key == 3:  # Right arrow
                self.jump_to_frame(self.current_frame_num + 1)
            elif key == ord('['):  # Jump back 10 frames
                self.jump_to_frame(self.current_frame_num - 10)
            elif key == ord(']'):  # Jump forward 10 frames
                self.jump_to_frame(self.current_frame_num + 10)
            elif key == ord('r'):  # Select region
                print("\nClick and drag to select position bar region")
            elif key == ord('l'):  # Label current frame
                self.label_current_frame()
            elif key == ord('s'):  # Save labels
                self.save_labels()
                print("✓ Labels saved")
            elif key == ord('j'):  # Jump to frame
                frame_str = input("Jump to frame number: ").strip()
                try:
                    frame_num = int(frame_str)
                    self.jump_to_frame(frame_num)
                except:
                    print("Invalid frame number")
            elif key == ord('e'):  # Extract digit samples
                self.extract_digit_samples()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final save
        self.save_labels()
        print(f"\n✓ Labeling complete. Saved {len(self.labels)} labels")

def main():
    parser = argparse.ArgumentParser(description="Position Bar Labeling Tool")
    parser.add_argument("video_path", help="Path to race video")
    parser.add_argument("--output-dir", default="data/position_labels", 
                       help="Directory to save labels")
    
    args = parser.parse_args()
    
    labeler = PositionBarLabeler(args.video_path, args.output_dir)
    labeler.run()

if __name__ == "__main__":
    main()