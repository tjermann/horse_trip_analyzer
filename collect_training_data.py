#!/usr/bin/env python3
"""
Collect training data for CNN position bar digit recognition.
Extracts digit regions from race videos and allows manual labeling.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict
from loguru import logger
import argparse
from tqdm import tqdm

class TrainingDataCollector:
    """Collect and label position bar digits for CNN training"""
    
    def __init__(self, output_dir: str = "data/position_digits"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each digit
        for digit in range(1, 21):  # Support up to 20 horses
            (self.output_dir / str(digit)).mkdir(exist_ok=True)
        
        self.annotations = []
        self.annotation_file = self.output_dir / "annotations.json"
        
        # Load existing annotations if they exist
        if self.annotation_file.exists():
            with open(self.annotation_file, 'r') as f:
                self.annotations = json.load(f)
            logger.info(f"Loaded {len(self.annotations)} existing annotations")
    
    def extract_digit_regions(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract potential digit regions from a frame"""
        height, width = frame.shape[:2]
        
        # Focus on position bar area (bottom 15% and top 15%)
        regions_to_check = [
            frame[int(height * 0.85):, :],  # Bottom
            frame[:int(height * 0.15), :],  # Top  
            frame[int(height * 0.75):int(height * 0.87), :],  # Original position
        ]
        
        all_digits = []
        
        for region_idx, region in enumerate(regions_to_check):
            if region.size == 0:
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            # Multiple preprocessing approaches
            preprocessed = [
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
            ]
            
            for proc_idx, binary in enumerate(preprocessed):
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio and size (digits are roughly square)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    area = w * h
                    
                    if (0.2 < aspect_ratio < 2.0 and 
                        h > 15 and w > 10 and 
                        area > 150 and area < 5000):
                        
                        # Extract digit region
                        digit = gray[y:y+h, x:x+w]
                        
                        # Resize to standard size for CNN (32x32)
                        digit_resized = cv2.resize(digit, (32, 32))
                        
                        # Calculate actual position in original frame
                        if region_idx == 0:  # Bottom region
                            actual_y = int(height * 0.85) + y
                        elif region_idx == 1:  # Top region
                            actual_y = y
                        else:  # Middle region
                            actual_y = int(height * 0.75) + y
                        
                        all_digits.append((digit_resized, (x, actual_y, w, h)))
        
        # Remove duplicates based on position
        unique_digits = []
        seen_positions = set()
        
        for digit, (x, y, w, h) in all_digits:
            pos_key = (x // 20, y // 20)  # Group nearby positions
            if pos_key not in seen_positions:
                unique_digits.append((digit, (x, y, w, h)))
                seen_positions.add(pos_key)
        
        # Sort by x position (left to right)
        unique_digits.sort(key=lambda d: d[1][0])
        
        return unique_digits
    
    def process_video(self, video_path: str, race_code: str = None):
        """Process a video to extract training samples"""
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if not race_code:
            race_code = Path(video_path).stem
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        # Sample frames throughout the video
        sample_interval = max(1, int(fps * 2))  # Sample every 2 seconds
        frame_indices = range(0, total_frames, sample_interval)
        
        samples_collected = 0
        
        for frame_idx in tqdm(frame_indices, desc="Extracting digits"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Extract digit regions
            digit_regions = self.extract_digit_regions(frame)
            
            if len(digit_regions) > 0:
                # Save samples for manual labeling
                for digit_idx, (digit_img, bbox) in enumerate(digit_regions):
                    sample_id = f"{race_code}_f{frame_idx}_d{digit_idx}"
                    sample_path = self.output_dir / "unlabeled" / f"{sample_id}.png"
                    sample_path.parent.mkdir(exist_ok=True)
                    
                    cv2.imwrite(str(sample_path), digit_img)
                    
                    # Store metadata
                    self.annotations.append({
                        'id': sample_id,
                        'video': video_path,
                        'frame': frame_idx,
                        'bbox': bbox,
                        'path': str(sample_path),
                        'label': None  # To be filled during labeling
                    })
                    
                    samples_collected += 1
        
        cap.release()
        
        # Save annotations
        with open(self.annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        logger.info(f"Collected {samples_collected} digit samples from {video_path}")
        return samples_collected
    
    def label_samples_interactive(self):
        """Interactive labeling interface for collected samples"""
        
        unlabeled = [a for a in self.annotations if a['label'] is None]
        
        if not unlabeled:
            logger.info("No unlabeled samples found!")
            return
        
        logger.info(f"Found {len(unlabeled)} unlabeled samples")
        logger.info("Instructions:")
        logger.info("  - Press 1-9 for digits 1-9")
        logger.info("  - Press 0 for digit 10, then 1-0 for 11-20")
        logger.info("  - Press 'n' for not a digit / unclear")
        logger.info("  - Press 's' to skip")
        logger.info("  - Press 'q' to quit")
        
        for idx, annotation in enumerate(unlabeled):
            img_path = annotation['path']
            
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Show enlarged version for easier viewing
            img_large = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
            
            # Add border and info
            bordered = cv2.copyMakeBorder(img_large, 20, 20, 20, 20, 
                                        cv2.BORDER_CONSTANT, value=128)
            
            cv2.putText(bordered, f"Sample {idx+1}/{len(unlabeled)}", 
                       (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            
            cv2.imshow("Digit Sample - Label this digit", bordered)
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    self._save_annotations()
                    return
                
                elif key == ord('s'):
                    break  # Skip this sample
                
                elif key == ord('n'):
                    # Not a digit
                    annotation['label'] = -1
                    logger.info(f"Labeled as: not a digit")
                    break
                
                elif ord('0') <= key <= ord('9'):
                    # Handle multi-digit input for 10-20
                    digit = key - ord('0')
                    
                    if digit == 0:
                        # Wait for second digit for 10-20
                        cv2.putText(bordered, "Enter second digit for 10-20", 
                                   (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
                        cv2.imshow("Digit Sample - Label this digit", bordered)
                        
                        key2 = cv2.waitKey(0) & 0xFF
                        if ord('0') <= key2 <= ord('9'):
                            digit = 10 + (key2 - ord('0'))
                    
                    if digit == 0:
                        digit = 10  # If just '0' pressed, interpret as 10
                    
                    annotation['label'] = digit
                    
                    # Move file to appropriate directory
                    new_path = self.output_dir / str(digit) / os.path.basename(img_path)
                    new_path.parent.mkdir(exist_ok=True)
                    os.rename(img_path, new_path)
                    annotation['path'] = str(new_path)
                    
                    logger.info(f"Labeled as: {digit}")
                    break
        
        cv2.destroyAllWindows()
        self._save_annotations()
        logger.info("Labeling session complete!")
    
    def _save_annotations(self):
        """Save annotations to file"""
        with open(self.annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        # Count labeled samples
        labeled_count = {}
        for ann in self.annotations:
            if ann['label'] is not None and ann['label'] > 0:
                label = ann['label']
                labeled_count[label] = labeled_count.get(label, 0) + 1
        
        logger.info("Labeled samples per digit:")
        for digit in sorted(labeled_count.keys()):
            logger.info(f"  Digit {digit}: {labeled_count[digit]} samples")
    
    def generate_dataset_summary(self):
        """Generate summary of collected dataset"""
        
        summary = {
            'total_samples': len(self.annotations),
            'labeled_samples': len([a for a in self.annotations if a['label'] is not None]),
            'unlabeled_samples': len([a for a in self.annotations if a['label'] is None]),
            'samples_per_digit': {}
        }
        
        for ann in self.annotations:
            if ann['label'] is not None and ann['label'] > 0:
                digit = ann['label']
                summary['samples_per_digit'][digit] = summary['samples_per_digit'].get(digit, 0) + 1
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Collect training data for position bar CNN")
    parser.add_argument('--video', type=str, help="Path to video file")
    parser.add_argument('--video-dir', type=str, default="data/videos", 
                       help="Directory containing videos")
    parser.add_argument('--label', action='store_true', 
                       help="Start interactive labeling session")
    parser.add_argument('--output-dir', type=str, default="data/position_digits",
                       help="Output directory for training data")
    
    args = parser.parse_args()
    
    collector = TrainingDataCollector(args.output_dir)
    
    if args.label:
        # Interactive labeling mode
        collector.label_samples_interactive()
    
    elif args.video:
        # Process single video
        collector.process_video(args.video)
        logger.info("Run with --label flag to label the extracted samples")
    
    elif args.video_dir:
        # Process all videos in directory
        video_files = list(Path(args.video_dir).glob("*.mp4"))
        logger.info(f"Found {len(video_files)} videos to process")
        
        for video_path in video_files:
            collector.process_video(str(video_path))
        
        logger.info("Run with --label flag to label the extracted samples")
    
    # Show summary
    summary = collector.generate_dataset_summary()
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total samples: {summary['total_samples']}")
    logger.info(f"  Labeled: {summary['labeled_samples']}")
    logger.info(f"  Unlabeled: {summary['unlabeled_samples']}")
    
    if summary['samples_per_digit']:
        logger.info(f"  Samples per digit: {summary['samples_per_digit']}")


if __name__ == "__main__":
    main()