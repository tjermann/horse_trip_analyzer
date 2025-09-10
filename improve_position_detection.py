#!/usr/bin/env python3
"""
Position Detection Improvement System
Uses ground truth labels to improve OCR and CNN accuracy
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import argparse

class PositionBarDataset(Dataset):
    """Dataset for position bar training from labeled data"""
    
    def __init__(self, video_path: str, labels_path: str, transform=None):
        self.video_path = video_path
        self.transform = transform
        
        # Load labels
        with open(labels_path, 'r') as f:
            data = json.load(f)
            self.labels = data['labels']
            self.default_region = data.get('default_region')
        
        self.cap = cv2.VideoCapture(video_path)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label_data = self.labels[idx]
        frame_num = label_data['frame_num']
        positions = label_data['positions']
        
        # Get frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_num}")
        
        # Extract position bar region
        if 'bar_region' in label_data and label_data['bar_region']:
            x, y, w, h = label_data['bar_region']
            bar_region = frame[y:y+h, x:x+w]
        else:
            # Use default region or bottom 15%
            height = frame.shape[0]
            bar_region = frame[int(height * 0.85):, :]
        
        if self.transform:
            bar_region = self.transform(bar_region)
        
        return bar_region, positions

class ImprovedPositionDetector:
    """Enhanced position detector that learns from ground truth"""
    
    def __init__(self):
        self.ocr_confidence_threshold = 0.5
        self.cnn_confidence_threshold = 0.6
        self.learned_patterns = {}
        self.position_templates = {}
        
    def learn_from_labels(self, video_path: str, labels_path: str):
        """Learn position bar patterns from labeled data"""
        logger.info("Learning from labeled data...")
        
        dataset = PositionBarDataset(video_path, labels_path)
        
        for i in range(len(dataset)):
            bar_region, positions = dataset[i]
            
            # Learn color patterns for each position
            self._learn_color_patterns(bar_region, positions)
            
            # Create templates for each digit
            self._create_digit_templates(bar_region, positions)
            
            # Learn spatial relationships
            self._learn_spatial_patterns(bar_region, positions)
        
        logger.info(f"Learned patterns from {len(dataset)} labeled frames")
    
    def _learn_color_patterns(self, bar_region: np.ndarray, positions: List[int]):
        """Learn color characteristics of position rectangles"""
        height, width = bar_region.shape[:2]
        num_positions = len(positions)
        
        if num_positions == 0:
            return
        
        segment_width = width // num_positions
        
        for i, horse_num in enumerate(positions):
            # Extract segment
            x_start = i * segment_width
            x_end = (i + 1) * segment_width if i < num_positions - 1 else width
            
            segment = bar_region[:, x_start:x_end]
            
            # Analyze color distribution
            hsv = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
            
            # Store color histogram as pattern
            if horse_num not in self.learned_patterns:
                self.learned_patterns[horse_num] = []
            
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            self.learned_patterns[horse_num].append(hist)
    
    def _create_digit_templates(self, bar_region: np.ndarray, positions: List[int]):
        """Create template images for each digit"""
        height, width = bar_region.shape[:2]
        num_positions = len(positions)
        
        if num_positions == 0:
            return
        
        segment_width = width // num_positions
        
        for i, horse_num in enumerate(positions):
            x_start = i * segment_width
            x_end = (i + 1) * segment_width if i < num_positions - 1 else width
            
            digit_region = bar_region[:, x_start:x_end]
            
            # Process for template matching
            gray = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
            
            # Multiple preprocessing methods
            templates = []
            
            # Method 1: Binary threshold
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            templates.append(binary)
            
            # Method 2: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            templates.append(adaptive)
            
            # Store templates
            if horse_num not in self.position_templates:
                self.position_templates[horse_num] = []
            
            self.position_templates[horse_num].extend(templates)
    
    def _learn_spatial_patterns(self, bar_region: np.ndarray, positions: List[int]):
        """Learn spatial relationships between positions"""
        # This could include learning:
        # - Typical spacing between position rectangles
        # - Height/width ratios
        # - Alignment patterns
        pass
    
    def detect_with_confidence(self, frame: np.ndarray) -> Dict[int, Tuple[int, float]]:
        """Detect positions using learned patterns with confidence scores"""
        height, width = frame.shape[:2]
        bar_region = frame[int(height * 0.85):, :]
        
        detections = {}
        
        # Use template matching for known patterns
        for horse_num, templates in self.position_templates.items():
            best_match = 0
            best_position = None
            
            for template in templates[:5]:  # Use top 5 templates
                if template.size == 0:
                    continue
                    
                # Resize template if needed
                if template.shape[1] > bar_region.shape[1]:
                    continue
                
                # Template matching
                result = cv2.matchTemplate(bar_region, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_match:
                    best_match = max_val
                    # Calculate position based on x-coordinate
                    position = int((max_loc[0] / bar_region.shape[1]) * 8) + 1
                    best_position = position
            
            if best_match > 0.6 and best_position:
                detections[horse_num] = (best_position, best_match)
        
        return detections
    
    def save_learned_patterns(self, output_path: str):
        """Save learned patterns for future use"""
        data = {
            'patterns': {k: [h.tolist() for h in v] for k, v in self.learned_patterns.items()},
            'confidence_thresholds': {
                'ocr': self.ocr_confidence_threshold,
                'cnn': self.cnn_confidence_threshold
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved learned patterns to {output_path}")

def create_improved_cnn_training_data(video_path: str, labels_path: str, output_dir: str):
    """Create high-quality CNN training data from labeled frames"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = PositionBarDataset(video_path, labels_path)
    
    samples_created = 0
    for i in range(len(dataset)):
        bar_region, positions = dataset[i]
        
        height, width = bar_region.shape[:2]
        num_positions = len(positions)
        
        if num_positions == 0:
            continue
        
        segment_width = width // num_positions
        
        for j, horse_num in enumerate(positions):
            # Extract digit region
            x_start = j * segment_width
            x_end = (j + 1) * segment_width if j < num_positions - 1 else width
            
            digit_region = bar_region[:, x_start:x_end]
            
            # Create multiple augmented versions
            augmentations = [
                digit_region,  # Original
                cv2.flip(digit_region, 0),  # Vertical flip
                cv2.GaussianBlur(digit_region, (3, 3), 0),  # Slight blur
            ]
            
            # Add brightness variations
            for alpha in [0.8, 1.2]:
                aug = cv2.convertScaleAbs(digit_region, alpha=alpha, beta=0)
                augmentations.append(aug)
            
            # Save all augmentations
            digit_dir = output_dir / str(horse_num)
            digit_dir.mkdir(exist_ok=True)
            
            for aug_idx, aug_img in enumerate(augmentations):
                filename = f"frame_{i}_pos_{j+1}_aug_{aug_idx}.png"
                cv2.imwrite(str(digit_dir / filename), aug_img)
                samples_created += 1
    
    logger.info(f"Created {samples_created} CNN training samples")

def main():
    parser = argparse.ArgumentParser(description="Improve position detection using ground truth")
    parser.add_argument("video_path", help="Path to race video")
    parser.add_argument("labels_path", help="Path to ground truth labels")
    parser.add_argument("--mode", choices=['learn', 'train_cnn', 'test'], 
                       default='learn', help="Operation mode")
    parser.add_argument("--output-dir", default="data/improved_training",
                       help="Output directory for training data")
    
    args = parser.parse_args()
    
    if args.mode == 'learn':
        detector = ImprovedPositionDetector()
        detector.learn_from_labels(args.video_path, args.labels_path)
        detector.save_learned_patterns("learned_patterns.json")
        
    elif args.mode == 'train_cnn':
        create_improved_cnn_training_data(args.video_path, args.labels_path, args.output_dir)
        
    elif args.mode == 'test':
        detector = ImprovedPositionDetector()
        detector.learn_from_labels(args.video_path, args.labels_path)
        
        # Test on a frame
        cap = cv2.VideoCapture(args.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 3979)  # Test frame
        ret, frame = cap.read()
        
        if ret:
            detections = detector.detect_with_confidence(frame)
            print("Detections with learned patterns:")
            for horse_num, (position, confidence) in detections.items():
                print(f"  Horse {horse_num}: Position {position} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()