#!/usr/bin/env python3
"""
Position Detection Accuracy Validator
Compares OCR/CNN predictions against ground truth labels
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
from loguru import logger
from src.position_bar_reader import PositionBarReader
from src.hybrid_position_detector import HybridPositionDetector

@dataclass
class ValidationResult:
    """Result of validating a single frame"""
    frame_num: int
    ground_truth: List[int]
    ocr_prediction: Optional[List[int]]
    cnn_prediction: Optional[List[int]]
    hybrid_prediction: Optional[Dict[int, Tuple[int, float]]]
    position_accuracy: float  # Percentage of correct positions
    horse_accuracy: float  # Percentage of horses in correct positions
    notes: str = ""

class PositionAccuracyValidator:
    """Validates position detection accuracy against ground truth"""
    
    def __init__(self, video_path: str, labels_path: str, num_horses: int = 8):
        self.video_path = video_path
        self.num_horses = num_horses
        
        # Load ground truth labels
        with open(labels_path, 'r') as f:
            data = json.load(f)
            self.labels = data['labels']
            self.default_region = data.get('default_region')
        
        # Initialize detectors
        self.position_reader = PositionBarReader(expected_horses=num_horses)
        self.hybrid_detector = HybridPositionDetector(num_horses=num_horses)
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Results storage
        self.validation_results: List[ValidationResult] = []
        
        logger.info(f"Loaded {len(self.labels)} ground truth labels")
    
    def validate_frame(self, frame_num: int) -> Optional[ValidationResult]:
        """Validate a single frame against ground truth"""
        # Find ground truth for this frame
        ground_truth = None
        for label in self.labels:
            if label['frame_num'] == frame_num:
                ground_truth = label['positions']
                break
        
        if not ground_truth:
            return None
        
        # Get frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Run OCR detection
        ocr_result = self.position_reader.read_position_bar(frame, frame_num, self.fps)
        ocr_prediction = ocr_result.positions if ocr_result else None
        
        # Run hybrid detection
        hybrid_positions = {}
        try:
            # Mock horse detections for testing (in real use, would come from horse detector)
            mock_horses = []
            for i in range(1, self.num_horses + 1):
                mock_horses.append({
                    'horse_id': i,
                    'bbox': [100*i, 200, 150*i, 300],
                    'confidence': 0.9
                })
            
            hybrid_positions = self.hybrid_detector.detect_positions(
                frame, mock_horses, frame_num, frame_num / self.fps
            )
        except Exception as e:
            logger.error(f"Hybrid detection failed: {e}")
        
        # Calculate accuracies
        position_accuracy = self.calculate_position_accuracy(ground_truth, ocr_prediction)
        horse_accuracy = self.calculate_horse_accuracy(ground_truth, hybrid_positions)
        
        result = ValidationResult(
            frame_num=frame_num,
            ground_truth=ground_truth,
            ocr_prediction=ocr_prediction,
            cnn_prediction=None,  # Could add separate CNN test
            hybrid_prediction=hybrid_positions,
            position_accuracy=position_accuracy,
            horse_accuracy=horse_accuracy
        )
        
        self.validation_results.append(result)
        return result
    
    def calculate_position_accuracy(self, ground_truth: List[int], prediction: Optional[List[int]]) -> float:
        """Calculate percentage of positions correctly identified"""
        if not prediction:
            return 0.0
        
        correct = 0
        for i in range(min(len(ground_truth), len(prediction))):
            if ground_truth[i] == prediction[i]:
                correct += 1
        
        return (correct / len(ground_truth)) * 100 if ground_truth else 0
    
    def calculate_horse_accuracy(self, ground_truth: List[int], 
                                predictions: Dict[int, Tuple[int, float]]) -> float:
        """Calculate percentage of horses in correct positions"""
        if not predictions:
            return 0.0
        
        correct = 0
        for pos_idx, horse_num in enumerate(ground_truth):
            if horse_num in predictions:
                predicted_pos, _ = predictions[horse_num]
                if predicted_pos == pos_idx + 1:  # Positions are 1-indexed
                    correct += 1
        
        return (correct / len(ground_truth)) * 100 if ground_truth else 0
    
    def validate_all(self):
        """Validate all labeled frames"""
        logger.info("Starting validation of all labeled frames...")
        
        for label in self.labels:
            frame_num = label['frame_num']
            result = self.validate_frame(frame_num)
            
            if result:
                logger.info(f"Frame {frame_num}: Position accuracy={result.position_accuracy:.1f}%, "
                          f"Horse accuracy={result.horse_accuracy:.1f}%")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive accuracy report"""
        if not self.validation_results:
            return {"error": "No validation results"}
        
        # Calculate overall statistics
        position_accuracies = [r.position_accuracy for r in self.validation_results]
        horse_accuracies = [r.horse_accuracy for r in self.validation_results]
        
        # Find problem frames
        problem_frames = []
        for result in self.validation_results:
            if result.position_accuracy < 50:
                problem_frames.append({
                    'frame': result.frame_num,
                    'accuracy': result.position_accuracy,
                    'ground_truth': result.ground_truth,
                    'prediction': result.ocr_prediction
                })
        
        report = {
            'total_frames_validated': len(self.validation_results),
            'average_position_accuracy': np.mean(position_accuracies),
            'average_horse_accuracy': np.mean(horse_accuracies),
            'min_position_accuracy': min(position_accuracies),
            'max_position_accuracy': max(position_accuracies),
            'perfect_frames': sum(1 for a in position_accuracies if a == 100),
            'problem_frames': problem_frames[:10],  # Top 10 worst frames
            'accuracy_distribution': {
                '0-25%': sum(1 for a in position_accuracies if a <= 25),
                '25-50%': sum(1 for a in position_accuracies if 25 < a <= 50),
                '50-75%': sum(1 for a in position_accuracies if 50 < a <= 75),
                '75-99%': sum(1 for a in position_accuracies if 75 < a < 100),
                '100%': sum(1 for a in position_accuracies if a == 100)
            }
        }
        
        return report
    
    def save_report(self, output_path: str):
        """Save validation report to file"""
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("POSITION DETECTION ACCURACY REPORT")
        print("="*60)
        print(f"Frames validated: {report['total_frames_validated']}")
        print(f"Average position accuracy: {report['average_position_accuracy']:.1f}%")
        print(f"Average horse accuracy: {report['average_horse_accuracy']:.1f}%")
        print(f"Perfect frames: {report['perfect_frames']}/{report['total_frames_validated']}")
        print("\nAccuracy distribution:")
        for range_name, count in report['accuracy_distribution'].items():
            print(f"  {range_name}: {count} frames")
        
        if report['problem_frames']:
            print("\nMost problematic frames:")
            for pf in report['problem_frames'][:5]:
                print(f"  Frame {pf['frame']}: {pf['accuracy']:.1f}% accuracy")
                print(f"    Expected: {pf['ground_truth']}")
                print(f"    Got: {pf['prediction']}")
        
        logger.info(f"Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Validate position detection accuracy")
    parser.add_argument("video_path", help="Path to race video")
    parser.add_argument("labels_path", help="Path to ground truth labels JSON")
    parser.add_argument("--num-horses", type=int, default=8, help="Number of horses in race")
    parser.add_argument("--output", default="validation_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    validator = PositionAccuracyValidator(args.video_path, args.labels_path, args.num_horses)
    validator.validate_all()
    validator.save_report(args.output)

if __name__ == "__main__":
    main()