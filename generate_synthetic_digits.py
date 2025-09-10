#!/usr/bin/env python3
"""
Generate synthetic digit samples for numbers 9-20 using image manipulation
"""

import cv2
import numpy as np
from pathlib import Path
import json
from loguru import logger

def create_synthetic_digits():
    """Create synthetic samples for digits 9-20"""
    
    data_dir = Path("data/position_digits")
    
    # Load existing annotations to see what digits we have
    annotations_file = data_dir / "annotations.json"
    if not annotations_file.exists():
        logger.error("No annotations file found. Label some samples first.")
        return
    
    with open(annotations_file) as f:
        annotations = json.load(f)
    
    # Count available digits
    available_digits = {}
    for ann in annotations:
        if ann['label'] and ann['label'] > 0:
            digit = ann['label']
            if digit not in available_digits:
                available_digits[digit] = []
            available_digits[digit].append(ann['path'])
    
    logger.info(f"Available labeled digits: {list(available_digits.keys())}")
    
    # Generate synthetic samples for missing digits
    synthetic_annotations = []
    
    for target_digit in range(9, 21):  # 9-20
        if target_digit in available_digits:
            continue
            
        logger.info(f"Generating synthetic samples for digit {target_digit}")
        
        # For two-digit numbers, combine single digits
        if target_digit >= 10:
            first_digit = 1
            second_digit = target_digit - 10
            
            if first_digit in available_digits and second_digit in available_digits:
                # Take samples from both digits
                first_samples = available_digits[first_digit][:5]
                second_samples = available_digits[second_digit][:5]
                
                for i, (first_path, second_path) in enumerate(zip(first_samples, second_samples)):
                    if Path(first_path).exists() and Path(second_path).exists():
                        # Load both digit images
                        first_img = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
                        second_img = cv2.imread(second_path, cv2.IMREAD_GRAYSCALE)
                        
                        if first_img is not None and second_img is not None:
                            # Combine horizontally
                            combined = np.hstack([first_img, second_img])
                            # Resize back to 32x32
                            combined = cv2.resize(combined, (32, 32))
                            
                            # Save synthetic sample
                            synthetic_dir = data_dir / str(target_digit)
                            synthetic_dir.mkdir(exist_ok=True)
                            synthetic_path = synthetic_dir / f"synthetic_{target_digit}_{i}.png"
                            cv2.imwrite(str(synthetic_path), combined)
                            
                            # Add to annotations
                            synthetic_annotations.append({
                                'id': f"synthetic_{target_digit}_{i}",
                                'video': 'synthetic',
                                'frame': -1,
                                'bbox': [0, 0, 32, 32],
                                'path': str(synthetic_path),
                                'label': target_digit
                            })
                            
                            logger.info(f"Created synthetic sample: {synthetic_path}")
    
    # Update annotations file
    if synthetic_annotations:
        annotations.extend(synthetic_annotations)
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        logger.info(f"Added {len(synthetic_annotations)} synthetic samples to annotations")
    
    return len(synthetic_annotations)

if __name__ == "__main__":
    create_synthetic_digits()