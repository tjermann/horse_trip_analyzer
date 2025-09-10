#!/usr/bin/env python3
"""
Fix OCR for white numbers on colored backgrounds in position bar
"""

import cv2
import numpy as np
import easyocr
from pathlib import Path
from loguru import logger

def preprocess_for_white_on_colored(image):
    """Enhanced preprocessing specifically for white numbers on colored backgrounds"""
    
    # Convert to different color spaces
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        gray = image.copy()
    
    processed_variants = []
    
    # 1. Invert for white-on-dark text
    inverted = cv2.bitwise_not(gray)
    processed_variants.append(("inverted", inverted))
    
    # 2. High threshold to isolate white text
    _, high_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    processed_variants.append(("high_threshold", high_thresh))
    
    # 3. Otsu with morphological operations
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    processed_variants.append(("otsu_clean", otsu_clean))
    
    # 4. Adaptive thresholding with different parameters
    adaptive1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_variants.append(("adaptive_11_2", adaptive1))
    
    adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    processed_variants.append(("adaptive_15_5", adaptive2))
    
    # 5. Color-based processing if color image
    if len(image.shape) == 3:
        # Extract channels that highlight white text
        b, g, r = cv2.split(image)
        
        # Combine channels to enhance white text
        combined = cv2.addWeighted(cv2.addWeighted(r, 0.33, g, 0.33, 0), 1, b, 0.34, 0)
        _, combined_thresh = cv2.threshold(combined, 180, 255, cv2.THRESH_BINARY)
        processed_variants.append(("color_combined", combined_thresh))
        
        # Use saturation channel (white text should have low saturation)
        s_channel = hsv[:, :, 1]
        _, sat_thresh = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY_INV)
        processed_variants.append(("saturation_inv", sat_thresh))
    
    # 6. Edge-enhanced version
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    processed_variants.append(("edges", edges_dilated))
    
    return processed_variants

def test_fixed_ocr_on_position_bar():
    """Test improved OCR on the position bar image"""
    
    # Load the debug image
    debug_img_path = "debug_output/region_Original_75-87pct.png"
    
    if not Path(debug_img_path).exists():
        logger.error(f"Debug image not found: {debug_img_path}")
        logger.info("Run debug_position_detection.py first")
        return
    
    image = cv2.imread(debug_img_path)
    logger.info(f"Testing improved OCR on position bar image: {image.shape}")
    
    # Expected: 2-7-5-4-6-3-8-1 based on visual inspection
    expected_order = [2, 7, 5, 4, 6, 3, 8, 1]
    logger.info(f"Expected order: {expected_order}")
    
    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Get preprocessed variants
    variants = preprocess_for_white_on_colored(image)
    
    logger.info("\n🔍 TESTING IMPROVED OCR PREPROCESSING:")
    
    best_result = None
    best_score = 0
    
    for variant_name, processed_img in variants:
        logger.info(f"\n--- Testing {variant_name} ---")
        
        # Save variant for inspection
        debug_dir = Path("debug_output")
        variant_path = debug_dir / f"ocr_variant_{variant_name}.png"
        cv2.imwrite(str(variant_path), processed_img)
        
        try:
            # Try OCR on this variant
            results = reader.readtext(processed_img, paragraph=False, width_ths=0.3, height_ths=0.3)
            
            detected_numbers = []
            for bbox, text, confidence in results:
                text = text.strip()
                if text.isdigit() and 1 <= int(text) <= 8:
                    x_center = (bbox[0][0] + bbox[2][0]) / 2
                    detected_numbers.append((int(text), x_center, confidence))
            
            # Sort by x position (left to right)
            detected_numbers.sort(key=lambda x: x[1])
            
            if detected_numbers:
                numbers_only = [num for num, _, _ in detected_numbers]
                confidences = [conf for _, _, conf in detected_numbers]
                avg_conf = sum(confidences) / len(confidences)
                
                logger.info(f"  Detected: {numbers_only}")
                logger.info(f"  Avg confidence: {avg_conf:.3f}")
                
                # Score based on how many correct positions
                correct_positions = sum(1 for i, num in enumerate(numbers_only) 
                                      if i < len(expected_order) and num == expected_order[i])
                score = correct_positions / len(expected_order) * 100
                
                logger.info(f"  Accuracy: {score:.1f}% ({correct_positions}/{len(expected_order)})")
                
                if score > best_score:
                    best_score = score
                    best_result = (variant_name, numbers_only, avg_conf)
            else:
                logger.info("  No valid numbers detected")
                
        except Exception as e:
            logger.error(f"  OCR failed: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("🎯 BEST RESULT:")
    if best_result:
        variant_name, numbers, avg_conf = best_result
        logger.info(f"  Best method: {variant_name}")
        logger.info(f"  Detected: {numbers}")
        logger.info(f"  Expected: {expected_order}")
        logger.info(f"  Accuracy: {best_score:.1f}%")
        logger.info(f"  Confidence: {avg_conf:.3f}")
        
        if best_score >= 80:
            logger.info("🎉 EXCELLENT! This method should work for position detection!")
        elif best_score >= 60:
            logger.info("✅ GOOD! This method shows promise")
        else:
            logger.info("⚠️  Still needs improvement")
    else:
        logger.info("❌ No successful OCR detection found")
        logger.info("💡 May need to try different approaches or manual digit extraction")
    
    logger.info(f"\n🔍 Debug variants saved to: debug_output/ocr_variant_*.png")

def extract_individual_digits():
    """Extract individual digit boxes for inspection"""
    
    debug_img_path = "debug_output/region_Original_75-87pct.png"
    image = cv2.imread(debug_img_path)
    
    if image is None:
        logger.error("Could not load debug image")
        return
    
    logger.info("\n📦 EXTRACTING INDIVIDUAL DIGITS:")
    
    # The position bar appears to have 8 evenly spaced digits
    height, width = image.shape[:2]
    
    # Estimate digit positions (8 digits evenly spaced)
    digit_width = width // 8
    
    debug_dir = Path("debug_output")
    
    for i in range(8):
        x_start = i * digit_width
        x_end = min((i + 1) * digit_width, width)
        
        # Extract digit region
        digit_region = image[:, x_start:x_end]
        
        # Save for inspection
        digit_path = debug_dir / f"digit_{i+1}_position.png"
        cv2.imwrite(str(digit_path), digit_region)
        
        logger.info(f"  Digit position {i+1}: saved to {digit_path}")
    
    logger.info("🎯 Check individual digit images to see the actual numbers")

if __name__ == "__main__":
    logger.info("🔧 FIXING POSITION BAR OCR")
    logger.info("="*50)
    
    test_fixed_ocr_on_position_bar()
    extract_individual_digits()