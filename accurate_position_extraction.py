#!/usr/bin/env python3
"""
Accurate position extraction - precisely target the colored position rectangles
"""

import cv2
import numpy as np
import easyocr
import os

def get_accurate_position_locations(region):
    """Based on careful visual inspection, define precise locations of colored position rectangles"""
    height, width = region.shape[:2]
    
    # From the visualization, I can see the colored rectangles are in a specific row
    # The rectangles appear to be around y=35-50 and are smaller than my previous estimates
    # Looking at the visualization more carefully:
    
    positions = [
        # (x, y, w, h) - much more precise based on visual inspection
        (45, 35, 20, 12),   # Position 1 - "2" (leftmost colored rectangle)
        (100, 35, 20, 12),  # Position 2 - "7" 
        (155, 35, 20, 12),  # Position 3 - "5"
        (210, 35, 20, 12),  # Position 4 - "4"
        (265, 35, 20, 12),  # Position 5 - "6"
        (320, 35, 20, 12),  # Position 6 - "3"
        (375, 35, 20, 12),  # Position 7 - "8"
        (430, 35, 20, 12),  # Position 8 - "1" (rightmost)
    ]
    
    # Ensure all positions are within bounds
    rectangles = []
    for x, y, w, h in positions:
        x = max(0, min(x, width - w))
        y = max(0, min(y, height - h))
        rectangles.append((x, y, w, h))
    
    return rectangles

def template_match_digits(region, debug_name):
    """Try template matching approach for small digit regions"""
    debug_dir = "debug_output"
    
    # Create templates for digits 1-8 (simple approach)
    # This is a fallback if OCR continues to fail
    
    # For now, let's focus on better OCR preprocessing
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Try different approaches specifically for very small regions
    results = []
    
    # Approach 1: Massive upscaling before OCR
    scale = 8  # 8x scaling
    upscaled = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{debug_dir}/{debug_name}_upscaled.png", upscaled)
    results.append(('upscaled', upscaled))
    
    # Approach 2: Upscaled grayscale with white text extraction
    upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, upscaled_white = cv2.threshold(upscaled_gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{debug_dir}/{debug_name}_upscaled_white.png", upscaled_white)
    results.append(('upscaled_white', upscaled_white))
    
    # Approach 3: Upscaled with morphological operations
    kernel = np.ones((2,2), np.uint8)
    upscaled_morph = cv2.morphologyEx(upscaled_white, cv2.MORPH_CLOSE, kernel)
    upscaled_morph = cv2.morphologyEx(upscaled_morph, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{debug_dir}/{debug_name}_upscaled_morph.png", upscaled_morph)
    results.append(('upscaled_morph', upscaled_morph))
    
    return results

def enhanced_small_digit_ocr(region, debug_name, reader):
    """Specialized OCR for very small digit regions"""
    if region.size == 0:
        return None, 0
    
    debug_dir = "debug_output"
    cv2.imwrite(f"{debug_dir}/{debug_name}_tiny_original.png", region)
    
    best_result = None
    best_confidence = 0
    best_method = None
    
    # Get different preprocessing approaches
    preprocessing_results = template_match_digits(region, debug_name)
    
    # Try OCR on each preprocessed version
    for method_name, processed_image in preprocessing_results:
        try:
            # Try with very permissive settings for small text
            results = reader.readtext(
                processed_image, 
                allowlist='12345678', 
                paragraph=False,
                width_ths=0.01,  # Very low width threshold
                height_ths=0.01, # Very low height threshold
                detail=1
            )
            
            for (bbox, text, confidence) in results:
                if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                    best_result = text
                    best_confidence = confidence
                    best_method = method_name
                    print(f"    {method_name}: '{text}' (conf: {confidence:.3f})")
        except Exception as e:
            print(f"    Error with {method_name}: {e}")
    
    return best_result, best_confidence, best_method

def main():
    debug_dir = "debug_output"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load the bottom 15% region
    region_path = f"{debug_dir}/region_Bottom_15pct.png"
    if not os.path.exists(region_path):
        print(f"Error: {region_path} not found")
        return
    
    region = cv2.imread(region_path)
    print(f"Loaded region: {region.shape}")
    
    # Get accurate position rectangles
    rectangles = get_accurate_position_locations(region)
    print(f"Using {len(rectangles)} accurate position locations")
    
    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    detected_positions = []
    
    for i, (x, y, w, h) in enumerate(rectangles):
        print(f"\nProcessing position {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        # Extract rectangle region
        rect_region = region[y:y+h, x:x+w]
        
        if rect_region.size == 0:
            print(f"  Empty region for position {i+1}")
            detected_positions.append('?')
            continue
        
        # Enhanced small digit OCR
        result, confidence, method = enhanced_small_digit_ocr(rect_region, f"accurate_pos_{i+1}", reader)
        
        if result:
            detected_positions.append(result)
            print(f"  FOUND: '{result}' (confidence: {confidence:.3f}, method: {method})")
        else:
            detected_positions.append('?')
            print(f"  No digit detected")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Detected sequence: {detected_positions}")
    print(f"Expected sequence: ['2', '7', '5', '4', '6', '3', '8', '1']")
    
    # Calculate accuracy
    expected = ['2', '7', '5', '4', '6', '3', '8', '1']
    correct = sum(1 for det, exp in zip(detected_positions, expected) if det == exp)
    accuracy = correct / len(expected) * 100
    print(f"Accuracy: {correct}/{len(expected)} = {accuracy:.1f}%")
    
    # Detailed comparison
    print(f"\nDetailed comparison:")
    for i, (det, exp) in enumerate(zip(detected_positions, expected)):
        status = "✓" if det == exp else "✗"
        print(f"  Position {i+1}: detected='{det}', expected='{exp}' {status}")
    
    # Create visualization
    result_image = region.copy()
    for i, (x, y, w, h) in enumerate(rectangles):
        color = (0, 255, 0) if detected_positions[i] != '?' and detected_positions[i] == expected[i] else (0, 0, 255)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_image, f"{detected_positions[i]}", (x-5, y-2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Add expected value
        cv2.putText(result_image, f"({expected[i]})", (x-5, y+h+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    cv2.imwrite(f"{debug_dir}/accurate_extraction_result.png", result_image)
    print(f"Saved visualization to {debug_dir}/accurate_extraction_result.png")

if __name__ == "__main__":
    main()