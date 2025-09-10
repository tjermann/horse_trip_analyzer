#!/usr/bin/env python3
"""
Final approach: Use visual inspection to manually locate position rectangles
"""

import cv2
import numpy as np
import easyocr
import os

def get_manual_position_locations(region):
    """Based on visual inspection, define approximate locations of each position rectangle"""
    height, width = region.shape[:2]
    
    # From visual inspection of the red banner, the 8 colored rectangles are approximately at:
    # These are rough estimates based on the visible positions in the image
    positions = [
        # (x_center_approx, y_center_approx, width_est, height_est)
        (65, 15, 25, 15),   # Position 1 (leftmost)
        (125, 15, 25, 15),  # Position 2
        (185, 15, 25, 15),  # Position 3  
        (245, 15, 25, 15),  # Position 4
        (305, 15, 25, 15),  # Position 5
        (365, 15, 25, 15),  # Position 6
        (425, 15, 25, 15),  # Position 7
        (485, 15, 25, 15),  # Position 8 (rightmost)
    ]
    
    rectangles = []
    for x_center, y_center, w, h in positions:
        x = x_center - w//2
        y = y_center - h//2
        
        # Ensure bounds are within image
        x = max(0, min(x, width - w))
        y = max(0, min(y, height - h))
        
        rectangles.append((x, y, w, h))
    
    return rectangles

def enhanced_digit_extraction(region, debug_name, reader):
    """Enhanced digit extraction with multiple preprocessing techniques"""
    if region.size == 0:
        return None, 0
    
    debug_dir = "debug_output"
    
    # Resize if too small
    original_region = region.copy()
    if region.shape[0] < 30 or region.shape[1] < 30:
        scale = max(2, 30 // min(region.shape[0], region.shape[1]))
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(f"{debug_dir}/{debug_name}_original.png", original_region)
    cv2.imwrite(f"{debug_dir}/{debug_name}_resized.png", region)
    
    best_result = None
    best_confidence = 0
    best_method = None
    
    # Method 1: Direct OCR on color image
    try:
        results = reader.readtext(region, allowlist='12345678', paragraph=False, width_ths=0.1, height_ths=0.1)
        for (bbox, text, confidence) in results:
            if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                best_result = text
                best_confidence = confidence
                best_method = "direct_color"
    except:
        pass
    
    # Method 2: White text extraction (high threshold)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Try multiple thresholds for white text
    for thresh_val in [180, 200, 220]:
        _, white_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"{debug_dir}/{debug_name}_white_{thresh_val}.png", white_mask)
        
        try:
            results = reader.readtext(white_mask, allowlist='12345678', paragraph=False)
            for (bbox, text, confidence) in results:
                if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                    best_result = text
                    best_confidence = confidence
                    best_method = f"white_{thresh_val}"
        except:
            pass
    
    # Method 3: Inverted with different thresholding
    inverted = cv2.bitwise_not(gray)
    cv2.imwrite(f"{debug_dir}/{debug_name}_inverted.png", inverted)
    
    # Otsu thresholding
    _, otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{debug_dir}/{debug_name}_otsu.png", otsu)
    
    try:
        results = reader.readtext(otsu, allowlist='12345678', paragraph=False)
        for (bbox, text, confidence) in results:
            if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                best_result = text
                best_confidence = confidence
                best_method = "otsu"
    except:
        pass
    
    # Method 4: Adaptive thresholding on inverted
    adaptive = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(f"{debug_dir}/{debug_name}_adaptive.png", adaptive)
    
    try:
        results = reader.readtext(adaptive, allowlist='12345678', paragraph=False)
        for (bbox, text, confidence) in results:
            if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                best_result = text
                best_confidence = confidence
                best_method = "adaptive"
    except:
        pass
    
    # Method 5: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, enhanced_thresh = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{debug_dir}/{debug_name}_clahe.png", enhanced_thresh)
    
    try:
        results = reader.readtext(enhanced_thresh, allowlist='12345678', paragraph=False)
        for (bbox, text, confidence) in results:
            if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                best_result = text
                best_confidence = confidence
                best_method = "clahe"
    except:
        pass
    
    print(f"  Best result: '{best_result}' (confidence: {best_confidence:.2f}, method: {best_method})")
    return best_result, best_confidence

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
    
    # Get manual position rectangles
    rectangles = get_manual_position_locations(region)
    print(f"Using {len(rectangles)} manual position locations")
    
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
        
        # Enhanced digit extraction
        result, confidence = enhanced_digit_extraction(rect_region, f"final_pos_{i+1}", reader)
        
        if result:
            detected_positions.append(result)
        else:
            detected_positions.append('?')
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Detected sequence: {detected_positions}")
    print(f"Expected sequence: ['2', '7', '5', '4', '6', '3', '8', '1']")
    
    # Calculate accuracy
    expected = ['2', '7', '5', '4', '6', '3', '8', '1']
    correct = sum(1 for det, exp in zip(detected_positions, expected) if det == exp)
    accuracy = correct / len(expected) * 100
    print(f"Accuracy: {correct}/{len(expected)} = {accuracy:.1f}%")
    
    # Create visualization
    result_image = region.copy()
    for i, (x, y, w, h) in enumerate(rectangles):
        color = (0, 255, 0) if detected_positions[i] != '?' else (0, 0, 255)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_image, f"{detected_positions[i]}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Add expected value in smaller text
        cv2.putText(result_image, f"({expected[i]})", (x, y + h + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(f"{debug_dir}/final_extraction_result.png", result_image)
    print(f"Saved visualization to {debug_dir}/final_extraction_result.png")

if __name__ == "__main__":
    main()