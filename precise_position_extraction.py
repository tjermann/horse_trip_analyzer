#!/usr/bin/env python3
"""
Precise position extraction - target the actual colored rectangles containing numbers
"""

import cv2
import numpy as np
import easyocr
import os

def find_position_rectangles(region):
    """Find the colored rectangles containing position numbers in the red banner"""
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Create mask for the red banner area (exclude the green/track area)
    # Red hue range in HSV
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 120])  
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Now find colored rectangles within the red area
    # Look for non-red colored areas (the position number backgrounds)
    
    # Create mask for colored areas that are NOT red and NOT green (track)
    # Focus on areas with moderate to high saturation but different hue
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    
    # Create mask for areas that could contain colored position numbers
    colored_mask = cv2.bitwise_and(
        cv2.threshold(saturation, 80, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(value, 100, 255, cv2.THRESH_BINARY)[1]
    )
    
    # Combine with red area mask to focus only on colored areas within red banner
    position_mask = cv2.bitwise_and(colored_mask, red_mask)
    
    # Find contours of potential position rectangles
    contours, _ = cv2.findContours(position_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Filter for small rectangular shapes typical of position numbers
        if 50 < area < 2000 and 0.4 < aspect_ratio < 2.5 and h > 8 and w > 8:
            rectangles.append((x, y, w, h))
    
    # Sort rectangles left to right
    rectangles.sort(key=lambda r: r[0])
    
    # Save debug masks
    debug_dir = "debug_output"
    cv2.imwrite(f"{debug_dir}/red_mask.png", red_mask)
    cv2.imwrite(f"{debug_dir}/colored_mask.png", colored_mask)
    cv2.imwrite(f"{debug_dir}/position_mask.png", position_mask)
    
    return rectangles

def extract_number_from_region(region, reader):
    """Extract number using multiple OCR approaches"""
    if region.size == 0:
        return None, 0
    
    # Resize region if too small
    if region.shape[0] < 20 or region.shape[1] < 20:
        scale = max(2, 20 // min(region.shape[0], region.shape[1]))
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    best_result = None
    best_confidence = 0
    
    # Method 1: Direct OCR on original
    try:
        results = reader.readtext(region, allowlist='12345678', paragraph=False)
        for (bbox, text, confidence) in results:
            if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                best_result = text
                best_confidence = confidence
    except:
        pass
    
    # Method 2: Grayscale with high threshold for white text
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    try:
        results = reader.readtext(white_thresh, allowlist='12345678', paragraph=False)
        for (bbox, text, confidence) in results:
            if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                best_result = text
                best_confidence = confidence
    except:
        pass
    
    # Method 3: Inverted with Otsu
    inverted = cv2.bitwise_not(gray)
    _, otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    try:
        results = reader.readtext(otsu, allowlist='12345678', paragraph=False)
        for (bbox, text, confidence) in results:
            if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                best_result = text
                best_confidence = confidence
    except:
        pass
    
    # Method 4: Enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, enhanced_thresh = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
    
    try:
        results = reader.readtext(enhanced_thresh, allowlist='12345678', paragraph=False)
        for (bbox, text, confidence) in results:
            if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                best_result = text
                best_confidence = confidence
    except:
        pass
    
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
    
    # Find position rectangles
    rectangles = find_position_rectangles(region)
    print(f"Found {len(rectangles)} potential position rectangles")
    
    if len(rectangles) == 0:
        print("No rectangles found! Check debug masks.")
        return
    
    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    detected_positions = []
    
    for i, (x, y, w, h) in enumerate(rectangles):
        print(f"\nProcessing rectangle {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        # Extract the rectangle with some padding
        padding = 3
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(region.shape[1], x + w + padding)
        y2 = min(region.shape[0], y + h + padding)
        
        rect_region = region[y1:y2, x1:x2]
        
        # Save original rectangle
        cv2.imwrite(f"{debug_dir}/precise_rect_{i+1}_original.png", rect_region)
        
        # Extract number
        result, confidence = extract_number_from_region(rect_region, reader)
        
        if result:
            detected_positions.append(result)
            print(f"  Found: '{result}' (confidence: {confidence:.2f})")
        else:
            detected_positions.append('?')
            print(f"  No digit detected")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Detected sequence: {detected_positions}")
    print(f"Expected sequence: ['2', '7', '5', '4', '6', '3', '8', '1']")
    
    # Calculate accuracy
    expected = ['2', '7', '5', '4', '6', '3', '8', '1']
    if len(detected_positions) == len(expected):
        correct = sum(1 for det, exp in zip(detected_positions, expected) if det == exp)
        accuracy = correct / len(expected) * 100
        print(f"Accuracy: {correct}/{len(expected)} = {accuracy:.1f}%")
    else:
        print(f"Length mismatch: detected {len(detected_positions)}, expected {len(expected)}")
    
    # Create visualization
    result_image = region.copy()
    for i, (x, y, w, h) in enumerate(rectangles):
        color = (0, 255, 0) if i < len(detected_positions) and detected_positions[i] != '?' else (0, 0, 255)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        if i < len(detected_positions):
            cv2.putText(result_image, f"{detected_positions[i]}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imwrite(f"{debug_dir}/precise_extraction_result.png", result_image)
    print(f"Saved visualization to {debug_dir}/precise_extraction_result.png")

if __name__ == "__main__":
    main()