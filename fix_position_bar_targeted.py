#!/usr/bin/env python3
"""
Targeted fix for position bar OCR - focuses on the red banner area with colored numbers
"""

import cv2
import numpy as np
import easyocr
import os

def extract_position_bar_region(frame):
    """Extract the red banner region containing position numbers"""
    height, width = frame.shape[:2]
    
    # Focus on bottom 15% where we found the position bar
    start_y = int(height * 0.85)
    end_y = height
    
    # Extract the region
    position_region = frame[start_y:end_y, 0:width]
    
    return position_region, start_y

def detect_colored_rectangles(region):
    """Detect the colored rectangles containing position numbers"""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Create mask for colored areas (exclude black/white/gray)
    # Focus on saturation to find colored rectangles
    saturation = hsv[:, :, 1]
    mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours of colored areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and aspect ratio
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Filter for rectangle-like shapes that could contain digits
        if area > 200 and 0.3 < aspect_ratio < 3.0 and h > 10:
            rectangles.append((x, y, w, h))
    
    # Sort rectangles left to right
    rectangles.sort(key=lambda r: r[0])
    
    return rectangles

def preprocess_digit_region(region):
    """Advanced preprocessing for white text on colored background"""
    # Try multiple approaches
    results = []
    
    # Method 1: Extract white text by color range
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # White text detection
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    results.append(('white_extraction', white_mask))
    
    # Method 2: Invert colors and threshold
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(('inverted_otsu', thresh))
    
    # Method 3: High contrast enhancement
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge back and convert to grayscale
    enhanced = cv2.merge([l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    
    # Threshold for white text
    _, enhanced_thresh = cv2.threshold(enhanced_gray, 200, 255, cv2.THRESH_BINARY)
    results.append(('enhanced_clahe', enhanced_thresh))
    
    return results

def main():
    # Load the test frame
    debug_dir = "debug_output"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load the bottom 15% region we identified
    region_path = f"{debug_dir}/region_Bottom_15pct.png"
    if not os.path.exists(region_path):
        print(f"Error: {region_path} not found")
        return
    
    region = cv2.imread(region_path)
    print(f"Loaded region: {region.shape}")
    
    # Detect colored rectangles
    rectangles = detect_colored_rectangles(region)
    print(f"Found {len(rectangles)} potential digit rectangles")
    
    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Process each rectangle
    detected_positions = []
    
    for i, (x, y, w, h) in enumerate(rectangles):
        # Extract digit region with padding
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(region.shape[1], x + w + padding)
        y2 = min(region.shape[0], y + h + padding)
        
        digit_region = region[y1:y2, x1:x2]
        
        if digit_region.size == 0:
            continue
            
        # Save original digit region
        cv2.imwrite(f"{debug_dir}/targeted_digit_{i+1}_original.png", digit_region)
        
        # Try different preprocessing methods
        preprocessing_results = preprocess_digit_region(digit_region)
        
        best_result = None
        best_confidence = 0
        
        for method_name, processed_image in preprocessing_results:
            # Save preprocessed image
            cv2.imwrite(f"{debug_dir}/targeted_digit_{i+1}_{method_name}.png", processed_image)
            
            try:
                # OCR on preprocessed image
                results = reader.readtext(processed_image, allowlist='0123456789')
                
                for (bbox, text, confidence) in results:
                    # Filter for single digits
                    if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                        best_result = text
                        best_confidence = confidence
                        print(f"  Digit {i+1} ({method_name}): '{text}' (confidence: {confidence:.2f})")
            
            except Exception as e:
                print(f"  Error processing digit {i+1} with {method_name}: {e}")
        
        if best_result:
            detected_positions.append(best_result)
        else:
            detected_positions.append('?')
    
    print(f"\nDetected position sequence: {detected_positions}")
    print(f"Expected sequence: ['2', '7', '5', '4', '6', '3', '8', '1']")
    
    # Draw rectangles on original for visualization
    result_image = region.copy()
    for i, (x, y, w, h) in enumerate(rectangles):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if i < len(detected_positions):
            cv2.putText(result_image, detected_positions[i], (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite(f"{debug_dir}/targeted_rectangles_detected.png", result_image)
    print(f"Saved visualization to {debug_dir}/targeted_rectangles_detected.png")

if __name__ == "__main__":
    main()