#!/usr/bin/env python3
"""
Manual position extraction - define approximate locations of each position number
"""

import cv2
import numpy as np
import easyocr
import os

def extract_manual_digit_regions(region):
    """Manually define approximate locations of position digits"""
    height, width = region.shape[:2]
    
    # Based on visual inspection, the position numbers are spread across the width
    # in the red banner area. Let's divide the width into 8 segments
    digit_width = width // 8
    
    digit_regions = []
    for i in range(8):
        x_start = i * digit_width
        x_end = (i + 1) * digit_width
        
        # Focus on the middle portion of height where numbers are likely
        y_start = height // 4
        y_end = height - 5
        
        digit_regions.append((x_start, y_start, x_end - x_start, y_end - y_start))
    
    return digit_regions

def preprocess_for_white_text(region):
    """Specialized preprocessing for white text on colored background"""
    # Convert to different color spaces and try to isolate white text
    
    # Method 1: HSV-based white detection
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Define range for white colors (high value, low saturation)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    return white_mask

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
    
    # Get manual digit regions
    digit_regions = extract_manual_digit_regions(region)
    print(f"Created {len(digit_regions)} manual digit regions")
    
    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    detected_positions = []
    
    for i, (x, y, w, h) in enumerate(digit_regions):
        print(f"\nProcessing region {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        # Extract the region
        digit_region = region[y:y+h, x:x+w]
        
        if digit_region.size == 0:
            print(f"  Empty region {i+1}")
            continue
        
        # Save original region
        cv2.imwrite(f"{debug_dir}/manual_digit_{i+1}_original.png", digit_region)
        
        # Try different preprocessing approaches
        approaches = []
        
        # Approach 1: White text extraction
        white_mask = preprocess_for_white_text(digit_region)
        approaches.append(('white_mask', white_mask))
        
        # Approach 2: Grayscale with high threshold
        gray = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
        _, high_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        approaches.append(('high_threshold', high_thresh))
        
        # Approach 3: Inverted with Otsu
        inverted = cv2.bitwise_not(gray)
        _, otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        approaches.append(('inverted_otsu', otsu))
        
        # Approach 4: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, enhanced_thresh = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
        approaches.append(('enhanced', enhanced_thresh))
        
        best_result = None
        best_confidence = 0
        
        for approach_name, processed in approaches:
            # Save processed image
            cv2.imwrite(f"{debug_dir}/manual_digit_{i+1}_{approach_name}.png", processed)
            
            try:
                # Try OCR
                results = reader.readtext(processed, allowlist='0123456789', width_ths=0.1, height_ths=0.1)
                
                for (bbox, text, confidence) in results:
                    if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                        best_result = text
                        best_confidence = confidence
                        print(f"  Found '{text}' with confidence {confidence:.2f} using {approach_name}")
                
                # Also try with different OCR parameters
                results2 = reader.readtext(processed, allowlist='12345678', paragraph=False)
                for (bbox, text, confidence) in results2:
                    if len(text) == 1 and text.isdigit() and confidence > best_confidence:
                        best_result = text
                        best_confidence = confidence
                        print(f"  Found '{text}' with confidence {confidence:.2f} using {approach_name} (alt params)")
                        
            except Exception as e:
                print(f"  Error with {approach_name}: {e}")
        
        if best_result:
            detected_positions.append(best_result)
            print(f"  Best result for region {i+1}: '{best_result}' (confidence: {best_confidence:.2f})")
        else:
            detected_positions.append('?')
            print(f"  No valid digit found for region {i+1}")
    
    print(f"\n=== RESULTS ===")
    print(f"Detected sequence: {detected_positions}")
    print(f"Expected sequence: ['2', '7', '5', '4', '6', '3', '8', '1']")
    
    # Calculate accuracy
    expected = ['2', '7', '5', '4', '6', '3', '8', '1']
    correct = sum(1 for i, (det, exp) in enumerate(zip(detected_positions, expected)) if det == exp)
    accuracy = correct / len(expected) * 100
    print(f"Accuracy: {correct}/{len(expected)} = {accuracy:.1f}%")
    
    # Create visualization
    result_image = region.copy()
    for i, (x, y, w, h) in enumerate(digit_regions):
        color = (0, 255, 0) if i < len(detected_positions) and detected_positions[i] != '?' else (0, 0, 255)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        if i < len(detected_positions):
            cv2.putText(result_image, f"{detected_positions[i]}", (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imwrite(f"{debug_dir}/manual_extraction_result.png", result_image)
    print(f"Saved visualization to {debug_dir}/manual_extraction_result.png")

if __name__ == "__main__":
    main()