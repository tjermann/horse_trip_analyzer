#!/usr/bin/env python3
"""
Final precise extraction - target the exact colored position rectangles
"""

import cv2
import numpy as np
import easyocr
import os

def get_final_precise_locations(region):
    """Final precise locations based on visual inspection of the colored rectangles"""
    height, width = region.shape[:2]
    
    # From the latest visualization, I can see the colored rectangles are positioned higher
    # The colored rectangles appear to be around y=15-25 (higher than my previous estimate)
    # And the spacing looks more even across the width
    
    positions = [
        # (x, y, w, h) - final precise positioning
        (50, 18, 22, 16),   # Position 1 - "2" 
        (105, 18, 22, 16),  # Position 2 - "7"
        (160, 18, 22, 16),  # Position 3 - "5" 
        (215, 18, 22, 16),  # Position 4 - "4"
        (270, 18, 22, 16),  # Position 5 - "6"
        (325, 18, 22, 16),  # Position 6 - "3"
        (380, 18, 22, 16),  # Position 7 - "8"
        (435, 18, 22, 16),  # Position 8 - "1"
    ]
    
    # Ensure all positions are within bounds
    rectangles = []
    for x, y, w, h in positions:
        x = max(0, min(x, width - w))
        y = max(0, min(y, height - h))
        rectangles.append((x, y, w, h))
    
    return rectangles

def ultimate_digit_extraction(region, debug_name, reader):
    """Ultimate digit extraction with maximum preprocessing techniques"""
    if region.size == 0:
        return None, 0, None
    
    debug_dir = "debug_output"
    cv2.imwrite(f"{debug_dir}/{debug_name}_final_original.png", region)
    
    best_result = None
    best_confidence = 0
    best_method = None
    
    # Massive upscaling first
    scale = 10  # 10x scaling for tiny digits
    upscaled = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Apply sharpening filter to upscaled image
    kernel_sharpen = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel_sharpen)
    cv2.imwrite(f"{debug_dir}/{debug_name}_sharpened.png", sharpened)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Direct on sharpened upscaled
    approaches = [
        ('sharpened_color', sharpened),
        ('sharpened_gray', gray),
    ]
    
    # Method 2: White text extraction with multiple thresholds
    for thresh in [180, 200, 220, 240]:
        _, white_thresh = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"{debug_dir}/{debug_name}_white_{thresh}.png", white_thresh)
        approaches.append((f'white_{thresh}', white_thresh))
    
    # Method 3: Inverted with different techniques
    inverted = cv2.bitwise_not(gray)
    
    # Otsu on inverted
    _, otsu_inv = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{debug_dir}/{debug_name}_otsu_inv.png", otsu_inv)
    approaches.append(('otsu_inv', otsu_inv))
    
    # Adaptive thresholding on inverted
    adaptive_inv = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(f"{debug_dir}/{debug_name}_adaptive_inv.png", adaptive_inv)
    approaches.append(('adaptive_inv', adaptive_inv))
    
    # Method 4: Morphological operations
    kernel = np.ones((3,3), np.uint8)
    for approach_name, img in approaches.copy():
        if len(img.shape) == 2:  # Only for grayscale images
            # Closing to fill gaps
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(f"{debug_dir}/{debug_name}_{approach_name}_closed.png", closed)
            approaches.append((f'{approach_name}_closed', closed))
    
    # Try OCR on all approaches
    for method_name, processed_image in approaches:
        try:
            # Very permissive OCR settings
            results = reader.readtext(
                processed_image, 
                allowlist='12345678', 
                paragraph=False,
                width_ths=0.001,  # Ultra-low thresholds
                height_ths=0.001,
                detail=1
            )
            
            for (bbox, text, confidence) in results:
                if len(text) == 1 and text.isdigit():
                    if confidence > best_confidence:
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
    
    # Get final precise rectangles
    rectangles = get_final_precise_locations(region)
    print(f"Using {len(rectangles)} final precise locations")
    
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
        
        # Ultimate digit extraction
        result, confidence, method = ultimate_digit_extraction(rect_region, f"final_pos_{i+1}", reader)
        
        if result:
            detected_positions.append(result)
            print(f"  FOUND: '{result}' (confidence: {confidence:.3f}, method: {method})")
        else:
            detected_positions.append('?')
            print(f"  No digit detected")
    
    print(f"\n{'='*50}")
    print(f"FINAL OCR RESULTS")
    print(f"{'='*50}")
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
        status = "✓ CORRECT" if det == exp else "✗ WRONG"
        print(f"  Position {i+1}: detected='{det}', expected='{exp}' {status}")
    
    if accuracy > 50:
        print(f"\n🎉 SUCCESS: {accuracy:.1f}% accuracy achieved!")
        print(f"OCR is now working on the position bar!")
    else:
        print(f"\n⚠️  Still need improvement: {accuracy:.1f}% accuracy")
        print(f"Check debug images in {debug_dir}/ for analysis")
    
    # Create visualization
    result_image = region.copy()
    for i, (x, y, w, h) in enumerate(rectangles):
        color = (0, 255, 0) if detected_positions[i] != '?' and detected_positions[i] == expected[i] else (0, 0, 255)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_image, f"{detected_positions[i]}", (x-2, y-2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Add expected value
        cv2.putText(result_image, f"({expected[i]})", (x-2, y+h+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    cv2.imwrite(f"{debug_dir}/final_precise_result.png", result_image)
    print(f"Visualization saved to {debug_dir}/final_precise_result.png")

if __name__ == "__main__":
    main()