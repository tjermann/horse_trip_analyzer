#!/usr/bin/env python3
"""
Debug position detection to understand why results are still inaccurate
"""

import cv2
import numpy as np
from pathlib import Path
from src.hybrid_position_detector import HybridPositionDetector
from src.known_results import KnownResults
from loguru import logger
import matplotlib.pyplot as plt

def debug_position_detection():
    """Debug what's happening with position detection"""
    
    race_code = "194367"
    video_path = f"data/videos/race_{race_code}.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return
    
    # Known results for comparison
    known_results = KnownResults.get_known_finish(race_code)
    logger.info(f"🎯 KNOWN RESULTS: {known_results} (2-7-5-4-6)")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Test frame near the finish (90% through)
    finish_frame = int(total_frames * 0.9)
    cap.set(cv2.CAP_PROP_POS_FRAMES, finish_frame)
    ret, frame = cap.read()
    
    if not ret:
        logger.error("Could not read finish frame")
        return
    
    logger.info(f"🔍 DEBUGGING FRAME {finish_frame} (near finish)")
    
    # Initialize detector
    detector = HybridPositionDetector(num_horses=8)
    
    # Debug each component separately
    height, width = frame.shape[:2]
    logger.info(f"Frame dimensions: {width}x{height}")
    
    # 1. Check position bar extraction regions
    logger.info("\n📍 POSITION BAR REGIONS:")
    regions = {
        "Original (75-87%)": (int(height * 0.75), int(height * 0.87)),
        "Bottom 15%": (int(height * 0.85), height),
        "Bottom 10%": (int(height * 0.90), height),
        "Top 15%": (0, int(height * 0.15))
    }
    
    x1 = int(width * 0.10)
    x2 = int(width * 0.95)
    
    for name, (y1, y2) in regions.items():
        region = frame[y1:y2, x1:x2]
        logger.info(f"  {name}: {region.shape[1]}x{region.shape[0]}px")
        
        # Save region for visual inspection
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(f"debug_output/region_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}.png", region)
    
    # 2. Test OCR on each region
    logger.info("\n🔤 OCR RESULTS BY REGION:")
    
    ocr_processor = detector.ocr_processor
    
    for name, (y1, y2) in regions.items():
        region = frame[y1:y2, x1:x2]
        try:
            ocr_results = ocr_processor.extract_numbers_with_confidence(region, 8)
            logger.info(f"  {name}: {ocr_results}")
        except Exception as e:
            logger.error(f"  {name}: OCR failed - {e}")
    
    # 3. Test CNN on main region  
    logger.info("\n🧠 CNN RESULTS:")
    main_region = frame[int(height * 0.75):int(height * 0.87), x1:x2]
    
    try:
        cnn_results = detector.cnn_reader.predict_with_confidence(main_region)
        logger.info(f"  CNN predictions: {cnn_results[:10]}...")  # First 10
        logger.info(f"  CNN trained: {detector.cnn_reader.is_trained}")
    except Exception as e:
        logger.error(f"  CNN failed: {e}")
    
    # 4. Test full hybrid detection
    logger.info("\n⚖️  HYBRID DETECTION:")
    try:
        positions = detector.detect_positions(frame, [], finish_frame, 2.0)
        logger.info(f"  Final positions: {positions}")
        
        # Convert to finish order
        finish_order = {}
        for horse_id, (position, confidence) in positions.items():
            finish_order[position] = horse_id
            
        logger.info(f"  Finish order by position:")
        for pos in sorted(finish_order.keys()):
            horse = finish_order[pos]
            conf = positions[horse][1]
            logger.info(f"    Position {pos}: Horse #{horse} (confidence: {conf:.3f})")
            
    except Exception as e:
        logger.error(f"  Hybrid detection failed: {e}")
    
    # 5. Compare to known results
    logger.info(f"\n📊 COMPARISON TO KNOWN RESULTS:")
    logger.info(f"  Known: 2-7-5-4-6 (positions 1-2-3-4-5)")
    if 'positions' in locals():
        predicted_order = [finish_order.get(i, 'missing') for i in range(1, 6)]
        logger.info(f"  Predicted: {'-'.join(map(str, predicted_order))}")
        
        # Check accuracy
        correct = 0
        for i, known_horse in enumerate(known_results, 1):
            predicted_horse = finish_order.get(i, None)
            if predicted_horse == known_horse:
                correct += 1
                logger.info(f"    ✅ Position {i}: Correct (Horse #{known_horse})")
            else:
                logger.info(f"    ❌ Position {i}: Expected #{known_horse}, got #{predicted_horse}")
        
        accuracy = correct / len(known_results) * 100
        logger.info(f"  🎯 Position Accuracy: {accuracy:.1f}% ({correct}/{len(known_results)})")
    
    cap.release()
    
    logger.info(f"\n🔍 Debug images saved to: debug_output/")
    logger.info(f"💡 RECOMMENDATIONS:")
    logger.info(f"   1. Check debug images to see if position bar is visible")
    logger.info(f"   2. Verify OCR is reading the correct numbers")
    logger.info(f"   3. Ensure position bar format matches expectation")
    logger.info(f"   4. Consider the position bar might be in a different location")

def analyze_position_bar_format():
    """Analyze the actual position bar format in the video"""
    
    logger.info("\n📋 POSITION BAR FORMAT ANALYSIS")
    logger.info("Expected format: Left-to-right order shows 1st-2nd-3rd-...-8th place")
    logger.info("Known finish: 2-7-5-4-6 means:")
    logger.info("  Position bar should show: [2][7][5][4][6][X][X][X] (left to right)")
    logger.info("  Where X are horses in 6th, 7th, 8th positions")

if __name__ == "__main__":
    debug_position_detection()
    analyze_position_bar_format()