#!/usr/bin/env python3
"""
Validate the trained CNN system performance on race_194367
"""

import cv2
import json
from src.video_processor import VideoProcessor
from src.known_results import KnownResults  
from loguru import logger
from pathlib import Path

def quick_validation_test():
    """Run a quick validation test on a few frames"""
    
    race_code = "194367"
    video_path = f"data/videos/race_{race_code}.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return
    
    logger.info("🚀 Starting Trained CNN Validation Test")
    logger.info("=" * 50)
    
    # Initialize hybrid detector directly
    from src.hybrid_position_detector import HybridPositionDetector
    hybrid_detector = HybridPositionDetector(num_horses=8)
    
    # Get known results for comparison
    known_results = KnownResults.get_known_finish(race_code)
    logger.info(f"Known results for race {race_code}: {known_results}")
    
    # Process just a subset of frames for validation
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Test frames from different parts of the race
    test_frames = [
        int(total_frames * 0.3),  # 30% through
        int(total_frames * 0.5),  # 50% through
        int(total_frames * 0.7),  # 70% through
        int(total_frames * 0.9),  # 90% through (near finish)
    ]
    
    results = []
    
    for i, frame_num in enumerate(test_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        logger.info(f"Testing frame {frame_num} ({(frame_num/total_frames)*100:.1f}% through race)")
        
        # Test hybrid detection
        positions = hybrid_detector.detect_positions(
            frame, [], frame_num, 2.0
        )
        
        if positions:
            logger.info(f"  Detected positions: {positions}")
            
            # Check for duplicates
            position_values = [pos for pos, conf in positions.values()]
            duplicates = len(position_values) != len(set(position_values))
            
            logger.info(f"  Unique positions: {not duplicates}")
            
            # Calculate confidence stats
            confidences = [conf for pos, conf in positions.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            logger.info(f"  Average confidence: {avg_confidence:.3f}")
            
            results.append({
                'frame': frame_num,
                'positions': positions,
                'unique_positions': not duplicates,
                'avg_confidence': avg_confidence,
                'num_horses_detected': len(positions)
            })
        else:
            logger.warning(f"  No positions detected!")
    
    cap.release()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 VALIDATION SUMMARY")
    logger.info("=" * 50)
    
    if results:
        unique_rate = sum(1 for r in results if r['unique_positions']) / len(results)
        avg_conf = sum(r['avg_confidence'] for r in results) / len(results)
        avg_detections = sum(r['num_horses_detected'] for r in results) / len(results)
        
        logger.info(f"✅ Frames tested: {len(results)}")
        logger.info(f"✅ Unique positions rate: {unique_rate*100:.1f}%")
        logger.info(f"✅ Average confidence: {avg_conf:.3f}")
        logger.info(f"✅ Average horses detected: {avg_detections:.1f}/8")
        
        # Check CNN status
        cnn_trained = hybrid_detector.cnn_reader.is_trained
        logger.info(f"✅ CNN trained status: {cnn_trained}")
        
        if unique_rate == 1.0:
            logger.info("🎉 EXCELLENT: No duplicate positions detected!")
        elif unique_rate >= 0.75:
            logger.info("✅ GOOD: Mostly unique positions")
        else:
            logger.info("⚠️  NEEDS IMPROVEMENT: Many duplicate positions")
            
        if avg_conf >= 0.7:
            logger.info("🎉 EXCELLENT: High confidence predictions!")
        elif avg_conf >= 0.5:
            logger.info("✅ GOOD: Moderate confidence")
        else:
            logger.info("⚠️  LOW: Low confidence predictions")
    
    return results

def compare_to_known_results():
    """Compare final predictions to known race results"""
    
    # This would require full race processing - for now just log the comparison
    race_code = "194367"
    known_finish = KnownResults.get_known_finish(race_code)
    
    logger.info(f"\n📊 KNOWN RESULTS COMPARISON")
    logger.info(f"Race {race_code} known finish: {known_finish}")
    logger.info(f"Expected: 2-7-5-4-6 (1st through 5th)")
    logger.info(f"Note: Full race analysis needed for complete comparison")

if __name__ == "__main__":
    results = quick_validation_test()
    compare_to_known_results()
    
    logger.info("\n🚀 Validation test complete!")
    logger.info("Run full analysis with: python main.py --race-code 194367 --num-horses 8")