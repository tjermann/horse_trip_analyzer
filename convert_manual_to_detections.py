#!/usr/bin/env python3
"""
Convert manual labels to detection format for finish-anchored tracking
Transforms manual labeling data into the format expected by the tracking system
"""

import json
import argparse
from pathlib import Path
from loguru import logger

def convert_manual_labels_to_detections(labels_file: str, output_file: str):
    """Convert manual labels format to detection format"""
    with open(labels_file) as f:
        manual_data = json.load(f)
    
    if 'labels' not in manual_data:
        logger.error("No labels found in manual data file")
        return
    
    labels = manual_data['labels']
    
    # Convert to detection format
    detections = {}
    total_detections = 0
    
    for frame_str, horse_data in labels.items():
        frame_num = int(frame_str)
        frame_detections = []
        
        for horse_id_str, bbox in horse_data.items():
            horse_id = int(horse_id_str)
            x, y, w, h = bbox
            
            detection = {
                'bbox': [x, y, w, h],
                'confidence': 0.9,  # High confidence for manual labels
                'horse_id': horse_id  # Include horse ID for reference
            }
            frame_detections.append(detection)
            total_detections += 1
        
        detections[frame_str] = frame_detections
    
    # Create detection output format
    output_data = {
        'race_code': manual_data.get('race_code', 'unknown'),
        'video_path': manual_data.get('video_path', ''),
        'created_at': manual_data.get('created_at', ''),
        'total_frames': manual_data.get('total_frames', 0),
        'fps': manual_data.get('fps', 25.0),
        'detections': detections,
        'summary': {
            'total_frames_with_detections': len(detections),
            'total_detections': total_detections,
            'horses_detected': list(set(
                int(horse_id) for frame_data in labels.values() 
                for horse_id in frame_data.keys()
            ))
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Converted manual labels to detections: {output_file}")
    logger.info(f"Total frames: {len(detections)}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Horses detected: {output_data['summary']['horses_detected']}")

def main():
    parser = argparse.ArgumentParser(description="Convert manual labels to detection format")
    parser.add_argument("--labels", required=True, help="Path to manual labels JSON file")
    parser.add_argument("--output", required=True, help="Output detection file path")
    
    args = parser.parse_args()
    
    convert_manual_labels_to_detections(args.labels, args.output)
    
    print(f"\n🔄 Manual Labels → Detections Conversion Complete!")
    print(f"📁 Output: {args.output}")

if __name__ == "__main__":
    main()