#!/usr/bin/env python3
"""
Prepare race 194367 for finish-anchored tracking validation
This script helps extract final positions or creates test data when manual labels are ready
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger

def extract_finish_positions_from_labels(labels_file: str) -> Dict[str, int]:
    """Extract final positions from manual labels by analyzing the last frame"""
    with open(labels_file) as f:
        data = json.load(f)
    
    if 'labels' not in data:
        logger.error("No labels found in the data file")
        return {}
    
    labels = data['labels']
    if not labels:
        logger.error("No labeled frames found")
        return {}
    
    # Find the last frame with labels (frames are stored as string keys)
    frame_numbers = [int(frame_str) for frame_str in labels.keys()]
    last_frame = max(frame_numbers)
    last_frame_str = str(last_frame)
    
    logger.info(f"Using frame {last_frame} as finish line")
    
    # Get all horses in the last frame
    finish_line_horses = []
    if last_frame_str in labels:
        for horse_id_str, bbox in labels[last_frame_str].items():
            horse_id = int(horse_id_str)
            # bbox is [x, y, w, h]
            center_x = bbox[0] + bbox[2] / 2  # x + width/2
            
            finish_line_horses.append({
                'horse_id': horse_id,
                'center_x': center_x
            })
    
    if not finish_line_horses:
        logger.error(f"No horses found in finish frame {last_frame}")
        return {}
    
    # Sort by horizontal position (left to right = 1st to last place)
    finish_line_horses.sort(key=lambda h: h['center_x'])
    
    # Create race results
    race_results = {}
    for position, horse_data in enumerate(finish_line_horses, 1):
        race_results[str(position)] = horse_data['horse_id']
        logger.info(f"{position}{'st' if position == 1 else 'nd' if position == 2 else 'rd' if position == 3 else 'th'} place: Horse #{horse_data['horse_id']}")
    
    return race_results

def create_mock_race_results(num_horses: int = 8) -> Dict[str, int]:
    """Create mock race results for testing (when real results not available)"""
    logger.warning("Creating mock race results for testing purposes")
    
    # Standard horse program numbers (1-8 for 8-horse race)
    race_results = {}
    for position in range(1, num_horses + 1):
        race_results[str(position)] = position
    
    logger.info(f"Created mock results for {num_horses} horses:")
    for pos, horse_num in race_results.items():
        logger.info(f"  {pos}{'st' if pos == '1' else 'nd' if pos == '2' else 'rd' if pos == '3' else 'th'} place: Horse #{horse_num}")
    
    return race_results

def save_race_results(race_results: Dict[str, int], output_file: str, race_code: str = "194367"):
    """Save race results in validation dataset format"""
    output_data = {
        "created_at": "2025-09-11T09:00:00.000000",
        "races": {
            race_code: {
                "race_code": race_code,
                "horse_count": len(race_results),
                "final_positions": {},
                "horse_numbers": race_results,
                "timing_data": [],
                "call_positions": []
            }
        },
        "summary": {
            "total_races": 1,
            "races_with_results": 1,
            "total_horses": len(race_results)
        }
    }
    
    # Add mock horse names for final positions
    for pos, horse_num in race_results.items():
        output_data["races"][race_code]["final_positions"][pos] = f"HORSE_{horse_num}({horse_num}) SK"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Race results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare race 194367 for finish-anchored tracking")
    parser.add_argument("--labels", help="Path to manual labels file (to extract finish positions)")
    parser.add_argument("--mock", action="store_true", help="Create mock race results for testing")
    parser.add_argument("--num-horses", type=int, default=8, help="Number of horses for mock results")
    parser.add_argument("--output", default="race_194367_results.json", help="Output file for race results")
    parser.add_argument("--race-code", default="194367", help="Race code")
    
    args = parser.parse_args()
    
    if args.labels and Path(args.labels).exists():
        # Extract from manual labels
        logger.info(f"Extracting finish positions from manual labels: {args.labels}")
        race_results = extract_finish_positions_from_labels(args.labels)
        
        if not race_results:
            logger.error("Failed to extract race results from labels")
            return
            
    elif args.mock:
        # Create mock results
        logger.info(f"Creating mock race results for {args.num_horses} horses")
        race_results = create_mock_race_results(args.num_horses)
        
    else:
        logger.error("Either --labels (with valid file) or --mock must be specified")
        return
    
    # Save results
    save_race_results(race_results, args.output, args.race_code)
    
    print(f"\n🏁 Race {args.race_code} Results Prepared!")
    print(f"📁 Saved to: {args.output}")
    print(f"🐎 Total horses: {len(race_results)}")
    print("\n🥇 Finish Order:")
    for pos, horse_num in race_results.items():
        print(f"   {pos}{'st' if pos == '1' else 'nd' if pos == '2' else 'rd' if pos == '3' else 'th'}: Horse #{horse_num}")

if __name__ == "__main__":
    main()