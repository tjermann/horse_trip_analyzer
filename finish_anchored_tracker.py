#!/usr/bin/env python3
"""
Finish-Line Anchored Horse Tracking
Revolutionary approach that uses known final positions to anchor horse identities,
then tracks backwards through the race to assign consistent IDs.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import json
from loguru import logger
from collections import defaultdict
import math

@dataclass
class AnchoredTrackResult:
    """Single frame tracking result with finish-line anchored identity"""
    frame_number: int
    program_number: int  # Actual horse program number (1-20)
    finish_position: int  # Final race position (1st, 2nd, etc.)
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    features: Dict[str, any] = field(default_factory=dict)

class FinishAnchoredTracker:
    """Horse tracking system anchored by known final positions"""
    
    def __init__(self, race_results: Dict[str, int]):
        """
        Initialize tracker with known race results
        race_results: {finish_position: program_number} e.g. {"1": 4, "2": 2, "3": 3, "4": 5}
        """
        self.race_results = race_results  # {"1": 4, "2": 2} = 1st place is horse #4
        self.program_numbers = list(race_results.values())  # [4, 2, 3, 5]
        
        # Tracking parameters
        self.feature_weight_color = 0.4
        self.feature_weight_position = 0.4  
        self.feature_weight_size = 0.2
        self.max_tracking_distance = 150
        self.min_confidence_threshold = 0.3
        
        logger.info(f"Initialized finish-anchored tracker with {len(self.race_results)} horses")
        for pos, horse_num in self.race_results.items():
            logger.info(f"  {pos}{'st' if pos == '1' else 'nd' if pos == '2' else 'rd' if pos == '3' else 'th'} place: Horse #{horse_num}")
    
    def extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, any]:
        """Extract visual features from bounding box region"""
        x, y, w, h = bbox
        
        # Ensure bbox is within frame bounds
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return {"color_hist": np.zeros(64), "dominant_color": (0, 0, 0)}
        
        # Color histogram
        roi_resized = cv2.resize(roi, (32, 32))
        color_hist = cv2.calcHist([roi_resized], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        
        # Dominant color
        roi_mean = cv2.mean(roi_resized)
        dominant_color = (int(roi_mean[0]), int(roi_mean[1]), int(roi_mean[2]))
        
        return {
            "color_hist": color_hist,
            "dominant_color": dominant_color,
            "size": (w, h),
            "center": (x + w//2, y + h//2)
        }
    
    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity score between two feature sets"""
        if not features1 or not features2:
            return 0.0
        
        # Color histogram similarity
        hist1 = features1.get("color_hist", np.zeros(64))
        hist2 = features2.get("color_hist", np.zeros(64))
        
        hist1_safe = hist1 + 1e-10
        hist2_safe = hist2 + 1e-10
        
        color_sim = 1.0 / (1.0 + cv2.compareHist(hist1_safe, hist2_safe, cv2.HISTCMP_CHISQR))
        
        # Position similarity
        center1 = features1.get("center", (0, 0))
        center2 = features2.get("center", (0, 0))
        position_dist = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        position_sim = 1.0 / (1.0 + position_dist / 100.0)
        
        # Size similarity
        size1 = features1.get("size", (0, 0))
        size2 = features2.get("size", (0, 0))
        size_ratio = min(size1[0] * size1[1], size2[0] * size2[1]) / max(size1[0] * size1[1], size2[0] * size2[1], 1)
        
        # Weighted similarity
        similarity = (self.feature_weight_color * color_sim + 
                     self.feature_weight_position * position_sim +
                     self.feature_weight_size * size_ratio)
        
        return min(1.0, similarity)
    
    def find_finish_line_frame(self, video_path: str, detections: Dict[int, List[Tuple]]) -> int:
        """Find the frame closest to the finish line (typically last 10% of frames)"""
        frame_numbers = sorted(detections.keys())
        if not frame_numbers:
            return 0
        
        # Use last 10% of frames as finish line area
        total_frames = max(frame_numbers)
        finish_area_start = int(total_frames * 0.9)
        
        # Find frame in finish area with number of detections matching our expected horses
        expected_horses = len(self.race_results)
        
        for frame_num in reversed(frame_numbers):
            if frame_num >= finish_area_start:
                detections_count = len(detections[frame_num])
                if detections_count >= expected_horses * 0.8:  # Allow some missing detections
                    logger.info(f"Selected finish line frame: {frame_num} with {detections_count} detections")
                    return frame_num
        
        # Fallback to last frame with detections
        last_frame = frame_numbers[-1]
        logger.warning(f"Using fallback finish frame: {last_frame}")
        return last_frame
    
    def anchor_finish_line_identities(self, finish_frame: int, detections: List[Tuple], 
                                    frame: np.ndarray) -> Dict[int, AnchoredTrackResult]:
        """Assign horse identities at the finish line based on position"""
        logger.info(f"Anchoring identities at finish line frame {finish_frame}")
        
        # Extract features for all detections
        detection_features = []
        for i, (x, y, w, h, conf) in enumerate(detections):
            features = self.extract_features(frame, (x, y, w, h))
            detection_features.append({
                'bbox': (x, y, w, h),
                'confidence': conf,
                'features': features,
                'detection_idx': i
            })
        
        # Sort detections by horizontal position (left to right across finish line)
        # This assumes finish line positions correspond to race order
        detection_features.sort(key=lambda d: d['bbox'][0])
        
        # Assign identities based on finish positions
        anchored_results = {}
        expected_horses = len(self.race_results)
        
        for rank, detection_data in enumerate(detection_features[:expected_horses]):
            finish_position = rank + 1  # 1st, 2nd, 3rd, etc.
            finish_pos_str = str(finish_position)
            
            if finish_pos_str in self.race_results:
                program_number = self.race_results[finish_pos_str]
                
                bbox = detection_data['bbox']
                result = AnchoredTrackResult(
                    frame_number=finish_frame,
                    program_number=program_number,
                    finish_position=finish_position,
                    bbox=bbox,
                    confidence=detection_data['confidence'],
                    features=detection_data['features']
                )
                
                anchored_results[program_number] = result
                logger.info(f"Anchored horse #{program_number} ({finish_position}{'st' if finish_position == 1 else 'nd' if finish_position == 2 else 'rd' if finish_position == 3 else 'th'} place) at {bbox}")
        
        return anchored_results
    
    def track_backwards_from_anchor(self, video_path: str, detections: Dict[int, List[Tuple]], 
                                  anchored_identities: Dict[int, AnchoredTrackResult]) -> Dict[int, List[AnchoredTrackResult]]:
        """Track backwards from finish line to assign consistent identities"""
        logger.info("Starting backward tracking from anchored identities...")
        
        cap = cv2.VideoCapture(video_path)
        backward_tracks = defaultdict(list)
        
        # Initialize tracks with anchored identities
        for program_num, anchored_result in anchored_identities.items():
            backward_tracks[program_num].append(anchored_result)
        
        # Get frames in reverse order
        frame_numbers = sorted([f for f in detections.keys() if f < anchored_result.frame_number], reverse=True)
        
        for frame_num in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
                
            current_detections = detections[frame_num]
            
            # Match current detections to existing tracks
            unmatched_detections = list(current_detections)
            active_horses = list(backward_tracks.keys())
            
            # Calculate similarity matrix
            similarity_matrix = {}
            for program_num in active_horses:
                if not backward_tracks[program_num]:
                    continue
                last_result = backward_tracks[program_num][-1]
                
                for det_idx, detection in enumerate(unmatched_detections):
                    x, y, w, h, conf = detection
                    current_features = self.extract_features(frame, (x, y, w, h))
                    
                    similarity = self.calculate_similarity(last_result.features, current_features)
                    similarity_matrix[(program_num, det_idx)] = similarity
            
            # Greedy assignment (highest similarity first)
            assigned_horses = set()
            assigned_detections = set()
            
            for (program_num, det_idx), similarity in sorted(similarity_matrix.items(), 
                                                           key=lambda x: x[1], reverse=True):
                if program_num in assigned_horses or det_idx in assigned_detections:
                    continue
                if similarity < self.min_confidence_threshold:
                    break
                    
                # Assign detection to horse
                detection = unmatched_detections[det_idx]
                x, y, w, h, conf = detection
                features = self.extract_features(frame, (x, y, w, h))
                
                # Find finish position for this program number
                finish_position = None
                for pos, prog_num in self.race_results.items():
                    if prog_num == program_num:
                        finish_position = int(pos)
                        break
                
                result = AnchoredTrackResult(
                    frame_number=frame_num,
                    program_number=program_num,
                    finish_position=finish_position,
                    bbox=(x, y, w, h),
                    confidence=similarity * conf,
                    features=features
                )
                backward_tracks[program_num].append(result)
                
                assigned_horses.add(program_num)
                assigned_detections.add(det_idx)
        
        # Reverse tracks to be in chronological order
        for program_num in backward_tracks:
            backward_tracks[program_num].reverse()
        
        cap.release()
        logger.info(f"Backward tracking complete: {len(backward_tracks)} horses tracked")
        return dict(backward_tracks)
    
    def process_race(self, video_path: str, detections: Dict[int, List[Tuple]]) -> Dict[int, List[AnchoredTrackResult]]:
        """Process entire race with finish-line anchored tracking"""
        logger.info(f"Starting finish-anchored tracking for {video_path}")
        
        # Find finish line frame
        finish_frame = self.find_finish_line_frame(video_path, detections)
        
        # Get finish line detections and frame
        if finish_frame not in detections:
            logger.error(f"Finish frame {finish_frame} not found in detections")
            return {}
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, finish_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Could not read finish frame {finish_frame}")
            return {}
        
        # Anchor identities at finish line
        anchored_identities = self.anchor_finish_line_identities(
            finish_frame, detections[finish_frame], frame)
        
        if not anchored_identities:
            logger.error("Failed to anchor identities at finish line")
            return {}
        
        # Track backwards from anchored identities
        final_tracks = self.track_backwards_from_anchor(video_path, detections, anchored_identities)
        
        logger.info(f"Finish-anchored tracking complete: {len(final_tracks)} consistent tracks")
        return final_tracks
    
    def save_results(self, results: Dict[int, List[AnchoredTrackResult]], output_path: str):
        """Save finish-anchored tracking results to file"""
        output_data = {
            "finish_anchored_tracking_results": [],
            "race_results": self.race_results,
            "summary": {
                "total_tracks": len(results),
                "program_numbers": list(results.keys()),
                "avg_track_length": sum(len(track) for track in results.values()) / len(results) if results else 0
            }
        }
        
        for program_num, track in results.items():
            track_data = {
                "program_number": program_num,
                "finish_position": track[0].finish_position if track else None,
                "track_length": len(track),
                "track": [
                    {
                        "frame_number": r.frame_number,
                        "bbox": r.bbox,
                        "confidence": r.confidence,
                        "finish_position": r.finish_position
                    }
                    for r in track
                ]
            }
            output_data["finish_anchored_tracking_results"].append(track_data)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Finish-Line Anchored Horse Tracking")
    parser.add_argument("--video", required=True, help="Path to race video")
    parser.add_argument("--detections", required=True, help="Path to detections JSON file")
    parser.add_argument("--race-results", required=True, help="Path to race results JSON file")
    parser.add_argument("--output", help="Output path for results")
    
    args = parser.parse_args()
    
    # Load detections
    with open(args.detections) as f:
        detections_data = json.load(f)
    
    # Convert to expected format
    detections = {}
    detection_dict = detections_data.get('detections', detections_data)
    
    for frame_str, frame_detections in detection_dict.items():
        try:
            frame_num = int(frame_str)
            detections[frame_num] = [(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']) 
                                    for d in frame_detections]
        except ValueError:
            continue
    
    # Load race results
    with open(args.race_results) as f:
        race_data = json.load(f)
    
    # Extract race results from validation data format
    race_results = {}
    if 'races' in race_data:
        # Find matching race
        for race_code, race_info in race_data['races'].items():
            if 'horse_numbers' in race_info:
                race_results = race_info['horse_numbers']
                break
    else:
        race_results = race_data.get('horse_numbers', race_data)
    
    if not race_results:
        logger.error("No race results found in data file")
        return
    
    # Create tracker and process
    tracker = FinishAnchoredTracker(race_results)
    results = tracker.process_race(args.video, detections)
    
    # Save results
    output_path = args.output or f"finish_anchored_tracking_{Path(args.video).stem}.json"
    tracker.save_results(results, output_path)
    
    # Print summary
    print(f"\n🏁 Finish-Anchored Tracking Complete!")
    print(f"📊 Total tracks: {len(results)}")
    print(f"🎯 Horses tracked:")
    for program_num, track in results.items():
        if track:
            finish_pos = track[0].finish_position
            print(f"   Horse #{program_num}: {finish_pos}{'st' if finish_pos == 1 else 'nd' if finish_pos == 2 else 'rd' if finish_pos == 3 else 'th'} place ({len(track)} frames)")

if __name__ == "__main__":
    main()