import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class InterpolatedPosition:
    frame_num: int
    track_id: int
    position: int
    confidence: float
    was_interpolated: bool = False


class PositionInterpolator:
    """
    Handles missing horse detections by interpolating positions when horses 
    are temporarily occluded or missed by the detector.
    
    Does NOT smooth legitimate position changes - horses can move rapidly in bunched fields.
    """
    
    def __init__(self, max_interpolation_gap: int = 3):
        self.max_gap = max_interpolation_gap  # Max frames to interpolate across
        
    def interpolate_missing_positions(self, 
                                    raw_positions: Dict[int, List[Tuple[int, int, float]]]) -> Dict[int, List[InterpolatedPosition]]:
        """
        Fill in missing positions for horses that temporarily disappear from detection
        
        Args:
            raw_positions: Dict[track_id, List[(frame_num, position, confidence)]]
        
        Returns:
            Dict[track_id, List[InterpolatedPosition]] - with gaps filled
        """
        interpolated_all = {}
        
        for track_id, positions in raw_positions.items():
            if len(positions) < 2:
                # Not enough data for interpolation
                interpolated_all[track_id] = [
                    InterpolatedPosition(frame, track_id, pos, conf, False)
                    for frame, pos, conf in positions
                ]
                continue
            
            # Sort by frame number
            positions.sort(key=lambda x: x[0])
            
            interpolated_positions = []
            
            for i in range(len(positions)):
                frame_num, position, confidence = positions[i]
                
                # Add the observed position
                interpolated_positions.append(
                    InterpolatedPosition(frame_num, track_id, position, confidence, False)
                )
                
                # Check if there's a gap to the next observation
                if i < len(positions) - 1:
                    next_frame, next_position, _ = positions[i + 1]
                    gap_size = next_frame - frame_num - 1
                    
                    # Only interpolate if gap is reasonable size
                    if 0 < gap_size <= self.max_gap:
                        # Linear interpolation for missing frames
                        for j in range(1, gap_size + 1):
                            interp_frame = frame_num + j
                            
                            # Linear interpolation
                            ratio = j / (gap_size + 1)
                            interp_position = position + (next_position - position) * ratio
                            interp_position = round(interp_position)
                            interp_position = max(1, min(8, interp_position))
                            
                            # Lower confidence for interpolated positions
                            interp_confidence = min(confidence, 0.6)
                            
                            interpolated_positions.append(
                                InterpolatedPosition(
                                    frame_num=interp_frame,
                                    track_id=track_id,
                                    position=interp_position,
                                    confidence=interp_confidence,
                                    was_interpolated=True
                                )
                            )
                            
                            logger.debug(f"Interpolated horse {track_id} position at frame {interp_frame}: {interp_position}")
            
            interpolated_all[track_id] = interpolated_positions
        
        return interpolated_all
    
    def detect_tracking_issues(self, positions: List[Tuple[int, int, float]]) -> List[Dict]:
        """
        Identify potential tracking issues without fixing them
        """
        if len(positions) < 3:
            return []
        
        issues = []
        positions.sort(key=lambda x: x[0])
        
        for i in range(len(positions) - 1):
            curr_frame, curr_pos, curr_conf = positions[i]
            next_frame, next_pos, next_conf = positions[i + 1]
            
            frame_gap = next_frame - curr_frame
            position_jump = abs(next_pos - curr_pos)
            
            # Flag potential issues
            if frame_gap == 1 and position_jump >= 5:
                issues.append({
                    "type": "large_single_frame_jump",
                    "from_frame": curr_frame,
                    "to_frame": next_frame,
                    "position_change": position_jump,
                    "description": f"Horse jumped from position {curr_pos} to {next_pos} in 1 frame"
                })
            
            elif frame_gap > 3:
                issues.append({
                    "type": "long_gap",
                    "from_frame": curr_frame,
                    "to_frame": next_frame,
                    "gap_size": frame_gap,
                    "description": f"Horse missing for {frame_gap} frames"
                })
            
            elif curr_conf < 0.3 or next_conf < 0.3:
                issues.append({
                    "type": "low_confidence",
                    "frame": curr_frame if curr_conf < next_conf else next_frame,
                    "confidence": min(curr_conf, next_conf),
                    "description": f"Very low detection confidence"
                })
        
        return issues


class SmartPositionTracker:
    """
    Combines interpolation with basic error detection while preserving 
    legitimate rapid position changes in bunched fields
    """
    
    def __init__(self):
        self.interpolator = PositionInterpolator(max_interpolation_gap=2)
        self.position_histories = {}
        
    def add_detection(self, frame_num: int, track_id: int, position: int, confidence: float):
        """Add a position detection"""
        if track_id not in self.position_histories:
            self.position_histories[track_id] = []
        
        self.position_histories[track_id].append((frame_num, position, confidence))
    
    def get_complete_positions(self) -> Dict[int, List[InterpolatedPosition]]:
        """Get position sequences with missing frames interpolated"""
        return self.interpolator.interpolate_missing_positions(self.position_histories)
    
    def get_tracking_quality_report(self) -> Dict[int, Dict]:
        """Get quality metrics for each horse's tracking"""
        report = {}
        
        for track_id, positions in self.position_histories.items():
            if not positions:
                continue
            
            positions.sort(key=lambda x: x[0])
            
            # Basic statistics
            frames = [p[0] for p in positions]
            confidences = [p[2] for p in positions]
            
            frame_gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
            max_gap = max(frame_gaps) if frame_gaps else 0
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Detect issues
            issues = self.interpolator.detect_tracking_issues(positions)
            
            report[track_id] = {
                "total_detections": len(positions),
                "frame_range": (min(frames), max(frames)) if frames else (0, 0),
                "max_gap": max_gap,
                "average_confidence": avg_confidence,
                "issues": issues,
                "quality_score": self._calculate_quality_score(positions, issues)
            }
        
        return report
    
    def _calculate_quality_score(self, positions: List[Tuple[int, int, float]], issues: List[Dict]) -> float:
        """Calculate a quality score (0-1) for tracking reliability"""
        if not positions:
            return 0.0
        
        # Base score from average confidence
        confidences = [p[2] for p in positions]
        confidence_score = np.mean(confidences)
        
        # Penalties for issues
        penalty = 0
        for issue in issues:
            if issue["type"] == "large_single_frame_jump":
                penalty += 0.1
            elif issue["type"] == "long_gap":
                penalty += 0.05 * issue["gap_size"]
            elif issue["type"] == "low_confidence":
                penalty += 0.05
        
        final_score = max(0.0, confidence_score - penalty)
        return final_score