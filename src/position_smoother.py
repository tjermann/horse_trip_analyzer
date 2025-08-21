import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, medfilt
from loguru import logger


@dataclass
class SmoothedPosition:
    frame_num: int
    track_id: int
    position: int
    confidence: float
    interpolated: bool = False


class PositionSmoother:
    """
    Smooths horse positions over time to handle tracking errors, occlusions, and missing detections.
    Uses interpolation, median filtering, and confidence-based smoothing.
    """
    
    def __init__(self, window_size: int = 5, max_gap: int = 3):
        self.window_size = window_size  # Frames to use for smoothing
        self.max_gap = max_gap         # Maximum frames to interpolate over
        self.position_histories = {}   # Dict[track_id, List[Tuple[frame, position, confidence]]]
        
    def add_position(self, frame_num: int, track_id: int, position: int, confidence: float = 1.0):
        """Add a position observation for a horse"""
        if track_id not in self.position_histories:
            self.position_histories[track_id] = []
        
        self.position_histories[track_id].append((frame_num, position, confidence))
    
    def get_smoothed_positions(self, track_id: int) -> List[SmoothedPosition]:
        """Get smoothed position sequence for a horse"""
        if track_id not in self.position_histories:
            return []
        
        history = self.position_histories[track_id]
        if len(history) < 3:
            # Not enough data for smoothing
            return [SmoothedPosition(frame, track_id, pos, conf) 
                   for frame, pos, conf in history]
        
        # Sort by frame number
        history.sort(key=lambda x: x[0])
        
        # Step 1: Fill gaps with interpolation
        filled_history = self._fill_gaps(history)
        
        # Step 2: Apply median filter to remove outliers
        median_filtered = self._apply_median_filter(filled_history)
        
        # Step 3: Apply Savitzky-Golay filter for final smoothing
        smoothed = self._apply_savgol_filter(median_filtered)
        
        return smoothed
    
    def _fill_gaps(self, history: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float, bool]]:
        """Fill missing frames using linear interpolation"""
        if len(history) < 2:
            return [(frame, pos, conf, False) for frame, pos, conf in history]
        
        filled = []
        frames, positions, confidences = zip(*history)
        
        # Find gaps and interpolate
        for i in range(len(history)):
            filled.append((frames[i], positions[i], confidences[i], False))
            
            # Check for gap to next frame
            if i < len(history) - 1:
                gap_size = frames[i + 1] - frames[i] - 1
                
                if 0 < gap_size <= self.max_gap:
                    # Interpolate missing positions
                    start_pos = positions[i]
                    end_pos = positions[i + 1]
                    
                    for j in range(1, gap_size + 1):
                        interp_frame = frames[i] + j
                        # Linear interpolation
                        ratio = j / (gap_size + 1)
                        interp_pos = round(start_pos + (end_pos - start_pos) * ratio)
                        interp_pos = max(1, min(8, interp_pos))  # Clamp to valid range
                        
                        filled.append((interp_frame, interp_pos, 0.5, True))  # Mark as interpolated
        
        return filled
    
    def _apply_median_filter(self, history: List[Tuple[int, int, float, bool]]) -> List[Tuple[int, int, float, bool]]:
        """Apply very light median filter to remove only obvious outliers"""
        if len(history) < 7:
            return history
        
        frames, positions, confidences, interpolated = zip(*history)
        positions = np.array(positions)
        
        # Only apply median filter to single-frame outliers (spikes)
        filtered_positions = positions.copy()
        
        for i in range(1, len(positions) - 1):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]
            next_pos = positions[i + 1]
            
            # Only smooth if current position is drastically different from both neighbors
            # AND the neighbors are similar (indicating the middle one is likely wrong)
            if abs(prev_pos - next_pos) <= 2 and abs(curr_pos - prev_pos) > 4 and abs(curr_pos - next_pos) > 4:
                # This looks like a tracking error - use median of the three
                filtered_positions[i] = np.median([prev_pos, curr_pos, next_pos])
                logger.debug(f"Smoothed obvious outlier: {curr_pos} -> {filtered_positions[i]} (neighbors: {prev_pos}, {next_pos})")
        
        # Ensure valid range
        filtered_positions = np.clip(filtered_positions, 1, 8)
        
        return [(frames[i], int(filtered_positions[i]), confidences[i], interpolated[i]) 
                for i in range(len(history))]
    
    def _apply_savgol_filter(self, history: List[Tuple[int, int, float, bool]]) -> List[SmoothedPosition]:
        """Apply Savitzky-Golay filter for final smoothing"""
        if len(history) < 7:
            return [SmoothedPosition(frame, track_id, pos, conf, interp) 
                   for frame, pos, conf, interp in history]
        
        frames, positions, confidences, interpolated = zip(*history)
        positions = np.array(positions, dtype=float)
        
        # Apply Savitzky-Golay filter
        window_length = min(7, len(positions))
        if window_length % 2 == 0:
            window_length -= 1  # Must be odd
        
        try:
            smoothed_positions = savgol_filter(positions, window_length, polyorder=2)
            smoothed_positions = np.clip(np.round(smoothed_positions), 1, 8)
        except:
            # Fallback to original positions if smoothing fails
            smoothed_positions = positions
        
        # Get track_id from the calling context (we'll need to pass this)
        # For now, we'll extract it from the position_histories
        track_id = None
        for tid, hist in self.position_histories.items():
            if len(hist) > 0 and hist[0][0] == frames[0]:
                track_id = tid
                break
        
        if track_id is None:
            track_id = 1  # Fallback
        
        return [SmoothedPosition(
            frame_num=frames[i], 
            track_id=track_id,
            position=int(smoothed_positions[i]), 
            confidence=confidences[i],
            interpolated=interpolated[i]
        ) for i in range(len(history))]
    
    def get_all_smoothed_positions(self) -> Dict[int, List[SmoothedPosition]]:
        """Get smoothed positions for all horses"""
        smoothed_all = {}
        
        for track_id in self.position_histories:
            smoothed_all[track_id] = self.get_smoothed_positions(track_id)
        
        return smoothed_all
    
    def detect_position_anomalies(self, track_id: int, threshold: float = 2.0) -> List[int]:
        """
        Detect frames with anomalous position changes that might indicate tracking errors
        """
        if track_id not in self.position_histories:
            return []
        
        history = self.position_histories[track_id]
        if len(history) < 5:
            return []
        
        history.sort(key=lambda x: x[0])
        frames, positions, _ = zip(*history)
        positions = np.array(positions)
        
        # Calculate position changes
        position_diffs = np.abs(np.diff(positions))
        
        # Find anomalous changes (large jumps)
        anomaly_indices = np.where(position_diffs > threshold)[0]
        
        # Return frame numbers with anomalies
        return [frames[i + 1] for i in anomaly_indices]
    
    def get_confidence_weighted_position(self, track_id: int, frame_num: int, 
                                       window: int = 3) -> Optional[Tuple[int, float]]:
        """
        Get position for a specific frame using confidence-weighted average of nearby frames
        """
        if track_id not in self.position_histories:
            return None
        
        history = self.position_histories[track_id]
        
        # Find observations within window
        nearby_obs = []
        for frame, pos, conf in history:
            if abs(frame - frame_num) <= window:
                distance = abs(frame - frame_num)
                weight = conf * (1.0 / (distance + 1))  # Closer frames get higher weight
                nearby_obs.append((pos, weight))
        
        if not nearby_obs:
            return None
        
        # Calculate weighted average position
        total_weight = sum(weight for _, weight in nearby_obs)
        if total_weight == 0:
            return None
        
        weighted_pos = sum(pos * weight for pos, weight in nearby_obs) / total_weight
        avg_confidence = total_weight / len(nearby_obs)
        
        return (round(weighted_pos), avg_confidence)


def smooth_horse_positions(raw_positions: Dict[int, List[Tuple[int, int, float]]]) -> Dict[int, List[SmoothedPosition]]:
    """
    Convenience function to smooth positions for all horses
    
    Args:
        raw_positions: Dict[track_id, List[(frame_num, position, confidence)]]
    
    Returns:
        Dict[track_id, List[SmoothedPosition]]
    """
    smoother = PositionSmoother()
    
    # Add all position observations
    for track_id, positions in raw_positions.items():
        for frame_num, position, confidence in positions:
            smoother.add_position(frame_num, track_id, position, confidence)
    
    # Get smoothed positions for all horses
    return smoother.get_all_smoothed_positions()