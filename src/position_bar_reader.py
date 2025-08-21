import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import torch


@dataclass
class PositionBarSnapshot:
    """Represents the position bar at a single point in time"""
    frame_num: int
    timestamp: float
    positions: List[int]  # List of horse numbers in order (1st place, 2nd place, etc.)
    confidence: float


class PositionBarReader:
    """
    Reads the position bar that shows live horse positions at the bottom of the screen.
    The leftmost number is 1st place, next is 2nd place, etc.
    """
    
    def __init__(self, expected_horses: int = 8):
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.expected_horses = expected_horses  # Only accept this many horses
        # Position bar: colored number squares overlaid on video feed (above info bar)
        self.bar_y_percent_start = 0.75  # Look in the video feed area
        self.bar_y_percent_end = 0.87    # Above the bottom info bar
        self.bar_x_percent_start = 0.10  # 10% from left - capture more width
        self.bar_x_percent_end = 0.95    # 5% from right - include rightmost box!
        
    def read_position_bar(self, frame: np.ndarray, frame_num: int = 0, fps: float = 30.0) -> Optional[PositionBarSnapshot]:
        """
        Read the position bar from a frame.
        Returns horse numbers in order from 1st place to last.
        """
        height, width = frame.shape[:2]
        
        # Extract the position bar region
        y1 = int(height * self.bar_y_percent_start)
        y2 = int(height * self.bar_y_percent_end)
        x1 = int(width * self.bar_x_percent_start)
        x2 = int(width * self.bar_x_percent_end)
        
        bar_region = frame[y1:y2, x1:x2]
        
        if bar_region.size == 0:
            return None
        
        # Try multiple preprocessing methods for colored numbers
        detected_numbers = self._detect_numbers_in_bar(bar_region)
        
        if not detected_numbers:
            return None
        
        # Sort by x-coordinate (left to right = 1st to last)
        detected_numbers.sort(key=lambda x: x[1])  # Sort by x position
        
        # Extract just the numbers in order, filtering to only valid horse numbers
        all_positions = [num for num, _, _ in detected_numbers]
        # Only keep numbers in valid range for this race (1 to expected_horses)
        ordered_positions = [p for p in all_positions if 1 <= p <= self.expected_horses]
        
        # Validate the filtered positions
        if not self._validate_positions(ordered_positions):
            logger.debug(f"Invalid position bar reading: {all_positions} -> {ordered_positions}")
            return None
        
        # Calculate average confidence
        avg_confidence = np.mean([conf for _, _, conf in detected_numbers])
        
        snapshot = PositionBarSnapshot(
            frame_num=frame_num,
            timestamp=frame_num / fps,
            positions=ordered_positions,
            confidence=avg_confidence
        )
        
        logger.debug(f"Frame {frame_num}: Position bar shows {ordered_positions} (conf: {avg_confidence:.2f})")
        
        return snapshot
    
    def _detect_numbers_in_bar(self, bar_region: np.ndarray) -> List[Tuple[int, float, float]]:
        """
        Detect numbers in the position bar.
        Returns list of (number, x_position, confidence)
        """
        detected = []
        
        # Try different preprocessing for colored numbers
        preprocessed_images = self._preprocess_for_colored_numbers(bar_region)
        
        for method_name, processed_img in preprocessed_images:
            try:
                # Read text with OCR - allow all digits for races with more than 9 horses
                results = self.reader.readtext(processed_img, allowlist='0123456789')
                
                for bbox, text, confidence in results:
                    text = text.strip()
                    
                    # Parse numbers (can be 1 or 2 digits)
                    import re
                    numbers_found = re.findall(r'\d+', text)
                    for num_str in numbers_found:
                        number = int(num_str)
                        if 1 <= number <= 20:  # Support up to 20 horses
                                # Get x-coordinate from bounding box
                                x_center = (bbox[0][0] + bbox[2][0]) / 2
                                
                                # Check if we already have this number at a similar position
                                duplicate = False
                                for existing_num, existing_x, _ in detected:
                                    if existing_num == number and abs(existing_x - x_center) < 20:
                                        duplicate = True
                                        break
                                
                                if not duplicate:
                                    detected.append((number, x_center, confidence))
                                    logger.debug(f"Detected {number} at x={x_center:.0f} using {method_name} (conf: {confidence:.2f})")
                
            except Exception as e:
                logger.debug(f"OCR failed with {method_name}: {e}")
                continue
        
        # Remove duplicates, keeping highest confidence for each number
        unique_numbers = {}
        for num, x_pos, conf in detected:
            if num not in unique_numbers or conf > unique_numbers[num][2]:
                unique_numbers[num] = (num, x_pos, conf)
        
        return list(unique_numbers.values())
    
    def _preprocess_for_colored_numbers(self, bar_region: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Preprocess the bar region to extract colored numbers"""
        processed = []
        
        # Original color image
        processed.append(("original", bar_region))
        
        # Convert to grayscale
        gray = cv2.cvtColor(bar_region, cv2.COLOR_BGR2GRAY)
        
        # Extract individual color channels (colored numbers might stand out in one channel)
        b, g, r = cv2.split(bar_region)
        processed.append(("blue_channel", b))
        processed.append(("green_channel", g))
        processed.append(("red_channel", r))
        
        # HSV value channel (good for bright colored text)
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        processed.append(("value_channel", v))
        
        # High contrast version
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed.append(("enhanced", enhanced))
        
        # Threshold for bright colors
        _, bright_thresh = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)
        processed.append(("bright_threshold", bright_thresh))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        processed.append(("adaptive", adaptive))
        
        return processed
    
    def _validate_positions(self, positions: List[int]) -> bool:
        """Validate that the positions make sense"""
        if len(positions) < 2:  # Need at least 2 horses for a race
            return False
        
        if len(positions) > self.expected_horses:  # Can't have more than expected
            return False
        
        # All should be in valid range for this race
        if not all(1 <= p <= self.expected_horses for p in positions):
            return False
        
        # Allow some duplicates since OCR may misread, but prefer unique readings
        return True


class RacePositionTracker:
    """
    Tracks horse positions throughout the race using the position bar.
    This is the PRIMARY source of truth for horse positions.
    """
    
    def __init__(self, expected_horses: int = 8):
        self.reader = PositionBarReader(expected_horses=expected_horses)
        self.position_history = []  # List of PositionBarSnapshot
        self.horses_in_race = set()
        self.expected_horses = expected_horses
        
    def process_frame(self, frame: np.ndarray, frame_num: int, fps: float = 30.0) -> Optional[PositionBarSnapshot]:
        """Process a frame and extract position information"""
        snapshot = self.reader.read_position_bar(frame, frame_num, fps)
        
        if snapshot and snapshot.positions:
            self.position_history.append(snapshot)
            self.horses_in_race.update(snapshot.positions)
            return snapshot
        
        return None
    
    def get_horse_position_at_frame(self, horse_num: int, frame_num: int) -> Optional[int]:
        """Get the position of a specific horse at a specific frame"""
        # Find closest snapshot
        closest_snapshot = None
        min_distance = float('inf')
        
        for snapshot in self.position_history:
            distance = abs(snapshot.frame_num - frame_num)
            if distance < min_distance:
                min_distance = distance
                closest_snapshot = snapshot
        
        if closest_snapshot and min_distance <= 30:  # Within 1 second
            try:
                position = closest_snapshot.positions.index(horse_num) + 1
                return position
            except ValueError:
                return None
        
        return None
    
    def get_race_summary(self) -> Dict:
        """Get summary of the race based on position bar readings"""
        if not self.position_history:
            logger.error("CRITICAL: No position bar readings found! The position bar is required for analysis.")
            raise ValueError("Position bar could not be read from video. Please check that the video shows the colored position numbers at the bottom of the screen.")
        
        # Get starting positions
        start_snapshot = self.position_history[0]
        
        # Get finishing positions - prioritize readings close to race end
        # Position bar disappears around 162s (2:42), so look for readings between 140-165s
        
        # First try: use readings from the very end (most recent)
        recent_readings = self.position_history[-5:] if len(self.position_history) >= 5 else self.position_history
        
        # Find the most complete recent reading
        if recent_readings:
            end_snapshot = max(recent_readings, key=lambda s: len(s.positions))
            logger.info(f"Using end snapshot from frame {end_snapshot.frame_num} with positions: {end_snapshot.positions}")
        else:
            end_snapshot = self.position_history[-1]
            logger.info(f"Using final snapshot from frame {end_snapshot.frame_num} with positions: {end_snapshot.positions}")
        
        # Track each horse's journey
        horse_journeys = {}
        for horse_num in self.horses_in_race:
            positions = []
            for snapshot in self.position_history[::10]:  # Sample every 10th frame
                try:
                    pos = snapshot.positions.index(horse_num) + 1
                    positions.append(pos)
                except ValueError:
                    pass  # Horse not visible in this frame
            
            if positions:
                # Try to get finish position from end_snapshot first, then fall back to last position
                try:
                    finish_pos = end_snapshot.positions.index(horse_num) + 1
                except (ValueError, AttributeError):
                    finish_pos = positions[-1] if positions else None
                
                horse_journeys[horse_num] = {
                    "positions": positions,
                    "start": positions[0] if positions else None,
                    "finish": finish_pos,
                    "best": min(positions) if positions else None,
                    "worst": max(positions) if positions else None,
                    "average": np.mean(positions) if positions else None
                }
        
        # Only include horses that are valid for this race (1 to expected_horses)
        valid_horses = {h for h in self.horses_in_race if 1 <= h <= self.expected_horses}
        
        return {
            "horses_detected": sorted(list(valid_horses)),
            "total_readings": len(self.position_history),
            "horse_journeys": {k: v for k, v in horse_journeys.items() if k in valid_horses},
            "winner": end_snapshot.positions[0] if end_snapshot.positions else None,
            "second": end_snapshot.positions[1] if len(end_snapshot.positions) > 1 else None,
            "third": end_snapshot.positions[2] if len(end_snapshot.positions) > 2 else None
        }
    
    def get_horse_position_chart(self, horse_num: int) -> List[int]:
        """Get the position progression for a specific horse"""
        positions = []
        
        for snapshot in self.position_history:
            try:
                pos = snapshot.positions.index(horse_num) + 1
                positions.append(pos)
            except ValueError:
                # Horse not found in this snapshot
                if positions:
                    # Use last known position
                    positions.append(positions[-1])
        
        return positions