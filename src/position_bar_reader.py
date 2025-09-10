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
        # Position bar: colored number squares in bottom red banner 
        self.bar_y_percent_start = 0.85  # Bottom 15% where position bar is located
        self.bar_y_percent_end = 1.0     # Very bottom of frame
        self.bar_x_percent_start = 0.0   # Full width to capture all position rectangles
        self.bar_x_percent_end = 1.0     # Full width
        
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
        
        # IMPROVED: Try to detect individual colored squares first
        # The position bar consists of colored squares with numbers
        detected_numbers = self._detect_numbers_with_segmentation(bar_region)
        
        # Fallback to original method if segmentation fails
        if not detected_numbers or len(detected_numbers) < 2:
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
    
    def _detect_numbers_with_segmentation(self, bar_region: np.ndarray) -> List[Tuple[int, float, float]]:
        """
        Improved detection using manual position locations discovered through debugging.
        Position bar consists of 8 colored squares in the red banner.
        """
        detected = []
        height, width = bar_region.shape[:2]
        
        # Manual position locations discovered through visual debugging
        # These are precise locations relative to the bottom 15% region
        manual_positions = [
            (50, 18, 22, 16),   # Position 1 
            (105, 18, 22, 16),  # Position 2
            (160, 18, 22, 16),  # Position 3  
            (215, 18, 22, 16),  # Position 4
            (270, 18, 22, 16),  # Position 5
            (325, 18, 22, 16),  # Position 6
            (380, 18, 22, 16),  # Position 7
            (435, 18, 22, 16),  # Position 8
        ]
        
        # Scale positions to match actual region size (since manual positions assume ~734 width, 73 height)
        expected_width = 734
        expected_height = 73
        
        width_scale = width / expected_width
        height_scale = height / expected_height
        
        for i, (x_manual, y_manual, w_manual, h_manual) in enumerate(manual_positions):
            if i >= self.expected_horses:  # Only process expected number of horses
                break
                
            # Scale the manual positions to current region size
            x = int(x_manual * width_scale)
            y = int(y_manual * height_scale)
            w = int(w_manual * width_scale)
            h = int(h_manual * height_scale)
            
            # Ensure bounds are valid
            x = max(0, min(x, width - w))
            y = max(0, min(y, height - h))
            
            # Extract this position rectangle
            square_region = bar_region[y:y+h, x:x+w]
            
            if square_region.size == 0:
                continue
            
            # Try OCR on this specific position with multiple preprocessing
            best_result = None
            best_confidence = 0
            
            for method_name, processed_img in self._preprocess_single_square(square_region):
                try:
                    results = self.reader.readtext(
                        processed_img, 
                        allowlist='12345678', 
                        paragraph=False,
                        width_ths=0.001,  # Ultra-low thresholds for small digits
                        height_ths=0.001,
                        detail=1
                    )
                    for bbox, text, confidence in results:
                        text = text.strip()
                        if len(text) == 1 and text.isdigit():
                            number = int(text)
                            if 1 <= number <= self.expected_horses and confidence > best_confidence:
                                best_result = number
                                best_confidence = confidence
                                logger.debug(f"Manual position {i+1}: {number} (conf: {confidence:.3f}, method: {method_name})")
                                break
                except:
                    continue
            
            if best_result:
                x_center = x + w/2
                detected.append((best_result, x_center, best_confidence))
        
        return detected
    
    def _preprocess_single_square(self, square: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Preprocess a single colored square for OCR with proven techniques"""
        processed = []
        
        # Massive upscaling first (10x for tiny digits)
        scale = 10
        upscaled = cv2.resize(square, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening to upscaled
        kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel_sharpen)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        
        # Original upscaled color
        processed.append(("upscaled_color", sharpened))
        
        # PROVEN METHOD: Adaptive threshold on inverted with morphological closing
        inverted = cv2.bitwise_not(gray)
        adaptive_inv = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        adaptive_inv_closed = cv2.morphologyEx(adaptive_inv, cv2.MORPH_CLOSE, kernel)
        processed.append(("adaptive_inv_closed", adaptive_inv_closed))  # This method works!
        
        # White text extraction with multiple thresholds  
        for thresh in [180, 200, 220]:
            _, white_thresh = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            processed.append((f"white_{thresh}", white_thresh))
        
        # Otsu on inverted
        _, otsu_inv = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(("otsu_inv", otsu_inv))
        
        return processed
    
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
                        if 1 <= number <= self.expected_horses:  # Only valid horse numbers for this race
                                # Get x-coordinate from bounding box
                                x_center = (bbox[0][0] + bbox[2][0]) / 2
                                
                                # Check if we already have this number at a similar position
                                duplicate = False
                                for existing_num, existing_x, _ in detected:
                                    if existing_num == number and abs(existing_x - x_center) < 20:
                                        duplicate = True
                                        break
                                
                                if not duplicate and confidence > 0.3:  # Higher confidence threshold
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
        """Preprocess the bar region to extract colored numbers - IMPROVED"""
        processed = []
        height, width = bar_region.shape[:2]
        
        # 1. Original color image first (works sometimes)
        processed.append(("original", bar_region))
        
        # 2. Convert to grayscale with proper weighting
        gray = cv2.cvtColor(bar_region, cv2.COLOR_BGR2GRAY)
        
        # 3. HSV processing - colored squares have distinct hue/saturation
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # High saturation areas (colored squares)
        _, sat_thresh = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)
        processed.append(("saturation", sat_thresh))
        
        # Value channel with better threshold
        _, val_thresh = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY)
        processed.append(("value_thresh", val_thresh))
        
        # 4. Individual color channels with OTSU thresholding
        b, g, r = cv2.split(bar_region)
        _, b_otsu = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, g_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, r_otsu = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(("blue_otsu", b_otsu))
        processed.append(("green_otsu", g_otsu))
        processed.append(("red_otsu", r_otsu))
        
        # 5. Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        processed.append(("enhanced", enhanced))
        
        # 6. Edge detection (numbers have strong edges)
        edges = cv2.Canny(enhanced, 50, 150)
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        processed.append(("edges", edges))
        
        # 7. Inverted (sometimes white text on colored background)
        inv_gray = cv2.bitwise_not(gray)
        processed.append(("inverted", inv_gray))
        
        # 8. Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(val_thresh, cv2.MORPH_CLOSE, kernel)
        processed.append(("morphological", morph))
        
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
        
        # Check for too many duplicates (indicates OCR error)
        unique_positions = len(set(positions))
        if unique_positions < len(positions) * 0.7:  # Less than 70% unique is suspicious
            return False
        
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
        
        # IMPROVED: Focus on final stretch for accurate finish positions
        # Look for readings in the last 20% of the race (final stretch)
        total_frames = self.position_history[-1].frame_num
        final_stretch_start = int(total_frames * 0.8)
        
        # Get all readings from final stretch
        final_stretch_readings = [s for s in self.position_history 
                                 if s.frame_num >= final_stretch_start]
        
        # Find the most complete reading from final stretch
        if final_stretch_readings:
            # Prefer readings with most horses detected
            end_snapshot = max(final_stretch_readings, 
                             key=lambda s: (len(s.positions), s.confidence))
            logger.info(f"Using final stretch snapshot from frame {end_snapshot.frame_num} "
                       f"with positions: {end_snapshot.positions} (conf: {end_snapshot.confidence:.2f})")
        else:
            # Fallback to last 10 readings
            recent_readings = self.position_history[-10:] if len(self.position_history) >= 10 else self.position_history
            if recent_readings:
                end_snapshot = max(recent_readings, key=lambda s: len(s.positions))
                logger.info(f"Using recent snapshot from frame {end_snapshot.frame_num} with positions: {end_snapshot.positions}")
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