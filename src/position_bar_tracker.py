import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from loguru import logger
import torch


@dataclass
class PositionBarReading:
    frame_num: int
    positions: Dict[int, int]  # position (1st, 2nd, etc.) -> horse_number
    confidence_scores: Dict[int, float]
    timestamp: float


class PositionBarTracker:
    """
    Tracks the virtual position bar that shows live horse positions throughout the race.
    This bar typically appears at ~15% from bottom of screen and updates continuously.
    """
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.position_history = []
        self.bar_location = None  # Will be auto-detected
        self.confidence_threshold = 0.4
        
    def detect_position_bar_location(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Auto-detect the position bar location in the frame
        Returns (x1, y1, x2, y2) coordinates of the bar
        """
        height, width = frame.shape[:2]
        
        # Based on screenshot: colored numbers 1-8 at ~10-12% from bottom
        estimated_y_start = int(height * 0.85)  # 15% from bottom
        estimated_y_end = int(height * 0.92)    # 8% from bottom  
        estimated_x_start = int(width * 0.15)   # 15% from left (numbers are centered)
        estimated_x_end = int(width * 0.85)     # 15% from right
        
        # Look for horizontal bars/rectangles in this region
        roi = frame[estimated_y_start:estimated_y_end, estimated_x_start:estimated_x_end]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal lines/bars
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal structures
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find contours of horizontal structures
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest horizontal structure (likely the position bar)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Convert back to full frame coordinates
            bar_x1 = estimated_x_start + x
            bar_y1 = estimated_y_start + y
            bar_x2 = bar_x1 + w
            bar_y2 = bar_y1 + h
            
            # Expand slightly for better OCR
            bar_y1 = max(0, bar_y1 - 10)
            bar_y2 = min(height, bar_y2 + 10)
            
            self.bar_location = (bar_x1, bar_y1, bar_x2, bar_y2)
            logger.info(f"Detected position bar at: {self.bar_location}")
            return self.bar_location
        
        # Fallback to estimated location based on screenshot analysis
        self.bar_location = (estimated_x_start, estimated_y_start, estimated_x_end, estimated_y_end)
        logger.info(f"Using position bar location based on screenshot analysis: {self.bar_location}")
        return self.bar_location
    
    def read_position_bar(self, frame: np.ndarray, frame_num: int, fps: float = 30.0) -> Optional[PositionBarReading]:
        """
        Read horse numbers and positions from the position bar
        """
        if self.bar_location is None:
            self.detect_position_bar_location(frame)
        
        x1, y1, x2, y2 = self.bar_location
        bar_roi = frame[y1:y2, x1:x2]
        
        if bar_roi.size == 0:
            return None
        
        # Enhanced preprocessing for position bar OCR
        positions_dict = {}
        confidence_dict = {}
        
        # Try multiple preprocessing approaches
        processed_images = self._preprocess_position_bar(bar_roi)
        
        for method_name, processed in processed_images:
            try:
                results = self.reader.readtext(processed, allowlist='0123456789')
                
                for (bbox, text, confidence) in results:
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Extract horse numbers
                    import re
                    numbers = re.findall(r'\b(\d{1,2})\b', text.strip())
                    
                    for num_str in numbers:
                        number = int(num_str)
                        if 1 <= number <= 20:  # Valid horse number range
                            # Estimate position based on x-coordinate in bar
                            bbox_center_x = (bbox[0][0] + bbox[2][0]) / 2
                            bar_width = x2 - x1
                            relative_position = bbox_center_x / bar_width
                            
                            # Convert to position (1st, 2nd, etc.)
                            estimated_position = int(relative_position * 8) + 1
                            estimated_position = max(1, min(8, estimated_position))
                            
                            # Store best confidence for each horse number
                            if number not in confidence_dict or confidence > confidence_dict[number]:
                                positions_dict[estimated_position] = number
                                confidence_dict[number] = confidence
                                
                                logger.debug(f"Frame {frame_num}: Horse #{number} in position {estimated_position} "
                                           f"(confidence: {confidence:.2f}, method: {method_name})")
                            
            except Exception as e:
                logger.debug(f"Position bar OCR failed with {method_name}: {e}")
                continue
        
        if positions_dict:
            reading = PositionBarReading(
                frame_num=frame_num,
                positions=positions_dict,
                confidence_scores=confidence_dict,
                timestamp=frame_num / fps
            )
            
            self.position_history.append(reading)
            return reading
        
        return None
    
    def _preprocess_position_bar(self, roi: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Specialized preprocessing for position bar OCR"""
        if roi.size == 0:
            return []
        
        # Resize if too small
        if min(roi.shape[:2]) < 30:
            scale = 30 / min(roi.shape[:2])
            new_width = int(roi.shape[1] * scale)
            new_height = int(roi.shape[0] * scale)
            roi = cv2.resize(roi, (new_width, new_height))
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        processed_images = []
        
        # For colored numbers on overlay, try different color space extractions
        
        # Extract different color channels that might highlight the colored numbers
        bgr_channels = cv2.split(roi)
        processed_images.append(("blue_channel", bgr_channels[0]))
        processed_images.append(("green_channel", bgr_channels[1]))
        processed_images.append(("red_channel", bgr_channels[2]))
        
        # HSV color space - good for colored text
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_channels = cv2.split(hsv)
        processed_images.append(("hue_channel", hsv_channels[0]))
        processed_images.append(("saturation_channel", hsv_channels[1]))
        processed_images.append(("value_channel", hsv_channels[2]))
        
        # High contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(("enhanced", enhanced))
        
        # Binary thresholds with different parameters for colored overlays
        _, thresh_high = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        processed_images.append(("thresh_high", thresh_high))
        
        _, thresh_low = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        processed_images.append(("thresh_low", thresh_low))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed_images.append(("adaptive", adaptive))
        
        return processed_images
    
    def get_horses_in_race(self, min_appearances: int = 3) -> Set[int]:
        """
        Get set of all horse numbers that have appeared in the position bar
        with minimum number of appearances for confidence
        """
        horse_counts = {}
        
        for reading in self.position_history:
            for position, horse_num in reading.positions.items():
                horse_counts[horse_num] = horse_counts.get(horse_num, 0) + 1
        
        # Return horses that appeared at least min_appearances times
        return {horse for horse, count in horse_counts.items() if count >= min_appearances}
    
    def get_position_at_frame(self, frame_num: int) -> Optional[Dict[int, int]]:
        """Get horse positions at a specific frame"""
        # Find closest reading to the requested frame
        closest_reading = None
        min_distance = float('inf')
        
        for reading in self.position_history:
            distance = abs(reading.frame_num - frame_num)
            if distance < min_distance:
                min_distance = distance
                closest_reading = reading
        
        if closest_reading and min_distance <= 30:  # Within 1 second at 30fps
            return closest_reading.positions
        
        return None
    
    def get_summary(self) -> Dict:
        """Get summary of position bar tracking"""
        if not self.position_history:
            return {"readings": 0, "horses_detected": 0}
        
        horses_detected = self.get_horses_in_race(min_appearances=1)
        
        return {
            "readings": len(self.position_history),
            "horses_detected": len(horses_detected),
            "horse_numbers": sorted(horses_detected),
            "bar_location": self.bar_location,
            "frames_analyzed": [r.frame_num for r in self.position_history]
        }