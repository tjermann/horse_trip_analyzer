import cv2
import numpy as np
import easyocr
from typing import List, Set, Optional, Tuple
from loguru import logger
import torch
from .position_bar_tracker import PositionBarTracker


class RaceStartDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.detected_numbers = set()
        self.frames_analyzed = 0
        self.confidence_threshold = 0.5
        
    def detect_horse_numbers_from_start(self, video_path: str, duration_seconds: int = 15) -> Set[int]:
        """
        Analyzes 1 frame per second from the first 15 seconds to detect horse numbers
        displayed on screen at race start.
        
        Args:
            video_path: Path to the race video
            duration_seconds: Duration in seconds to analyze (default 15)
            
        Returns:
            Set of detected horse numbers
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return set()
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        logger.info(f"Analyzing {duration_seconds} frames (1 per second) from first {duration_seconds} seconds...")
        
        from tqdm import tqdm
        all_detected_numbers = set()
        
        # Progress bar for auto-detection
        pbar = tqdm(total=duration_seconds, desc="Auto-detecting horses", unit="seconds")
        
        # Sample 1 frame per second for the specified duration
        for second in range(duration_seconds):
            # Jump to the exact second
            frame_number = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Look for number displays in different regions
            numbers = self._detect_numbers_in_frame(frame, frame_number)
            all_detected_numbers.update(numbers)
            
            pbar.update(1)
            
            # Update progress bar description with current findings
            if len(all_detected_numbers) > 0:
                pbar.set_description(f"Found {len(all_detected_numbers)} horses: {sorted(all_detected_numbers)}")
            
            # Early exit if we found a reasonable number of horses
            if len(all_detected_numbers) >= 12:  # Stop if we find more than expected
                logger.info(f"Found {len(all_detected_numbers)} horses, stopping analysis")
                break
        
        pbar.close()
        cap.release()
        
        # Filter to reasonable horse numbers (1-20 typically)
        valid_numbers = {n for n in all_detected_numbers if 1 <= n <= 20}
        
        logger.info(f"Detected horse numbers: {sorted(valid_numbers)}")
        return valid_numbers
    
    def _detect_numbers_in_frame(self, frame: np.ndarray, frame_num: int) -> Set[int]:
        """Detect horse numbers specifically from the virtual position bar"""
        height, width = frame.shape[:2]
        detected = set()
        
        # Focus on the virtual position bar area (15% from bottom, spanning full width)
        bar_height = int(height * 0.08)  # Bar is roughly 8% of screen height
        bar_y_start = int(height * 0.82)  # Start at 82% down (18% from bottom)
        bar_y_end = bar_y_start + bar_height
        
        # The position bar spans most of the width but might have margins
        bar_x_start = int(width * 0.05)  # 5% margin from left
        bar_x_end = int(width * 0.95)    # 5% margin from right
        
        position_bar_roi = frame[bar_y_start:bar_y_end, bar_x_start:bar_x_end]
        
        if position_bar_roi.size > 0:
            numbers = self._extract_numbers_from_roi(position_bar_roi, frame_num, region_name="position_bar")
            detected.update(numbers)
            
            if len(numbers) > 0:
                logger.debug(f"Frame {frame_num}: Found numbers {sorted(numbers)} in position bar")
        
        # Also check a slightly wider area around the position bar as backup
        extended_bar_y_start = int(height * 0.78)  # 22% from bottom
        extended_bar_y_end = int(height * 0.92)    # 8% from bottom
        
        extended_roi = frame[extended_bar_y_start:extended_bar_y_end, bar_x_start:bar_x_end]
        
        if extended_roi.size > 0:
            backup_numbers = self._extract_numbers_from_roi(extended_roi, frame_num, region_name="extended_bar")
            # Only add numbers that weren't already found
            new_numbers = backup_numbers - detected
            if len(new_numbers) > 0:
                detected.update(new_numbers)
                logger.debug(f"Frame {frame_num}: Found additional numbers {sorted(new_numbers)} in extended bar area")
        
        return detected
    
    def _extract_numbers_from_roi(self, roi: np.ndarray, frame_num: int, region_name: str = "roi") -> Set[int]:
        """Extract numbers from a region of interest"""
        numbers = set()
        
        # Try different preprocessing approaches
        processed_images = self._preprocess_for_ocr(roi)
        
        for img_name, processed in processed_images:
            try:
                results = self.reader.readtext(processed, allowlist='0123456789')
                
                for (_, text, confidence) in results:
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Look for single or double digit numbers
                    text = text.strip()
                    if text.isdigit():
                        number = int(text)
                        if 1 <= number <= 20:  # Reasonable horse number range
                            numbers.add(number)
                            logger.debug(f"Frame {frame_num}: Found number {number} "
                                       f"(confidence: {confidence:.2f}, method: {img_name}, region: {region_name})")
                    
                    # Also check for numbers in longer strings
                    import re
                    digit_matches = re.findall(r'\b(\d{1,2})\b', text)
                    for match in digit_matches:
                        number = int(match)
                        if 1 <= number <= 20:
                            numbers.add(number)
                            logger.debug(f"Frame {frame_num}: Extracted number {number} "
                                       f"from '{text}' (method: {img_name}, region: {region_name})")
                            
            except Exception as e:
                logger.debug(f"OCR failed on {img_name}: {e}")
                continue
        
        return numbers
    
    def _preprocess_for_ocr(self, roi: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Apply different preprocessing techniques for better OCR"""
        if roi.size == 0:
            return []
        
        # Resize if too small
        if min(roi.shape[:2]) < 50:
            scale = 50 / min(roi.shape[:2])
            new_width = int(roi.shape[1] * scale)
            new_height = int(roi.shape[0] * scale)
            roi = cv2.resize(roi, (new_width, new_height))
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        processed_images = [
            ("original", gray)
        ]
        
        # Binary threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("binary", thresh))
        
        # Inverted binary
        _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(("binary_inv", thresh_inv))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed_images.append(("adaptive", adaptive))
        
        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed_images.append(("cleaned", cleaned))
        
        # Edge enhancement
        edges = cv2.Canny(gray, 50, 150)
        processed_images.append(("edges", edges))
        
        # High contrast regions (good for overlaid text)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(("enhanced", enhanced))
        
        return processed_images
    
    def detect_starting_lineup_display(self, video_path: str) -> Optional[int]:
        """
        Specifically look for frames that show the complete starting lineup,
        often displayed as a grid or list at race start.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        frame_count = 0
        max_numbers_found = 0
        
        # Look in first 10 seconds
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames = int(10 * fps)
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Look for frames with many numbers (indicating lineup display)
            numbers = self._detect_numbers_in_frame(frame, frame_count)
            
            if len(numbers) > max_numbers_found:
                max_numbers_found = len(numbers)
                logger.info(f"Frame {frame_count}: Found {len(numbers)} numbers: {sorted(numbers)}")
            
            frame_count += 1
        
        cap.release()
        
        if max_numbers_found >= 4:  # At least 4 horses to be considered valid
            logger.success(f"Detected {max_numbers_found} horses from starting lineup")
            return max_numbers_found
        
        return None
    
    def get_horse_count_and_numbers(self, video_path: str) -> Tuple[int, Set[int]]:
        """
        Main method to get both the count and specific numbers of horses
        Uses the position bar tracker for accurate detection
        
        Returns:
            Tuple of (number_of_horses, set_of_horse_numbers)
        """
        # Use position bar tracker for more accurate detection
        logger.info("Using position bar tracker for horse detection...")
        position_tracker = PositionBarTracker()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return 8, set(range(1, 9))
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Sample frames throughout the first 60 seconds to catch position bar updates
        duration_seconds = 60
        sample_interval = 3  # Every 3 seconds
        
        from tqdm import tqdm
        samples = duration_seconds // sample_interval
        pbar = tqdm(total=samples, desc="Tracking position bar", unit="samples")
        
        all_horses = set()
        
        for second in range(0, duration_seconds, sample_interval):
            frame_number = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            reading = position_tracker.read_position_bar(frame, frame_number, fps)
            if reading:
                frame_horses = set(reading.positions.values())
                all_horses.update(frame_horses)
                
                pbar.set_description(f"Found {len(all_horses)} horses: {sorted(all_horses)}")
            
            pbar.update(1)
            
            # Early exit if we've found a stable set of horses
            if len(all_horses) >= 8 and second > 15:  # After 15 seconds, if we have 8+ horses
                logger.info(f"Found stable set of {len(all_horses)} horses, stopping early")
                break
        
        pbar.close()
        cap.release()
        
        if len(all_horses) >= 4:
            logger.success(f"Position bar detected {len(all_horses)} horses: {sorted(all_horses)}")
            return len(all_horses), all_horses
        
        # Fallback to original method if position bar detection failed
        logger.info("Position bar detection insufficient, trying fallback method...")
        horse_numbers = self.detect_horse_numbers_from_start(video_path)
        
        if len(horse_numbers) >= 4:
            logger.success(f"Fallback method found {len(horse_numbers)} horses: {sorted(horse_numbers)}")
            return len(horse_numbers), horse_numbers
        
        # Final fallback
        logger.warning("Could not auto-detect horse count, defaulting to 8")
        return 8, set(range(1, 9))


def detect_race_horses(video_path: str) -> Tuple[int, Set[int]]:
    """
    Convenience function to detect horses in a race video
    
    Returns:
        Tuple of (number_of_horses, set_of_horse_numbers)
    """
    detector = RaceStartDetector()
    return detector.get_horse_count_and_numbers(video_path)