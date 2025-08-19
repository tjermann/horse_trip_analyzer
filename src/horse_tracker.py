import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import easyocr
from scipy.spatial import distance
from collections import defaultdict
from loguru import logger


@dataclass
class TrackedHorse:
    horse_id: int  # The actual horse number (1-8 in this race)
    current_bbox: Optional[Tuple[int, int, int, int]] = None
    last_seen_frame: int = 0
    appearance_features: List[np.ndarray] = field(default_factory=list)
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    missing_frames: int = 0
    total_frames_tracked: int = 0
    
    def update(self, bbox: Tuple[int, int, int, int], features: np.ndarray, 
               frame_num: int, confidence: float):
        self.current_bbox = bbox
        self.last_seen_frame = frame_num
        self.appearance_features.append(features)
        if len(self.appearance_features) > 50:
            self.appearance_features.pop(0)
        
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.position_history.append(center)
        if len(self.position_history) > 100:
            self.position_history.pop(0)
        
        self.confidence_scores.append(confidence)
        self.missing_frames = 0
        self.total_frames_tracked += 1
    
    def mark_missing(self):
        self.missing_frames += 1
        self.current_bbox = None
    
    @property
    def is_lost(self) -> bool:
        return self.missing_frames > 30  # Lost if not seen for 1 second at 30fps
    
    @property
    def average_features(self) -> np.ndarray:
        if not self.appearance_features:
            return np.zeros(512)
        return np.mean(self.appearance_features, axis=0)


class HorseNumberReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.number_cache = {}
        
    def read_horse_number(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        x1, y1, x2, y2 = bbox
        
        # Expand region to capture saddle cloth area
        height = y2 - y1
        width = x2 - x1
        
        # Saddle cloth is typically on the side of the horse
        cloth_y1 = y1 + int(height * 0.3)
        cloth_y2 = y1 + int(height * 0.7)
        cloth_x1 = x1
        cloth_x2 = x2
        
        cloth_region = frame[cloth_y1:cloth_y2, cloth_x1:cloth_x2]
        
        if cloth_region.size == 0:
            return None
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(cloth_region, cv2.COLOR_BGR2GRAY)
        
        # Try multiple preprocessing approaches
        results = []
        
        # Approach 1: Direct OCR
        result1 = self._try_read_number(gray)
        if result1:
            results.append(result1)
        
        # Approach 2: Threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        result2 = self._try_read_number(thresh)
        if result2:
            results.append(result2)
        
        # Approach 3: Inverted threshold
        _, thresh_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        result3 = self._try_read_number(thresh_inv)
        if result3:
            results.append(result3)
        
        # Return most common result
        if results:
            from collections import Counter
            most_common = Counter(results).most_common(1)[0][0]
            return most_common
        
        return None
    
    def _try_read_number(self, image: np.ndarray) -> Optional[int]:
        try:
            results = self.reader.readtext(image, allowlist='0123456789')
            
            for (_, text, confidence) in results:
                if confidence > 0.5 and text.isdigit():
                    number = int(text)
                    if 1 <= number <= 20:  # Reasonable horse number range
                        return number
        except:
            pass
        
        return None


class ImprovedHorseTracker:
    def __init__(self, expected_horses: int = 8):
        self.expected_horses = expected_horses
        self.horses = {}  # Dict[int, TrackedHorse] - keyed by horse number
        self.number_reader = HorseNumberReader()
        self.frame_count = 0
        self.unidentified_detections = []
        
        # Initialize expected horses
        for i in range(1, expected_horses + 1):
            self.horses[i] = TrackedHorse(horse_id=i)
    
    def update(self, detections: List, frame: np.ndarray) -> Dict[int, TrackedHorse]:
        self.frame_count += 1
        
        # Mark all horses as missing initially
        for horse in self.horses.values():
            horse.mark_missing()
        
        # Process each detection
        assignments = {}  # detection_idx -> horse_id
        
        for det_idx, detection in enumerate(detections):
            bbox = detection.bbox
            features = self._extract_features(frame, bbox)
            
            # Try to read horse number
            horse_number = self.number_reader.read_horse_number(frame, bbox)
            
            if horse_number and horse_number in self.horses:
                # Direct number match
                assignments[det_idx] = horse_number
            else:
                # Use appearance matching
                best_match = self._find_best_match(features, bbox)
                if best_match:
                    assignments[det_idx] = best_match
        
        # Handle unassigned detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in assignments:
                # Try harder to identify the horse
                horse_id = self._identify_unassigned(detection, frame, features)
                if horse_id:
                    assignments[det_idx] = horse_id
        
        # Update tracked horses
        for det_idx, horse_id in assignments.items():
            detection = detections[det_idx]
            features = self._extract_features(frame, detection.bbox)
            self.horses[horse_id].update(
                bbox=detection.bbox,
                features=features,
                frame_num=self.frame_count,
                confidence=detection.confidence
            )
            
            # Update the detection with the correct horse ID
            detection.track_id = horse_id
        
        return self.horses
    
    def _extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(512)
        
        # Resize to fixed size
        roi = cv2.resize(roi, (128, 128))
        
        # Extract color histogram features
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Extract texture features using Gabor filters
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gabor_features = []
        
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_features.append(filtered.mean())
            gabor_features.append(filtered.std())
        
        # Combine all features
        features = np.concatenate([
            hist_h[:170],
            hist_s[:170],
            hist_v[:170],
            gabor_features
        ])
        
        # Pad or truncate to 512 dimensions
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)))
        else:
            features = features[:512]
        
        return features
    
    def _find_best_match(self, features: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        best_horse_id = None
        best_score = float('inf')
        
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        for horse_id, horse in self.horses.items():
            if horse.is_lost:
                continue
            
            # Skip if horse was just seen (avoid double assignment)
            if horse.missing_frames == 0:
                continue
            
            # Appearance similarity
            if len(horse.appearance_features) > 0:
                appearance_dist = distance.cosine(features, horse.average_features)
            else:
                appearance_dist = 1.0
            
            # Position prediction
            if len(horse.position_history) > 1:
                # Simple linear prediction
                last_pos = horse.position_history[-1]
                if len(horse.position_history) > 2:
                    prev_pos = horse.position_history[-2]
                    velocity = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
                    predicted_pos = (
                        last_pos[0] + velocity[0] * horse.missing_frames,
                        last_pos[1] + velocity[1] * horse.missing_frames
                    )
                else:
                    predicted_pos = last_pos
                
                position_dist = distance.euclidean(center, predicted_pos) / 100.0
            else:
                position_dist = 1.0
            
            # Combined score (weighted)
            score = appearance_dist * 0.7 + position_dist * 0.3
            
            if score < best_score and score < 0.5:  # Threshold for match
                best_score = score
                best_horse_id = horse_id
        
        return best_horse_id
    
    def _identify_unassigned(self, detection, frame: np.ndarray, features: np.ndarray) -> Optional[int]:
        # Find horses that haven't been seen recently
        missing_horses = [
            horse_id for horse_id, horse in self.horses.items()
            if horse.missing_frames > 5 and not horse.is_lost
        ]
        
        if not missing_horses:
            return None
        
        # If only one horse is missing, assign it
        if len(missing_horses) == 1:
            return missing_horses[0]
        
        # Otherwise, try to make best guess based on position
        # (Horses at the back are more likely to be higher numbers, etc.)
        bbox = detection.bbox
        x_center = (bbox[0] + bbox[2]) / 2
        frame_width = frame.shape[1]
        
        # Simple heuristic: horses on left tend to have lower numbers early in race
        if x_center < frame_width * 0.3:
            return min(missing_horses)
        elif x_center > frame_width * 0.7:
            return max(missing_horses)
        else:
            return missing_horses[len(missing_horses) // 2]
    
    def get_summary(self) -> Dict:
        summary = {
            "total_frames": self.frame_count,
            "horses_tracked": {}
        }
        
        for horse_id, horse in self.horses.items():
            summary["horses_tracked"][horse_id] = {
                "frames_tracked": horse.total_frames_tracked,
                "tracking_percentage": (horse.total_frames_tracked / self.frame_count * 100) if self.frame_count > 0 else 0,
                "currently_visible": horse.missing_frames == 0,
                "average_confidence": np.mean(horse.confidence_scores) if horse.confidence_scores else 0
            }
        
        return summary


import torch  # Add this import at the top