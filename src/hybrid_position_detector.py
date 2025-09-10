#!/usr/bin/env python3
"""
Hybrid position detection system combining:
1. Enhanced OCR with multiple preprocessing techniques
2. Custom CNN model for position bar numbers
3. Visual tracking verification
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import easyocr
from scipy import stats
import os


@dataclass
class PositionReading:
    """A position reading from any source"""
    horse_number: int
    position: int
    confidence: float
    source: str  # 'ocr', 'cnn', 'visual'
    frame_num: int


class EnhancedOCRProcessor:
    """
    Advanced OCR with multiple preprocessing techniques
    """
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
    def preprocess_advanced(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply multiple advanced preprocessing techniques
        """
        preprocessed = []
        
        # 1. Adaptive histogram equalization
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        preprocessed.append(("clahe", enhanced))
        
        # 2. Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        preprocessed.append(("morphological", cleaned))
        
        # 3. Bilateral filter to preserve edges while reducing noise
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        preprocessed.append(("bilateral", bilateral))
        
        # 4. Adaptive thresholding for different lighting conditions
        adaptive = cv2.adaptiveThreshold(gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(("adaptive", adaptive))
        
        # 5. Color channel separation (if color image)
        if len(image.shape) == 3:
            # Try each color channel separately
            b, g, r = cv2.split(image)
            preprocessed.append(("blue_channel", b))
            preprocessed.append(("green_channel", g))
            preprocessed.append(("red_channel", r))
            
            # HSV processing
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            preprocessed.append(("saturation", s))
            preprocessed.append(("value", v))
        
        # 6. Edge-enhanced version
        edges = cv2.Canny(enhanced, 50, 150)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        preprocessed.append(("edges", dilated_edges))
        
        return preprocessed
    
    def extract_numbers_with_confidence(self, image: np.ndarray, 
                                       expected_horses: int) -> List[Tuple[int, float]]:
        """
        Extract numbers with confidence scores using multiple preprocessing
        """
        all_detections = []
        
        # Try multiple ROI regions for position bar
        height, width = image.shape[:2]
        roi_regions = [
            image,  # Full region
            image[int(height*0.7):, :],  # Bottom 30%
            image[int(height*0.6):, :],  # Bottom 40%
            image[:int(height*0.3), :],  # Top 30% (sometimes position bar is at top)
        ]
        
        for roi in roi_regions:
            preprocessed_images = self.preprocess_advanced(roi)
        
            for method_name, processed_img in preprocessed_images:
                try:
                    # Try OCR with different parameters
                    for allowlist in ['123456789', '0123456789']:
                        results = self.reader.readtext(processed_img, 
                                                      allowlist=allowlist,
                                                      paragraph=False,
                                                      width_ths=0.7,
                                                      height_ths=0.7)
                        
                        for bbox, text, confidence in results:
                            text = text.strip()
                            if text.isdigit():
                                number = int(text)
                                if 1 <= number <= expected_horses:
                                    x_center = (bbox[0][0] + bbox[2][0]) / 2
                                    all_detections.append({
                                        'number': number,
                                        'x_pos': x_center,
                                        'confidence': confidence,
                                        'method': method_name
                                    })
                except:
                    continue
        
        # Aggregate detections by number and position
        aggregated = self._aggregate_detections(all_detections)
        return aggregated
    
    def _aggregate_detections(self, detections: List[Dict]) -> List[Tuple[int, float]]:
        """
        Aggregate multiple detections into consensus readings
        """
        if not detections:
            return []
        
        # Group by approximate x position
        position_groups = {}
        for det in detections:
            # Round x position to nearest 50 pixels
            x_bucket = int(det['x_pos'] / 50) * 50
            if x_bucket not in position_groups:
                position_groups[x_bucket] = []
            position_groups[x_bucket].append(det)
        
        # For each position, find most likely number
        results = []
        for x_bucket, group in position_groups.items():
            # Count occurrences of each number
            number_counts = {}
            confidence_sum = {}
            
            for det in group:
                num = det['number']
                conf = det['confidence']
                if num not in number_counts:
                    number_counts[num] = 0
                    confidence_sum[num] = 0
                number_counts[num] += 1
                confidence_sum[num] += conf
            
            # Choose number with highest total confidence
            best_num = max(confidence_sum.keys(), 
                          key=lambda n: confidence_sum[n])
            avg_confidence = confidence_sum[best_num] / number_counts[best_num]
            
            # Boost confidence if detected by multiple methods
            if number_counts[best_num] > 1:
                avg_confidence = min(1.0, avg_confidence * 1.2)
            
            results.append((best_num, x_bucket, avg_confidence))
        
        # Sort by x position and extract numbers with confidence
        results.sort(key=lambda x: x[1])
        return [(num, conf) for num, _, conf in results]


class PositionBarCNN(nn.Module):
    """
    Custom CNN for position bar number recognition
    """
    
    def __init__(self, num_classes=10):
        super(PositionBarCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate size after convolutions
        # Assuming 32x32 input: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolution blocks
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)


class CNNPositionReader:
    """
    CNN-based position bar reader
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default to trained model if it exists
        if model_path is None:
            default_model_path = "models/position_cnn_best.pth"
            if os.path.exists(default_model_path):
                model_path = default_model_path
        
        # Initialize model - will adjust num_classes based on trained model
        model_classes = 20  # Default
        
        # Check if trained model exists and get its size
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'fc3.weight' in checkpoint:
                    model_classes = checkpoint['fc3.weight'].shape[0]
                    logger.info(f"Found trained model with {model_classes} classes")
            except:
                pass
        
        self.model = PositionBarCNN(num_classes=model_classes).to(self.device)
        
        if model_path and os.path.exists(model_path):
            # Load pre-trained model
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info(f"Loaded pre-trained CNN model from {model_path}")
                self.is_trained = True
            except Exception as e:
                logger.warning(f"Could not load CNN model: {e}")
                self.is_trained = False
        else:
            logger.info("Using untrained CNN (run train_position_cnn.py to train)")
            self.is_trained = False
    
    def extract_digit_regions(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Extract individual digit regions from position bar
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find contours of potential digits
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (digits should be roughly square)
            aspect_ratio = w / float(h)
            if 0.3 < aspect_ratio < 3.0 and h > 10 and w > 5:
                # Extract and resize to 32x32
                digit = gray[y:y+h, x:x+w]
                digit_resized = cv2.resize(digit, (32, 32))
                digit_regions.append((digit_resized, x))
        
        # Sort by x position
        digit_regions.sort(key=lambda x: x[1])
        return digit_regions
    
    def predict_with_confidence(self, image: np.ndarray) -> List[Tuple[int, float]]:
        """
        Predict numbers with confidence scores
        """
        digit_regions = self.extract_digit_regions(image)
        predictions = []
        
        for digit_img, x_pos in digit_regions:
            # Prepare for CNN
            digit_tensor = torch.FloatTensor(digit_img).unsqueeze(0).unsqueeze(0)
            digit_tensor = digit_tensor.to(self.device) / 255.0
            
            with torch.no_grad():
                output = self.model(digit_tensor)
                probabilities = torch.exp(output)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Convert 0-indexed prediction back to 1-indexed horse number
                horse_number = predicted.item() + 1
                
                # Adjust confidence based on whether model is trained
                if not self.is_trained:
                    confidence = confidence.item() * 0.1  # Very low confidence for untrained
                else:
                    confidence = confidence.item()
                
                predictions.append((horse_number, x_pos, confidence))
        
        # Sort by x position and return
        predictions.sort(key=lambda x: x[1])
        return [(num, conf) for num, _, conf in predictions]


class VisualPositionVerifier:
    """
    Verify positions using visual tracking consistency
    """
    
    def __init__(self):
        self.position_history = {}  # horse_id -> list of positions
        self.velocity_history = {}  # horse_id -> list of velocities
        
    def verify_position_change(self, horse_id: int, 
                               current_pos: int, 
                               proposed_pos: int,
                               time_delta: float) -> float:
        """
        Verify if a position change is physically plausible
        Returns confidence (0-1) in the proposed position
        """
        
        if horse_id not in self.position_history:
            self.position_history[horse_id] = []
            self.velocity_history[horse_id] = []
            return 0.5  # Neutral confidence for first reading
        
        history = self.position_history[horse_id]
        
        if len(history) < 2:
            return 0.7  # Moderate confidence with little history
        
        # Calculate expected position based on recent velocity
        recent_positions = history[-3:]
        if len(recent_positions) >= 2:
            # Simple linear extrapolation
            velocity = (recent_positions[-1] - recent_positions[-2]) / time_delta
            expected_pos = recent_positions[-1] + velocity * time_delta
            
            # Calculate confidence based on deviation from expected
            deviation = abs(proposed_pos - expected_pos)
            
            # Horses typically don't jump more than 2 positions per second
            max_reasonable_change = 2.0 * time_delta
            
            if deviation <= 0.5:
                confidence = 0.95  # Very close to expected
            elif deviation <= max_reasonable_change:
                confidence = 0.7 - (deviation / max_reasonable_change) * 0.5
            else:
                confidence = 0.2  # Unlikely position jump
            
            return confidence
        
        return 0.5
    
    def update_history(self, horse_id: int, position: int):
        """Update position history for a horse"""
        if horse_id not in self.position_history:
            self.position_history[horse_id] = []
        self.position_history[horse_id].append(position)
        
        # Keep only recent history
        if len(self.position_history[horse_id]) > 10:
            self.position_history[horse_id] = self.position_history[horse_id][-10:]


class HybridPositionDetector:
    """
    Main hybrid system combining all three approaches
    """
    
    def __init__(self, num_horses: int = 8):
        self.num_horses = num_horses
        self.ocr_processor = EnhancedOCRProcessor()
        self.cnn_reader = CNNPositionReader()
        self.visual_verifier = VisualPositionVerifier()
        
        # Confidence weights for fusion
        # CNN weight reduced since model is untrained
        self.ocr_weight = 0.5  # Increased - OCR is more reliable
        self.cnn_weight = 0.1  # Reduced - untrained network gives random output
        self.visual_weight = 0.4  # Increased - visual tracking is fairly reliable
        
    def detect_positions(self, 
                         frame: np.ndarray,
                         visual_detections: List,
                         frame_num: int,
                         fps: float) -> Dict[int, Tuple[int, float]]:
        """
        Detect horse positions using all three methods
        
        Returns:
            Dict[horse_number, (position, confidence)]
        """
        
        # Extract position bar region - try multiple areas
        height, width = frame.shape[:2]
        
        # Primary region (original)
        y1 = int(height * 0.75)
        y2 = int(height * 0.87)
        x1 = int(width * 0.10)
        x2 = int(width * 0.95)
        position_bar = frame[y1:y2, x1:x2]
        
        # Also prepare alternative regions for fallback
        bottom_bar = frame[int(height * 0.85):, x1:x2]  # Very bottom
        full_width_bar = frame[y1:y2, :]  # Full width
        
        # Method 1: Enhanced OCR
        ocr_readings = self.ocr_processor.extract_numbers_with_confidence(
            position_bar, self.num_horses)
        
        # Method 2: CNN predictions
        cnn_readings = self.cnn_reader.predict_with_confidence(position_bar)
        
        # Method 3: Visual tracking order
        visual_positions = self._get_visual_positions(visual_detections)
        
        # Fuse all three sources
        fused_positions = self._fuse_position_sources(
            ocr_readings, cnn_readings, visual_positions, frame_num, fps)
        
        return fused_positions
    
    def _get_visual_positions(self, detections: List) -> Dict[int, int]:
        """
        Get positions based on visual tracking (left-to-right order)
        """
        if not detections:
            return {}
        
        # Sort by x-coordinate
        sorted_detections = sorted(detections, 
                                 key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)
        
        positions = {}
        for idx, det in enumerate(sorted_detections, 1):
            if hasattr(det, 'track_id'):
                positions[det.track_id] = idx
        
        return positions
    
    def _fuse_position_sources(self,
                               ocr_readings: List[Tuple[int, float]],
                               cnn_readings: List[Tuple[int, float]],
                               visual_positions: Dict[int, int],
                               frame_num: int,
                               fps: float) -> Dict[int, Tuple[int, float]]:
        """
        Fuse multiple position sources using adaptive weighted confidence
        """
        
        # Convert readings to position mappings
        ocr_positions = {}
        for idx, (horse_num, conf) in enumerate(ocr_readings):
            ocr_positions[horse_num] = (idx + 1, conf)
        
        cnn_positions = {}
        for idx, (horse_num, conf) in enumerate(cnn_readings):
            cnn_positions[horse_num] = (idx + 1, conf)
        
        # Adaptive weights based on data quality
        # If OCR has high confidence, increase its weight
        avg_ocr_conf = np.mean([conf for _, conf in ocr_readings]) if ocr_readings else 0
        avg_cnn_conf = np.mean([conf for _, conf in cnn_readings]) if cnn_readings else 0
        
        # Dynamic weight adjustment based on model training status and confidence
        if self.cnn_reader.is_trained:
            # CNN is trained - can give it more weight
            if avg_ocr_conf > 0.8 and avg_cnn_conf > 0.8:
                # Both high confidence
                ocr_weight = 0.4
                cnn_weight = 0.4
                visual_weight = 0.2
            elif avg_ocr_conf > 0.8:
                ocr_weight = 0.5
                cnn_weight = 0.3
                visual_weight = 0.2
            elif avg_cnn_conf > 0.8:
                ocr_weight = 0.3
                cnn_weight = 0.5
                visual_weight = 0.2
            else:
                # Use balanced weights
                ocr_weight = 0.35
                cnn_weight = 0.35
                visual_weight = 0.3
        else:
            # CNN untrained - minimize its weight
            if avg_ocr_conf > 0.8:
                ocr_weight = 0.7  # High confidence OCR gets more weight
                cnn_weight = 0.05  # Reduce untrained CNN
                visual_weight = 0.25
            elif avg_ocr_conf < 0.4:
                ocr_weight = 0.2  # Low confidence OCR gets less weight
                cnn_weight = 0.05  # Keep CNN minimal
                visual_weight = 0.75  # Rely more on visual tracking
            else:
                ocr_weight = self.ocr_weight  # Use default weights
                cnn_weight = self.cnn_weight
                visual_weight = self.visual_weight
        
        # Combine all horses detected
        all_horses = set()
        all_horses.update(ocr_positions.keys())
        all_horses.update(cnn_positions.keys())
        all_horses.update(visual_positions.keys())
        
        fused_positions = {}
        
        for horse_id in all_horses:
            position_votes = []
            
            # OCR vote
            if horse_id in ocr_positions:
                pos, conf = ocr_positions[horse_id]
                position_votes.append((pos, conf * ocr_weight, 'ocr'))
            
            # CNN vote
            if horse_id in cnn_positions:
                pos, conf = cnn_positions[horse_id]
                position_votes.append((pos, conf * cnn_weight, 'cnn'))
            
            # Visual tracking vote with verification
            if horse_id in visual_positions:
                pos = visual_positions[horse_id]
                # Verify against history
                visual_conf = self.visual_verifier.verify_position_change(
                    horse_id, 
                    self.visual_verifier.position_history.get(horse_id, [pos])[-1] if horse_id in self.visual_verifier.position_history else pos,
                    pos,
                    1.0 / fps
                )
                position_votes.append((pos, visual_conf * visual_weight, 'visual'))
            
            if position_votes:
                # Weighted average position
                total_weight = sum(w for _, w, _ in position_votes)
                if total_weight > 0:
                    weighted_pos = sum(p * w for p, w, _ in position_votes) / total_weight
                    final_pos = round(weighted_pos)
                    
                    # Calculate fusion confidence
                    # Higher confidence if sources agree
                    positions_only = [p for p, _, _ in position_votes]
                    if len(set(positions_only)) == 1:
                        # All sources agree
                        fusion_confidence = min(0.95, total_weight * 1.5)
                    else:
                        # Sources disagree - use variance
                        if len(positions_only) > 1:
                            variance = np.var(positions_only)
                            fusion_confidence = max(0.3, total_weight * (1 - variance/10))
                        else:
                            fusion_confidence = total_weight
                    
                    fused_positions[horse_id] = (final_pos, fusion_confidence)
                    
                    # Update visual history
                    self.visual_verifier.update_history(horse_id, final_pos)
                    
                    # Log fusion decision
                    sources_str = ', '.join([f"{s}={p}" for p, _, s in position_votes])
                    logger.debug(f"Horse {horse_id}: {sources_str} → Position {final_pos} (conf: {fusion_confidence:.2f})")
        
        # Ensure unique positions
        fused_positions = self._enforce_unique_positions(fused_positions)
        
        return fused_positions
    
    def _enforce_unique_positions(self, 
                                 positions: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
        """
        Ensure each position is assigned to only one horse
        """
        # Sort by confidence
        sorted_horses = sorted(positions.items(), 
                             key=lambda x: x[1][1], reverse=True)
        
        assigned_positions = set()
        final_positions = {}
        
        for horse_id, (pos, conf) in sorted_horses:
            if pos not in assigned_positions:
                final_positions[horse_id] = (pos, conf)
                assigned_positions.add(pos)
            else:
                # Find next available position
                for alt_pos in range(1, self.num_horses + 1):
                    if alt_pos not in assigned_positions:
                        final_positions[horse_id] = (alt_pos, conf * 0.7)  # Reduce confidence
                        assigned_positions.add(alt_pos)
                        logger.debug(f"Reassigned Horse {horse_id} from {pos} to {alt_pos}")
                        break
        
        return final_positions