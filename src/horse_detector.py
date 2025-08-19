import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import supervision as sv
from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass
class HorseDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    track_id: Optional[int] = None
    jersey_number: Optional[str] = None
    color_features: Optional[np.ndarray] = None


class HorseDetector:
    def __init__(self, model_path: Optional[str] = None):
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8x.pt')
            logger.info("Using pretrained YOLOv8x model")
        
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        
        self.horse_class_id = None
        self.person_class_id = None
        self._identify_class_ids()
        
    def _identify_class_ids(self):
        class_names = self.model.names
        for idx, name in class_names.items():
            if 'horse' in name.lower():
                self.horse_class_id = idx
            elif 'person' in name.lower():
                self.person_class_id = idx
        
        if self.horse_class_id is None:
            self.horse_class_id = 17
            logger.warning("Horse class not found, using default ID 17")
        if self.person_class_id is None:
            self.person_class_id = 0
            logger.warning("Person class not found, using default ID 0")
    
    def detect_horses(self, frame: np.ndarray) -> List[HorseDetection]:
        results = self.model(frame, conf=0.3)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    
                    if class_id == self.horse_class_id:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        detection = HorseDetection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=conf
                        )
                        
                        detection.color_features = self._extract_color_features(
                            frame, detection.bbox
                        )
                        
                        detections.append(detection)
        
        return detections
    
    def _extract_color_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(12)
        
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        hist_h = cv2.calcHist([roi_hsv], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([roi_hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([roi_hsv], [2], None, [32], [0, 256])
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        dominant_colors = self._get_dominant_colors(roi)
        
        features = np.concatenate([
            hist_h[:4],
            hist_s[:4],
            hist_v[:4],
            dominant_colors.flatten()
        ])
        
        return features[:12]
    
    def _get_dominant_colors(self, roi: np.ndarray, k: int = 3) -> np.ndarray:
        if roi.size == 0:
            return np.zeros((k, 3))
        
        pixels = roi.reshape(-1, 3).astype(np.float32)
        
        if len(pixels) < k:
            return np.zeros((k, 3))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers / 255.0
    
    def track_horses(self, detections: List[HorseDetection], frame_shape: Tuple[int, int]) -> List[HorseDetection]:
        if not detections:
            return detections
        
        detection_array = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence]
            for d in detections
        ])
        
        sv_detections = sv.Detections(
            xyxy=detection_array[:, :4],
            confidence=detection_array[:, 4],
            class_id=np.full(len(detections), self.horse_class_id)
        )
        
        tracked = self.tracker.update_with_detections(sv_detections)
        
        for i, track_id in enumerate(tracked.tracker_id):
            if i < len(detections):
                detections[i].track_id = int(track_id) if track_id is not None else None
        
        return detections
    
    def annotate_frame(self, frame: np.ndarray, detections: List[HorseDetection]) -> np.ndarray:
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            color = (0, 255, 0) if det.confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"Horse"
            if det.track_id is not None:
                label += f" #{det.track_id}"
            if det.jersey_number:
                label += f" ({det.jersey_number})"
            label += f" {det.confidence:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated


class JockeyColorIdentifier:
    def __init__(self):
        self.known_colors = {}
        
    def identify_jersey(self, frame: np.ndarray, horse_bbox: Tuple[int, int, int, int]) -> Optional[str]:
        x1, y1, x2, y2 = horse_bbox
        
        jockey_y1 = max(0, y1 - int((y2 - y1) * 0.3))
        jockey_y2 = y1 + int((y2 - y1) * 0.3)
        jockey_x1 = x1 + int((x2 - x1) * 0.2)
        jockey_x2 = x2 - int((x2 - x1) * 0.2)
        
        if jockey_y2 <= jockey_y1 or jockey_x2 <= jockey_x1:
            return None
        
        jockey_roi = frame[jockey_y1:jockey_y2, jockey_x1:jockey_x2]
        
        if jockey_roi.size == 0:
            return None
        
        dominant_color = self._get_dominant_color(jockey_roi)
        
        return self._match_to_known_jersey(dominant_color)
    
    def _get_dominant_color(self, roi: np.ndarray) -> np.ndarray:
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        pixels = roi_hsv.reshape(-1, 3).astype(np.float32)
        
        if len(pixels) < 10:
            return np.array([0, 0, 0])
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers[0]
    
    def _match_to_known_jersey(self, color: np.ndarray) -> Optional[str]:
        return None
    
    def register_jersey(self, jersey_number: str, color_sample: np.ndarray):
        self.known_colors[jersey_number] = color_sample