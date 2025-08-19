import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import cv2
from scipy.spatial import distance
from scipy.signal import savgol_filter
from loguru import logger
import json


class TripEvent(Enum):
    BOXED_IN = "boxed_in"
    WIDE_TRIP = "wide_trip"
    BUMPED = "bumped"
    STEADIED = "steadied"
    CHECKED = "checked"
    FRONT_RUNNING = "front_running"
    STALKING = "stalking"
    CLOSING = "closing"
    TRAFFIC_TROUBLE = "traffic_trouble"
    RAIL_TRIP = "rail_trip"
    PACE_PRESSURE = "pace_pressure"


@dataclass
class HorsePosition:
    frame_num: int
    track_id: int
    bbox: Tuple[int, int, int, int]
    position_in_field: int = 0
    lateral_position: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    distance_traveled: float = 0.0
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass 
class TripAnalysis:
    track_id: int
    horse_name: Optional[str] = None
    events: List[Dict] = field(default_factory=list)
    trip_difficulty_score: float = 0.0
    pace_scenario: str = ""
    ground_loss: float = 0.0
    energy_distribution: Dict = field(default_factory=dict)
    position_chart: List[int] = field(default_factory=list)
    final_position: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "horse_name": self.horse_name,
            "trip_difficulty_score": self.trip_difficulty_score,
            "events": self.events,
            "pace_scenario": self.pace_scenario,
            "ground_loss": self.ground_loss,
            "energy_distribution": self.energy_distribution,
            "position_chart": self.position_chart,
            "final_position": self.final_position
        }


class TripAnalyzer:
    def __init__(self, track_width_meters: float = 12.0, fps: int = 30):
        self.track_width = track_width_meters
        self.fps = fps
        self.frame_buffer = {}
        self.horse_histories = {}
        
    def update_frame(self, frame_num: int, detections: List, frame_shape: Tuple[int, int]):
        frame_positions = []
        
        for det in detections:
            if det.track_id is None:
                continue
                
            pos = HorsePosition(
                frame_num=frame_num,
                track_id=det.track_id,
                bbox=det.bbox
            )
            
            pos.lateral_position = self._calculate_lateral_position(det.bbox, frame_shape)
            
            if det.track_id not in self.horse_histories:
                self.horse_histories[det.track_id] = []
            
            if len(self.horse_histories[det.track_id]) > 0:
                prev_pos = self.horse_histories[det.track_id][-1]
                pos.speed = self._calculate_speed(pos, prev_pos)
                pos.distance_traveled = prev_pos.distance_traveled + self._calculate_distance(pos, prev_pos)
                
                if len(self.horse_histories[det.track_id]) > 1:
                    prev_prev_pos = self.horse_histories[det.track_id][-2]
                    pos.acceleration = self._calculate_acceleration(pos, prev_pos, prev_prev_pos)
            
            self.horse_histories[det.track_id].append(pos)
            frame_positions.append(pos)
        
        self._calculate_relative_positions(frame_positions)
        self.frame_buffer[frame_num] = frame_positions
        
        self._detect_events(frame_num, frame_positions)
    
    def _calculate_lateral_position(self, bbox: Tuple[int, int, int, int], 
                                   frame_shape: Tuple[int, int]) -> float:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        return center_x / frame_shape[1]
    
    def _calculate_speed(self, current: HorsePosition, previous: HorsePosition) -> float:
        dist = distance.euclidean(current.center, previous.center)
        time_diff = (current.frame_num - previous.frame_num) / self.fps
        if time_diff > 0:
            return dist / time_diff
        return 0.0
    
    def _calculate_distance(self, current: HorsePosition, previous: HorsePosition) -> float:
        return distance.euclidean(current.center, previous.center)
    
    def _calculate_acceleration(self, current: HorsePosition, prev: HorsePosition, 
                               prev_prev: HorsePosition) -> float:
        if current.speed > 0 and prev.speed > 0:
            time_diff = (current.frame_num - prev.frame_num) / self.fps
            if time_diff > 0:
                return (current.speed - prev.speed) / time_diff
        return 0.0
    
    def _calculate_relative_positions(self, positions: List[HorsePosition]):
        if not positions:
            return
        
        sorted_positions = sorted(positions, key=lambda p: p.center[0])
        
        for i, pos in enumerate(sorted_positions):
            pos.position_in_field = i + 1
    
    def _detect_events(self, frame_num: int, positions: List[HorsePosition]):
        for pos in positions:
            history = self.horse_histories[pos.track_id]
            
            if len(history) < 10:
                continue
            
            recent_positions = history[-10:]
            
            if self._is_boxed_in(pos, positions):
                self._add_event(pos.track_id, frame_num, TripEvent.BOXED_IN)
            
            if self._is_wide_trip(recent_positions):
                self._add_event(pos.track_id, frame_num, TripEvent.WIDE_TRIP)
            
            if self._detect_bump(recent_positions):
                self._add_event(pos.track_id, frame_num, TripEvent.BUMPED)
            
            if self._detect_steadied(recent_positions):
                self._add_event(pos.track_id, frame_num, TripEvent.STEADIED)
    
    def _is_boxed_in(self, horse_pos: HorsePosition, all_positions: List[HorsePosition]) -> bool:
        if len(all_positions) < 3:
            return False
        
        x1, y1, x2, y2 = horse_pos.bbox
        horse_center = horse_pos.center
        
        horses_ahead = []
        horses_behind = []
        horses_side = []
        
        for other in all_positions:
            if other.track_id == horse_pos.track_id:
                continue
            
            other_center = other.center
            
            if other_center[0] < horse_center[0] - 50:
                horses_ahead.append(other)
            elif other_center[0] > horse_center[0] + 50:
                horses_behind.append(other)
            else:
                if abs(other_center[1] - horse_center[1]) < 100:
                    horses_side.append(other)
        
        return len(horses_ahead) > 0 and len(horses_behind) > 0 and len(horses_side) > 1
    
    def _is_wide_trip(self, recent_positions: List[HorsePosition]) -> bool:
        if len(recent_positions) < 5:
            return False
        
        lateral_positions = [p.lateral_position for p in recent_positions]
        avg_lateral = np.mean(lateral_positions)
        
        return avg_lateral > 0.7 or avg_lateral < 0.3
    
    def _detect_bump(self, recent_positions: List[HorsePosition]) -> bool:
        if len(recent_positions) < 3:
            return False
        
        accelerations = [p.acceleration for p in recent_positions[-3:]]
        
        if any(abs(a) > 50 for a in accelerations):
            lateral_changes = []
            for i in range(1, len(recent_positions)):
                lateral_change = abs(recent_positions[i].lateral_position - 
                                   recent_positions[i-1].lateral_position)
                lateral_changes.append(lateral_change)
            
            return max(lateral_changes) > 0.1
        
        return False
    
    def _detect_steadied(self, recent_positions: List[HorsePosition]) -> bool:
        if len(recent_positions) < 5:
            return False
        
        speeds = [p.speed for p in recent_positions]
        
        if len(speeds) >= 5:
            smoothed = savgol_filter(speeds, 5, 2)
            deceleration = np.diff(smoothed)
            
            return any(d < -10 for d in deceleration)
        
        return False
    
    def _add_event(self, track_id: int, frame_num: int, event_type: TripEvent):
        if track_id not in self.horse_histories:
            return
        
        event = {
            "frame": frame_num,
            "time": frame_num / self.fps,
            "type": event_type.value,
            "severity": self._calculate_severity(event_type)
        }
        
        if not hasattr(self, 'events'):
            self.events = {}
        
        if track_id not in self.events:
            self.events[track_id] = []
        
        self.events[track_id].append(event)
    
    def _calculate_severity(self, event_type: TripEvent) -> float:
        severity_map = {
            TripEvent.BOXED_IN: 0.8,
            TripEvent.WIDE_TRIP: 0.6,
            TripEvent.BUMPED: 0.9,
            TripEvent.STEADIED: 0.7,
            TripEvent.CHECKED: 0.85,
            TripEvent.TRAFFIC_TROUBLE: 0.75,
            TripEvent.PACE_PRESSURE: 0.5
        }
        return severity_map.get(event_type, 0.5)
    
    def analyze_trips(self) -> List[TripAnalysis]:
        analyses = []
        
        for track_id, history in self.horse_histories.items():
            if len(history) < 30:
                continue
            
            analysis = TripAnalysis(track_id=track_id)
            
            analysis.ground_loss = self._calculate_ground_loss(history)
            
            analysis.pace_scenario = self._determine_pace_scenario(history)
            
            analysis.energy_distribution = self._analyze_energy_distribution(history)
            
            analysis.position_chart = [p.position_in_field for p in history[::30]]
            
            if history:
                analysis.final_position = history[-1].position_in_field
            
            if hasattr(self, 'events') and track_id in self.events:
                analysis.events = self.events[track_id]
            
            analysis.trip_difficulty_score = self._calculate_trip_difficulty(analysis)
            
            analyses.append(analysis)
        
        return analyses
    
    def _calculate_ground_loss(self, history: List[HorsePosition]) -> float:
        if len(history) < 2:
            return 0.0
        
        total_distance = history[-1].distance_traveled
        
        start_pos = history[0].center
        end_pos = history[-1].center
        direct_distance = distance.euclidean(start_pos, end_pos)
        
        if direct_distance > 0:
            return (total_distance - direct_distance) / direct_distance
        return 0.0
    
    def _determine_pace_scenario(self, history: List[HorsePosition]) -> str:
        if not history:
            return "unknown"
        
        early_positions = [p.position_in_field for p in history[:len(history)//3]]
        late_positions = [p.position_in_field for p in history[-len(history)//3:]]
        
        if not early_positions or not late_positions:
            return "unknown"
        
        avg_early = np.mean(early_positions)
        avg_late = np.mean(late_positions)
        
        if avg_early <= 2:
            if avg_late <= 3:
                return "wire_to_wire"
            else:
                return "faded"
        elif avg_early <= 4:
            if avg_late <= avg_early - 1:
                return "stalker_win"
            else:
                return "stalker_held"
        else:
            if avg_late <= 3:
                return "closer_win"
            else:
                return "closer_mild"
    
    def _analyze_energy_distribution(self, history: List[HorsePosition]) -> Dict:
        if len(history) < 10:
            return {}
        
        speeds = [p.speed for p in history if p.speed > 0]
        
        if not speeds:
            return {}
        
        quarters = len(speeds) // 4
        
        return {
            "first_quarter_avg_speed": np.mean(speeds[:quarters]) if quarters > 0 else 0,
            "second_quarter_avg_speed": np.mean(speeds[quarters:2*quarters]) if quarters > 0 else 0,
            "third_quarter_avg_speed": np.mean(speeds[2*quarters:3*quarters]) if quarters > 0 else 0,
            "final_quarter_avg_speed": np.mean(speeds[3*quarters:]) if quarters > 0 else 0,
            "speed_variance": np.var(speeds),
            "max_speed": max(speeds),
            "avg_speed": np.mean(speeds)
        }
    
    def _calculate_trip_difficulty(self, analysis: TripAnalysis) -> float:
        score = 0.0
        
        for event in analysis.events:
            score += event.get("severity", 0.5) * 10
        
        score += analysis.ground_loss * 20
        
        if analysis.pace_scenario == "wire_to_wire":
            score += 15
        elif analysis.pace_scenario == "faded":
            score += 25
        
        if analysis.energy_distribution:
            variance = analysis.energy_distribution.get("speed_variance", 0)
            score += min(variance / 10, 10)
        
        return min(score, 100)