#!/usr/bin/env python3
"""
Maps visual horse detections to position bar horse numbers.
This allows us to track horses visually even when position bar OCR fails.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class HorseIdentity:
    """Links a visual detection track_id to a horse number"""
    track_id: int  # Visual tracking ID from detector
    horse_number: int  # Horse number from position bar
    confidence: float  # Confidence in this mapping
    last_position: int  # Last known position from position bar
    
@dataclass
class PositionMapping:
    """Maps position in field to horse identity"""
    position: int  # 1st place, 2nd place, etc.
    horse_number: int  # Horse number
    track_id: Optional[int]  # Visual track ID (if available)
    confidence: float

class HorseIdentityMapper:
    """
    Fuses position bar data with visual horse detection to maintain
    horse identities throughout the race.
    """
    
    def __init__(self):
        self.horse_identities: Dict[int, HorseIdentity] = {}  # track_id -> HorseIdentity
        self.number_to_track: Dict[int, int] = {}  # horse_number -> track_id
        self.position_history: List[List[PositionMapping]] = []  # Frame-by-frame position mappings
        self.confidence_threshold = 0.3
        
    def update_frame(self, detections: List, position_snapshot, frame_num: int):
        """
        Update horse identities by fusing visual detections with position bar data.
        
        Args:
            detections: List of horse detections from YOLO (with track_ids)
            position_snapshot: Position bar reading (horse numbers in order)
            frame_num: Current frame number
        """
        
        if not detections:
            return
            
        # Sort detections by x-coordinate (left to right)
        sorted_detections = sorted(detections, key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)
        
        current_mappings = []
        
        if position_snapshot and position_snapshot.positions:
            # We have position bar data - use it to establish/update identities
            horse_numbers = position_snapshot.positions
            
            logger.debug(f"Frame {frame_num}: Position bar shows {horse_numbers}, "
                        f"Visual detections: {len(sorted_detections)} horses")
            
            # Try to match visual detections to position bar
            for pos_idx, horse_num in enumerate(horse_numbers):
                position = pos_idx + 1  # 1st place, 2nd place, etc.
                
                # Find the best matching visual detection for this position
                if pos_idx < len(sorted_detections):
                    detection = sorted_detections[pos_idx]
                    track_id = detection.track_id
                    
                    # Update or create horse identity
                    if horse_num in self.number_to_track:
                        # We already know this horse - verify it's the same track_id
                        existing_track_id = self.number_to_track[horse_num]
                        if existing_track_id == track_id:
                            # Confirmed identity
                            self.horse_identities[track_id].confidence = min(1.0, 
                                self.horse_identities[track_id].confidence + 0.1)
                            self.horse_identities[track_id].last_position = position
                        else:
                            # Track ID changed - horse might have been re-assigned
                            logger.warning(f"Horse #{horse_num} track_id changed from {existing_track_id} to {track_id}")
                            # Update mapping with lower confidence
                            if existing_track_id in self.horse_identities:
                                del self.horse_identities[existing_track_id]
                            self._create_identity(track_id, horse_num, position, confidence=0.5)
                    else:
                        # New horse identity
                        self._create_identity(track_id, horse_num, position, confidence=0.8)
                    
                    current_mappings.append(PositionMapping(
                        position=position,
                        horse_number=horse_num,
                        track_id=track_id,
                        confidence=position_snapshot.confidence
                    ))
                else:
                    # Position bar shows more horses than we detected visually
                    current_mappings.append(PositionMapping(
                        position=position,
                        horse_number=horse_num,
                        track_id=None,
                        confidence=position_snapshot.confidence * 0.5  # Lower confidence
                    ))
        
        else:
            # No position bar data - use existing identities to infer positions
            logger.debug(f"Frame {frame_num}: No position bar, inferring from {len(sorted_detections)} visual detections")
            
            for pos_idx, detection in enumerate(sorted_detections):
                position = pos_idx + 1
                track_id = detection.track_id
                
                # Look up horse number from existing identity
                if track_id in self.horse_identities:
                    horse_num = self.horse_identities[track_id].horse_number
                    # Reduce confidence since we're inferring
                    self.horse_identities[track_id].confidence *= 0.98
                    self.horse_identities[track_id].last_position = position
                    
                    current_mappings.append(PositionMapping(
                        position=position,
                        horse_number=horse_num,
                        track_id=track_id,
                        confidence=self.horse_identities[track_id].confidence
                    ))
                else:
                    # Unknown horse - can't identify without position bar
                    current_mappings.append(PositionMapping(
                        position=position,
                        horse_number=-1,  # Unknown
                        track_id=track_id,
                        confidence=0.1
                    ))
        
        self.position_history.append(current_mappings)
        
        # Log current identities
        if frame_num % 30 == 0:  # Every 30 frames
            self._log_current_identities(frame_num)
    
    def _create_identity(self, track_id: int, horse_number: int, position: int, confidence: float):
        """Create a new horse identity mapping"""
        self.horse_identities[track_id] = HorseIdentity(
            track_id=track_id,
            horse_number=horse_number,
            confidence=confidence,
            last_position=position
        )
        self.number_to_track[horse_number] = track_id
        logger.info(f"Created identity: Track {track_id} = Horse #{horse_number} (confidence: {confidence:.2f})")
    
    def _log_current_identities(self, frame_num: int):
        """Log current horse identities for debugging"""
        logger.debug(f"Frame {frame_num} - Current horse identities:")
        for track_id, identity in self.horse_identities.items():
            if identity.confidence > self.confidence_threshold:
                logger.debug(f"  Track {track_id} = Horse #{identity.horse_number} "
                           f"(pos: {identity.last_position}, conf: {identity.confidence:.2f})")
    
    def get_final_positions(self) -> Dict[int, int]:
        """
        Get final race positions based on visual tracking.
        Returns: Dict[horse_number, final_position]
        """
        if not self.position_history:
            return {}
        
        # Use the last frame with reliable position data
        final_mappings = None
        
        # Look backwards for the most complete position mapping
        for mappings in reversed(self.position_history[-10:]):  # Check last 10 frames
            # Count horses with known identities
            known_horses = sum(1 for m in mappings if m.horse_number > 0 and m.confidence > self.confidence_threshold)
            if known_horses >= 3:  # Need at least 3 horses to be reliable
                final_mappings = mappings
                break
        
        if not final_mappings:
            final_mappings = self.position_history[-1]  # Last resort
        
        # Create final position mapping
        final_positions = {}
        for mapping in final_mappings:
            if mapping.horse_number > 0 and mapping.confidence > self.confidence_threshold:
                final_positions[mapping.horse_number] = mapping.position
        
        logger.info(f"Final positions from visual tracking: {final_positions}")
        return final_positions
    
    def get_horse_journey(self, horse_number: int) -> List[int]:
        """Get position progression for a specific horse"""
        positions = []
        
        for frame_mappings in self.position_history:
            for mapping in frame_mappings:
                if mapping.horse_number == horse_number and mapping.confidence > self.confidence_threshold:
                    positions.append(mapping.position)
                    break
        
        return positions
    
    def get_winner(self) -> Optional[int]:
        """Get the winning horse number based on final visual positions"""
        final_positions = self.get_final_positions()
        
        # Find horse in 1st position
        for horse_num, position in final_positions.items():
            if position == 1:
                return horse_num
        
        return None