#!/usr/bin/env python3
"""
Position validation and consensus logic for horse race position tracking.
Ensures each position is only assigned to one horse at any given time.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidatedPosition:
    """A validated position assignment for horses"""
    horse_number: int
    position: int
    confidence: float
    source: str  # 'ocr', 'visual', 'inferred'


class PositionValidator:
    """
    Validates and disambiguates horse positions to ensure:
    1. Each position is assigned to exactly one horse
    2. Each horse has exactly one position
    3. Positions are consistent with race physics
    """
    
    def __init__(self, num_horses: int = 8):
        self.num_horses = num_horses
        self.position_history: List[List[ValidatedPosition]] = []
        self.horse_position_counts: Dict[int, Counter] = defaultdict(Counter)  # Track common positions per horse
        self.last_known_positions: Dict[int, int] = {}  # Last known position for each horse
        self.position_confidence: Dict[int, float] = {}  # Running confidence for each horse's position
        
    def validate_positions(self, 
                          ocr_positions: Optional[List[int]], 
                          visual_order: Optional[List[int]],
                          frame_num: int) -> List[ValidatedPosition]:
        """
        Validate and disambiguate positions from multiple sources.
        
        Args:
            ocr_positions: Horse numbers in order from OCR (1st place horse, 2nd place horse, etc.)
            visual_order: Horse numbers in left-to-right visual order
            frame_num: Current frame number
            
        Returns:
            List of validated position assignments
        """
        validated = []
        
        # Case 1: We have OCR data
        if ocr_positions and len(ocr_positions) >= 2:
            # Check for duplicates in OCR
            if len(ocr_positions) != len(set(ocr_positions)):
                logger.warning(f"Frame {frame_num}: OCR has duplicate horses: {ocr_positions}")
                ocr_positions = self._resolve_ocr_duplicates(ocr_positions, frame_num)
            
            # Validate each position
            assigned_positions = set()
            assigned_horses = set()
            
            for pos_idx, horse_num in enumerate(ocr_positions):
                position = pos_idx + 1
                
                # Skip if this horse or position already assigned
                if horse_num in assigned_horses:
                    logger.debug(f"Frame {frame_num}: Horse {horse_num} already assigned, skipping duplicate")
                    continue
                if position in assigned_positions:
                    logger.debug(f"Frame {frame_num}: Position {position} already assigned, skipping")
                    continue
                
                # Check position continuity
                position_jump = 0
                if horse_num in self.last_known_positions:
                    position_jump = abs(position - self.last_known_positions[horse_num])
                
                # Adjust confidence based on position continuity
                confidence = 0.8
                if position_jump > 3:  # Large jump is suspicious
                    confidence *= 0.5
                    logger.debug(f"Frame {frame_num}: Horse {horse_num} jumped {position_jump} positions, reducing confidence")
                elif position_jump <= 1:  # Small change is good
                    confidence *= 1.2
                    confidence = min(confidence, 0.95)
                
                # Valid assignment
                validated.append(ValidatedPosition(
                    horse_number=horse_num,
                    position=position,
                    confidence=confidence,
                    source='ocr'
                ))
                assigned_positions.add(position)
                assigned_horses.add(horse_num)
                
                # Update history for this horse
                self.horse_position_counts[horse_num][position] += 1
                self.last_known_positions[horse_num] = position
                self.position_confidence[horse_num] = confidence
        
        # Case 2: Only visual data available
        elif visual_order and len(visual_order) >= 2:
            # Use visual order with lower confidence
            assigned_horses = set()
            for pos_idx, horse_num in enumerate(visual_order):
                if horse_num > 0 and horse_num not in assigned_horses:  # Valid horse number
                    validated.append(ValidatedPosition(
                        horse_number=horse_num,
                        position=pos_idx + 1,
                        confidence=0.5,  # Lower confidence for visual-only
                        source='visual'
                    ))
                    assigned_horses.add(horse_num)
        
        # Case 3: Need to infer from history
        else:
            validated = self._infer_from_history(frame_num)
        
        # Store in history
        if validated:
            self.position_history.append(validated)
        
        return validated
    
    def _resolve_ocr_duplicates(self, ocr_positions: List[int], frame_num: int) -> List[int]:
        """
        Resolve duplicate horses in OCR reading using historical data.
        """
        # Count occurrences
        horse_counts = Counter(ocr_positions)
        duplicates = {h for h, count in horse_counts.items() if count > 1}
        
        if not duplicates:
            return ocr_positions
        
        resolved = []
        used_horses = set()
        
        for pos_idx, horse_num in enumerate(ocr_positions):
            if horse_num in duplicates:
                # This horse appears multiple times - use history to decide if this position is correct
                if horse_num not in used_horses:
                    # First occurrence - check if this position is historically likely
                    position = pos_idx + 1
                    historical_positions = self.horse_position_counts[horse_num]
                    
                    if historical_positions and position in historical_positions:
                        # This position makes sense historically
                        resolved.append(horse_num)
                        used_horses.add(horse_num)
                    else:
                        # Skip this occurrence
                        logger.debug(f"Frame {frame_num}: Skipping duplicate horse {horse_num} at position {position}")
                        continue
                else:
                    # Already used this horse, skip
                    continue
            else:
                # No duplicate, use as is
                if horse_num not in used_horses:
                    resolved.append(horse_num)
                    used_horses.add(horse_num)
        
        return resolved
    
    def _infer_from_history(self, frame_num: int) -> List[ValidatedPosition]:
        """
        Infer positions from recent history when no current data available.
        """
        if not self.position_history:
            return []
        
        # Use recent positions (last 5 frames)
        recent = self.position_history[-5:]
        
        # Aggregate positions for each horse
        horse_positions = defaultdict(list)
        for frame_positions in recent:
            for vp in frame_positions:
                horse_positions[vp.horse_number].append(vp.position)
        
        # Calculate most likely position for each horse
        inferred = []
        used_positions = set()
        
        for horse_num, positions in horse_positions.items():
            if positions:
                # Use mode (most common) position
                most_common = Counter(positions).most_common(1)[0][0]
                if most_common not in used_positions:
                    inferred.append(ValidatedPosition(
                        horse_number=horse_num,
                        position=most_common,
                        confidence=0.3,  # Low confidence for inferred
                        source='inferred'
                    ))
                    used_positions.add(most_common)
        
        return inferred
    
    def get_consensus_positions(self, window_size: int = 10) -> Dict[int, int]:
        """
        Get consensus positions over recent frames.
        
        Returns:
            Dict[horse_number, most_likely_position]
        """
        if len(self.position_history) < window_size:
            window_size = len(self.position_history)
        
        if window_size == 0:
            return {}
        
        # Look at recent window
        recent = self.position_history[-window_size:]
        
        # Aggregate positions for each horse
        horse_positions = defaultdict(list)
        for frame_positions in recent:
            for vp in frame_positions:
                if vp.confidence > 0.3:  # Only use confident readings
                    horse_positions[vp.horse_number].append(vp.position)
        
        # Calculate consensus
        consensus = {}
        for horse_num, positions in horse_positions.items():
            if positions:
                # Use weighted mode based on recency
                position_counts = Counter(positions)
                consensus[horse_num] = position_counts.most_common(1)[0][0]
        
        return consensus
    
    def get_final_positions(self, last_percent: float = 0.1) -> Dict[int, int]:
        """
        Get final race positions from the last portion of the race.
        
        Args:
            last_percent: Portion of race to consider (0.1 = last 10%)
            
        Returns:
            Dict[horse_number, final_position]
        """
        if not self.position_history:
            return {}
        
        # Calculate how many frames to look at
        total_frames = len(self.position_history)
        frames_to_check = max(1, int(total_frames * last_percent))
        
        # Get positions from final stretch
        final_stretch = self.position_history[-frames_to_check:]
        
        # Aggregate with higher weight for later frames
        weighted_positions = defaultdict(lambda: defaultdict(float))
        
        for frame_idx, frame_positions in enumerate(final_stretch):
            # Weight increases linearly towards the end
            weight = (frame_idx + 1) / len(final_stretch)
            
            for vp in frame_positions:
                if vp.confidence > 0.3:
                    weighted_positions[vp.horse_number][vp.position] += weight * vp.confidence
        
        # Determine final positions
        final_positions = {}
        used_positions = set()
        
        # Sort horses by their most likely position
        horse_rankings = []
        for horse_num, position_weights in weighted_positions.items():
            if position_weights:
                # Get weighted average position
                total_weight = sum(position_weights.values())
                avg_position = sum(pos * weight for pos, weight in position_weights.items()) / total_weight
                horse_rankings.append((horse_num, avg_position, total_weight))
        
        # Sort by average position
        horse_rankings.sort(key=lambda x: x[1])
        
        # Assign positions ensuring uniqueness
        for rank, (horse_num, avg_pos, confidence) in enumerate(horse_rankings, 1):
            final_positions[horse_num] = rank
            logger.info(f"Horse #{horse_num}: Final position {rank} (avg: {avg_pos:.2f}, conf: {confidence:.2f})")
        
        return final_positions
    
    def validate_sequence(self, positions: List[int], max_jump: int = 3) -> bool:
        """
        Validate that a position sequence is physically possible.
        Horses shouldn't jump more than max_jump positions between frames.
        """
        if len(positions) < 2:
            return True
        
        for i in range(1, len(positions)):
            if abs(positions[i] - positions[i-1]) > max_jump:
                return False
        
        return True
    
    def smooth_position_sequence(self, positions: List[int], window: int = 3) -> List[int]:
        """
        Smooth a position sequence to remove impossible jumps.
        """
        if len(positions) < window:
            return positions
        
        smoothed = []
        for i in range(len(positions)):
            # Get window around current position
            start = max(0, i - window // 2)
            end = min(len(positions), i + window // 2 + 1)
            window_positions = positions[start:end]
            
            # Use median for smoothing
            smoothed.append(int(np.median(window_positions)))
        
        return smoothed