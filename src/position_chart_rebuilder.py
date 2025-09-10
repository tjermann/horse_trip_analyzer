#!/usr/bin/env python3
"""
Rebuilds position charts to ensure no duplicate positions at any point in time.
"""

from typing import List, Dict, Set
from collections import defaultdict
from loguru import logger


class PositionChartRebuilder:
    """
    Rebuilds position charts ensuring each position is unique at every time point.
    """
    
    def __init__(self, num_horses: int = 8):
        self.num_horses = num_horses
    
    def rebuild_charts(self, raw_charts: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Rebuild position charts to ensure uniqueness at each time point.
        
        Args:
            raw_charts: Dict[horse_num, List[positions]] - raw position data
            
        Returns:
            Dict[horse_num, List[positions]] - cleaned position data
        """
        
        if not raw_charts:
            return {}
        
        # Find the maximum chart length
        max_length = max(len(chart) for chart in raw_charts.values() if chart)
        if max_length == 0:
            return raw_charts
        
        logger.info(f"Rebuilding position charts for {len(raw_charts)} horses, max length {max_length}")
        
        # Build cleaned charts
        cleaned_charts = defaultdict(list)
        
        # Process each time point
        for time_idx in range(max_length):
            # Get all horses' positions at this time point
            time_positions = {}
            for horse_num, positions in raw_charts.items():
                if time_idx < len(positions):
                    time_positions[horse_num] = positions[time_idx]
            
            # Resolve duplicates at this time point
            cleaned_time = self._resolve_time_point(time_positions, time_idx)
            
            # Add to cleaned charts
            for horse_num, position in cleaned_time.items():
                cleaned_charts[horse_num].append(position)
        
        # Fill in missing data points
        for horse_num in raw_charts.keys():
            if horse_num in cleaned_charts:
                chart = cleaned_charts[horse_num]
                # Fill gaps with interpolation
                if len(chart) < max_length:
                    last_pos = chart[-1] if chart else self.num_horses
                    while len(chart) < max_length:
                        chart.append(last_pos)
        
        return dict(cleaned_charts)
    
    def _resolve_time_point(self, positions: Dict[int, int], time_idx: int) -> Dict[int, int]:
        """
        Resolve duplicate positions at a single time point.
        
        Args:
            positions: Dict[horse_num, claimed_position]
            time_idx: Time index for debugging
            
        Returns:
            Dict[horse_num, resolved_position] with no duplicates
        """
        
        # Track assignments
        assigned_positions = set()
        resolved = {}
        
        # Group horses by claimed position
        position_claims = defaultdict(list)
        for horse_num, pos in positions.items():
            position_claims[pos].append(horse_num)
        
        # Process positions in order (1st place first)
        for position in range(1, self.num_horses + 1):
            if position in position_claims:
                claimants = position_claims[position]
                
                if len(claimants) == 1:
                    # No conflict
                    horse = claimants[0]
                    if position not in assigned_positions:
                        resolved[horse] = position
                        assigned_positions.add(position)
                else:
                    # Multiple horses claim this position
                    logger.debug(f"Time {time_idx}: Horses {claimants} all claim position {position}")
                    
                    # Assign to first horse (arbitrary but consistent)
                    for horse in claimants:
                        # Find next available position
                        for alt_pos in range(1, self.num_horses + 1):
                            if alt_pos not in assigned_positions:
                                resolved[horse] = alt_pos
                                assigned_positions.add(alt_pos)
                                if alt_pos != position:
                                    logger.debug(f"  Reassigned Horse #{horse} from position {position} to {alt_pos}")
                                break
        
        # Handle any remaining horses
        remaining_horses = set(positions.keys()) - set(resolved.keys())
        remaining_positions = set(range(1, self.num_horses + 1)) - assigned_positions
        
        for horse, pos in zip(remaining_horses, remaining_positions):
            resolved[horse] = pos
            logger.debug(f"Time {time_idx}: Assigned Horse #{horse} to remaining position {pos}")
        
        return resolved
    
    def validate_charts(self, charts: Dict[int, List[int]]) -> bool:
        """
        Validate that charts have no duplicate positions at any time point.
        
        Returns:
            True if valid, False if duplicates found
        """
        
        if not charts:
            return True
        
        max_length = max(len(chart) for chart in charts.values() if chart)
        
        for time_idx in range(max_length):
            positions_at_time = []
            for horse_num, positions in charts.items():
                if time_idx < len(positions):
                    positions_at_time.append(positions[time_idx])
            
            # Check for duplicates
            if len(positions_at_time) != len(set(positions_at_time)):
                logger.error(f"Time {time_idx}: Duplicate positions found: {positions_at_time}")
                return False
        
        return True