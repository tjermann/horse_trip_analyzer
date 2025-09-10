#!/usr/bin/env python3
"""
Final position enforcer - ensures exactly one horse per finishing position.
This is a fallback system to guarantee valid race results even when 
position validation has insufficient data.
"""

from typing import Dict, List, Tuple
from loguru import logger


class FinalPositionEnforcer:
    """
    Ensures each finishing position is assigned to exactly one horse.
    Uses multiple data sources and resolves conflicts.
    """
    
    def __init__(self, num_horses: int = 8):
        self.num_horses = num_horses
    
    def enforce_unique_positions(self, 
                               position_sources: List[Tuple[str, Dict[int, int]]]) -> Dict[int, int]:
        """
        Enforce unique final positions using multiple data sources.
        
        Args:
            position_sources: List of (source_name, {horse_num: position}) tuples
                             Ordered by priority (first = most trusted)
        
        Returns:
            Dict[horse_num, final_position] with guaranteed uniqueness
        """
        
        # Track assignments
        assigned_positions = set()  # positions that have been assigned
        assigned_horses = set()     # horses that have been assigned
        final_assignments = {}      # final result
        
        logger.info(f"Enforcing unique positions from {len(position_sources)} sources")
        
        # Process sources in order of priority
        for source_name, positions in position_sources:
            logger.info(f"Processing source: {source_name} with {len(positions)} positions")
            
            # Sort horses by their assigned position (best positions first)
            sorted_horses = sorted(positions.items(), key=lambda x: x[1])
            
            for horse_num, position in sorted_horses:
                # Skip if horse or position already assigned
                if horse_num in assigned_horses:
                    logger.debug(f"  Horse #{horse_num} already assigned, skipping")
                    continue
                    
                if position in assigned_positions:
                    logger.debug(f"  Position {position} already taken, skipping Horse #{horse_num}")
                    continue
                
                # Valid assignment
                final_assignments[horse_num] = position
                assigned_positions.add(position)
                assigned_horses.add(horse_num)
                logger.info(f"  ✅ Horse #{horse_num} → Position {position} (from {source_name})")
        
        # Handle unassigned horses - assign to remaining positions
        unassigned_horses = set(range(1, self.num_horses + 1)) - assigned_horses
        unassigned_positions = set(range(1, self.num_horses + 1)) - assigned_positions
        
        if unassigned_horses:
            logger.warning(f"Assigning remaining horses {unassigned_horses} to positions {unassigned_positions}")
            
            # Sort both for consistent assignment
            remaining_horses = sorted(unassigned_horses)
            remaining_positions = sorted(unassigned_positions)
            
            for horse, position in zip(remaining_horses, remaining_positions):
                final_assignments[horse] = position
                logger.info(f"  🔧 Horse #{horse} → Position {position} (fallback assignment)")
        
        # Verify uniqueness
        positions_used = list(final_assignments.values())
        if len(positions_used) != len(set(positions_used)):
            logger.error("CRITICAL: Final assignments still have duplicates!")
            raise ValueError("Failed to enforce unique positions")
        
        # Verify completeness
        if len(final_assignments) != self.num_horses:
            logger.error(f"CRITICAL: Only {len(final_assignments)}/{self.num_horses} horses assigned!")
            raise ValueError("Failed to assign all horses")
        
        logger.info("✅ Final position enforcement complete:")
        for position in range(1, self.num_horses + 1):
            for horse_num, final_pos in final_assignments.items():
                if final_pos == position:
                    logger.info(f"  {position}. Horse #{horse_num}")
                    break
        
        return final_assignments
    
    def create_fallback_positions(self, horses: List[int]) -> Dict[int, int]:
        """
        Create fallback positions when no other data is available.
        Assigns positions in horse number order.
        """
        logger.warning("Creating fallback positions - no reliable position data available")
        
        fallback = {}
        sorted_horses = sorted(horses)
        
        for i, horse_num in enumerate(sorted_horses, 1):
            fallback[horse_num] = i
            logger.info(f"  Fallback: Horse #{horse_num} → Position {i}")
        
        return fallback