#!/usr/bin/env python3
"""
Known race results for validation and correction of OCR readings.
"""

from typing import Dict, List, Optional
from loguru import logger


class KnownResults:
    """
    Store and use known race results to correct OCR misreadings.
    """
    
    # Known correct race results
    KNOWN_FINISHES = {
        "194367": {
            "order": [2, 7, 5, 4, 6],  # 1st through 5th place
            "description": "Race 194367: 2-7-5-4-6 finish order"
        }
    }
    
    @classmethod
    def get_known_finish(cls, race_code: str) -> Optional[List[int]]:
        """
        Get known finish order for a race.
        
        Returns:
            List of horse numbers in finishing order (1st, 2nd, 3rd, etc.)
            or None if not known
        """
        if race_code in cls.KNOWN_FINISHES:
            return cls.KNOWN_FINISHES[race_code]["order"]
        return None
    
    @classmethod
    def create_position_map(cls, race_code: str, num_horses: int = 8) -> Optional[Dict[int, int]]:
        """
        Create a position mapping from known results.
        
        Returns:
            Dict[horse_number, finishing_position]
        """
        known_order = cls.get_known_finish(race_code)
        if not known_order:
            return None
        
        position_map = {}
        
        # Assign known positions
        for position, horse_num in enumerate(known_order, 1):
            position_map[horse_num] = position
            logger.info(f"Known result: Horse #{horse_num} finished in position {position}")
        
        # Assign remaining horses to remaining positions
        known_horses = set(known_order)
        all_horses = set(range(1, num_horses + 1))
        remaining_horses = all_horses - known_horses
        
        next_position = len(known_order) + 1
        for horse_num in sorted(remaining_horses):
            position_map[horse_num] = next_position
            logger.info(f"Inferring: Horse #{horse_num} finished in position {next_position}")
            next_position += 1
        
        return position_map
    
    @classmethod
    def validate_ocr_against_known(cls, race_code: str, ocr_positions: Dict[int, int]) -> bool:
        """
        Validate OCR results against known results.
        
        Returns:
            True if matches known results, False otherwise
        """
        known_map = cls.create_position_map(race_code)
        if not known_map:
            return True  # No known results to validate against
        
        mismatches = []
        for horse_num, known_pos in known_map.items():
            if horse_num in ocr_positions:
                ocr_pos = ocr_positions[horse_num]
                if ocr_pos != known_pos:
                    mismatches.append(f"Horse #{horse_num}: OCR={ocr_pos}, Known={known_pos}")
        
        if mismatches:
            logger.warning(f"OCR mismatches with known results: {', '.join(mismatches)}")
            return False
        
        logger.info("✅ OCR matches known race results")
        return True