#!/usr/bin/env python3
"""
Quick script to collect training data from multiple races by trying different race codes
"""

from src.video_scraper import TJKVideoScraper
from collect_training_data import TrainingDataCollector
from loguru import logger
from pathlib import Path
import time

def collect_from_multiple_races():
    """Try multiple race codes and collect data from successful downloads"""
    
    # Try a range of race codes around the working one
    base_code = 194367
    race_codes_to_try = [
        base_code + 1,  # 194368
        base_code + 2,  # 194369  
        base_code + 3,  # 194370
        base_code - 1,  # 194366
        base_code - 2,  # 194365
        base_code + 10, # 194377
        base_code + 20, # 194387
    ]
    
    scraper = TJKVideoScraper()
    collector = TrainingDataCollector("data/position_digits")
    
    successful_downloads = 0
    total_new_samples = 0
    
    for race_code in race_codes_to_try:
        try:
            race_code_str = str(race_code)
            video_path = f"data/videos/race_{race_code_str}.mp4"
            
            logger.info(f"Trying race {race_code_str}...")
            
            # Check if already exists
            if Path(video_path).exists():
                logger.info(f"Race {race_code_str} already downloaded")
            else:
                # Try to download
                logger.info(f"Downloading race {race_code_str}...")
                success = scraper.download_race_video(race_code_str)
                
                if not success:
                    logger.warning(f"Failed to download race {race_code_str}")
                    continue
                
                # Small delay between downloads
                time.sleep(3)
            
            # If video exists, collect training data
            if Path(video_path).exists():
                logger.info(f"Collecting samples from race {race_code_str}...")
                samples = collector.process_video(video_path, race_code_str)
                total_new_samples += samples
                successful_downloads += 1
                logger.info(f"✓ Collected {samples} samples from race {race_code_str}")
            else:
                logger.warning(f"✗ Video not found after download: {video_path}")
                
        except Exception as e:
            logger.error(f"Error processing race {race_code}: {str(e)[:200]}")
            continue
        
        # Don't overwhelm the server
        time.sleep(2)
        
        # Stop after 3 successful downloads to not take too long
        if successful_downloads >= 3:
            logger.info("Got 3 successful races, stopping collection")
            break
    
    logger.info(f"Collection complete!")
    logger.info(f"  Successful downloads: {successful_downloads}")
    logger.info(f"  Total new samples: {total_new_samples}")
    
    # Show updated summary
    summary = collector.generate_dataset_summary()
    logger.info(f"  Dataset now has: {summary['total_samples']} total samples")
    logger.info(f"  Unlabeled: {summary['unlabeled_samples']}")
    
    return total_new_samples

if __name__ == "__main__":
    collect_from_multiple_races()