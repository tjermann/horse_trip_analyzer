#!/usr/bin/env python3
"""
Collect training data from multiple races with different horse counts
"""

import os
from pathlib import Path
from collect_training_data import TrainingDataCollector
from src.video_scraper import VideoScraper
from loguru import logger

def collect_diverse_training_data():
    """Collect training data from races with different horse counts"""
    
    # Example race codes with different horse counts (you'd need to verify these)
    target_races = [
        # Format: (race_code, expected_horses, description)
        ("194367", 8, "8-horse race"),
        # Add more races here with different horse counts
        # ("194368", 10, "10-horse race"), 
        # ("194369", 12, "12-horse race"),
        # ("194370", 6, "6-horse race"),
    ]
    
    collector = TrainingDataCollector("data/position_digits")
    scraper = VideoScraper()
    
    total_samples = 0
    
    for race_code, expected_horses, description in target_races:
        logger.info(f"Processing {description} - Race {race_code}")
        
        video_path = f"data/videos/race_{race_code}.mp4"
        
        # Download if not exists
        if not Path(video_path).exists():
            logger.info(f"Downloading race {race_code}...")
            try:
                scraper.download_race_video(race_code)
            except Exception as e:
                logger.error(f"Failed to download race {race_code}: {e}")
                continue
        
        # Process video
        if Path(video_path).exists():
            samples = collector.process_video(video_path, race_code)
            total_samples += samples
            logger.info(f"Collected {samples} samples from {description}")
        else:
            logger.warning(f"Video not found: {video_path}")
    
    logger.info(f"Total samples collected: {total_samples}")
    
    # Show summary
    summary = collector.generate_dataset_summary()
    logger.info(f"Dataset now contains {summary['total_samples']} total samples")
    logger.info(f"Unlabeled: {summary['unlabeled_samples']}")
    
    return total_samples

if __name__ == "__main__":
    collect_diverse_training_data()