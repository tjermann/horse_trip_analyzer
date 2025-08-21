#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from loguru import logger

from src.video_scraper import TJKVideoScraper
from src.video_processor import VideoProcessor


def setup_logging(verbose: bool = False):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    logger.add("logs/horse_analyzer.log", rotation="10 MB", level="DEBUG")


def main():
    parser = argparse.ArgumentParser(description="Horse Race Trip Analyzer")
    parser.add_argument("--race-code", type=str, help="Race code to download from TJK")
    parser.add_argument("--video-path", type=str, help="Path to existing video file")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed files")
    parser.add_argument("--save-annotated", action="store_true", 
                       help="Save annotated video with detections")
    parser.add_argument("--num-horses", type=int, default=None,
                       help="Expected number of horses in the race (auto-detect if not specified)")
    parser.add_argument("--no-auto-detect", action="store_true",
                       help="Disable automatic horse count detection")
    parser.add_argument("--target-fps", type=float, default=1.0,
                       help="Target processing frame rate (default: 1.0 fps)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if not args.race_code and not args.video_path:
        logger.error("Please provide either --race-code or --video-path")
        sys.exit(1)
    
    video_path = None
    race_code = None
    
    if args.race_code:
        logger.info(f"Downloading race video for code: {args.race_code}")
        scraper = TJKVideoScraper()
        try:
            video_path = scraper.scrape_race(args.race_code)
            race_code = args.race_code
        finally:
            scraper.close()
        
        if not video_path:
            logger.error("Failed to download video")
            sys.exit(1)
    else:
        video_path = args.video_path
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
        
        # Extract race code from filename if not provided
        if race_code is None:
            video_name = Path(video_path).stem
            if "race_" in video_name:
                race_code = video_name.split("race_")[1].split("_")[0]
    
    logger.info("Processing video...")
    processor = VideoProcessor(
        output_dir=args.output_dir,
        save_annotated=args.save_annotated,
        expected_horses=args.num_horses,
        auto_detect_horses=not args.no_auto_detect,
        target_fps=args.target_fps
    )
    
    results = processor.process_video(video_path, race_code)
    
    if results:
        report = processor.generate_report(results)
        print("\n" + report)
        
        logger.success(f"Analysis complete! Processed {results['num_horses_detected']} horses")
        
        top_difficulty = max(results['trip_analyses'], 
                           key=lambda x: x['trip_difficulty_score'])
        logger.info(f"Most difficult trip: Horse #{top_difficulty['track_id']} "
                   f"with score {top_difficulty['trip_difficulty_score']:.1f}")
    else:
        logger.error("Failed to process video")
        sys.exit(1)


if __name__ == "__main__":
    main()