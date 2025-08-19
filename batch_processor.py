#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
import pandas as pd
from loguru import logger
import sys

from src.video_processor import VideoProcessor
from src.video_scraper import TJKVideoScraper


def process_single_race(race_code: str, output_dir: str, download: bool = True) -> Dict:
    try:
        logger.info(f"Processing race: {race_code}")
        
        video_path = None
        
        if download:
            scraper = TJKVideoScraper()
            try:
                video_path = scraper.scrape_race(race_code)
            finally:
                scraper.close()
        else:
            video_path = f"data/videos/race_{race_code}.mp4"
            if not Path(video_path).exists():
                logger.error(f"Video not found: {video_path}")
                return None
        
        if not video_path:
            logger.error(f"Failed to get video for race {race_code}")
            return None
        
        processor = VideoProcessor(output_dir=output_dir, save_annotated=False, auto_detect_horses=True)
        results = processor.process_video(video_path, race_code)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing race {race_code}: {e}")
        return None


def aggregate_results(results_list: List[Dict]) -> pd.DataFrame:
    all_trips = []
    
    for result in results_list:
        if not result:
            continue
            
        race_code = result.get('race_code', 'unknown')
        
        for trip in result.get('trip_analyses', []):
            trip_data = {
                'race_code': race_code,
                'track_id': trip['track_id'],
                'difficulty_score': trip['trip_difficulty_score'],
                'pace_scenario': trip['pace_scenario'],
                'ground_loss': trip['ground_loss'],
                'num_events': len(trip['events']),
                'final_position': trip.get('final_position'),
            }
            
            event_types = [e['type'] for e in trip['events']]
            for event_type in ['boxed_in', 'wide_trip', 'bumped', 'steadied']:
                trip_data[f'had_{event_type}'] = event_type in event_types
            
            all_trips.append(trip_data)
    
    return pd.DataFrame(all_trips)


def main():
    parser = argparse.ArgumentParser(description="Batch process horse races")
    parser.add_argument("--race-codes", nargs="+", help="List of race codes")
    parser.add_argument("--race-file", type=str, help="File with race codes (one per line)")
    parser.add_argument("--output-dir", type=str, default="data/batch_processed",
                       help="Output directory")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Maximum parallel workers")
    parser.add_argument("--no-download", action="store_true",
                       help="Skip downloading, use existing videos")
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/batch_processor.log", rotation="10 MB", level="DEBUG")
    
    race_codes = []
    
    if args.race_codes:
        race_codes.extend(args.race_codes)
    
    if args.race_file:
        with open(args.race_file, 'r') as f:
            race_codes.extend([line.strip() for line in f if line.strip()])
    
    if not race_codes:
        logger.error("No race codes provided")
        sys.exit(1)
    
    logger.info(f"Processing {len(race_codes)} races")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_single_race, 
                race_code, 
                args.output_dir,
                not args.no_download
            ): race_code 
            for race_code in race_codes
        }
        
        for future in as_completed(futures):
            race_code = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.success(f"Completed: {race_code}")
                else:
                    logger.warning(f"No results for: {race_code}")
            except Exception as e:
                logger.error(f"Failed {race_code}: {e}")
    
    logger.info(f"Successfully processed {len(results)}/{len(race_codes)} races")
    
    if results:
        df = aggregate_results(results)
        
        csv_path = Path(args.output_dir) / "aggregate_analysis.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved aggregate results to: {csv_path}")
        
        logger.info("\nSummary Statistics:")
        logger.info(f"Total horses analyzed: {len(df)}")
        logger.info(f"Average trip difficulty: {df['difficulty_score'].mean():.2f}")
        logger.info(f"Horses with boxed-in trouble: {df['had_boxed_in'].sum()}")
        logger.info(f"Horses with wide trips: {df['had_wide_trip'].sum()}")
        
        all_results_path = Path(args.output_dir) / "all_results.json"
        with open(all_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved all results to: {all_results_path}")


if __name__ == "__main__":
    main()