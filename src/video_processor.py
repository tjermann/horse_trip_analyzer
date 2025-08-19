import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
import json
from loguru import logger

from .horse_detector import HorseDetector, JockeyColorIdentifier
from .horse_tracker import ImprovedHorseTracker
from .trip_analyzer import TripAnalyzer, TripAnalysis
from .race_start_detector import detect_race_horses


class VideoProcessor:
    def __init__(self, 
                 output_dir: str = "data/processed",
                 save_annotated: bool = True,
                 expected_horses: Optional[int] = None,
                 auto_detect_horses: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_annotated = save_annotated
        self.expected_horses = expected_horses
        self.auto_detect_horses = auto_detect_horses
        
        self.detector = HorseDetector()
        self.jockey_identifier = JockeyColorIdentifier()
        self.analyzer = TripAnalyzer()
        
        # Tracker will be initialized after horse count detection
        self.tracker = None
        
    def process_video(self, video_path: str, race_code: Optional[str] = None) -> Dict:
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        logger.info(f"Processing video: {video_path}")
        
        # Auto-detect number of horses if not specified
        if self.auto_detect_horses and self.expected_horses is None:
            logger.info("Auto-detecting number of horses from race start...")
            horse_count, horse_numbers = detect_race_horses(str(video_path))
            self.expected_horses = horse_count
            logger.info(f"Detected {horse_count} horses: {sorted(horse_numbers)}")
        elif self.expected_horses is None:
            self.expected_horses = 8
            logger.info("Using default of 8 horses")
        
        # Initialize tracker with detected horse count
        self.tracker = ImprovedHorseTracker(expected_horses=self.expected_horses)
        
        cap = cv2.VideoCapture(str(video_path))
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
        
        if self.save_annotated:
            output_video_path = self.output_dir / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_num = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detector.detect_horses(frame)
            
            # Use improved tracker instead of basic tracking
            tracked_horses = self.tracker.update(detections, frame)
            
            # The tracker has already updated track_ids in detections
            # Now update the analyzer with properly tracked horses
            self.analyzer.update_frame(frame_num, detections, (height, width))
            
            if self.save_annotated:
                annotated_frame = self.detector.annotate_frame(frame, detections)
                out.write(annotated_frame)
            
            frame_num += 1
            pbar.update(1)
            
            if frame_num % 100 == 0:
                logger.debug(f"Processed {frame_num}/{total_frames} frames")
        
        pbar.close()
        cap.release()
        
        if self.save_annotated:
            out.release()
            logger.info(f"Saved annotated video to: {output_video_path}")
        
        trip_analyses = self.analyzer.analyze_trips()
        
        # Get tracking summary
        tracking_summary = self.tracker.get_summary()
        
        # Filter analyses to only include actual horses and limit to expected count
        # Sort by total frames tracked (most reliable tracking first)
        sorted_analyses = sorted(trip_analyses, 
                               key=lambda a: len(self.analyzer.horse_histories.get(a.track_id, [])), 
                               reverse=True)
        
        valid_analyses = []
        for analysis in sorted_analyses:
            if len(valid_analyses) >= self.expected_horses:
                break
            if analysis.track_id <= 20:  # Reasonable horse number
                valid_analyses.append(analysis)
        
        results = {
            "video_path": str(video_path),
            "race_code": race_code,
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "duration": total_frames / fps,
                "resolution": f"{width}x{height}"
            },
            "num_horses_detected": len(valid_analyses),
            "tracking_summary": tracking_summary,
            "trip_analyses": [analysis.to_dict() for analysis in valid_analyses]
        }
        
        results_path = self.output_dir / f"{video_path.stem}_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved analysis results to: {results_path}")
        
        return results
    
    def generate_report(self, analysis_results: Dict) -> str:
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HORSE RACE TRIP ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        if analysis_results.get("race_code"):
            report_lines.append(f"Race Code: {analysis_results['race_code']}")
        
        report_lines.append(f"Video: {Path(analysis_results['video_path']).name}")
        report_lines.append(f"Duration: {analysis_results['video_info']['duration']:.1f} seconds")
        report_lines.append(f"Horses Detected: {analysis_results['num_horses_detected']}")
        report_lines.append("")
        
        sorted_analyses = sorted(
            analysis_results['trip_analyses'],
            key=lambda x: x['trip_difficulty_score'],
            reverse=True
        )
        
        for i, analysis in enumerate(sorted_analyses, 1):
            report_lines.append("-" * 40)
            report_lines.append(f"Horse #{analysis['track_id']}")
            if analysis.get('horse_name'):
                report_lines.append(f"Name: {analysis['horse_name']}")
            
            report_lines.append(f"Trip Difficulty Score: {analysis['trip_difficulty_score']:.1f}/100")
            report_lines.append(f"Pace Scenario: {analysis['pace_scenario']}")
            report_lines.append(f"Ground Loss: {analysis['ground_loss']:.2%}")
            
            if analysis.get('final_position'):
                report_lines.append(f"Final Position: {analysis['final_position']}")
            
            if analysis['events']:
                report_lines.append("\nTrip Events:")
                for event in analysis['events']:
                    report_lines.append(f"  - {event['type']} at {event['time']:.1f}s (severity: {event['severity']:.1f})")
            
            if analysis.get('position_chart'):
                positions = analysis['position_chart']
                report_lines.append(f"\nPosition Chart: {' -> '.join(map(str, positions))}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("KEY INSIGHTS:")
        report_lines.append("-" * 40)
        
        most_difficult = sorted_analyses[0] if sorted_analyses else None
        if most_difficult:
            report_lines.append(f"• Horse #{most_difficult['track_id']} had the most difficult trip")
            report_lines.append(f"  (difficulty score: {most_difficult['trip_difficulty_score']:.1f})")
        
        troubled_trips = [a for a in sorted_analyses if len(a['events']) > 2]
        if troubled_trips:
            report_lines.append(f"• {len(troubled_trips)} horses experienced significant traffic trouble")
        
        wide_runners = [a for a in sorted_analyses 
                       if any(e['type'] == 'wide_trip' for e in a['events'])]
        if wide_runners:
            report_lines.append(f"• {len(wide_runners)} horses ran wide trips")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        race_code = analysis_results.get('race_code') or 'unknown'
        report_path = self.output_dir / f"race_report_{race_code}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Generated report: {report_path}")
        
        return report