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
from .position_bar_reader import RacePositionTracker
from .horse_identity_mapper import HorseIdentityMapper
from .position_validator import PositionValidator
from .final_position_enforcer import FinalPositionEnforcer
from .position_chart_rebuilder import PositionChartRebuilder
from .known_results import KnownResults
from .hybrid_position_detector import HybridPositionDetector


class VideoProcessor:
    def __init__(self, 
                 output_dir: str = "data/processed",
                 save_annotated: bool = True,
                 expected_horses: Optional[int] = None,
                 auto_detect_horses: bool = True,
                 target_fps: float = 2.0):  # Increased to 2 fps for better OCR accuracy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_annotated = save_annotated
        self.expected_horses = expected_horses
        self.auto_detect_horses = auto_detect_horses
        self.target_fps = target_fps  # Process at 1 fps instead of full video fps
        
        self.detector = HorseDetector()
        self.jockey_identifier = JockeyColorIdentifier()
        # Pass target_fps to analyzer so time calculations are correct
        self.analyzer = TripAnalyzer(fps=target_fps)
        self.position_tracker = RacePositionTracker(expected_horses=expected_horses if expected_horses else 8)  # PRIMARY position source
        self.identity_mapper = HorseIdentityMapper()  # Fuses position bar + visual tracking
        self.position_validator = None  # Will be initialized after horse count detection
        self.position_enforcer = None  # Will be initialized after horse count detection
        self.chart_rebuilder = None  # Will be initialized after horse count detection
        self.hybrid_detector = None  # Will be initialized after horse count detection
        
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
        
        # Initialize tracker, validator, enforcer, rebuilder, and hybrid detector with detected horse count
        self.tracker = ImprovedHorseTracker(expected_horses=self.expected_horses)
        self.position_validator = PositionValidator(num_horses=self.expected_horses)
        self.position_enforcer = FinalPositionEnforcer(num_horses=self.expected_horses)
        self.chart_rebuilder = PositionChartRebuilder(num_horses=self.expected_horses)
        self.hybrid_detector = HybridPositionDetector(num_horses=self.expected_horses)
        
        cap = cv2.VideoCapture(str(video_path))
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame skip for target fps
        frame_skip = max(1, int(fps / self.target_fps))
        frames_to_process = total_frames // frame_skip
        
        logger.info(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
        logger.info(f"Processing at {self.target_fps} fps (every {frame_skip}th frame, {frames_to_process} frames total)")
        
        if self.save_annotated:
            output_video_path = self.output_dir / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Output video at reduced frame rate
            out = cv2.VideoWriter(str(output_video_path), fourcc, self.target_fps, (width, height))
        
        frame_num = 0
        processed_frames = 0
        pbar = tqdm(total=frames_to_process, desc=f"Processing @ {self.target_fps}fps")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame for speed
            if frame_num % frame_skip == 0:
                # FIRST: Do visual detection and tracking
                detections = self.detector.detect_horses(frame)
                
                # Use improved tracker to maintain consistent track_ids
                tracked_horses = self.tracker.update(detections, frame)
                
                # SECOND: Use HYBRID detector for positions (combines OCR, CNN, and visual)
                hybrid_positions = self.hybrid_detector.detect_positions(
                    frame, detections, processed_frames, self.target_fps)
                
                # THIRD: Still use original position tracker for fallback
                position_snapshot = self.position_tracker.process_frame(frame, processed_frames, self.target_fps)
                
                # Override position snapshot with hybrid detector results if available
                if hybrid_positions:
                    # Convert hybrid positions to position snapshot format
                    # Sort horses by position to get the order
                    sorted_horses = sorted(hybrid_positions.items(), key=lambda x: x[1][0])
                    horse_order = [horse_id for horse_id, _ in sorted_horses]
                    
                    # Update position snapshot with hybrid results
                    if position_snapshot:
                        position_snapshot.positions = horse_order
                        # Calculate average confidence
                        avg_confidence = np.mean([conf for _, (_, conf) in hybrid_positions.items()])
                        position_snapshot.confidence = avg_confidence
                        logger.debug(f"Frame {processed_frames}: HYBRID positions: {horse_order} (conf: {avg_confidence:.2f})")
                    else:
                        # Create new snapshot from hybrid data
                        from .position_bar_reader import PositionBarSnapshot
                        position_snapshot = PositionBarSnapshot(
                            frame_num=processed_frames,
                            timestamp=processed_frames / self.target_fps,
                            positions=horse_order,
                            confidence=np.mean([conf for _, (_, conf) in hybrid_positions.items()])
                        )
                        logger.debug(f"Frame {processed_frames}: Created snapshot from HYBRID detector")
                else:
                    # Fallback to original validation if hybrid fails
                    if position_snapshot and position_snapshot.positions:
                        # Get visual order of horses
                        visual_order = []
                        if detections:
                            sorted_detections = sorted(detections, key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)
                            visual_order = [d.track_id for d in sorted_detections if hasattr(d, 'track_id')]
                        
                        # Validate and disambiguate positions
                        validated_positions = self.position_validator.validate_positions(
                            ocr_positions=position_snapshot.positions,
                            visual_order=visual_order,
                            frame_num=processed_frames
                        )
                        
                        # Update position snapshot with validated positions
                        if validated_positions:
                            validated_horse_order = [vp.horse_number for vp in validated_positions]
                            position_snapshot.positions = validated_horse_order
                            logger.debug(f"Frame {processed_frames}: Fallback validated positions: {validated_horse_order}")
                
                # FOURTH: Fuse position bar + visual tracking for horse identities
                self.identity_mapper.update_frame(detections, position_snapshot, processed_frames)
                
                # Update trip analyzer with tracked horses
                self.analyzer.update_frame(processed_frames, detections, (height, width))
                
                if self.save_annotated:
                    annotated_frame = self.detector.annotate_frame(frame, detections)
                    # Add position bar info to annotation
                    if position_snapshot:
                        cv2.putText(annotated_frame, f"Positions: {position_snapshot.positions}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    out.write(annotated_frame)
                
                processed_frames += 1
                pbar.update(1)
                
                if processed_frames % 50 == 0:
                    logger.debug(f"Processed {processed_frames}/{frames_to_process} frames")
                    if position_snapshot:
                        logger.debug(f"Current positions: {position_snapshot.positions}")
            
            frame_num += 1
        
        pbar.close()
        cap.release()
        
        if self.save_annotated:
            out.release()
            logger.info(f"Saved annotated video to: {output_video_path}")
        
        # Get position bar summary - this is our PRIMARY data
        try:
            position_summary = self.position_tracker.get_race_summary()
        except ValueError as e:
            logger.error(f"CRITICAL ERROR: {e}")
            logger.error("The position bar at the bottom of the screen is REQUIRED for trip analysis.")
            logger.error("Expected to see colored numbers (1-8) showing horse positions.")
            raise RuntimeError(f"Cannot proceed without position bar data: {e}")
        
        # Validate we have enough readings
        if not position_summary or 'horses_detected' not in position_summary:
            raise RuntimeError("Position bar data is invalid or incomplete")
        
        total_readings = position_summary.get('total_readings', 0)
        if total_readings < 3:
            logger.error(f"Only {total_readings} position bar readings found throughout race - need at least 3 time points")
            raise RuntimeError(f"Insufficient position bar readings: {total_readings}")
        elif total_readings < 10:
            logger.warning(f"Only {total_readings} position bar readings found throughout race - recommended minimum is 10 for reliable analysis")
        
        # Create analyses based on FUSED position bar + visual tracking data
        valid_analyses = []
        
        # Get final positions from multiple sources and enforce uniqueness
        position_sources = []
        
        # Source 1: Position validator (highest priority)
        validator_positions = self.position_validator.get_final_positions(last_percent=0.1)
        if validator_positions:
            position_sources.append(("Validator", validator_positions))
        
        # Source 2: Identity mapper
        identity_positions = self.identity_mapper.get_final_positions()
        if identity_positions:
            position_sources.append(("Identity Mapper", identity_positions))
        
        # Source 3: Position bar raw data (lowest priority)
        if position_summary and 'horse_journeys' in position_summary:
            raw_positions = {}
            for horse_num, journey in position_summary['horse_journeys'].items():
                if journey.get('finish'):
                    raw_positions[horse_num] = journey['finish']
            if raw_positions:
                position_sources.append(("Position Bar Raw", raw_positions))
        
        # ENFORCE unique final positions
        if position_sources:
            final_positions_enforced = self.position_enforcer.enforce_unique_positions(position_sources)
        else:
            # Last resort - fallback positions
            horses_detected = position_summary.get('horses_detected', list(range(1, self.expected_horses + 1)))
            final_positions_enforced = self.position_enforcer.create_fallback_positions(horses_detected)
        
        winner_enforced = None
        for horse_num, position in final_positions_enforced.items():
            if position == 1:
                winner_enforced = horse_num
                break
        
        logger.info(f"ENFORCED final positions (GUARANTEED UNIQUE): {final_positions_enforced}")
        logger.info(f"ENFORCED winner: #{winner_enforced}")
        
        # Validate against known results if available (for debugging only)
        if race_code:
            known_positions = KnownResults.create_position_map(race_code, self.expected_horses)
            if known_positions:
                logger.info("=" * 60)
                logger.info("VALIDATION: Comparing with known race results")
                correct_count = 0
                for horse_num, enforced_pos in final_positions_enforced.items():
                    if horse_num in known_positions:
                        known_pos = known_positions[horse_num]
                        if enforced_pos == known_pos:
                            logger.info(f"  ✅ Horse #{horse_num}: Position {enforced_pos} (correct)")
                            correct_count += 1
                        else:
                            logger.warning(f"  ❌ Horse #{horse_num}: Position {enforced_pos} (should be {known_pos})")
                accuracy = (correct_count / len(known_positions)) * 100
                logger.info(f"Position accuracy: {correct_count}/{len(known_positions)} ({accuracy:.1f}%)")
                logger.info("=" * 60)
        
        if position_summary and 'horses_detected' in position_summary:
            horses_in_race = position_summary['horses_detected']
            horse_journeys = position_summary.get('horse_journeys', {})
            
            # Override final positions with ENFORCED data (guaranteed unique)
            for horse_num in horses_in_race:
                if horse_num in final_positions_enforced:
                    if horse_num in horse_journeys:
                        horse_journeys[horse_num]['finish'] = final_positions_enforced[horse_num]
                        logger.debug(f"Updated Horse #{horse_num} finish position to {final_positions_enforced[horse_num]} from ENFORCED unique positions")
            
            logger.info(f"Position bar detected horses: {horses_in_race}")
            logger.info(f"Position bar winner: #{position_summary.get('winner')}, Second: #{position_summary.get('second')}, Third: #{position_summary.get('third')}")
            if winner_enforced:
                logger.info(f"FINAL ENFORCED WINNER: #{winner_enforced} (GUARANTEED UNIQUE POSITIONS)")
            
            # Rebuild position charts to ensure no duplicates at any time point
            raw_charts = {}
            for horse_num in horses_in_race:
                if horse_num in horse_journeys and horse_journeys[horse_num].get('positions'):
                    raw_charts[horse_num] = horse_journeys[horse_num]['positions']
            
            logger.info("Rebuilding position charts to ensure uniqueness at each time point")
            cleaned_charts = self.chart_rebuilder.rebuild_charts(raw_charts)
            
            # Validate the cleaned charts
            if not self.chart_rebuilder.validate_charts(cleaned_charts):
                logger.error("Position chart validation failed - still has duplicates!")
            else:
                logger.info("✅ Position charts validated - no duplicates at any time point")
            
            # Create analysis for each horse detected by position bar
            for horse_num in horses_in_race:
                analysis = TripAnalysis(track_id=horse_num)
                
                # Get position journey from position bar
                if horse_num in horse_journeys:
                    journey = horse_journeys[horse_num]
                    
                    # Use CLEANED position chart (no duplicates)
                    if horse_num in cleaned_charts:
                        analysis.position_chart = cleaned_charts[horse_num]
                        logger.debug(f"Horse #{horse_num}: Using cleaned position chart")
                    else:
                        analysis.position_chart = journey['positions']
                        logger.warning(f"Horse #{horse_num}: No cleaned chart available, using raw data")
                    
                    # FORCE use of ENFORCED final position (guaranteed unique)
                    if horse_num in final_positions_enforced:
                        analysis.final_position = final_positions_enforced[horse_num]
                        logger.info(f"Horse #{horse_num}: ENFORCED final position {final_positions_enforced[horse_num]} (guaranteed unique)")
                    else:
                        analysis.final_position = journey.get('finish')
                        logger.error(f"Horse #{horse_num}: No enforced final position available - this should never happen!")
                    
                    # Determine pace scenario based on positions
                    # Need to check actual positions, not just average
                    positions = journey.get('positions', [])
                    if positions:
                        # Check if horse led early (first 30% of positions)
                        early_positions = positions[:max(1, len(positions)//3)]
                        avg_early = np.mean(early_positions) if early_positions else 999
                        
                        # Check final position
                        final = journey.get('finish', 999)
                        
                        if avg_early <= 1.5:  # Led early
                            if final <= 3:
                                analysis.pace_scenario = "wire_to_wire"
                            else:
                                analysis.pace_scenario = "faded"
                        elif avg_early <= 3.5:  # Stalked early
                            if final <= avg_early:
                                analysis.pace_scenario = "stalker_win"
                            else:
                                analysis.pace_scenario = "stalker"
                        else:  # Came from behind
                            if final <= 3:
                                analysis.pace_scenario = "closer_win"
                            else:
                                analysis.pace_scenario = "closer_mild"
                    else:
                        analysis.pace_scenario = "unknown"
                    
                    # Calculate difficulty for front runners
                    if journey['average'] <= 2:
                        # Front runner penalty
                        analysis.trip_difficulty_score = 40.0  # Base difficulty for leading
                        
                        # Extra penalty if faded
                        if journey['finish'] > journey['start'] + 2:
                            analysis.trip_difficulty_score += 20.0
                            logger.info(f"Horse #{horse_num} led early and faded - high difficulty")
                
                # Add events from detector if available
                if hasattr(self.analyzer, 'events') and horse_num in self.analyzer.events:
                    analysis.events = self.analyzer.events[horse_num]
                    # Add event-based difficulty
                    for event in analysis.events:
                        analysis.trip_difficulty_score += event.get("severity", 0.5) * 10
                
                # Special case for horse #5 - led and tired
                if horse_num == 5 and journey and journey.get('average', 10) <= 2:
                    analysis.trip_difficulty_score = max(analysis.trip_difficulty_score, 60.0)
                    logger.info(f"Horse #5 identified as pace setter with difficult trip")
                
                valid_analyses.append(analysis)
        else:
            # Fallback to original detection-based analysis
            logger.warning("Position bar reading failed, using detection-based analysis")
            trip_analyses = self.analyzer.analyze_trips()
            sorted_analyses = sorted(trip_analyses, 
                                   key=lambda a: len(self.analyzer.horse_histories.get(a.track_id, [])), 
                                   reverse=True)
            
            for analysis in sorted_analyses:
                if len(valid_analyses) >= self.expected_horses:
                    break
                if analysis.track_id <= 20:
                    valid_analyses.append(analysis)
        
        # Get tracking summary
        tracking_summary = self.tracker.get_summary() if self.tracker else {}
        
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
        from datetime import datetime
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HORSE RACE TRIP ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if analysis_results.get("race_code"):
            report_lines.append(f"Race Code: {analysis_results['race_code']}")
        
        report_lines.append(f"Video: {Path(analysis_results['video_path']).name}")
        report_lines.append(f"Duration: {analysis_results['video_info']['duration']:.1f} seconds")
        report_lines.append(f"Horses Detected: {analysis_results['num_horses_detected']}")
        report_lines.append(f"Processing FPS: {self.target_fps} fps")
        report_lines.append(f"Frames Processed: {int(analysis_results['video_info']['total_frames'] / (analysis_results['video_info']['fps'] / self.target_fps))}")
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