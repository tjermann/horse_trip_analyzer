#!/usr/bin/env python3
"""
Validation Script for Finish-Anchored Tracking
Compares finish-anchored tracking results against manual ground truth labels
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
import numpy as np
from collections import defaultdict
import math

def load_manual_labels(labels_file: str) -> Dict[int, List[Dict]]:
    """Load manual labeling ground truth data"""
    with open(labels_file) as f:
        data = json.load(f)
    
    labels_by_frame = defaultdict(list)
    
    # Handle the actual manual labels format
    labels = data.get('labels', {})
    for frame_str, horse_data in labels.items():
        frame_num = int(frame_str)
        
        for horse_id_str, bbox in horse_data.items():
            horse_id = int(horse_id_str)
            
            labels_by_frame[frame_num].append({
                'horse_id': horse_id,
                'bbox': tuple(bbox),  # [x, y, w, h]
                'ground_truth': True
            })
    
    return dict(labels_by_frame)

def load_tracking_results(results_file: str) -> Dict[int, List[Dict]]:
    """Load finish-anchored tracking results"""
    with open(results_file) as f:
        data = json.load(f)
    
    tracking_by_frame = defaultdict(list)
    
    for track_data in data.get('finish_anchored_tracking_results', []):
        program_number = track_data['program_number']
        finish_position = track_data['finish_position']
        
        for detection in track_data['track']:
            frame_num = detection['frame_number']
            bbox = tuple(detection['bbox'])  # [x, y, w, h]
            confidence = detection['confidence']
            
            tracking_by_frame[frame_num].append({
                'program_number': program_number,
                'finish_position': finish_position,
                'bbox': bbox,
                'confidence': confidence
            })
    
    return dict(tracking_by_frame)

def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections(ground_truth: List[Dict], predictions: List[Dict], iou_threshold: float = 0.5) -> Tuple[List, List, List]:
    """Match ground truth labels with predictions using IoU threshold"""
    matches = []
    unmatched_gt = list(range(len(ground_truth)))
    unmatched_pred = list(range(len(predictions)))
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(ground_truth), len(predictions)))
    for i, gt in enumerate(ground_truth):
        for j, pred in enumerate(predictions):
            iou_matrix[i, j] = calculate_iou(gt['bbox'], pred['bbox'])
    
    # Greedy matching (highest IoU first)
    while len(unmatched_gt) > 0 and len(unmatched_pred) > 0:
        # Find highest IoU among unmatched
        best_iou = 0.0
        best_gt_idx = None
        best_pred_idx = None
        
        for gt_idx in unmatched_gt:
            for pred_idx in unmatched_pred:
                if iou_matrix[gt_idx, pred_idx] > best_iou:
                    best_iou = iou_matrix[gt_idx, pred_idx]
                    best_gt_idx = gt_idx
                    best_pred_idx = pred_idx
        
        if best_iou >= iou_threshold:
            matches.append({
                'gt_idx': best_gt_idx,
                'pred_idx': best_pred_idx,
                'iou': best_iou,
                'gt_horse_id': ground_truth[best_gt_idx]['horse_id'],
                'pred_program_number': predictions[best_pred_idx]['program_number'],
                'pred_confidence': predictions[best_pred_idx]['confidence']
            })
            unmatched_gt.remove(best_gt_idx)
            unmatched_pred.remove(best_pred_idx)
        else:
            break
    
    return matches, unmatched_gt, unmatched_pred

def validate_tracking(labels_file: str, results_file: str, race_results: Dict[str, int] = None) -> Dict:
    """Validate finish-anchored tracking against manual labels"""
    logger.info(f"Validating tracking results: {results_file}")
    logger.info(f"Against manual labels: {labels_file}")
    
    # Load data
    manual_labels = load_manual_labels(labels_file)
    tracking_results = load_tracking_results(results_file)
    
    # Initialize metrics
    total_matches = 0
    total_gt_detections = 0
    total_pred_detections = 0
    frame_metrics = []
    identity_matches = defaultdict(list)  # Track identity consistency
    
    # Common frames between both datasets
    common_frames = set(manual_labels.keys()) & set(tracking_results.keys())
    logger.info(f"Analyzing {len(common_frames)} common frames")
    
    for frame_num in sorted(common_frames):
        gt_detections = manual_labels[frame_num]
        pred_detections = tracking_results[frame_num]
        
        matches, unmatched_gt, unmatched_pred = match_detections(gt_detections, pred_detections)
        
        frame_metrics.append({
            'frame': frame_num,
            'gt_count': len(gt_detections),
            'pred_count': len(pred_detections),
            'matches': len(matches),
            'precision': len(matches) / len(pred_detections) if pred_detections else 0.0,
            'recall': len(matches) / len(gt_detections) if gt_detections else 0.0
        })
        
        # Track identity consistency
        for match in matches:
            gt_horse_id = match['gt_horse_id']
            pred_program_number = match['pred_program_number']
            identity_matches[gt_horse_id].append(pred_program_number)
        
        total_matches += len(matches)
        total_gt_detections += len(gt_detections)
        total_pred_detections += len(pred_detections)
        
        if len(matches) > 0:
            logger.info(f"Frame {frame_num}: {len(matches)} matches out of {len(gt_detections)} GT, {len(pred_detections)} pred")
    
    # Calculate overall metrics
    overall_precision = total_matches / total_pred_detections if total_pred_detections > 0 else 0.0
    overall_recall = total_matches / total_gt_detections if total_gt_detections > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Identity consistency analysis
    identity_consistency = {}
    for gt_horse_id, pred_program_numbers in identity_matches.items():
        if pred_program_numbers:
            # Most common prediction for this horse
            from collections import Counter
            counter = Counter(pred_program_numbers)
            most_common_pred, count = counter.most_common(1)[0]
            consistency = count / len(pred_program_numbers)
            identity_consistency[gt_horse_id] = {
                'most_common_prediction': most_common_pred,
                'consistency_score': consistency,
                'total_detections': len(pred_program_numbers),
                'prediction_distribution': dict(counter)
            }
    
    # Compile results
    validation_results = {
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'total_matches': total_matches,
            'total_gt_detections': total_gt_detections,
            'total_pred_detections': total_pred_detections
        },
        'identity_consistency': identity_consistency,
        'frame_metrics': frame_metrics,
        'common_frames_count': len(common_frames),
        'validation_summary': {
            'avg_precision': np.mean([m['precision'] for m in frame_metrics]),
            'avg_recall': np.mean([m['recall'] for m in frame_metrics]),
            'frames_with_matches': len([m for m in frame_metrics if m['matches'] > 0])
        }
    }
    
    return validation_results

def print_validation_summary(results: Dict):
    """Print a human-readable validation summary"""
    print("\n🏁 FINISH-ANCHORED TRACKING VALIDATION RESULTS")
    print("=" * 60)
    
    overall = results['overall_metrics']
    print(f"📊 Overall Performance:")
    print(f"   Precision: {overall['precision']:.3f} ({overall['total_matches']}/{overall['total_pred_detections']})")
    print(f"   Recall:    {overall['recall']:.3f} ({overall['total_matches']}/{overall['total_gt_detections']})")
    print(f"   F1-Score:  {overall['f1_score']:.3f}")
    
    summary = results['validation_summary']
    print(f"\n📈 Frame-by-Frame Analysis:")
    print(f"   Frames analyzed: {results['common_frames_count']}")
    print(f"   Frames with matches: {summary['frames_with_matches']}")
    print(f"   Average precision: {summary['avg_precision']:.3f}")
    print(f"   Average recall: {summary['avg_recall']:.3f}")
    
    print(f"\n🎯 Identity Consistency:")
    identity = results['identity_consistency']
    for horse_id, data in identity.items():
        pred_num = data['most_common_prediction']
        consistency = data['consistency_score']
        detections = data['total_detections']
        print(f"   Horse #{horse_id} → #{pred_num} ({consistency:.1%} consistent, {detections} detections)")
        
        if len(data['prediction_distribution']) > 1:
            print(f"     Distribution: {data['prediction_distribution']}")
    
    # Problem frames
    problem_frames = [m for m in results['frame_metrics'] if m['recall'] < 0.8 or m['precision'] < 0.8]
    if problem_frames:
        print(f"\n⚠️  Problem Frames (precision or recall < 80%):")
        for frame in problem_frames[:10]:  # Show first 10 problem frames
            print(f"   Frame {frame['frame']}: P={frame['precision']:.2f}, R={frame['recall']:.2f} "
                  f"({frame['matches']}/{frame['pred_count']} pred, {frame['gt_count']} GT)")

def main():
    parser = argparse.ArgumentParser(description="Validate finish-anchored tracking against manual labels")
    parser.add_argument("--labels", required=True, help="Path to manual labels JSON file")
    parser.add_argument("--results", required=True, help="Path to tracking results JSON file")
    parser.add_argument("--output", help="Path to save validation results JSON")
    parser.add_argument("--race-results", help="Path to race results for additional validation")
    
    args = parser.parse_args()
    
    # Load race results if provided
    race_results = None
    if args.race_results:
        with open(args.race_results) as f:
            race_data = json.load(f)
            if 'races' in race_data:
                # Extract race results from validation data format
                for race_info in race_data['races'].values():
                    if 'horse_numbers' in race_info:
                        race_results = race_info['horse_numbers']
                        break
    
    # Run validation
    validation_results = validate_tracking(args.labels, args.results, race_results)
    
    # Print summary
    print_validation_summary(validation_results)
    
    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Detailed validation results saved to {args.output}")

if __name__ == "__main__":
    main()