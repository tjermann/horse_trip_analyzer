# Horse Race Trip Analyzer - Project Context

## Project Overview
Deep learning computer vision system for analyzing horse racing trips at scale. Processes video feeds to evaluate trip difficulty and identify factors affecting performance beyond just finishing position.

## Core Objective
Quantify trip difficulty (0-100 scale) by detecting and analyzing various racing scenarios that impact a horse's ability to perform, helping identify horses that ran better/worse than their finishing position suggests.

## Key Technical Components

### 1. Video Scraper (`src/video_scraper.py`)
- Scrapes videos from TJK website (Turkish Jockey Club)
- Uses Selenium for dynamic content handling
- Downloads videos with yt-dlp or direct requests
- Extracts race metadata (horses, jockeys, etc.)

### 2. Race Start Detector (`src/race_start_detector.py`) **NEW**
- **Automatically detects number of horses** from race start screens
- Multi-region OCR analysis (top, bottom, sides, center)
- Advanced preprocessing (binary, adaptive threshold, edge enhancement)
- Validates against reasonable horse number ranges (1-20)
- Analyzes first 10-15 seconds where lineup info is displayed

### 3. Horse Detector (`src/horse_detector.py`)
- YOLOv8x model for horse detection
- Basic ByteTrack for initial object detection
- Color feature extraction for horse/jockey identification
- Confidence-based filtering and annotation

### 4. Improved Horse Tracker (`src/horse_tracker.py`) **NEW**
- **Maintains consistent horse IDs (1-8)** throughout race
- **EasyOCR integration** for reading saddle cloth numbers
- **Re-identification features** using color histograms + Gabor filters
- **Handles temporary occlusions** with position prediction
- **Object permanence** - tracks horses even when off-screen

### 5. Trip Analyzer (`src/trip_analyzer.py`)
- Real-time position tracking for each horse
- Event detection algorithms:
  - **Boxed In**: Horse surrounded with no clear path
  - **Wide Trip**: Forced to run outside optimal path
  - **Bumped**: Physical interference detected via acceleration spikes
  - **Steadied**: Forced to slow in traffic
- Calculates metrics:
  - Ground loss (extra distance vs optimal path)
  - Pace scenario (wire-to-wire, closer, stalker, etc.)
  - Energy distribution across race quarters
  - Speed variance and acceleration patterns

### 6. Video Processor (`src/video_processor.py`)
- Main pipeline orchestrating detection → tracking → analysis
- Generates annotated videos with bounding boxes
- Produces JSON analysis and human-readable reports
- Handles frame-by-frame processing with progress tracking

### 7. Batch Processor (`batch_processor.py`)
- Parallel processing of multiple races
- Aggregates results into CSV/JSON
- Statistical analysis across races

## Trip Difficulty Scoring Algorithm

```
Score = (Event Severity × 10) + (Ground Loss × 20) + Pace Penalty + (Speed Variance / 10)
```

- Event severity: 0.5-0.9 based on type
- Pace penalties: +15 for wire-to-wire, +25 for faded
- Capped at 100

## Critical Detection Logic

### Boxing Detection
- Checks for horses ahead, behind, and to sides within proximity thresholds
- Triggers when surrounded on 3+ sides

### Wide Trip Detection
- Lateral position >70% or <30% of track width
- Sustained over multiple frames

### Bump Detection
- Acceleration spike >50 units
- Combined with lateral position change >10%

### Steadying Detection
- Speed deceleration <-10 units in smoothed data
- Uses Savitzky-Golay filter for noise reduction

## Current State & Next Steps

### Completed
- Full pipeline from video download to analysis report
- Multi-horse tracking and individual trip analysis
- Batch processing capability
- Trip difficulty scoring system

### Limitations & Improvements Needed
1. Using general YOLOv8 (not horse-specific)
2. Jockey color identification incomplete
3. No integration with official race results
4. Track geometry assumptions (straight vs turns)
5. No stride analysis or biomechanics

### Potential Enhancements
- Fine-tune YOLO specifically on horse racing data
- Implement turn detection for accurate distance calculations
- Add pace pressure analysis (early speed competition)
- Integrate with betting odds/results for ROI analysis
- Real-time streaming capability

## Usage Examples

```bash
# Auto-detect horse count and analyze (recommended)
python main.py --race-code 194367 --save-annotated

# Force specific horse count
python main.py --race-code 194367 --num-horses 12 --save-annotated

# Disable auto-detection
python main.py --race-code 194367 --no-auto-detect

# Batch processing with auto-detection
python batch_processor.py --race-file races.txt --no-download --max-workers 4

# Process existing video
python main.py --video-path /path/to/video.mp4 --output-dir custom_output/
```

## Key Insights for Trip Analysis
- Horses with high difficulty scores but good finishes = talented
- Horses with low difficulty scores but poor finishes = limited ability
- Ground loss >5% typically indicates significant traffic trouble
- Multiple steadying events correlate with compromised performance
- Wide trip on turns adds ~2-3 lengths of extra distance

## Technical Requirements
- Python 3.11 recommended for performance
- GPU with CUDA for faster processing (2-5 fps vs 0.5-1 fps CPU)
- ~4GB RAM for 1080p video processing
- Chrome/Chromium for web scraping

## File Structure
```
horse_trip_analyzer/
├── src/                     # Core modules
├── data/videos/            # Downloaded race videos  
├── data/processed/         # Analysis outputs
├── main.py                 # Single race CLI
├── batch_processor.py      # Multi-race processing
└── requirements.txt        # Dependencies
```

## Debug Commands
```bash
# Test detection on single frame
python -c "from src.horse_detector import HorseDetector; import cv2; d = HorseDetector(); frame = cv2.imread('test.jpg'); print(d.detect_horses(frame))"

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Contact with TJK API
Base URL: https://www.tjk.org/EN/YarisSever/Info/YarisVideoKosu/Kosu?KosuKodu={race_code}
- Videos are embedded as HTML5 video elements
- May require session handling for bulk downloads
- Consider rate limiting to avoid blocking