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
- **Maintains consistent horse IDs (1-20)** throughout race
- **EasyOCR integration** for reading saddle cloth numbers
- **Re-identification features** using color histograms + Gabor filters
- **Handles temporary occlusions** with position prediction
- **Object permanence** - tracks horses even when off-screen

### 5. Position Bar Reader (`src/position_bar_reader.py`) **CRITICAL**
- **PRIMARY data source** for horse positions throughout race
- Reads colored numbers from bottom 15% of screen (left-to-right = 1st to last)
- **Supports 1-20 horses** (previously limited to 8)
- Advanced OCR preprocessing for colored numbers
- **Required for analysis** - system fails without position bar data
- Validates position sequences and maintains race summary

### 6. Trip Analyzer (`src/trip_analyzer.py`)
- Real-time position tracking for each horse
- Event detection algorithms:
  - **Boxed In**: Horse surrounded with no clear path
  - **Wide Trip**: Forced to run outside optimal path
  - **Bumped**: Physical interference detected via acceleration spikes
  - **Steadied**: Forced to slow in traffic
  - **Wind Resistance**: Front-running penalty for leading horses
- Calculates metrics:
  - Ground loss (extra distance vs optimal path)
  - Pace scenario (wire-to-wire, closer, stalker, etc.)
  - Energy distribution across race quarters
  - Speed variance and acceleration patterns

### 7. Video Processor (`src/video_processor.py`)
- Main pipeline orchestrating detection → tracking → analysis
- **Optimized frame processing** (1 fps default for 25x speedup)
- Generates annotated videos with bounding boxes
- Produces JSON analysis and human-readable reports
- Handles frame-by-frame processing with progress tracking

### 8. Batch Processor (`batch_processor.py`)
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

## **CRITICAL: Position Bar System**

The position bar is the **PRIMARY and REQUIRED** data source for horse positions. Without it, analysis cannot proceed.

### Position Bar Requirements
- **Location**: Colored numbers at bottom 15% of video screen
- **Format**: Left-to-right order indicates 1st place to last place
- **Visibility**: Must be clearly visible throughout race (first 15 seconds critical)
- **Support**: Handles 1-20 horses (extended from original 8-horse limit)

### Position Bar Processing
- Advanced OCR with multiple preprocessing methods (hue, saturation, brightness channels)
- Validates position sequences for consistency
- **Critical error handling**: System stops if position bar cannot be read
- Used for final race results (winner, second, third) and trip difficulty scoring

### Error Messages
```
CRITICAL ERROR: Position bar could not be read from video
Expected to see colored numbers (1-8) showing horse positions
Cannot proceed without position bar data
```

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

### Wind Resistance Detection
- Front-runners (1st/2nd position) tracked for sustained periods
- 80% of recent positions must be in front (1st/2nd)
- Adds difficulty penalty for leading horses breaking wind

## Current State & Next Steps

### Completed
- Full pipeline from video download to analysis report
- Multi-horse tracking and individual trip analysis
- **Position bar as primary data source** (critical improvement)
- **Support for 1-20 horses** (extended from 8 horse limit)
- **Critical error handling** for missing position bar data
- Batch processing capability
- Trip difficulty scoring system
- **Front-runner difficulty recognition** (wind resistance penalty)
- **Race report naming fixed** (was saving as "_unknown")

### Current Critical Issues
1. **OCR Reading Invalid Numbers** - Detecting horses 11, 15, 17 etc. when only 8 exist
   - Position bar coordinates are correct (75-87% from top)
   - OCR is misreading or picking up extra numbers
   - Now filtering to only accept horses 1-8 for 8-horse races
2. **Pace Scenario Classification Wrong** - Horses labeled wire-to-wire incorrectly
   - Horse #2 and #7 did not lead but classified as wire-to-wire
   - Need better position progression analysis
3. **No Trip Events Detected** - Zero events like bumping, boxing, etc.
   - Event detection thresholds may be too strict
   - Need to verify event detection is working

### Limitations & Improvements Needed
1. Using general YOLOv8 (not horse-specific)
2. **Position bar OCR accuracy** - CPU processing is slow (90+ minutes/race)
3. No integration with official race results
4. Track geometry assumptions (straight vs turns)
5. No stride analysis or biomechanics
6. **GPU highly recommended** for reasonable processing times

### Potential Enhancements
- Fine-tune YOLO specifically on horse racing data
- Implement turn detection for accurate distance calculations
- Add pace pressure analysis (early speed competition)
- Integrate with betting odds/results for ROI analysis
- Real-time streaming capability

## Usage Examples

```bash
# Process existing local video (skips download)
python main.py --video-path data/videos/race_194367.mp4 --race-code 194367 --num-horses 8 --no-auto-detect

# Download and process new race
python main.py --race-code 194367 --num-horses 8 --no-auto-detect --save-annotated

# Auto-detect horse count (slow but thorough) 
python main.py --race-code 194367 --save-annotated

# Debug position bar detection
python debug_position_bar.py  # Creates debug images

# Batch processing with manual horse counts (faster)
python batch_processor.py --race-file races.txt --no-download --max-workers 4

# Speed options
python main.py --race-code 194367 --target-fps 2.0  # Higher quality
python main.py --race-code 194367 --target-fps 0.5  # Ultra-fast
```

## Key Insights for Trip Analysis
- Horses with high difficulty scores but good finishes = talented
- Horses with low difficulty scores but poor finishes = limited ability
- Ground loss >5% typically indicates significant traffic trouble
- Multiple steadying events correlate with compromised performance
- Wide trip on turns adds ~2-3 lengths of extra distance

## Technical Requirements
- Python 3.11 recommended for performance
- **Position bar analysis is CPU-intensive** (~30 seconds per frame)
- **GPU highly recommended** (provides 10-20x speedup for OCR)
- ~2GB RAM for 1080p video processing (reduced due to frame skipping)
- Chrome/Chromium for web scraping
- **Critical**: Position bar must be visible in video (colored numbers at bottom)

### Model Downloads Required
YOLOv8 model files are too large for git repositories. After cloning, run:
```bash
# The system will automatically download YOLOv8x on first run
# Or manually download:
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
```

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
# Debug position bar detection (creates debug images)
python debug_position_bar.py

# Test detection on single frame
python -c "from src.horse_detector import HorseDetector; import cv2; d = HorseDetector(); frame = cv2.imread('test.jpg'); print(d.detect_horses(frame))"

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test position bar reader directly
python -c "from src.position_bar_reader import PositionBarReader; import cv2; r = PositionBarReader(); frame = cv2.imread('test.jpg'); print(r.read_position_bar(frame))"
```

## Contact with TJK API
Base URL: https://www.tjk.org/EN/YarisSever/Info/YarisVideoKosu/Kosu?KosuKodu={race_code}
- Videos are embedded as HTML5 video elements
- May require session handling for bulk downloads
- Consider rate limiting to avoid blocking