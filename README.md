# Horse Race Trip Analyzer

A deep learning computer vision system for analyzing horse racing trips at scale. This system processes video feeds to evaluate trip difficulty, identifying various scenarios that affect a horse's performance.

## Features

### Trip Detection Capabilities
- **Boxing In**: Detects when horses are trapped with no clear running lane
- **Wide Trips**: Identifies horses forced to run wide on turns (extra distance)
- **Bumping/Interference**: Detects physical contact between horses
- **Steadying/Checking**: Identifies when horses must slow due to traffic
- **Pace Scenarios**: Analyzes running styles (front-runner, stalker, closer)
- **Ground Loss**: Calculates extra distance traveled vs optimal path
- **Energy Distribution**: Analyzes speed and acceleration patterns

### Technical Features
- YOLOv8-based horse detection and tracking
- **Hybrid Position Detection System** (NEW)
  - Enhanced OCR with 10+ preprocessing techniques
  - Custom CNN model for position bar digit recognition
  - Visual tracking verification for physical plausibility
  - Weighted fusion with confidence scoring
- **Guaranteed Unique Position Assignment**
  - Multiple validation layers
  - Position chart rebuilding
  - Final position enforcement
- **Automatic horse count detection** from race start screens
- **Advanced multi-object tracking** with re-identification features
- **Trip event detection throughout entire race** (not just start)
- **Optimized frame processing** (0.5-2 fps for speed/accuracy balance)
- Automated video scraping from TJK website
- Batch processing for multiple races
- Detailed trip difficulty scoring (0-100 scale)
- Visual annotations and reporting

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd horse_trip_analyzer

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model (required - too large for git)
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
# This will download yolov8x.pt (~140MB) to the current directory

# Alternative: Manual download
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# Create necessary directories
mkdir -p data/{videos,processed} logs models
```

### Important: YOLOv8 Model Required
The YOLOv8x model file (`yolov8x.pt`) is required but not included in the repository due to its size (~140MB). The model will be automatically downloaded on first run, or you can download it manually using the commands above.

## Usage

### Single Race Analysis

```bash
# Auto-detect horse count and analyze (recommended)
python main.py --race-code 194367 --save-annotated

# Specify horse count manually
python main.py --race-code 194367 --num-horses 8

# Disable auto-detection (uses default 8 horses)
python main.py --race-code 194367 --no-auto-detect

# Process at higher quality (slower)
python main.py --race-code 194367 --target-fps 4.0

# Ultra-fast processing for testing
python main.py --race-code 194367 --target-fps 0.5

# Analyze existing video
python main.py --video-path data/videos/race_194367.mp4
```

### Batch Processing

```bash
# Process multiple races (auto-detects horse count for each)
python batch_processor.py --race-codes 194367 194368 194369

# Process from file
echo "194367\n194368\n194369" > race_list.txt
python batch_processor.py --race-file race_list.txt --max-workers 4
```

## Output Files

### For each race:
- `*_analysis.json`: Detailed trip analysis data
- `*_annotated.mp4`: Video with detection overlays (optional)
- `race_report_*.txt`: Human-readable report

### Batch processing:
- `aggregate_analysis.csv`: Summary statistics for all races
- `all_results.json`: Complete analysis data

## Trip Difficulty Scoring

The system calculates a 0-100 difficulty score based on:
- Number and severity of trip events (40%)
- Ground loss percentage (20%)
- Pace scenario difficulty (15%)
- Speed variance (10%)
- Traffic trouble frequency (15%)

## Architecture

```
horse_trip_analyzer/
├── src/
│   ├── video_scraper.py             # TJK website video downloading
│   ├── horse_detector.py            # YOLOv8 detection & basic tracking
│   ├── horse_tracker.py             # Advanced tracking with re-identification
│   ├── race_start_detector.py       # Auto-detect horse count from race start
│   ├── hybrid_position_detector.py  # NEW: Multi-method position detection
│   ├── position_validator.py        # Position validation & consensus
│   ├── final_position_enforcer.py   # Guarantee unique final positions
│   ├── position_chart_rebuilder.py  # Clean position charts
│   ├── known_results.py             # Validation against known results
│   ├── trip_analyzer.py             # Trip event detection & scoring
│   └── video_processor.py           # Main processing pipeline
├── main.py                          # Single race CLI
├── batch_processor.py               # Batch processing CLI
└── requirements.txt                 # Dependencies
```

## Key Metrics Explained

- **Ground Loss**: Extra distance traveled compared to optimal path (as %)
- **Pace Scenario**: Running style classification (wire-to-wire, closer, etc.)
- **Trip Events**: Specific incidents affecting the horse's run
- **Energy Distribution**: Speed consistency across race quarters

## Performance Considerations

- **Processing speed**: ~30-60 seconds per 3-minute race (1 fps mode)
- **Memory usage**: ~2GB for 1080p video (reduced due to frame skipping)
- **CPU-friendly**: Optimized for CPU processing, GPU optional
- **Speed options**: 0.5-4 fps (25x to 6x speedup vs full framerate)

## Recent Major Improvements (2025)

- [x] **Hybrid Position Detection System** 🆕
  - Enhanced OCR with 10+ preprocessing techniques
  - Custom CNN model for position bar digit recognition  
  - Visual tracking verification for physical plausibility
  - Weighted fusion with confidence scoring
- [x] **Guaranteed Unique Position Assignment** 🆕
  - Position validation and consensus algorithms
  - Position chart rebuilding to prevent impossible sequences
  - Final position enforcement ensuring 1-8 unique assignments
- [x] **Fixed Trip Event Detection** 🆕
  - Events now detected throughout entire race (not just first 10 seconds)
  - Proper fps calculations for accurate timing
- [x] **Automatic horse count detection** from race start screens
- [x] **Advanced multi-object tracking** with re-identification
- [x] **Optimized frame processing** (0.5-2 fps for speed/accuracy balance)

## Future Enhancements

- [ ] Train CNN model on horse racing position bar data
- [ ] Fine-tune YOLO model specifically for horses  
- [ ] Enhanced jockey silk color identification
- [ ] Turn radius analysis for more accurate distance calculations
- [ ] Stride length estimation and biomechanics
- [ ] GPU acceleration for CNN position detection
- [ ] Real-time processing capability

## Troubleshooting

If Chrome driver issues occur:
```bash
# Install Chrome
sudo apt-get update
sudo apt-get install google-chrome-stable
```

For CUDA/GPU issues:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Example Output

```
Trip Difficulty Score: 72.3/100
Pace Scenario: closer_win
Ground Loss: 3.2%
Events:
  - boxed_in at 45.2s (severity: 0.8)
  - wide_trip at 62.1s (severity: 0.6)
  - steadied at 71.3s (severity: 0.7)
Position Chart: 8 -> 7 -> 5 -> 3 -> 1
```