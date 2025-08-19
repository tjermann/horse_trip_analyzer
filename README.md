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
- Multi-object tracking across frames
- Automated video scraping from TJK website
- Batch processing for multiple races
- Detailed trip difficulty scoring (0-100 scale)
- Visual annotations and reporting

## Installation

```bash
# Clone the repository
cd /home/tyler/Documents/Horse/horse_trip_analyzer

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{videos,processed} logs models
```

## Usage

### Single Race Analysis

```bash
# Download and analyze a race by code
python main.py --race-code 194367 --save-annotated

# Analyze existing video
python main.py --video-path data/videos/race_194367.mp4
```

### Batch Processing

```bash
# Process multiple races
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
│   ├── video_scraper.py     # TJK website video downloading
│   ├── horse_detector.py    # YOLOv8 detection & tracking
│   ├── trip_analyzer.py     # Trip event detection & scoring
│   └── video_processor.py   # Main processing pipeline
├── main.py                   # Single race CLI
├── batch_processor.py        # Batch processing CLI
└── requirements.txt          # Dependencies
```

## Key Metrics Explained

- **Ground Loss**: Extra distance traveled compared to optimal path (as %)
- **Pace Scenario**: Running style classification (wire-to-wire, closer, etc.)
- **Trip Events**: Specific incidents affecting the horse's run
- **Energy Distribution**: Speed consistency across race quarters

## Performance Considerations

- Processing speed: ~2-5 fps on GPU, ~0.5-1 fps on CPU
- Memory usage: ~4GB for 1080p video
- Recommended: CUDA-capable GPU for faster processing

## Future Enhancements

- [ ] Fine-tune YOLO model specifically for horses
- [ ] Jockey silk color identification
- [ ] Turn radius analysis
- [ ] Stride length estimation
- [ ] Integration with race results data
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