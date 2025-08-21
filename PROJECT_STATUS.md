# Project Status - Horse Race Trip Analyzer

## Last Updated
2025-08-19

## Project Goal
Build a deep learning system to analyze horse racing videos and quantify trip difficulty, identifying horses whose performance was helped or hindered by racing circumstances.

## What's Built

### ✅ Core Infrastructure
- [x] Project structure and dependencies defined
- [x] Video scraping from TJK website
- [x] YOLOv8-based horse detection
- [x] **Automatic horse count detection from race start** 🆕
- [x] **Advanced multi-object tracking with re-identification** 🆕
- [x] **OCR-based horse number recognition** 🆕
- [x] **Optimized frame processing (1 fps default)** 🆕
- [x] Trip event detection system
- [x] Difficulty scoring algorithm
- [x] Batch processing pipeline
- [x] Report generation

### ✅ Trip Analysis Features
- [x] Boxing in detection
- [x] Wide trip identification  
- [x] Bump/interference detection
- [x] Steadying/checking detection
- [x] Ground loss calculation
- [x] Pace scenario classification
- [x] Energy distribution analysis
- [x] Position charting

### ✅ Output Formats
- [x] JSON analysis data
- [x] Human-readable text reports
- [x] Annotated videos (optional)
- [x] CSV aggregate statistics
- [x] Frame-by-frame tracking data

## What's Not Built Yet

### 🔴 Model Training
- [ ] Horse-specific YOLO training
- [ ] Jockey silk color classifier
- [ ] Track geometry model

### 🔴 Advanced Analytics  
- [ ] Stride length estimation
- [ ] Turn vs straightaway analysis
- [ ] Pace pressure quantification
- [ ] Momentum loss calculations

### 🔴 Data Integration
- [ ] Official results scraping
- [ ] Historical performance database
- [ ] Betting odds integration
- [ ] Weather/track condition data

### 🔴 Production Features
- [ ] Real-time streaming analysis
- [ ] Web UI dashboard
- [ ] API endpoints
- [ ] Docker containerization

## Current Limitations

1. **Model Accuracy**: Using generic YOLOv8, not trained on horses specifically
2. **Track Geometry**: Assumes uniform track, doesn't account for turns  
3. **Video Quality**: Dependent on source video resolution
4. **OCR Accuracy**: Horse number detection works ~70-80% of the time
5. **Frame Sampling**: May miss very brief events (sub-second incidents)

## Quick Start Commands

```bash
# Setup environment
conda create -n horse_racing python=3.11 -y
conda activate horse_racing
pip install -r requirements.txt

# Test single race (auto-detects horse count, 1 fps default)
python main.py --race-code 194367 --save-annotated

# Force specific horse count
python main.py --race-code 194367 --num-horses 8

# Speed options
python main.py --race-code 194367 --target-fps 2.0  # Higher quality
python main.py --race-code 194367 --target-fps 0.5  # Ultra-fast

# Run batch analysis (auto-detects for each race)
python batch_processor.py --race-codes 194367 194368 194369
```

## Next Session Priorities

1. **Test New Features**: Validate auto-detection and improved tracking on multiple races
2. **Track Geometry**: Implement turn detection for accurate distance calculations
3. **Results Integration**: Scrape and correlate with official results  
4. **Validation**: Compare analysis with expert trip notes
5. **UI Development**: Build web interface for visualization
6. **OCR Improvement**: Fine-tune number recognition for better accuracy

## Key Files to Review
- `src/race_start_detector.py` - **NEW**: Auto horse count detection
- `src/horse_tracker.py` - **NEW**: Improved tracking with re-ID
- `src/trip_analyzer.py` - Core analysis logic
- `src/horse_detector.py` - Basic detection
- `src/video_processor.py` - Main pipeline (updated)
- `main.py` - Entry point (updated with new options)
- `requirements.txt` - Dependencies (added EasyOCR)

## Performance Metrics
- Processing Speed: ~30-60 seconds per 3-minute race (1 fps mode)
- Detection Confidence: ~0.3-0.8 (needs improvement)
- Memory Usage: ~2GB for 1080p video (reduced due to frame skipping)
- Batch Capacity: 4-8 parallel videos (CPU optimized)

## Known Issues
1. Selenium ChromeDriver may need manual setup
2. GPU memory errors on videos >5 minutes
3. Track ID persistence issues across scene cuts
4. False positives on crowd/barriers

## Success Criteria
- Accurately identify 90%+ of trip troubles
- Correlation between difficulty score and underperformance
- Process full race card (10 races) in <30 minutes
- Generate actionable insights for betting strategy