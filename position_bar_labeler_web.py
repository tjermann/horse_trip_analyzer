#!/usr/bin/env python3
"""
Web-based Position Bar Labeling Tool
Allows frame-by-frame labeling of position bar for ground truth data collection
"""

from flask import Flask, render_template_string, jsonify, request, send_file
import cv2
import numpy as np
import json
import os
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import argparse
from loguru import logger

app = Flask(__name__)

# Global state
video_path = None
cap = None
total_frames = 0
fps = 0
current_frame_num = 0
labels = []
labels_file = None
position_bar_region = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Position Bar Labeler</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #2a2a2a;
            color: #fff;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        .main-container {
            display: flex;
            gap: 20px;
        }
        .video-section {
            flex: 2;
            background: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
        }
        .controls-section {
            flex: 1;
            background: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
        }
        #frameCanvas {
            width: 100%;
            border: 2px solid #444;
            cursor: crosshair;
            background: #000;
        }
        .frame-info {
            margin: 10px 0;
            padding: 10px;
            background: #333;
            border-radius: 5px;
        }
        .navigation {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: #764ba2;
        }
        button:disabled {
            background: #444;
            cursor: not-allowed;
        }
        .small-btn {
            padding: 5px 10px;
            font-size: 12px;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            background: #333;
            border: 1px solid #555;
            color: white;
            border-radius: 5px;
        }
        .position-input {
            margin: 20px 0;
            padding: 15px;
            background: #333;
            border-radius: 10px;
        }
        .labeled-frames {
            max-height: 200px;
            overflow-y: auto;
            padding: 10px;
            background: #333;
            border-radius: 5px;
            margin-top: 20px;
        }
        .labeled-frame {
            padding: 5px;
            margin: 2px 0;
            background: #444;
            border-radius: 3px;
            cursor: pointer;
        }
        .labeled-frame:hover {
            background: #555;
        }
        .region-display {
            padding: 10px;
            background: #444;
            border-radius: 5px;
            margin: 10px 0;
        }
        #regionCanvas {
            width: 100%;
            max-height: 150px;
            border: 1px solid #666;
            margin-top: 10px;
        }
        .status-message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            text-align: center;
        }
        .success {
            background: #28a745;
        }
        .error {
            background: #dc3545;
        }
        .info {
            background: #17a2b8;
        }
        .shortcuts {
            font-size: 12px;
            color: #888;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏇 Position Bar Labeler</h1>
        <p>Label horse positions frame by frame for ground truth data</p>
    </div>
    
    <div class="main-container">
        <div class="video-section">
            <canvas id="frameCanvas"></canvas>
            
            <div class="frame-info">
                <strong>Frame:</strong> <span id="frameNum">0</span> / <span id="totalFrames">0</span>
                | <strong>Time:</strong> <span id="timestamp">0.00</span>s
                | <strong>Status:</strong> <span id="labelStatus">Not labeled</span>
            </div>
            
            <div class="navigation">
                <button onclick="previousFrame()" title="Previous frame">◀</button>
                <button onclick="skip(-10)" class="small-btn">-10</button>
                <button onclick="skip(-50)" class="small-btn">-50</button>
                <button onclick="jumpToFrame()" title="Jump to frame">Jump</button>
                <input type="number" id="jumpInput" placeholder="Frame #" style="width: 100px;">
                <button onclick="skip(50)" class="small-btn">+50</button>
                <button onclick="skip(10)" class="small-btn">+10</button>
                <button onclick="nextFrame()" title="Next frame">▶</button>
            </div>
            
            <div class="region-display">
                <strong>Position Bar Region:</strong>
                <div id="regionInfo">Click and drag on the main image to select the position bar region</div>
                <canvas id="regionCanvas"></canvas>
            </div>
        </div>
        
        <div class="controls-section">
            <h3>Label Current Frame</h3>
            
            <div class="position-input">
                <label>Horse positions (left to right = 1st to last):</label>
                <input type="text" id="positionsInput" placeholder="e.g., 2,7,5,4,6,3,8,1">
                <small>Enter horse numbers separated by commas</small>
                
                <label style="margin-top: 10px;">Confidence (0-1):</label>
                <input type="number" id="confidenceInput" value="1.0" min="0" max="1" step="0.1">
                
                <label style="margin-top: 10px;">Notes (optional):</label>
                <input type="text" id="notesInput" placeholder="Any observations">
                
                <button onclick="labelFrame()" style="width: 100%; margin-top: 15px;">
                    Label This Frame
                </button>
            </div>
            
            <button onclick="saveLabels()" style="width: 100%; background: #28a745;">
                💾 Save All Labels
            </button>
            
            <button onclick="extractDigits()" style="width: 100%; margin-top: 10px;">
                📦 Extract Digit Samples
            </button>
            
            <div class="labeled-frames">
                <h4>Labeled Frames (<span id="labelCount">0</span>)</h4>
                <div id="labeledList"></div>
            </div>
            
            <div class="shortcuts">
                <h4>Keyboard Shortcuts:</h4>
                <p>← → : Navigate frames</p>
                <p>[ ] : Skip 10 frames</p>
                <p>L : Label current frame</p>
                <p>S : Save labels</p>
                <p>E : Extract digits</p>
            </div>
        </div>
    </div>
    
    <div id="statusMessage"></div>
    
    <script>
        let currentFrame = 0;
        let totalFrames = 0;
        let labels = [];
        let isSelecting = false;
        let selectionStart = null;
        let selectionEnd = null;
        let positionBarRegion = null;
        
        const canvas = document.getElementById('frameCanvas');
        const ctx = canvas.getContext('2d');
        const regionCanvas = document.getElementById('regionCanvas');
        const regionCtx = regionCanvas.getContext('2d');
        
        // Mouse events for region selection
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            isSelecting = true;
            selectionStart = {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
            selectionEnd = {...selectionStart};
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!isSelecting) return;
            
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            selectionEnd = {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
            
            drawFrame();
        });
        
        canvas.addEventListener('mouseup', (e) => {
            if (!isSelecting) return;
            isSelecting = false;
            
            // Calculate region
            const x1 = Math.min(selectionStart.x, selectionEnd.x);
            const y1 = Math.min(selectionStart.y, selectionEnd.y);
            const x2 = Math.max(selectionStart.x, selectionEnd.x);
            const y2 = Math.max(selectionStart.y, selectionEnd.y);
            
            if (Math.abs(x2 - x1) > 20 && Math.abs(y2 - y1) > 10) {
                positionBarRegion = {
                    x: Math.floor(x1),
                    y: Math.floor(y1),
                    width: Math.floor(x2 - x1),
                    height: Math.floor(y2 - y1)
                };
                
                updateRegionDisplay();
                showStatus('Position bar region selected', 'success');
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowLeft': previousFrame(); break;
                case 'ArrowRight': nextFrame(); break;
                case '[': skip(-10); break;
                case ']': skip(10); break;
                case 'l': case 'L': document.getElementById('positionsInput').focus(); break;
                case 's': case 'S': saveLabels(); break;
                case 'e': case 'E': extractDigits(); break;
            }
        });
        
        function loadFrame(frameNum) {
            fetch(`/get_frame/${frameNum}`)
                .then(response => response.json())
                .then(data => {
                    currentFrame = data.frame_num;
                    totalFrames = data.total_frames;
                    
                    // Update display
                    document.getElementById('frameNum').textContent = currentFrame;
                    document.getElementById('totalFrames').textContent = totalFrames;
                    document.getElementById('timestamp').textContent = data.timestamp.toFixed(2);
                    
                    // Draw frame
                    const img = new Image();
                    img.onload = () => {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        drawFrame();
                    };
                    img.src = 'data:image/jpeg;base64,' + data.frame;
                    
                    // Check if labeled
                    const label = labels.find(l => l.frame_num === currentFrame);
                    if (label) {
                        document.getElementById('labelStatus').textContent = 
                            'Labeled: ' + label.positions.join(',');
                        document.getElementById('labelStatus').style.color = '#28a745';
                    } else {
                        document.getElementById('labelStatus').textContent = 'Not labeled';
                        document.getElementById('labelStatus').style.color = '#ffc107';
                    }
                    
                    // Update region if exists
                    if (data.region) {
                        positionBarRegion = data.region;
                        updateRegionDisplay();
                    }
                });
        }
        
        function drawFrame() {
            // Draw selection rectangle if selecting
            if (isSelecting && selectionStart && selectionEnd) {
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.strokeRect(
                    selectionStart.x, selectionStart.y,
                    selectionEnd.x - selectionStart.x,
                    selectionEnd.y - selectionStart.y
                );
            }
            
            // Draw position bar region if set
            if (positionBarRegion) {
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.strokeRect(
                    positionBarRegion.x, positionBarRegion.y,
                    positionBarRegion.width, positionBarRegion.height
                );
                ctx.fillStyle = '#00ff00';
                ctx.font = '14px Arial';
                ctx.fillText('Position Bar', positionBarRegion.x, positionBarRegion.y - 5);
            }
        }
        
        function updateRegionDisplay() {
            if (!positionBarRegion) return;
            
            document.getElementById('regionInfo').textContent = 
                `Region: ${positionBarRegion.x}, ${positionBarRegion.y} - ` +
                `${positionBarRegion.width}x${positionBarRegion.height}px`;
            
            // Show region preview
            fetch(`/get_region/${currentFrame}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(positionBarRegion)
            })
            .then(response => response.json())
            .then(data => {
                if (data.region) {
                    const img = new Image();
                    img.onload = () => {
                        regionCanvas.width = img.width;
                        regionCanvas.height = img.height;
                        regionCtx.drawImage(img, 0, 0);
                    };
                    img.src = 'data:image/jpeg;base64,' + data.region;
                }
            });
        }
        
        function previousFrame() {
            if (currentFrame > 0) loadFrame(currentFrame - 1);
        }
        
        function nextFrame() {
            if (currentFrame < totalFrames - 1) loadFrame(currentFrame + 1);
        }
        
        function skip(frames) {
            const newFrame = Math.max(0, Math.min(currentFrame + frames, totalFrames - 1));
            loadFrame(newFrame);
        }
        
        function jumpToFrame() {
            const frameNum = parseInt(document.getElementById('jumpInput').value);
            if (!isNaN(frameNum)) {
                loadFrame(Math.max(0, Math.min(frameNum, totalFrames - 1)));
            }
        }
        
        function labelFrame() {
            const positionsStr = document.getElementById('positionsInput').value.trim();
            if (!positionsStr) {
                showStatus('Please enter horse positions', 'error');
                return;
            }
            
            const positions = positionsStr.split(',').map(x => parseInt(x.trim()));
            const confidence = parseFloat(document.getElementById('confidenceInput').value);
            const notes = document.getElementById('notesInput').value.trim();
            
            fetch('/label_frame', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    frame_num: currentFrame,
                    positions: positions,
                    confidence: confidence,
                    notes: notes,
                    bar_region: positionBarRegion
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Frame labeled successfully!', 'success');
                    updateLabelsList();
                    
                    // Clear inputs
                    document.getElementById('positionsInput').value = '';
                    document.getElementById('notesInput').value = '';
                    
                    // Auto-advance to next frame
                    setTimeout(() => nextFrame(), 500);
                }
            });
        }
        
        function saveLabels() {
            fetch('/save_labels')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showStatus(`Saved ${data.count} labels!`, 'success');
                    }
                });
        }
        
        function extractDigits() {
            if (!positionBarRegion) {
                showStatus('Please select position bar region first', 'error');
                return;
            }
            
            fetch('/extract_digits', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    frame_num: currentFrame,
                    region: positionBarRegion
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`Extracted ${data.count} digit samples`, 'success');
                } else {
                    showStatus('Frame must be labeled first', 'error');
                }
            });
        }
        
        function updateLabelsList() {
            fetch('/get_labels')
                .then(response => response.json())
                .then(data => {
                    labels = data.labels;
                    document.getElementById('labelCount').textContent = labels.length;
                    
                    const list = document.getElementById('labeledList');
                    list.innerHTML = '';
                    
                    labels.slice(-10).reverse().forEach(label => {
                        const div = document.createElement('div');
                        div.className = 'labeled-frame';
                        div.textContent = `Frame ${label.frame_num}: ${label.positions.join(',')}`;
                        div.onclick = () => loadFrame(label.frame_num);
                        list.appendChild(div);
                    });
                });
        }
        
        function showStatus(message, type) {
            const elem = document.getElementById('statusMessage');
            elem.textContent = message;
            elem.className = 'status-message ' + type;
            elem.style.display = 'block';
            setTimeout(() => elem.style.display = 'none', 3000);
        }
        
        // Initialize
        loadFrame(0);
        updateLabelsList();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_frame/<int:frame_num>')
def get_frame(frame_num):
    global cap, current_frame_num, total_frames, fps, position_bar_region
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        return jsonify({'error': 'Could not read frame'}), 404
    
    current_frame_num = frame_num
    
    # Convert frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'frame_num': frame_num,
        'total_frames': total_frames,
        'timestamp': frame_num / fps,
        'frame': frame_base64,
        'region': position_bar_region
    })

@app.route('/get_region/<int:frame_num>', methods=['POST'])
def get_region(frame_num):
    region = request.json
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        return jsonify({'error': 'Could not read frame'}), 404
    
    # Extract region
    x, y, w, h = region['x'], region['y'], region['width'], region['height']
    region_img = frame[y:y+h, x:x+w]
    
    # Enlarge for better visibility
    scale = 3
    enlarged = cv2.resize(region_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    _, buffer = cv2.imencode('.jpg', enlarged)
    region_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'region': region_base64})

@app.route('/label_frame', methods=['POST'])
def label_frame():
    global labels, position_bar_region
    
    data = request.json
    frame_num = data['frame_num']
    
    # Remove any existing label for this frame
    labels = [l for l in labels if l['frame_num'] != frame_num]
    
    # Add new label
    label = {
        'frame_num': frame_num,
        'timestamp': frame_num / fps,
        'positions': data['positions'],
        'bar_region': data.get('bar_region'),
        'confidence': data.get('confidence', 1.0),
        'notes': data.get('notes', '')
    }
    labels.append(label)
    
    # Update global region if provided
    if data.get('bar_region'):
        position_bar_region = data['bar_region']
    
    return jsonify({'success': True})

@app.route('/get_labels')
def get_labels():
    return jsonify({'labels': labels})

@app.route('/save_labels')
def save_labels():
    global labels_file, position_bar_region
    
    data = {
        'video_path': video_path,
        'total_frames': total_frames,
        'fps': fps,
        'default_region': position_bar_region,
        'labels': labels
    }
    
    with open(labels_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(labels)} labels to {labels_file}")
    
    return jsonify({'success': True, 'count': len(labels)})

@app.route('/extract_digits', methods=['POST'])
def extract_digits():
    data = request.json
    frame_num = data['frame_num']
    region = data['region']
    
    # Find label for this frame
    label = next((l for l in labels if l['frame_num'] == frame_num), None)
    
    if not label:
        return jsonify({'success': False, 'error': 'Frame not labeled'})
    
    # Get frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        return jsonify({'success': False, 'error': 'Could not read frame'})
    
    # Extract position bar region
    x, y, w, h = region['x'], region['y'], region['width'], region['height']
    bar_img = frame[y:y+h, x:x+w]
    
    # Create output directory
    samples_dir = Path("data/cnn_training_samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract individual digits
    positions = label['positions']
    num_positions = len(positions)
    segment_width = w // num_positions
    
    count = 0
    for i, horse_num in enumerate(positions):
        x_start = i * segment_width
        x_end = (i + 1) * segment_width if i < num_positions - 1 else w
        
        digit_img = bar_img[:, x_start:x_end]
        
        # Save to appropriate digit folder
        digit_dir = samples_dir / str(horse_num)
        digit_dir.mkdir(exist_ok=True)
        
        filename = f"frame_{frame_num}_pos_{i+1}.png"
        filepath = digit_dir / filename
        
        cv2.imwrite(str(filepath), digit_img)
        count += 1
    
    logger.info(f"Extracted {count} digit samples from frame {frame_num}")
    
    return jsonify({'success': True, 'count': count})

def main():
    global video_path, cap, total_frames, fps, labels_file, labels, position_bar_region
    
    parser = argparse.ArgumentParser(description="Web-based Position Bar Labeler")
    parser.add_argument("video", help="Path to race video")
    parser.add_argument("--port", type=int, default=5002, help="Port for web server")
    
    args = parser.parse_args()
    
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup labels file
    output_dir = Path("data/position_labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_file = output_dir / f"{Path(video_path).stem}_labels.json"
    
    # Load existing labels if available
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            data = json.load(f)
            labels = data.get('labels', [])
            position_bar_region = data.get('default_region')
            logger.info(f"Loaded {len(labels)} existing labels")
    
    logger.info(f"Starting web labeler for {video_path}")
    logger.info(f"Video: {total_frames} frames at {fps} fps")
    logger.info(f"Open browser at http://localhost:{args.port}")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == "__main__":
    main()