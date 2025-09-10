#!/usr/bin/env python3
"""
Web-based labeling interface for position bar digits.
Works without OpenCV GUI support.
"""

import os
import json
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
from loguru import logger
import numpy as np

app = Flask(__name__)

# Global state
current_index = 0
annotations = []
annotation_file = None
unlabeled = []


def load_annotations(data_dir="data/position_digits"):
    """Load existing annotations"""
    global annotations, annotation_file, unlabeled
    
    annotation_file = Path(data_dir) / "annotations.json"
    
    if annotation_file.exists():
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = []
    
    # Filter unlabeled samples
    unlabeled = [a for a in annotations if a['label'] is None]
    
    logger.info(f"Loaded {len(annotations)} total annotations")
    logger.info(f"Found {len(unlabeled)} unlabeled samples")
    
    return len(unlabeled)


def save_annotations():
    """Save annotations to file"""
    if annotation_file:
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        logger.info(f"Saved annotations to {annotation_file}")


def get_image_base64(image_path):
    """Convert image to base64 for web display"""
    if not os.path.exists(image_path):
        return None
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize for better visibility
    img_large = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    # Encode to base64
    _, buffer = cv2.imencode('.png', img_large)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64


@app.route('/')
def index():
    """Main labeling interface"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Position Digit Labeling</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .image-container {
            text-align: center;
            margin: 30px 0;
        }
        #digit-image {
            border: 3px solid #ddd;
            background: #fff;
            padding: 10px;
            image-rendering: pixelated;
            image-rendering: -moz-crisp-edges;
            image-rendering: crisp-edges;
        }
        .button-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin: 20px 0;
        }
        .button-grid button {
            padding: 20px;
            font-size: 24px;
            cursor: pointer;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            transition: all 0.3s;
        }
        .button-grid button:hover {
            background: #45a049;
            transform: scale(1.05);
        }
        .control-buttons {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        .control-buttons button {
            padding: 15px 30px;
            font-size: 18px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            transition: all 0.3s;
        }
        #skip-btn {
            background: #ff9800;
            color: white;
        }
        #skip-btn:hover {
            background: #e68900;
        }
        #not-digit-btn {
            background: #f44336;
            color: white;
        }
        #not-digit-btn:hover {
            background: #da190b;
        }
        #progress {
            text-align: center;
            font-size: 18px;
            margin: 20px 0;
            color: #666;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 20px 0;
            font-size: 12px;
        }
        .stat-item {
            text-align: center;
            padding: 5px;
            background: #f5f5f5;
            border-radius: 3px;
        }
        .instructions {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .shortcuts {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Position Bar Digit Labeling</h1>
        
        <div id="progress">Loading...</div>
        
        <div class="image-container">
            <img id="digit-image" src="" alt="Digit sample" width="256" height="256">
        </div>
        
        <div class="button-grid">
            <button onclick="labelDigit(1)">1</button>
            <button onclick="labelDigit(2)">2</button>
            <button onclick="labelDigit(3)">3</button>
            <button onclick="labelDigit(4)">4</button>
            <button onclick="labelDigit(5)">5</button>
            <button onclick="labelDigit(6)">6</button>
            <button onclick="labelDigit(7)">7</button>
            <button onclick="labelDigit(8)">8</button>
            <button onclick="labelDigit(9)">9</button>
            <button onclick="labelDigit(10)">10</button>
            <button onclick="labelDigit(11)">11</button>
            <button onclick="labelDigit(12)">12</button>
            <button onclick="labelDigit(13)">13</button>
            <button onclick="labelDigit(14)">14</button>
            <button onclick="labelDigit(15)">15</button>
            <button onclick="labelDigit(16)">16</button>
            <button onclick="labelDigit(17)">17</button>
            <button onclick="labelDigit(18)">18</button>
            <button onclick="labelDigit(19)">19</button>
            <button onclick="labelDigit(20)">20</button>
        </div>
        
        <div class="control-buttons">
            <button id="skip-btn" onclick="skipSample()">Skip (S)</button>
            <button id="not-digit-btn" onclick="labelNotDigit()">Not a Digit (N)</button>
        </div>
        
        <div class="instructions">
            <h3>Instructions:</h3>
            <div class="shortcuts">
                <div>• Click the number you see in the image</div>
                <div>• Press 1-9 keys for digits 1-9</div>
                <div>• Click "Skip" or press S to skip unclear samples</div>
                <div>• Click "Not a Digit" or press N for non-digit images</div>
            </div>
        </div>
        
        <div id="stats" class="stats"></div>
    </div>
    
    <script>
        let currentSample = null;
        let totalSamples = 0;
        let currentIndex = 0;
        let labelCounts = {};
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            if (event.key >= '1' && event.key <= '9') {
                labelDigit(parseInt(event.key));
            } else if (event.key.toLowerCase() === 's') {
                skipSample();
            } else if (event.key.toLowerCase() === 'n') {
                labelNotDigit();
            }
        });
        
        function loadNextSample() {
            fetch('/get_sample')
                .then(response => response.json())
                .then(data => {
                    if (data.done) {
                        document.getElementById('progress').innerHTML = 
                            '<h2 style="color: green;">All samples labeled! 🎉</h2>';
                        document.getElementById('digit-image').style.display = 'none';
                        updateStats();
                    } else {
                        currentSample = data;
                        currentIndex = data.index;
                        totalSamples = data.total;
                        document.getElementById('progress').innerHTML = 
                            `Sample ${currentIndex + 1} of ${totalSamples}`;
                        document.getElementById('digit-image').src = 
                            'data:image/png;base64,' + data.image;
                        updateStats();
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }
        
        function labelDigit(digit) {
            if (!currentSample) return;
            
            fetch('/label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    sample_id: currentSample.id,
                    label: digit
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadNextSample();
                }
            });
        }
        
        function skipSample() {
            loadNextSample();
        }
        
        function labelNotDigit() {
            if (!currentSample) return;
            
            fetch('/label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    sample_id: currentSample.id,
                    label: -1
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadNextSample();
                }
            });
        }
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    labelCounts = data.label_counts;
                    let statsHtml = '';
                    for (let i = 1; i <= 20; i++) {
                        let count = labelCounts[i] || 0;
                        statsHtml += `<div class="stat-item">Digit ${i}: ${count}</div>`;
                    }
                    document.getElementById('stats').innerHTML = statsHtml;
                });
        }
        
        // Load first sample on page load
        loadNextSample();
    </script>
</body>
</html>
    '''


@app.route('/get_sample')
def get_sample():
    """Get next unlabeled sample"""
    global current_index
    
    if current_index >= len(unlabeled):
        return jsonify({'done': True})
    
    sample = unlabeled[current_index]
    img_base64 = get_image_base64(sample['path'])
    
    if img_base64:
        return jsonify({
            'id': sample['id'],
            'image': img_base64,
            'index': current_index,
            'total': len(unlabeled),
            'done': False
        })
    else:
        # Skip this sample if image not found
        current_index += 1
        return get_sample()


@app.route('/label', methods=['POST'])
def label_sample():
    """Label a sample"""
    global current_index
    
    data = request.json
    sample_id = data.get('sample_id')
    label = data.get('label')
    
    # Find and update the sample
    for ann in annotations:
        if ann['id'] == sample_id:
            ann['label'] = label
            
            # Move file to appropriate directory if it's a valid digit
            if label > 0:
                old_path = Path(ann['path'])
                new_dir = old_path.parent.parent / str(label)
                new_dir.mkdir(exist_ok=True)
                new_path = new_dir / old_path.name
                
                if old_path.exists():
                    old_path.rename(new_path)
                    ann['path'] = str(new_path)
            
            break
    
    # Save annotations
    save_annotations()
    
    # Move to next sample
    current_index += 1
    
    return jsonify({'success': True})


@app.route('/stats')
def get_stats():
    """Get labeling statistics"""
    label_counts = {}
    
    for ann in annotations:
        if ann['label'] is not None and ann['label'] > 0:
            label = ann['label']
            label_counts[label] = label_counts.get(label, 0) + 1
    
    return jsonify({
        'total_labeled': len([a for a in annotations if a['label'] is not None]),
        'total_unlabeled': len([a for a in annotations if a['label'] is None]),
        'label_counts': label_counts
    })


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Web-based digit labeling interface")
    parser.add_argument('--data-dir', type=str, default='data/position_digits',
                       help='Directory containing annotations')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run web server on')
    
    args = parser.parse_args()
    
    # Load annotations
    num_unlabeled = load_annotations(args.data_dir)
    
    if num_unlabeled == 0:
        logger.warning("No unlabeled samples found!")
        logger.info("Run: python collect_training_data.py --video <video_file>")
        return
    
    logger.info(f"Starting web server on http://localhost:{args.port}")
    logger.info(f"Open this URL in your browser to start labeling")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()