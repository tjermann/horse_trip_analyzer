#!/usr/bin/env python3
"""
Web-Based Manual Horse Labeling Tool
Interactive tool for creating ground truth horse tracking data via web interface
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
from flask import Flask, render_template_string, request, jsonify, send_file
import base64
import io

class WebHorseLabeler:
    """Web-based horse labeling interface"""
    
    def __init__(self, video_path: str, race_code: str, validation_data: Dict = None):
        self.video_path = video_path
        self.race_code = race_code
        self.validation_data = validation_data or {}
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Labeling data
        self.labels = {}  # frame_number -> {horse_id: bbox}
        
        # Load validation data if available
        self.ground_truth = {}
        if validation_data and race_code in validation_data.get('races', {}):
            race_data = validation_data['races'][race_code]
            self.ground_truth = race_data.get('horse_numbers', {})
            print(f"Ground truth positions loaded:")
            for pos, horse_num in self.ground_truth.items():
                print(f"  Position {pos}: Horse #{horse_num}")
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get specific frame from video"""
        if frame_number < 0 or frame_number >= self.total_frames:
            return None
        
        try:
            # Use threading lock to prevent opencv threading issues
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            return frame if ret else None
        except Exception as e:
            print(f"Error reading frame {frame_number}: {e}")
            return None
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 for web display"""
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    
    def draw_labels_on_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Draw existing labels on frame"""
        display_frame = frame.copy()
        
        # Colors for different horses
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green  
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 255),  # Purple
            (0, 165, 255),  # Orange
        ]
        
        if frame_number in self.labels:
            for horse_id, bbox in self.labels[frame_number].items():
                x, y, w, h = bbox
                color = colors[(horse_id - 1) % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw horse ID with background
                text = f"#{horse_id}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(display_frame, (x, y - text_size[1] - 10), 
                            (x + text_size[0] + 10, y), color, -1)
                cv2.putText(display_frame, text, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return display_frame
    
    def save_labels(self):
        """Save current labels to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"manual_labels_{self.race_code}_{timestamp}.json"
        
        label_data = {
            'race_code': self.race_code,
            'video_path': self.video_path,
            'created_at': datetime.now().isoformat(),
            'total_frames': self.total_frames,
            'fps': self.fps,
            'labels': {},
            'validation_data': self.validation_data.get('races', {}).get(self.race_code, {})
        }
        
        # Convert frame numbers to strings for JSON
        for frame_num, frame_labels in self.labels.items():
            label_data['labels'][str(frame_num)] = frame_labels
        
        with open(filename, 'w') as f:
            json.dump(label_data, f, indent=2)
        
        return filename

def create_flask_app(labeler: WebHorseLabeler):
    """Create Flask app for web interface"""
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE, 
                                    race_code=labeler.race_code,
                                    total_frames=labeler.total_frames,
                                    fps=labeler.fps,
                                    ground_truth=labeler.ground_truth)
    
    @app.route('/get_frame/<int:frame_number>')
    def get_frame(frame_number):
        frame = labeler.get_frame(frame_number)
        if frame is None:
            return jsonify({'error': 'Frame not found'}), 404
        
        # Draw existing labels
        display_frame = labeler.draw_labels_on_frame(frame, frame_number)
        
        # Convert to base64
        img_base64 = labeler.frame_to_base64(display_frame)
        
        return jsonify({
            'image': img_base64,
            'frame_number': frame_number,
            'labels': labeler.labels.get(frame_number, {}),
            'time': f"{frame_number / labeler.fps:.1f}s"
        })
    
    @app.route('/add_label', methods=['POST'])
    def add_label():
        data = request.json
        frame_number = data['frame_number']
        horse_id = int(data['horse_id'])
        bbox = data['bbox']  # [x, y, width, height]
        
        if frame_number not in labeler.labels:
            labeler.labels[frame_number] = {}
        
        labeler.labels[frame_number][horse_id] = bbox
        
        return jsonify({'success': True})
    
    @app.route('/remove_label', methods=['POST'])
    def remove_label():
        data = request.json
        frame_number = data['frame_number']
        horse_id = int(data['horse_id'])
        
        if frame_number in labeler.labels and horse_id in labeler.labels[frame_number]:
            del labeler.labels[frame_number][horse_id]
            
            # Clean up empty frame entries
            if not labeler.labels[frame_number]:
                del labeler.labels[frame_number]
        
        return jsonify({'success': True})
    
    @app.route('/clear_frame', methods=['POST'])
    def clear_frame():
        data = request.json
        frame_number = data['frame_number']
        
        if frame_number in labeler.labels:
            del labeler.labels[frame_number]
        
        return jsonify({'success': True})
    
    @app.route('/save_labels', methods=['POST'])
    def save_labels():
        filename = labeler.save_labels()
        
        # Get summary stats
        total_labels = sum(len(frame_labels) for frame_labels in labeler.labels.values())
        horse_ids = set()
        for frame_labels in labeler.labels.values():
            horse_ids.update(frame_labels.keys())
        
        return jsonify({
            'success': True,
            'filename': filename,
            'stats': {
                'total_labels': total_labels,
                'unique_horses': len(horse_ids),
                'frames_labeled': len(labeler.labels)
            }
        })
    
    @app.route('/get_stats')
    def get_stats():
        total_labels = sum(len(frame_labels) for frame_labels in labeler.labels.values())
        horse_ids = set()
        for frame_labels in labeler.labels.values():
            horse_ids.update(frame_labels.keys())
        
        return jsonify({
            'total_labels': total_labels,
            'unique_horses': len(horse_ids),
            'frames_labeled': len(labeler.labels),
            'current_labels': labeler.labels
        })
    
    return app

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Horse Labeler - Race {{ race_code }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .controls { background: #e8e8e8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .video-container { text-align: center; margin: 20px 0; }
        #raceVideo { 
            max-width: 100%; 
            border: 2px solid #333; 
            cursor: crosshair;
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            pointer-events: auto;
        }
        .button-group { margin: 10px 0; }
        button { padding: 8px 15px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .horse-btn { background: #4CAF50; color: white; }
        .horse-btn.active { background: #2E7D32; }
        .nav-btn { background: #2196F3; color: white; }
        .action-btn { background: #FF9800; color: white; }
        .info { background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .stats { background: #f3e5f5; padding: 10px; border-radius: 5px; }
        .ground-truth { background: #e8f5e8; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
        #frameInfo { font-size: 18px; font-weight: bold; margin: 10px 0; }
        .instructions { background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Horse Labeler - Race {{ race_code }}</h1>
        
        <div class="ground-truth">
            <h3>🎯 Ground Truth (Final Results):</h3>
            {% for pos, horse_num in ground_truth.items() %}
            <strong>{{ pos }}{{ "st" if pos == "1" else "nd" if pos == "2" else "rd" if pos == "3" else "th" }}: Horse #{{ horse_num }}</strong>{{ " | " if not loop.last }}
            {% endfor %}
        </div>
        
        <div class="instructions">
            <h3>📋 Instructions:</h3>
            <ol>
                <li><strong>Select Horse ID</strong>: Click buttons 1-9 below to choose which horse to label</li>
                <li><strong>Draw Box</strong>: Click and drag on the video to draw a bounding box around the selected horse</li>
                <li><strong>Navigate</strong>: Use Previous/Next buttons or the frame slider</li>
                <li><strong>Key Frames</strong>: Focus on frames ~25, ~63, ~88, ~113, ~138 (roughly every 15-20 seconds)</li>
                <li><strong>Save Often</strong>: Click Save Labels frequently to preserve your work</li>
            </ol>
        </div>
        
        <div class="controls">
            <div class="button-group">
                <strong>Select Horse ID:</strong>
                {% if ground_truth %}
                    {% for pos, horse_num in ground_truth.items() %}
                    <button class="horse-btn" onclick="selectHorse({{ horse_num }})">Horse #{{ horse_num }}</button>
                    {% endfor %}
                {% else %}
                    <!-- Default buttons when no ground truth available -->
                    <button class="horse-btn" onclick="selectHorse(1)">Horse #1</button>
                    <button class="horse-btn" onclick="selectHorse(2)">Horse #2</button>
                    <button class="horse-btn" onclick="selectHorse(3)">Horse #3</button>
                    <button class="horse-btn" onclick="selectHorse(4)">Horse #4</button>
                    <button class="horse-btn" onclick="selectHorse(5)">Horse #5</button>
                    <button class="horse-btn" onclick="selectHorse(6)">Horse #6</button>
                    <button class="horse-btn" onclick="selectHorse(7)">Horse #7</button>
                    <button class="horse-btn" onclick="selectHorse(8)">Horse #8</button>
                {% endif %}
            </div>
            
            <div class="button-group">
                <strong>Navigation:</strong>
                <button class="nav-btn" onclick="changeFrame(-1)">⬅ Previous</button>
                <button class="nav-btn" onclick="changeFrame(1)">Next ➡</button>
                <button class="nav-btn" onclick="goToFrame()">Go to Frame</button>
                <input type="number" id="frameInput" min="0" max="{{ total_frames - 1 }}" style="width: 80px;">
            </div>
            
            <div class="button-group">
                <strong>Actions:</strong>
                <button class="action-btn" onclick="saveLabels()">💾 Save Labels</button>
                <button class="action-btn" onclick="clearFrame()">🗑 Clear Frame</button>
                <button class="action-btn" onclick="updateStats()">📊 Update Stats</button>
            </div>
        </div>
        
        <div class="info">
            <div id="frameInfo">Frame: 0 / {{ total_frames }} | Time: 0.0s</div>
            <div>Total Video Duration: {{ "%.1f"|format((total_frames / fps)) }}s | FPS: {{ fps }}</div>
            <div>Currently Selected: <span id="selectedHorse">None</span></div>
        </div>
        
        <div class="video-container">
            <img id="raceVideo" src="" alt="Race video frame" draggable="false">
        </div>
        
        <div>
            <strong>Frame Slider:</strong>
            <input type="range" id="frameSlider" min="0" max="{{ total_frames - 1 }}" value="0" 
                   style="width: 100%;" oninput="debouncedLoadFrame(this.value)">
        </div>
        
        <div class="stats" id="stats">
            <strong>📈 Labeling Stats:</strong> Loading...
        </div>
    </div>

    <script>
        let currentFrame = 0;
        let selectedHorse = null;
        let isDrawing = false;
        let startX, startY;
        let loadFrameTimeout;
        
        // Load first frame on page load
        window.onload = function() {
            loadFrame(0);
            updateStats();
        };
        
        function selectHorse(horseId) {
            selectedHorse = horseId;
            document.getElementById('selectedHorse').textContent = `Horse #${horseId}`;
            
            // Update button styling
            document.querySelectorAll('.horse-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        function loadFrame(frameNumber) {
            currentFrame = parseInt(frameNumber);
            document.getElementById('frameSlider').value = currentFrame;
            document.getElementById('frameInput').value = currentFrame;
            
            fetch(`/get_frame/${currentFrame}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                        return;
                    }
                    
                    document.getElementById('raceVideo').src = data.image;
                    document.getElementById('frameInfo').textContent = 
                        `Frame: ${data.frame_number} / {{ total_frames }} | Time: ${data.time}`;
                })
                .catch(error => {
                    console.error('Error loading frame:', error);
                });
        }
        
        // Debounced version for slider to prevent crashes
        function debouncedLoadFrame(frameNumber) {
            clearTimeout(loadFrameTimeout);
            loadFrameTimeout = setTimeout(() => {
                loadFrame(frameNumber);
            }, 200); // Wait 200ms after user stops moving slider
        }
        
        function changeFrame(delta) {
            let newFrame = currentFrame + delta;
            if (newFrame >= 0 && newFrame < {{ total_frames }}) {
                loadFrame(newFrame);
            }
        }
        
        function goToFrame() {
            let frame = parseInt(document.getElementById('frameInput').value);
            if (frame >= 0 && frame < {{ total_frames }}) {
                loadFrame(frame);
            }
        }
        
        // Mouse events for drawing bounding boxes
        document.getElementById('raceVideo').addEventListener('mousedown', function(e) {
            e.preventDefault(); // Prevent default drag behavior
            
            if (!selectedHorse) {
                alert('Please select a horse ID first (click the horse buttons above)!');
                return;
            }
            
            console.log('Mouse down - starting to draw box for Horse #' + selectedHorse);
            isDrawing = true;
            let rect = this.getBoundingClientRect();
            
            // Calculate coordinates relative to actual image size
            let scaleX = this.naturalWidth / rect.width;
            let scaleY = this.naturalHeight / rect.height;
            
            startX = (e.clientX - rect.left) * scaleX;
            startY = (e.clientY - rect.top) * scaleY;
            
            console.log('Start coordinates:', startX, startY, 'Scale:', scaleX, scaleY);
        });
        
        document.getElementById('raceVideo').addEventListener('mouseup', function(e) {
            e.preventDefault(); // Prevent default drag behavior
            
            if (!isDrawing || !selectedHorse) {
                console.log('Mouse up but not drawing or no horse selected');
                return;
            }
            
            console.log('Mouse up - finishing box draw');
            isDrawing = false;
            let rect = this.getBoundingClientRect();
            
            // Calculate coordinates relative to actual image size
            let scaleX = this.naturalWidth / rect.width;
            let scaleY = this.naturalHeight / rect.height;
            
            let endX = (e.clientX - rect.left) * scaleX;
            let endY = (e.clientY - rect.top) * scaleY;
            
            // Calculate bounding box
            let x = Math.min(startX, endX);
            let y = Math.min(startY, endY);
            let width = Math.abs(endX - startX);
            let height = Math.abs(endY - startY);
            
            console.log('Bounding box:', x, y, width, height);
            
            // Only add if box is large enough (at least 20x20 pixels)
            if (width > 20 && height > 20) {
                console.log('Adding label for Horse #' + selectedHorse);
                addLabel(currentFrame, selectedHorse, [Math.round(x), Math.round(y), Math.round(width), Math.round(height)]);
            } else {
                console.log('Box too small:', width, height);
                alert('Bounding box too small. Please draw a larger box around the horse.');
            }
        });
        
        // Prevent context menu on right click
        document.getElementById('raceVideo').addEventListener('contextmenu', function(e) {
            e.preventDefault();
        });
        
        // Prevent all drag-related events
        document.getElementById('raceVideo').addEventListener('dragstart', function(e) {
            e.preventDefault();
        });
        
        document.getElementById('raceVideo').addEventListener('drag', function(e) {
            e.preventDefault();
        });
        
        document.getElementById('raceVideo').addEventListener('dragend', function(e) {
            e.preventDefault();
        });
        
        // Additional event to prevent selection
        document.getElementById('raceVideo').addEventListener('selectstart', function(e) {
            e.preventDefault();
        });
        
        function addLabel(frame, horseId, bbox) {
            fetch('/add_label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    frame_number: frame,
                    horse_id: horseId,
                    bbox: bbox
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadFrame(currentFrame); // Reload to show the new label
                    updateStats();
                    console.log(`Added label for Horse #${horseId} at frame ${frame}`);
                }
            })
            .catch(error => console.error('Error adding label:', error));
        }
        
        function clearFrame() {
            if (confirm('Clear all labels for this frame?')) {
                fetch('/clear_frame', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        frame_number: currentFrame
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadFrame(currentFrame); // Reload to show cleared frame
                        updateStats();
                        console.log(`Cleared all labels for frame ${currentFrame}`);
                    }
                })
                .catch(error => console.error('Error clearing frame:', error));
            }
        }
        
        function saveLabels() {
            fetch('/save_labels', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`Labels saved to: ${data.filename}\\n\\nStats:\\n- Total labels: ${data.stats.total_labels}\\n- Unique horses: ${data.stats.unique_horses}\\n- Frames labeled: ${data.stats.frames_labeled}`);
                    updateStats();
                }
            })
            .catch(error => console.error('Error saving labels:', error));
        }
        
        function updateStats() {
            fetch('/get_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('stats').innerHTML = 
                        `<strong>📈 Labeling Stats:</strong> ${data.total_labels} total labels | ${data.unique_horses} horses | ${data.frames_labeled} frames`;
                })
                .catch(error => console.error('Error updating stats:', error));
        }
        
        // Get valid horse IDs from ground truth or default to 1-8
        let validHorseIds = {% if ground_truth %}{{ ground_truth.values() | list }}{% else %}[1, 2, 3, 4, 5, 6, 7, 8]{% endif %};
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            let keyNum = parseInt(e.key);
            if (e.key >= '1' && e.key <= '8' && validHorseIds.includes(keyNum)) {
                selectHorse(keyNum);
            } else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
                changeFrame(-1);
            } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
                changeFrame(1);
            } else if (e.key === 's' || e.key === 'S') {
                e.preventDefault();
                saveLabels();
            }
        });
    </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Web-Based Manual Horse Labeling Tool")
    parser.add_argument("--video", required=True, help="Path to race video")
    parser.add_argument("--race-code", required=True, help="Race code")
    parser.add_argument("--validation-data", help="Path to validation dataset JSON")
    parser.add_argument("--port", type=int, default=5001, help="Port for web server")
    
    args = parser.parse_args()
    
    # Load validation data if provided
    validation_data = {}
    if args.validation_data and Path(args.validation_data).exists():
        with open(args.validation_data) as f:
            validation_data = json.load(f)
    
    # Create labeler and Flask app
    labeler = WebHorseLabeler(args.video, args.race_code, validation_data)
    app = create_flask_app(labeler)
    
    print(f"\n🚀 Starting Web Horse Labeler for Race {args.race_code}")
    print(f"📹 Video: {args.video}")
    print(f"🌐 Open your browser to: http://localhost:{args.port}")
    print(f"📊 Total frames: {labeler.total_frames}, FPS: {labeler.fps}")
    print("\n🎯 Ground Truth Final Positions:")
    for pos, horse_num in labeler.ground_truth.items():
        print(f"   {pos}{'st' if pos == '1' else 'nd' if pos == '2' else 'rd' if pos == '3' else 'th'}: Horse #{horse_num}")
    print(f"\n💡 Focus on key frames: ~25, ~63, ~88, ~113, ~138")
    print(f"🎮 Use keyboard shortcuts: 1-5 (select horse), A/D (prev/next), S (save)")
    
    app.run(debug=True, host='0.0.0.0', port=args.port)

if __name__ == "__main__":
    main()