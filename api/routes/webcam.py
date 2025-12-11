"""
============================================================
WEBCAM ROUTES - Webcam Interface & Video Stream
============================================================
"""

from flask import Blueprint, Response, render_template_string, jsonify
import cv2
import threading
import time
import torch
from PIL import Image

from ..model import face_model
from ..config import CLASS_NAMES, CONFIDENCE_THRESHOLD, ANTISPOOF_ENABLED, ANTISPOOF_THRESHOLD
from ..antispoof import anti_spoof

webcam_bp = Blueprint('webcam', __name__)

# Camera singleton
camera = None
camera_lock = threading.Lock()


def get_camera():
    """Get camera instance"""
    global camera
    if camera is None or not camera.isOpened():
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(0.5)  # Wait for camera to initialize
    return camera


def release_camera():
    """Release camera"""
    global camera
    if camera is not None:
        camera.release()
        camera = None


def generate_frames():
    """Generate frames with face detection overlay - optimized for low latency"""
    cam = get_camera()
    
    if not cam.isOpened():
        print("❌ Camera not available")
        return
    
    print("✅ Stream started")
    
    frame_count = 0
    detect_interval = 3  # Deteksi setiap 3 frame (balance speed vs accuracy)
    cached_detections = []  # Cache: [(x1,y1,x2,y2,label,color), ...]
    
    try:
        while True:
            success, frame = cam.read()
            
            if not success or frame is None:
                time.sleep(0.05)
                continue
            
            frame_count += 1
            
            # Deteksi hanya setiap N frame untuk kurangi latency
            if frame_count % detect_interval == 0:
                try:
                    # Gunakan detect_faces langsung (lebih cepat dari process_frame)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    faces = face_model.detect_faces(pil_img)
                    
                    cached_detections = []
                    for x1, y1, x2, y2, conf in faces:
                        # Quick classify
                        face_crop = pil_img.crop((x1, y1, x2, y2))
                        
                        # Anti-spoofing check
                        is_real = True
                        if ANTISPOOF_ENABLED:
                            is_real, spoof_score, spoof_reason = anti_spoof.check_liveness(face_crop)
                            is_real = spoof_score >= ANTISPOOF_THRESHOLD
                        
                        if not is_real:
                            # Spoof detected
                            color = (0, 0, 255)  # Red
                            label = f"SPOOF! {spoof_score*100:.0f}%"
                            cached_detections.append((x1, y1, x2, y2, label, color))
                            continue
                        
                        # Real face - classify
                        face_tensor = face_model.transform(face_crop).unsqueeze(0).to(face_model.device)
                        
                        with torch.no_grad():
                            outputs = face_model.model(face_tensor).logits
                            probs = torch.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probs, 1)
                        
                        conf_val = confidence.item()
                        
                        if conf_val >= CONFIDENCE_THRESHOLD:
                            name = CLASS_NAMES[predicted.item()]
                            full_label, _, authorized = face_model.get_full_label(name)
                        else:
                            full_label = "Unknown (Guest)"
                            authorized = False
                        
                        color = (0, 255, 0) if authorized else (0, 0, 255)
                        label = f"{full_label} {conf_val*100:.1f}%"
                        cached_detections.append((x1, y1, x2, y2, label, color))
                        
                except Exception as e:
                    print(f"⚠️ Detect error: {e}")
            
            # Gambar cached detections ke frame
            for x1, y1, x2, y2, label, color in cached_detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1 + len(label)*10, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Encode dengan kualitas lebih rendah
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    except GeneratorExit:
        print("✅ Stream stopped")


@webcam_bp.route('/api/stream')
def video_stream():
    """Video stream with face detection overlay"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@webcam_bp.route('/api/webcam')
def webcam_page():
    """Webcam interface page with Privacy Mask & Ignore Zone"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition - Webcam</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                color: white;
                padding: 20px;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { text-align: center; margin-bottom: 20px; font-size: 1.8em; }
            .main-content { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
            @media (max-width: 1000px) { .main-content { grid-template-columns: 1fr; } }
            
            .video-section {
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
                padding: 20px;
            }
            .video-container {
                position: relative;
                width: 100%;
                background: #000;
                border-radius: 10px;
                overflow: hidden;
            }
            #video, #streamImg { width: 100%; display: block; }
            #overlay {
                position: absolute;
                top: 0; left: 0;
                width: 100%; height: 100%;
                cursor: crosshair;
            }
            #streamImg { display: none; }
            
            .toolbar {
                display: flex;
                gap: 8px;
                margin: 15px 0;
                flex-wrap: wrap;
                justify-content: center;
            }
            button {
                padding: 10px 20px;
                font-size: 14px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s;
                font-weight: 600;
            }
            button:hover { transform: scale(1.03); }
            .btn-green { background: #00b894; color: white; }
            .btn-red { background: #d63031; color: white; }
            .btn-blue { background: #0984e3; color: white; }
            .btn-orange { background: #e17055; color: white; }
            .btn-purple { background: #6c5ce7; color: white; }
            .btn-gray { background: #636e72; color: white; }
            .btn-active { box-shadow: 0 0 0 3px rgba(255,255,255,0.5); }
            
            .zone-tools {
                background: rgba(0,0,0,0.3);
                border-radius: 10px;
                padding: 15px;
                margin-top: 15px;
            }
            .zone-tools h3 { font-size: 1em; margin-bottom: 10px; color: #dfe6e9; }
            .zone-list { max-height: 150px; overflow-y: auto; }
            .zone-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 12px;
                background: rgba(255,255,255,0.1);
                border-radius: 6px;
                margin-bottom: 5px;
                font-size: 0.85em;
            }
            .zone-item.privacy { border-left: 3px solid #e17055; }
            .zone-item.ignore { border-left: 3px solid #6c5ce7; }
            .zone-item.detect { border-left: 3px solid #00b894; }
            .zone-delete { 
                background: none; 
                border: none; 
                color: #ff7675; 
                cursor: pointer;
                padding: 2px 8px;
            }
            
            .results-section {
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
                padding: 20px;
            }
            h2 { margin-bottom: 15px; font-size: 1.2em; border-bottom: 2px solid rgba(255,255,255,0.2); padding-bottom: 10px; }
            #results { max-height: 300px; overflow-y: auto; }
            .result-card {
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 12px;
                margin-bottom: 8px;
                border-left: 4px solid;
            }
            .result-card.authorized { border-color: #00b894; }
            .result-card.unauthorized { border-color: #d63031; }
            .result-name { font-size: 1.1em; font-weight: bold; }
            .result-info { font-size: 0.8em; color: #b2bec3; margin-top: 5px; }
            .status-badge {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 15px;
                font-size: 0.75em;
                margin-top: 5px;
            }
            .status-badge.authorized { background: #00b894; }
            .status-badge.unauthorized { background: #d63031; }
            
            .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 15px; }
            .stat-card { background: rgba(255,255,255,0.1); border-radius: 8px; padding: 12px; text-align: center; }
            .stat-value { font-size: 1.5em; font-weight: bold; }
            .stat-label { font-size: 0.7em; color: #b2bec3; }
            
            .no-results { text-align: center; color: #b2bec3; padding: 30px; }
            .mode-indicator {
                text-align: center;
                padding: 8px;
                border-radius: 8px;
                margin-bottom: 10px;
                font-weight: bold;
            }
            .mode-indicator.capture { background: rgba(9, 132, 227, 0.3); }
            .mode-indicator.stream { background: rgba(0, 184, 148, 0.3); }
            .mode-indicator.drawing { background: rgba(225, 112, 85, 0.3); }
            
            .log-section { margin-top: 20px; }
            #logList { max-height: 200px; overflow-y: auto; font-size: 0.8em; }
            .log-item { padding: 5px 10px; border-bottom: 1px solid rgba(255,255,255,0.1); }
            .log-item.auth { color: #00b894; }
            .log-item.unauth { color: #d63031; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Face Recognition System</h1>
            
            <div class="main-content">
                <div class="video-section">
                    <div id="modeIndicator" class="mode-indicator capture">📸 CAPTURE MODE - Click to detect | <span id="fpsCounter">FPS: --</span></div>
                    
                    <div class="video-container">
                        <video id="video" autoplay playsinline></video>
                        <img id="streamImg" src="">
                        <canvas id="overlay"></canvas>
                    </div>
                    
                    <div class="toolbar">
                        <button class="btn-green" onclick="startCamera()">▶ Start</button>
                        <button class="btn-red" onclick="stopAll()">⏹ Stop</button>
                        <button class="btn-blue" onclick="captureAndDetect()">📸 Detect</button>
                        <button class="btn-blue" onclick="toggleAutoDetect()" id="autoBtn">🔄 Auto: OFF</button>
                        <span style="width: 20px;"></span>
                        <button class="btn-orange" onclick="setDrawMode('privacy')" id="privacyBtn">🔒 Privacy Mask</button>
                        <button class="btn-purple" onclick="setDrawMode('ignore')" id="ignoreBtn">🚫 Ignore Zone</button>
                        <button class="btn-green" onclick="setDrawMode('detect')" id="detectBtn" style="background:#00b894;">🎯 Detect Zone</button>
                        <button class="btn-gray" onclick="setDrawMode(null)">✋ Select</button>
                        <button class="btn-gray" onclick="clearAllZones()">🗑️ Clear All</button>
                        <span style="width: 20px;"></span>
                        <button class="btn-blue" onclick="toggleDwellTime()" id="dwellBtn">⏱️ Dwell: 3s ON</button>
                    </div>
                    
                    <div class="zone-tools">
                        <h3>📍 Zones (<span id="zoneCount">0</span>)</h3>
                        <div class="zone-list" id="zoneList">
                            <div class="no-results">No zones defined. Draw on video to add.</div>
                        </div>
                    </div>
                </div>
                
                <div class="results-section">
                    <h2>📋 Detection Results</h2>
                    <div id="results">
                        <div class="no-results">
                            <p>No faces detected yet.</p>
                            <p>Click "Detect" or enable "Auto".</p>
                        </div>
                    </div>
                    
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-value" id="totalDetections">0</div>
                            <div class="stat-label">Total</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="authorizedCount" style="color: #00b894;">0</div>
                            <div class="stat-label">Authorized</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="unauthorizedCount" style="color: #d63031;">0</div>
                            <div class="stat-label">Unauthorized</div>
                        </div>
                    </div>
                    
                    <div class="log-section">
                        <h2>📜 Access Log</h2>
                        <div id="logList"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Elements
            const video = document.getElementById('video');
            const streamImg = document.getElementById('streamImg');
            const overlay = document.getElementById('overlay');
            const ctx = overlay.getContext('2d');
            
            // State
            let stream = null;
            let autoDetect = false;
            let autoInterval = null;
            let drawMode = null; // 'privacy', 'ignore', or null
            let zones = []; // {type, x, y, w, h}
            let isDrawing = false;
            let startX, startY;
            let stats = { total: 0, authorized: 0, unauthorized: 0 };
            let logs = [];
            let lastDetectedFaces = new Set(); // Track last detected faces to avoid spam
            let lastDetectionTime = {}; // Track when each face was last logged
            
            // Dwell time tracking - untuk detect zone
            const DWELL_TIME_MS = 3000; // 3 detik harus diam di zona
            let faceTracker = {}; // { faceId: { firstSeen, lastSeen, bbox, classified } }
            let dwellTimeEnabled = true; // Enable/disable dwell time feature
            
            // FPS counter
            let frameCount = 0;
            let lastFpsTime = Date.now();
            let currentFps = 0;
            let videoFps = 0;
            let isProcessing = false; // Prevent overlapping requests
            let lastResults = []; // Cache last detection results
            
            // Initialize
            function init() {
                resizeOverlay();
                window.addEventListener('resize', resizeOverlay);
                setupDrawing();
                startCamera();
            }
            
            function resizeOverlay() {
                const rect = video.getBoundingClientRect();
                overlay.width = rect.width;
                overlay.height = rect.height;
                drawZones();
            }
            
            // Camera
            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 1280, height: 720, facingMode: 'user' } 
                    });
                    video.srcObject = stream;
                    video.style.display = 'block';
                    streamImg.style.display = 'none';
                    updateMode('capture');
                    video.onloadedmetadata = () => {
                        resizeOverlay();
                        requestAnimationFrame(renderLoop);
                    };
                } catch (err) {
                    alert('Camera error: ' + err.message + '\\n\\nTry using localhost or HTTPS.');
                }
            }
            
            function stopAll() {
                if (stream) {
                    stream.getTracks().forEach(t => t.stop());
                    video.srcObject = null;
                }
                if (autoInterval) {
                    clearInterval(autoInterval);
                    autoInterval = null;
                    autoDetect = false;
                    document.getElementById('autoBtn').textContent = '🔄 Auto: OFF';
                }
                streamImg.src = '';
                streamImg.style.display = 'none';
                updateMode('capture');
            }
            
            function updateMode(mode) {
                const indicator = document.getElementById('modeIndicator');
                indicator.className = 'mode-indicator ' + mode;
                const fpsText = '<span id="fpsCounter">Video: ' + videoFps + ' | Detect: ' + currentFps + ' FPS</span>';
                if (mode === 'capture') indicator.innerHTML = '📸 CAPTURE MODE - Click Detect | ' + fpsText;
                else if (mode === 'stream') indicator.innerHTML = '🎬 STREAM MODE - Live detection | ' + fpsText;
                else if (mode === 'drawing') indicator.innerHTML = '✏️ DRAWING MODE - Draw zones on video | ' + fpsText;
            }
            
            // Auto detect
            function toggleAutoDetect() {
                autoDetect = !autoDetect;
                document.getElementById('autoBtn').textContent = autoDetect ? '🔄 Auto: ON' : '🔄 Auto: OFF';
                
                if (autoDetect) {
                    autoInterval = setInterval(captureAndDetect, 100); // Every 100ms (~10 FPS)
                } else if (autoInterval) {
                    clearInterval(autoInterval);
                    autoInterval = null;
                }
            }
            
            // Toggle dwell time feature
            function toggleDwellTime() {
                dwellTimeEnabled = !dwellTimeEnabled;
                document.getElementById('dwellBtn').textContent = dwellTimeEnabled ? '⏱️ Dwell: 3s ON' : '⏱️ Dwell: OFF';
                document.getElementById('dwellBtn').className = dwellTimeEnabled ? 'btn-blue' : 'btn-gray';
                faceTracker = {}; // Reset tracker
            }
            
            // Calculate face center for tracking
            function getFaceCenter(bbox) {
                return {
                    x: (bbox.x1 + bbox.x2) / 2,
                    y: (bbox.y1 + bbox.y2) / 2
                };
            }
            
            // Check if face is inside any detect zone
            function isInDetectZone(bbox, videoWidth, videoHeight) {
                const detectZones = zones.filter(z => z.type === 'detect');
                if (detectZones.length === 0) return true; // No detect zones = detect everywhere
                
                const centerX = (bbox.x1 + bbox.x2) / 2 / videoWidth;
                const centerY = (bbox.y1 + bbox.y2) / 2 / videoHeight;
                
                return detectZones.some(z => 
                    centerX >= z.x && centerX <= z.x + z.w &&
                    centerY >= z.y && centerY <= z.y + z.h
                );
            }
            
            // Generate face ID based on position (for tracking same face)
            function generateFaceId(bbox, videoWidth, videoHeight) {
                // Divide screen into grid cells for tracking
                const gridSize = 5; // 5x5 grid
                const cellX = Math.floor((bbox.x1 + bbox.x2) / 2 / videoWidth * gridSize);
                const cellY = Math.floor((bbox.y1 + bbox.y2) / 2 / videoHeight * gridSize);
                return `face_${cellX}_${cellY}`;
            }
            
            // Clean up old face trackers (faces that left the zone)
            function cleanupFaceTracker() {
                const now = Date.now();
                const timeout = 1000; // Remove if not seen for 1 second
                
                Object.keys(faceTracker).forEach(faceId => {
                    if (now - faceTracker[faceId].lastSeen > timeout) {
                        delete faceTracker[faceId];
                    }
                });
            }
            
            // FPS update
            function updateDetectFps() {
                frameCount++;
                const now = Date.now();
                const elapsed = now - lastFpsTime;
                
                if (elapsed >= 1000) {
                    currentFps = Math.round(frameCount * 1000 / elapsed);
                    const fpsEl = document.getElementById('fpsCounter');
                    if (fpsEl) fpsEl.textContent = 'Video: ' + videoFps + ' | Detect: ' + currentFps + ' FPS';
                    frameCount = 0;
                    lastFpsTime = now;
                }
            }
            
            // Video render loop - runs at full speed, redraws cached results
            let videoFrameCount = 0;
            let lastVideoFpsTime = Date.now();
            
            function renderLoop() {
                videoFrameCount++;
                const now = Date.now();
                if (now - lastVideoFpsTime >= 1000) {
                    videoFps = videoFrameCount;
                    videoFrameCount = 0;
                    lastVideoFpsTime = now;
                    const fpsEl = document.getElementById('fpsCounter');
                    if (fpsEl) fpsEl.textContent = 'Video: ' + videoFps + ' | Detect: ' + currentFps + ' FPS';
                }
                
                // Redraw zones and cached detection results
                drawZones();
                if (lastResults.length > 0) {
                    drawDetectionBoxes(lastResults);
                }
                
                if (stream) {
                    requestAnimationFrame(renderLoop);
                }
            }
            
            function drawDetectionBoxes(results) {
                results.forEach(r => {
                    if (!r.bbox) return;
                    
                    const scaleX = overlay.width / video.videoWidth;
                    const scaleY = overlay.height / video.videoHeight;
                    const x = r.bbox.x1 * scaleX;
                    const y = r.bbox.y1 * scaleY;
                    const w = (r.bbox.x2 - r.bbox.x1) * scaleX;
                    const h = (r.bbox.y2 - r.bbox.y1) * scaleY;
                    
                    // Different styling for pending (dwell time) vs classified
                    if (r.isPending) {
                        // Pending - yellow/orange with progress bar
                        ctx.strokeStyle = '#f39c12';
                        ctx.lineWidth = 3;
                        ctx.setLineDash([5, 5]);
                        ctx.strokeRect(x, y, w, h);
                        ctx.setLineDash([]);
                        
                        // Progress bar at bottom of box
                        const progressWidth = w * (r.dwellProgress || 0);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                        ctx.fillRect(x, y + h - 8, w, 8);
                        ctx.fillStyle = '#f39c12';
                        ctx.fillRect(x, y + h - 8, progressWidth, 8);
                        
                        // Label
                        const label = r.full_label;
                        ctx.font = 'bold 14px Arial';
                        const textWidth = ctx.measureText(label).width;
                        ctx.fillStyle = '#f39c12';
                        ctx.fillRect(x, y - 25, textWidth + 15, 22);
                        ctx.fillStyle = '#fff';
                        ctx.fillText(label, x + 7, y - 8);
                    } else {
                        // Classified - normal green/red
                        ctx.strokeStyle = r.authorized ? '#00b894' : '#d63031';
                        ctx.lineWidth = 3;
                        ctx.strokeRect(x, y, w, h);
                        
                        const label = r.full_label + ' ' + (r.confidence || '') + '%';
                        ctx.font = 'bold 14px Arial';
                        const textWidth = ctx.measureText(label).width;
                        ctx.fillStyle = r.authorized ? '#00b894' : '#d63031';
                        ctx.fillRect(x, y - 25, textWidth + 15, 22);
                        ctx.fillStyle = '#fff';
                        ctx.fillText(label, x + 7, y - 8);
                    }
                });
            }
            
            // Capture & Detect
            async function captureAndDetect() {
                if (!stream) {
                    alert('Start camera first!');
                    return;
                }
                
                // Skip if still processing previous frame
                if (isProcessing) return;
                isProcessing = true;
                
                updateDetectFps();
                cleanupFaceTracker(); // Clean old trackers
                
                // Create temp canvas
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(video, 0, 0);
                
                // Apply privacy mask (black out areas)
                zones.filter(z => z.type === 'privacy').forEach(zone => {
                    const sx = zone.x * video.videoWidth;
                    const sy = zone.y * video.videoHeight;
                    const sw = zone.w * video.videoWidth;
                    const sh = zone.h * video.videoHeight;
                    tempCtx.fillStyle = '#000';
                    tempCtx.fillRect(sx, sy, sw, sh);
                });
                
                const imageData = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                
                // Get ignore zones (normalized coordinates)
                const ignoreZones = zones.filter(z => z.type === 'ignore').map(z => ({
                    x1: z.x, y1: z.y, x2: z.x + z.w, y2: z.y + z.h
                }));
                
                // Get detect zones (only detect faces inside these zones)
                const detectZones = zones.filter(z => z.type === 'detect').map(z => ({
                    x1: z.x, y1: z.y, x2: z.x + z.w, y2: z.y + z.h
                }));
                
                try {
                    const response = await fetch('/api/detect', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            image: imageData,
                            ignore_zones: ignoreZones,
                            detect_zones: detectZones,
                            dwell_time_enabled: dwellTimeEnabled,
                            dwell_time_ms: DWELL_TIME_MS
                        })
                    });
                    const data = await response.json();
                    
                    // Process results with dwell time logic
                    if (dwellTimeEnabled && detectZones.length > 0) {
                        processWithDwellTime(data, video.videoWidth, video.videoHeight);
                    } else {
                        displayResults(data);
                        lastResults = data.results || [];
                    }
                } catch (err) {
                    console.error('Detection error:', err);
                } finally {
                    isProcessing = false;
                }
            }
            
            // Process detection results with dwell time
            function processWithDwellTime(data, videoWidth, videoHeight) {
                const now = Date.now();
                const results = data.results || [];
                const processedResults = [];
                const pendingResults = [];
                
                results.forEach(r => {
                    if (!r.bbox) return;
                    
                    // Check if face is in detect zone
                    if (!isInDetectZone(r.bbox, videoWidth, videoHeight)) {
                        return; // Skip faces outside detect zone
                    }
                    
                    // Generate face ID for tracking
                    const faceId = generateFaceId(r.bbox, videoWidth, videoHeight);
                    
                    if (!faceTracker[faceId]) {
                        // New face - start tracking
                        faceTracker[faceId] = {
                            firstSeen: now,
                            lastSeen: now,
                            bbox: r.bbox,
                            classified: false,
                            result: r
                        };
                    } else {
                        // Existing face - update
                        faceTracker[faceId].lastSeen = now;
                        faceTracker[faceId].bbox = r.bbox;
                        faceTracker[faceId].result = r;
                    }
                    
                    const tracker = faceTracker[faceId];
                    const dwellTime = now - tracker.firstSeen;
                    
                    if (dwellTime >= DWELL_TIME_MS) {
                        // Face has been in zone long enough - classify!
                        if (!tracker.classified) {
                            tracker.classified = true;
                            processedResults.push(r);
                        } else {
                            // Already classified, just show result
                            processedResults.push(r);
                        }
                    } else {
                        // Still waiting - show countdown
                        const remaining = Math.ceil((DWELL_TIME_MS - dwellTime) / 1000);
                        pendingResults.push({
                            ...r,
                            full_label: `⏱️ Wait ${remaining}s...`,
                            authorized: null, // Pending state
                            isPending: true,
                            dwellProgress: dwellTime / DWELL_TIME_MS
                        });
                    }
                });
                
                // Combine results
                const allResults = [...processedResults, ...pendingResults];
                
                // Display only classified results in the results panel
                displayResults({ results: processedResults });
                
                // But show all (including pending) in overlay
                lastResults = allResults;
            }
            
            // Drawing zones
            function setDrawMode(mode) {
                drawMode = mode;
                document.getElementById('privacyBtn').classList.toggle('btn-active', mode === 'privacy');
                document.getElementById('ignoreBtn').classList.toggle('btn-active', mode === 'ignore');
                document.getElementById('detectBtn').classList.toggle('btn-active', mode === 'detect');
                overlay.style.cursor = mode ? 'crosshair' : 'default';
                updateMode(mode ? 'drawing' : 'capture');
            }
            
            function setupDrawing() {
                overlay.addEventListener('mousedown', (e) => {
                    if (!drawMode) return;
                    isDrawing = true;
                    const rect = overlay.getBoundingClientRect();
                    startX = (e.clientX - rect.left) / rect.width;
                    startY = (e.clientY - rect.top) / rect.height;
                });
                
                overlay.addEventListener('mousemove', (e) => {
                    if (!isDrawing) return;
                    const rect = overlay.getBoundingClientRect();
                    const currentX = (e.clientX - rect.left) / rect.width;
                    const currentY = (e.clientY - rect.top) / rect.height;
                    
                    drawZones();
                    
                    // Draw current selection
                    const colors = { privacy: '#e17055', ignore: '#6c5ce7', detect: '#00b894' };
                    ctx.strokeStyle = colors[drawMode] || '#fff';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(
                        startX * overlay.width,
                        startY * overlay.height,
                        (currentX - startX) * overlay.width,
                        (currentY - startY) * overlay.height
                    );
                    ctx.setLineDash([]);
                });
                
                overlay.addEventListener('mouseup', (e) => {
                    if (!isDrawing) return;
                    isDrawing = false;
                    
                    const rect = overlay.getBoundingClientRect();
                    const endX = (e.clientX - rect.left) / rect.width;
                    const endY = (e.clientY - rect.top) / rect.height;
                    
                    const x = Math.min(startX, endX);
                    const y = Math.min(startY, endY);
                    const w = Math.abs(endX - startX);
                    const h = Math.abs(endY - startY);
                    
                    if (w > 0.02 && h > 0.02) { // Min size
                        zones.push({ type: drawMode, x, y, w, h, id: Date.now() });
                        updateZoneList();
                        drawZones();
                    }
                });
            }
            
            function drawZones() {
                ctx.clearRect(0, 0, overlay.width, overlay.height);
                
                // Check if there are any detect zones
                const detectZones = zones.filter(z => z.type === 'detect');
                
                // If detect zones exist, fill everything with ignore overlay first
                if (detectZones.length > 0) {
                    ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
                    ctx.fillRect(0, 0, overlay.width, overlay.height);
                    
                    // Cut out the detect zones (make them clear)
                    ctx.globalCompositeOperation = 'destination-out';
                    detectZones.forEach(zone => {
                        const x = zone.x * overlay.width;
                        const y = zone.y * overlay.height;
                        const w = zone.w * overlay.width;
                        const h = zone.h * overlay.height;
                        ctx.fillRect(x, y, w, h);
                    });
                    ctx.globalCompositeOperation = 'source-over';
                }
                
                zones.forEach(zone => {
                    const x = zone.x * overlay.width;
                    const y = zone.y * overlay.height;
                    const w = zone.w * overlay.width;
                    const h = zone.h * overlay.height;
                    
                    if (zone.type === 'privacy') {
                        ctx.fillStyle = 'rgba(225, 112, 85, 0.5)';
                        ctx.fillRect(x, y, w, h);
                        ctx.strokeStyle = '#e17055';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, w, h);
                        ctx.fillStyle = '#fff';
                        ctx.font = '12px Arial';
                        ctx.fillText('🔒 PRIVACY', x + 5, y + 15);
                    } else if (zone.type === 'ignore') {
                        ctx.fillStyle = 'rgba(108, 92, 231, 0.3)';
                        ctx.fillRect(x, y, w, h);
                        ctx.strokeStyle = '#6c5ce7';
                        ctx.lineWidth = 2;
                        ctx.setLineDash([5, 5]);
                        ctx.strokeRect(x, y, w, h);
                        ctx.setLineDash([]);
                        ctx.fillStyle = '#fff';
                        ctx.font = '12px Arial';
                        ctx.fillText('🚫 IGNORE', x + 5, y + 15);
                    } else if (zone.type === 'detect') {
                        ctx.fillStyle = 'rgba(0, 184, 148, 0.15)';
                        ctx.fillRect(x, y, w, h);
                        ctx.strokeStyle = '#00b894';
                        ctx.lineWidth = 3;
                        ctx.strokeRect(x, y, w, h);
                        ctx.fillStyle = '#fff';
                        ctx.font = 'bold 12px Arial';
                        ctx.fillText('🎯 DETECT ZONE', x + 5, y + 18);
                    }
                });
            }
            
            function drawDetections(results) {
                // Clear and redraw zones first
                drawZones();
                
                if (!results || results.length === 0) return;
                
                results.forEach(r => {
                    if (!r.bbox) return;
                    
                    // Convert bbox to overlay coordinates
                    const scaleX = overlay.width / video.videoWidth;
                    const scaleY = overlay.height / video.videoHeight;
                    const x = r.bbox.x1 * scaleX;
                    const y = r.bbox.y1 * scaleY;
                    const w = (r.bbox.x2 - r.bbox.x1) * scaleX;
                    const h = (r.bbox.y2 - r.bbox.y1) * scaleY;
                    
                    ctx.strokeStyle = r.authorized ? '#00b894' : '#d63031';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x, y, w, h);
                    
                    // Label background
                    const label = r.full_label + ' ' + r.confidence + '%';
                    ctx.font = 'bold 14px Arial';
                    const textWidth = ctx.measureText(label).width;
                    ctx.fillStyle = r.authorized ? '#00b894' : '#d63031';
                    ctx.fillRect(x, y - 25, textWidth + 15, 22);
                    
                    // Label text
                    ctx.fillStyle = '#fff';
                    ctx.fillText(label, x + 7, y - 8);
                });
            }
            
            function updateZoneList() {
                const list = document.getElementById('zoneList');
                document.getElementById('zoneCount').textContent = zones.length;
                
                if (zones.length === 0) {
                    list.innerHTML = '<div class="no-results">No zones defined.</div>';
                    return;
                }
                
                const icons = { privacy: '🔒', ignore: '🚫', detect: '🎯' };
                list.innerHTML = zones.map((z, i) => `
                    <div class="zone-item ${z.type}">
                        <span>${icons[z.type]} ${z.type.toUpperCase()} ${i + 1} (${(z.w * 100).toFixed(0)}% x ${(z.h * 100).toFixed(0)}%)</span>
                        <button class="zone-delete" onclick="deleteZone(${z.id})">✕</button>
                    </div>
                `).join('');
            }
            
            function deleteZone(id) {
                zones = zones.filter(z => z.id !== id);
                updateZoneList();
                drawZones();
            }
            
            function clearAllZones() {
                zones = [];
                updateZoneList();
                drawZones();
            }
            
            // Results
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                
                if (!data.results || data.results.length === 0) {
                    resultsDiv.innerHTML = '<div class="no-results">No faces detected.</div>';
                    return;
                }
                
                let html = '';
                data.results.forEach(r => {
                    const statusClass = r.authorized ? 'authorized' : 'unauthorized';
                    const statusText = r.authorized ? '✅ AUTHORIZED' : '❌ DENIED';
                    
                    html += `
                        <div class="result-card ${statusClass}">
                            <div class="result-name">${r.full_label}</div>
                            <div class="result-info">Confidence: ${r.confidence}%</div>
                            <span class="status-badge ${statusClass}">${statusText}</span>
                        </div>
                    `;
                    
                    // Add to log (returns true if actually logged, false if cooldown)
                    const wasLogged = addLog(r);
                    
                    // Only update stats if actually logged (not duplicate)
                    if (wasLogged) {
                        stats.total++;
                        if (r.authorized) stats.authorized++;
                        else stats.unauthorized++;
                    }
                });
                
                resultsDiv.innerHTML = html;
                updateStats();
            }
            
            function updateStats() {
                document.getElementById('totalDetections').textContent = stats.total;
                document.getElementById('authorizedCount').textContent = stats.authorized;
                document.getElementById('unauthorizedCount').textContent = stats.unauthorized;
            }
            
            function addLog(result) {
                const now = Date.now();
                const faceKey = result.name.toLowerCase(); // Use name as key
                const cooldownMs = 10000; // 10 seconds cooldown per face
                
                // Check if this face was logged recently
                if (lastDetectionTime[faceKey] && (now - lastDetectionTime[faceKey]) < cooldownMs) {
                    // Skip logging - same face detected within cooldown
                    return false;
                }
                
                // Update last detection time
                lastDetectionTime[faceKey] = now;
                
                const time = new Date().toLocaleTimeString();
                const logClass = result.authorized ? 'auth' : 'unauth';
                const status = result.authorized ? '✅' : '❌';
                
                logs.unshift({ time, result, logClass, status });
                if (logs.length > 50) logs.pop();
                
                const logList = document.getElementById('logList');
                logList.innerHTML = logs.map(l => 
                    `<div class="log-item ${l.logClass}">${l.time} ${l.status} ${l.result.full_label} (${l.result.confidence}%)</div>`
                ).join('');
                
                return true; // Log was added
            }
            
            // Start
            init();
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)
