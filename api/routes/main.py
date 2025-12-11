"""
============================================================
MAIN ROUTES - Home & Health Check & Docs
============================================================
"""

from flask import Blueprint, jsonify, render_template_string
from datetime import datetime
import os

from ..config import MODEL_PATH

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page with API documentation"""
    return jsonify({
        "name": "Face Recognition API",
        "version": "1.0",
        "description": "Real-time face recognition using MTCNN + ViT",
        "endpoints": {
            "/": "API documentation (JSON)",
            "/docs": "API documentation (HTML)",
            "/api/health": "GET - Health check",
            "/api/detect": "POST - Detect faces from base64 image",
            "/api/detect/upload": "POST - Detect faces from uploaded file",
            "/api/webcam": "GET - Webcam interface",
            "/api/stream": "GET - Video stream with detection overlay",
            "/api/logs": "GET - Get access logs",
            "/api/stats": "GET - Get daily statistics"
        }
    })


@main_bp.route('/docs')
def docs():
    """API Documentation page"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition API - Documentation</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                color: #e0e0e0;
                padding: 40px 20px;
            }
            .container { max-width: 900px; margin: 0 auto; }
            h1 { 
                text-align: center; 
                margin-bottom: 10px;
                font-size: 2.5em;
                color: #00b894;
            }
            .subtitle {
                text-align: center;
                color: #b2bec3;
                margin-bottom: 40px;
            }
            .card {
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                border: 1px solid rgba(255,255,255,0.1);
            }
            h2 { 
                color: #00b894;
                margin-bottom: 20px;
                font-size: 1.4em;
            }
            .endpoint {
                background: rgba(0,0,0,0.3);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
            }
            .endpoint:last-child { margin-bottom: 0; }
            .method {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 5px;
                font-size: 0.8em;
                font-weight: bold;
                margin-right: 10px;
            }
            .method.get { background: #0984e3; }
            .method.post { background: #00b894; }
            .path {
                font-family: monospace;
                font-size: 1.1em;
                color: #fff;
            }
            .desc {
                margin-top: 10px;
                color: #b2bec3;
                font-size: 0.9em;
            }
            .example {
                background: rgba(0,0,0,0.4);
                border-radius: 8px;
                padding: 12px;
                margin-top: 10px;
                font-family: monospace;
                font-size: 0.85em;
                overflow-x: auto;
            }
            .example-label {
                color: #fdcb6e;
                font-size: 0.8em;
                margin-bottom: 5px;
            }
            a { color: #74b9ff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .quick-links {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            .quick-link {
                background: #0984e3;
                color: white;
                padding: 12px 25px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                transition: all 0.3s;
            }
            .quick-link:hover { 
                background: #0874c9; 
                transform: scale(1.05);
                text-decoration: none;
            }
            .status { 
                display: inline-block;
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 0.75em;
                background: #00b894;
                color: white;
                margin-left: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Face Recognition API</h1>
            <p class="subtitle">Real-time face recognition using MTCNN + ViT</p>
            
            <div class="quick-links">
                <a href="/api/webcam" class="quick-link">📹 Open Webcam</a>
                <a href="/api/health" class="quick-link">💚 Health Check</a>
                <a href="/api/logs" class="quick-link">📋 View Logs</a>
                <a href="/api/stats" class="quick-link">📊 Statistics</a>
            </div>
            
            <div class="card">
                <h2>📡 API Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/</span>
                    <div class="desc">API documentation in JSON format</div>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/docs</span>
                    <div class="desc">API documentation page (this page)</div>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/api/health</span>
                    <div class="desc">Health check - returns server status and device info</div>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="path">/api/detect</span>
                    <div class="desc">Detect faces from base64 encoded image</div>
                    <div class="example-label">Request Body:</div>
                    <div class="example">{"image": "base64_encoded_image_string"}</div>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="path">/api/detect/upload</span>
                    <div class="desc">Detect faces from uploaded image file (multipart/form-data)</div>
                    <div class="example-label">Form Field:</div>
                    <div class="example">file: [image file]</div>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/api/webcam</span>
                    <div class="desc">Web interface for webcam face detection</div>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/api/stream</span>
                    <div class="desc">Live video stream with face detection overlay</div>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/api/logs</span>
                    <div class="desc">Get access logs with pagination</div>
                    <div class="example-label">Query Parameters:</div>
                    <div class="example">?limit=100&offset=0</div>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/api/stats</span>
                    <div class="desc">Get daily statistics</div>
                    <div class="example-label">Query Parameters:</div>
                    <div class="example">?days=7</div>
                </div>
            </div>
            
            <div class="card">
                <h2>📤 Response Format</h2>
                <div class="example-label">Detection Response:</div>
                <div class="example">{
  "success": true,
  "timestamp": "2025-12-07T18:30:00",
  "faces_detected": 1,
  "results": [
    {
      "name": "Iksan",
      "role": "Aslab",
      "full_label": "Iksan (Aslab)",
      "authorized": true,
      "confidence": 98.5,
      "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 180},
      "detection_score": 0.99
    }
  ]
}</div>
            </div>
            
            <div class="card">
                <h2>👥 Recognized Classes</h2>
                <div class="example">
Iksan (Aslab) ✅ | Akbar (Aslab) ✅ | Aprilianza (Aslab) ✅
Bian (Dosen) ✅ | Fadhilah (Aslab) ✅ | Falah (Aslab) ✅
Imelda (Aslab) ✅ | Rifqy (Aslab) ✅ | Yolanda (Aslab) ✅
                </div>
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html)


@main_bp.route('/api/health')
def health():
    """Health check endpoint"""
    from ..model import face_model
    
    return jsonify({
        "status": "healthy",
        "device": str(face_model.device),
        "model_loaded": os.path.exists(MODEL_PATH),
        "timestamp": datetime.now().isoformat()
    })
