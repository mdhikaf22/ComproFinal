"""
============================================================
FACE RECOGNITION API - FLASK BACKEND
============================================================
Webcam → Backend (Flask) → Model → Response JSON
                    ↓
               Database log akses
============================================================

Struktur Modular:
- api/config.py      : Konfigurasi
- api/database.py    : Database operations
- api/model.py       : Face detection & classification
- api/routes/        : API endpoints
============================================================
"""

from flask import Flask
from flask_cors import CORS

from api.config import HOST, PORT, DEBUG
from api.database import init_database
from api.model import face_model
from api.routes import main_bp, detection_bp, logs_bp, webcam_bp


def create_app():
    """Create Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(logs_bp)
    app.register_blueprint(webcam_bp)
    
    return app


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    import ssl
    import os
    
    # Initialize
    init_database()
    face_model.initialize()
    
    # Check for SSL certificates
    cert_file = 'cert.pem'
    key_file = 'key.pem'
    use_ssl = os.path.exists(cert_file) and os.path.exists(key_file)
    
    print("\n" + "=" * 60)
    print("      STARTING FACE RECOGNITION API SERVER")
    print("=" * 60)
    
    if use_ssl:
        print(f"🔒 HTTPS Mode (SSL enabled)")
        print(f"🌐 Server: https://localhost:{PORT}")
        print(f"📹 Webcam UI: https://localhost:{PORT}/api/webcam")
    else:
        print(f"🔓 HTTP Mode")
        print(f"🌐 Server: http://localhost:{PORT}")
        print(f"📹 Webcam UI: http://localhost:{PORT}/api/webcam")
        print(f"")
        print(f"⚠️  Webcam hanya bisa diakses dari localhost!")
        print(f"   Untuk akses dari device lain, generate SSL cert:")
        print(f"   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes")
    
    print(f"📊 API Docs: http://localhost:{PORT}/docs")
    print("=" * 60 + "\n")
    
    app = create_app()
    
    if use_ssl:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, ssl_context=context)
    else:
        app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
