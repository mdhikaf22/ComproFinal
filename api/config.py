"""
============================================================
CONFIG - Konfigurasi Aplikasi
============================================================
Sesuai dengan: Model build/Compro_YOLOv8_Training.ipynb
"""

import os

# ============================================================
# ROLE MAPPING - Untuk authorization
# ============================================================
ROLE_MAPPING = {
    "iksan": "Aslab",
    "akbar": "Aslab",
    "aprilianza": "Aslab",
    "bian": "Dosen",
    "fadhilah": "Aslab",
    "falah": "Aslab",
    "imelda": "Aslab",
    "rifqy": "Aslab",
    "yolanda": "Aslab",
}

# ============================================================
# CLASS NAMES - Sesuai urutan di ImageFolder dataset
# ============================================================
CLASS_NAMES = ['akbar', 'aprilianza', 'bian', 'fadhilah', 'falah', 'iksan', 'imelda', 'rifqy', 'yolanda']

# ============================================================
# THRESHOLDS
# ============================================================
CONFIDENCE_THRESHOLD = 0.5       # Min confidence untuk klasifikasi ViT
FACE_DETECTION_THRESHOLD = 0.35  # Min confidence untuk deteksi wajah YOLOv8

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "access_logs.db")

# Model paths - sesuai output dari notebook training
VIT_MODEL_PATH = os.path.join(BASE_DIR, "Model build", "best_vit_yolo.pth")  # ViT classifier (YOLOv8 version)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8-face.pt")                    # YOLOv8-face detector

# Fallback ke model lama jika model baru belum ada
if not os.path.exists(VIT_MODEL_PATH):
    VIT_MODEL_PATH = os.path.join(BASE_DIR, "best_vit_mtcnn.pth")
if not os.path.exists(VIT_MODEL_PATH):
    VIT_MODEL_PATH = os.path.join(BASE_DIR, "Model build", "best_vit_mtcnn.pth")

# Legacy alias
MODEL_PATH = VIT_MODEL_PATH

SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
CONFIG_PATH = os.path.join(BASE_DIR, "Model build", "model_config.json")

# ============================================================
# DETECTION BACKEND
# ============================================================
# Options: 'yolo' (faster, ~15-30 FPS) or 'mtcnn' (slower, ~2-3 FPS)
DETECTION_BACKEND = os.environ.get('DETECTION_BACKEND', 'yolo')

# ============================================================
# ANTI-SPOOFING CONFIG (Silent Face Anti-Spoofing)
# ============================================================
ANTISPOOF_ENABLED = True          # Enable/disable anti-spoofing
ANTISPOOF_THRESHOLD = 0.5         # Min score untuk dianggap real (0.0-1.0)
                                  # Deep learning model punya weight 60%
                                  # Traditional methods punya weight 40%

# ============================================================
# SERVER CONFIG
# ============================================================
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False
