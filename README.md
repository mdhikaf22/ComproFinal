# Face Recognition System - YOLOv8 + ViT + Anti-Spoofing

Sistem pengenalan wajah real-time menggunakan **YOLOv8-face** untuk deteksi wajah, **Vision Transformer (ViT)** untuk klasifikasi, dan **Silent Face Anti-Spoofing** untuk mencegah manipulasi foto/layar HP.

## ⚡ Quick Start

```bash
# 1. Clone repository (dengan LFS untuk download model)
git lfs install
git clone https://github.com/mdhikaf22/Compro.git
cd Compro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan server
python app.py

# 4. Buka browser
# http://localhost:5000/api/webcam
```

> **Note:** Repository ini menggunakan Git LFS untuk file model (`best_vit_yolo.pth`). Pastikan `git lfs` terinstall sebelum clone.

## 📋 Deskripsi

Proyek ini adalah sistem pengenalan wajah yang dirancang untuk keperluan akses kontrol (misalnya di depan pintu lab). Sistem dapat mendeteksi wajah dari webcam secara real-time dan mengklasifikasikan apakah orang tersebut **Authorized** atau **Not Authorized**.

### Fitur Utama
- **Face Detection**: YOLOv8-face (default) atau MTCNN (fallback)
- **Face Classification**: ViT (Vision Transformer) dari Google
- **🛡️ Anti-Spoofing**: Silent Face Anti-Spoofing (Deep Learning + Traditional Methods)
- **Dual Backend**: Switch antara YOLOv8 dan MTCNN via environment variable
- **Real-time Inference**: ~5 FPS untuk detection + classification
- **Smooth Video**: Render loop terpisah untuk video 30+ FPS
- **REST API**: Backend Flask dengan endpoint untuk integrasi
- **Database Logging**: SQLite untuk menyimpan log akses
- **Web Interface**: UI webcam berbasis browser
- **Authorization System**: Menampilkan status otorisasi berdasarkan role

## 👥 Kelas yang Dikenali

| Nama | Role | Status |
|------|------|--------|
| Iksan | Aslab | ✅ Authorized |
| Akbar | Aslab | ✅ Authorized |
| Aprilianza | Aslab | ✅ Authorized |
| Bian | Dosen | ✅ Authorized |
| Fadhilah | Aslab | ✅ Authorized |
| Falah | Aslab | ✅ Authorized |
| Imelda | Aslab | ✅ Authorized |
| Rifqy | Aslab | ✅ Authorized |
| Yolanda | Aslab | ✅ Authorized |

## 🛠️ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
ultralytics>=8.0.0          # YOLOv8 face detection
facenet-pytorch>=2.5.0      # MTCNN fallback
onnxruntime>=1.15.0         # Anti-spoofing optimization
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0               # Anti-spoofing analysis
flask>=2.3.0
flask-cors>=4.0.0
```

### Instalasi Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Struktur Proyek

```
compro/
├── app.py                      # Main entry point (Flask)
├── requirements.txt            # Dependencies
├── yolov8-face.pt              # YOLOv8 face detection model
├── access_logs.db              # SQLite database (auto-generated)
│
├── api/                        # API Package (Modular)
│   ├── __init__.py
│   ├── config.py               # Konfigurasi (class names, roles, thresholds)
│   ├── database.py             # Database operations
│   ├── model.py                # Face detection & classification (YOLOv8/MTCNN + ViT)
│   ├── antispoof.py            # 🛡️ Silent Face Anti-Spoofing module
│   └── routes/                 # API Routes
│       ├── __init__.py
│       ├── main.py             # Home & health check
│       ├── detection.py        # Face detection endpoints
│       ├── logs.py             # Access logs & statistics
│       └── webcam.py           # Webcam interface & stream
│
├── resources/                  # Model resources
│   └── anti_spoof_models/      # Anti-spoofing models
│       ├── 2.7_80x80_MiniFASNetV2.pth
│       └── 4_0_0_80x80_MiniFASNetV1SE.pth
│
├── Model build/                # Training notebooks & artifacts
│   ├── Compro_YOLOv8_Training.ipynb  # YOLOv8 + ViT training notebook
│   ├── Compro_MTCNN.ipynb            # MTCNN + ViT training notebook
│   ├── best_vit_yolo.pth             # Model weights (Git LFS)
│   ├── yolov8-face.pt                # YOLOv8 face detection
│   ├── model_config.json             # Model configuration
│   ├── yolov8_config.json            # YOLOv8 configuration
│   ├── confusion_matrix.png          # Training results
│   └── training_history.png          # Training history
│
├── vit_dataset/                # Dataset untuk training
│   └── train/
│       ├── akbar/
│       ├── aprilianza/
│       └── ...
├── screenshots/                # Folder screenshot webcam
└── README.md
```

## 🚀 Cara Penggunaan

### 1. Training Model (Notebook)

**YOLOv8 + ViT (Recommended):**
1. Buka `Model build/Compro_YOLOv8_Training.ipynb`
2. Jalankan semua cell secara berurutan
3. Model akan disimpan di `Model build/best_vit_yolo.pth`

**MTCNN + ViT (Alternative):**
1. Buka `Model build/Compro_MTCNN.ipynb`
2. Jalankan semua cell secara berurutan

### 2. Menjalankan API Server (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan server (default: YOLOv8 backend)
python app.py

# Atau dengan MTCNN backend
set DETECTION_BACKEND=mtcnn
python app.py
```

Server akan berjalan di `http://localhost:5000`

### 3. Switch Detection Backend

```bash
# Windows - YOLOv8 (default, faster ~5 FPS)
set DETECTION_BACKEND=yolo
python app.py

# Windows - MTCNN (more accurate)
set DETECTION_BACKEND=mtcnn
python app.py
```

### 4. Webcam Inference

#### Option A: Via Web Browser (Recommended)
1. Jalankan `python app.py`
2. Buka browser ke `http://localhost:5000/api/webcam`
3. Klik "Start Camera" lalu "Capture & Detect"

#### Option B: Di Notebook
1. Buka `Model build/Compro_YOLOv8_Training.ipynb`
2. Jalankan cell "WEBCAM INFERENCE" di bagian akhir
3. Tekan Stop (interrupt kernel) untuk berhenti

### 5. Kontrol Webcam

| Key | Fungsi |
|-----|--------|
| `q` | Keluar dari webcam |
| `s` | Simpan screenshot |

## 📊 Hasil Training

- **Detection Model**: YOLOv8-face (pretrained)
- **Classification Model**: ViT (google/vit-base-patch16-224-in21k)
- **Epochs**: 30
- **Best Validation Accuracy**: ~100%
- **Total Classes**: 9
- **Inference Speed**: ~5 FPS (detection + classification)

## 🔧 Konfigurasi

### Environment Variables
```bash
DETECTION_BACKEND=yolo    # 'yolo' atau 'mtcnn'
```

### Threshold (api/config.py)
```python
CONFIDENCE_THRESHOLD = 0.5      # Threshold klasifikasi ViT
FACE_DETECTION_THRESHOLD = 0.35 # Threshold deteksi wajah

# Anti-Spoofing
ANTISPOOF_ENABLED = True        # Enable/disable anti-spoofing
ANTISPOOF_THRESHOLD = 0.5       # Min score untuk dianggap real (0.0-1.0)
```

### YOLOv8 Parameters
```python
conf = 0.35                     # Confidence threshold
imgsz = 480                     # Image size untuk inference
```

### MTCNN Parameters (fallback)
```python
min_face_size = 20              # Ukuran minimum wajah
thresholds = [0.5, 0.6, 0.6]    # Threshold per stage
```

## 📝 Catatan

- Pastikan webcam terhubung dan berfungsi dengan baik
- Model membutuhkan GPU untuk performa optimal (CUDA)
- Jika menggunakan CPU, inference akan lebih lambat
- YOLOv8 lebih cepat (~5 FPS), MTCNN lebih akurat (~2 FPS)
- Video render tetap smooth (30+ FPS) karena render loop terpisah

## 🏗️ Arsitektur

```
Input Image/Frame
    ↓
┌─────────────────────────────────┐
│  Detection Backend (selectable) │
│  ├── YOLOv8-face (default)      │
│  └── MTCNN (fallback)           │
└─────────────────────────────────┘
    ↓
Face Cropping + Padding (5%)
    ↓
┌─────────────────────────────────┐
│  🛡️ Anti-Spoofing Check         │
│  ├── MiniFASNet V2 (60%)        │
│  ├── MiniFASNet V1SE            │
│  ├── Texture Analysis (15%)     │
│  ├── Color Analysis (10%)       │
│  ├── Frequency Analysis (10%)   │
│  └── Edge Analysis (5%)         │
└─────────────────────────────────┘
    ↓
[If Real Face]
    ↓
Preprocessing (Resize 224x224, Normalize)
    ↓
ViT Classification
    ↓
Output: Name, Role, Authorization Status, Liveness Score
```

## ⏱️ Dwell Time Detection

Fitur untuk mencegah deteksi orang yang hanya lewat. Hanya wajah yang **berdiri diam di detect zone selama 3 detik** yang akan diproses untuk klasifikasi.

### Cara Kerja
1. Gambar **Detect Zone** di area yang ingin dimonitor (misal: depan pintu)
2. Aktifkan **Auto Detect**
3. Wajah yang masuk zona akan menampilkan countdown "⏱️ Wait 3s..."
4. Setelah 3 detik diam, baru akan diklasifikasi
5. Orang yang hanya lewat tidak akan terdeteksi

### UI Indicator
- **🟡 Orange dashed box**: Wajah sedang menunggu (belum 3 detik)
- **Progress bar**: Menunjukkan sisa waktu tunggu
- **🟢 Green/🔴 Red box**: Wajah sudah diklasifikasi

### Toggle
Klik tombol **"⏱️ Dwell: 3s ON"** untuk enable/disable fitur ini.

---

## 🛡️ Anti-Spoofing

Sistem menggunakan **Silent Face Anti-Spoofing** untuk mencegah manipulasi menggunakan foto atau layar HP.

### Metode yang Digunakan

| Method | Weight | Description |
|--------|--------|-------------|
| **Deep Learning (MiniFASNet)** | 60% | Neural network untuk deteksi spoof |
| **Texture Analysis (LBP)** | 15% | Local Binary Pattern - foto punya tekstur smooth |
| **Color Distribution** | 10% | Layar HP punya warna oversaturated |
| **Frequency Analysis** | 10% | Deteksi Moire pattern dari layar |
| **Edge Analysis** | 5% | Karakteristik edge berbeda |

### Output Anti-Spoofing

```json
{
  "is_spoof": false,
  "liveness_score": 75.5,
  "spoof_reason": ""
}
```

Jika spoof terdeteksi:
```json
{
  "is_spoof": true,
  "full_label": "⚠️ SPOOF DETECTED",
  "spoof_reason": "DL: Spoof detected; Texture too smooth"
}
```

## 🔄 Performance Comparison

| Backend | Detection FPS | Accuracy | Notes |
|---------|--------------|----------|-------|
| YOLOv8-face | ~5 FPS | Good | Faster, recommended |
| MTCNN | ~2 FPS | Better | More accurate, slower |

## 📜 License

Project ini dibuat untuk keperluan Computing Project.

## 👨‍💻 Author

MAHARDHIKA
