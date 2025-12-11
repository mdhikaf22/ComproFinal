"""
============================================================
MODEL - Face Detection & Classification
============================================================
Sesuai dengan: Model build/Compro_YOLOv8_Training.ipynb

Supports two detection backends:
- YOLOv8-face (default, faster ~15-30 FPS)
- MTCNN (slower ~2-3 FPS, more accurate)
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
import cv2
import os

from .config import (
    CLASS_NAMES, ROLE_MAPPING, 
    CONFIDENCE_THRESHOLD, FACE_DETECTION_THRESHOLD,
    VIT_MODEL_PATH, YOLO_MODEL_PATH, DETECTION_BACKEND,
    ANTISPOOF_ENABLED, ANTISPOOF_THRESHOLD
)
from .database import log_access
from .antispoof import anti_spoof


class FaceRecognitionModel:
    """Face Recognition Model using YOLOv8/MTCNN + ViT"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = None
        self.detector = None
        self.model = None
        self.backend = DETECTION_BACKEND
        self._initialized = False
    
    def initialize(self):
        """Initialize model and detector"""
        if self._initialized:
            return
        
        print("=" * 60)
        print("      INITIALIZING FACE RECOGNITION MODEL")
        print("=" * 60)
        print(f"[OK] Device: {self.device}")
        print(f"[OK] Detection Backend: {self.backend.upper()}")
        
        # Transform for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        # Initialize detector based on backend
        if self.backend == 'yolo':
            self._init_yolo()
        else:
            self._init_mtcnn()
        
        # ViT Model for classification
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=len(CLASS_NAMES),
            ignore_mismatched_sizes=True
        )
        
        if os.path.exists(VIT_MODEL_PATH):
            self.model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=self.device))
            print(f"[OK] ViT Model loaded from {VIT_MODEL_PATH}")
        else:
            print(f"[WARN] ViT model not found at {VIT_MODEL_PATH}")
        
        self.model.to(self.device)
        self.model.eval()
        print("[OK] Model ready")
        print("=" * 60)
        
        self._initialized = True
    
    def _init_yolo(self):
        """Initialize YOLOv8 face detector"""
        try:
            from ultralytics import YOLO
            
            # Use YOLO_MODEL_PATH from config
            if os.path.exists(YOLO_MODEL_PATH):
                self.detector = YOLO(YOLO_MODEL_PATH)
                self._yolo_is_face_model = True
                print(f"[OK] YOLOv8-face loaded from {YOLO_MODEL_PATH}")
            else:
                # Fallback to YOLOv8n (detects person, not face directly)
                self.detector = YOLO('yolov8n.pt')
                self._yolo_is_face_model = False
                print("[OK] YOLOv8n loaded (person detection mode)")
            
            # Warm up
            self.detector.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
            print("[OK] YOLOv8 warmed up")
            
        except Exception as e:
            print(f"[ERROR] YOLOv8 init failed: {e}")
            raise RuntimeError(f"YOLOv8 initialization failed: {e}. Please install ultralytics: pip install ultralytics")
    
    def _init_mtcnn(self):
        """Initialize MTCNN face detector"""
        from facenet_pytorch import MTCNN
        
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=20,
            thresholds=[0.5, 0.6, 0.6],
            factor=0.709,
            post_process=False
        )
        self.backend = 'mtcnn'
        print("[OK] MTCNN initialized")
    
    def detect_faces(self, image):
        """Detect faces using selected backend. Returns list of (x1,y1,x2,y2,conf)"""
        if self.backend == 'yolo':
            return self._detect_yolo(image)
        else:
            return self._detect_mtcnn(image)
    
    def _detect_yolo(self, image):
        """Detect faces using YOLOv8"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Run detection - use smaller image size for speed
        results = self.detector.predict(img_np, verbose=False, conf=0.35, imgsz=480)
        
        faces = []
        img_h, img_w = img_np.shape[:2]
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # If using face model, class 0 is face
                if getattr(self, '_yolo_is_face_model', False):
                    # Face model - direct face detection
                    faces.append((x1, y1, x2, y2, conf))
                else:
                    # Person model - estimate face from upper body
                    if cls == 0:  # person class
                        person_h = y2 - y1
                        person_w = x2 - x1
                        
                        # Estimasi wajah: 60% lebar, 35% tinggi dari bagian atas
                        face_w = int(person_w * 0.6)
                        face_h = int(person_h * 0.35)
                        
                        center_x = (x1 + x2) // 2
                        face_x1 = max(0, center_x - face_w // 2)
                        face_x2 = min(img_w, center_x + face_w // 2)
                        
                        # Geser sedikit ke bawah dari y1 untuk melewati rambut
                        face_y1 = max(0, y1 + int(person_h * 0.05))
                        face_y2 = min(img_h, face_y1 + face_h)
                        
                        if face_x2 - face_x1 > 20 and face_y2 - face_y1 > 20:
                            faces.append((face_x1, face_y1, face_x2, face_y2, conf))
        
        return faces
    
    def _detect_mtcnn(self, image):
        """Detect faces using MTCNN"""
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        boxes, probs = self.detector.detect(pil_image)
        
        faces = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob is not None and prob >= FACE_DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    faces.append((x1, y1, x2, y2, float(prob)))
        
        return faces
    
    @staticmethod
    def get_full_label(name):
        """Get full label with role and authorization status"""
        name_lower = name.lower()
        if name_lower in ROLE_MAPPING:
            role = ROLE_MAPPING[name_lower]
            return f"{name.capitalize()} ({role})", role, True
        else:
            return f"{name} (Guest)", "Guest", False
    
    def process_image(self, image, save_log=True):
        """Process image and return detection results"""
        if not self._initialized:
            self.initialize()
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Detect faces using selected backend
        faces = self.detect_faces(pil_image)
        
        results = []
        
        if not faces:
            return results
        
        img_w, img_h = pil_image.size
        
        for x1, y1, x2, y2, prob in faces:
            # Add minimal padding for classification
            w, h = x2 - x1, y2 - y1
            pad = int(max(w, h) * 0.05)
            x1_crop = max(0, x1 - pad)
            y1_crop = max(0, y1 - pad)
            x2_crop = min(img_w, x2 + pad)
            y2_crop = min(img_h, y2 + pad)
            
            # Crop and classify (use padded coords for better classification)
            face = pil_image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
            
            # Anti-spoofing check
            is_real = True
            spoof_score = 1.0
            spoof_reason = ""
            
            if ANTISPOOF_ENABLED:
                is_real, spoof_score, spoof_reason = anti_spoof.check_liveness(face)
                is_real = spoof_score >= ANTISPOOF_THRESHOLD
            
            # If spoof detected, mark as unauthorized
            if not is_real:
                name = "Spoof"
                full_label = f"⚠️ SPOOF DETECTED"
                role = "Spoof"
                authorized = False
                confidence = spoof_score
                
                result = {
                    "name": name,
                    "role": role,
                    "full_label": full_label,
                    "authorized": False,
                    "confidence": round(spoof_score * 100, 2),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "detection_score": round(float(prob), 2),
                    "is_spoof": True,
                    "spoof_reason": spoof_reason
                }
                results.append(result)
                
                if save_log:
                    log_access("Spoof", "Spoof", False, spoof_score)
                continue
            
            # Normal classification for real faces
            face_tensor = self.transform(face).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(face_tensor).logits
                probs_cls = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs_cls, 1)
            
            confidence = confidence.item()
            
            if confidence >= CONFIDENCE_THRESHOLD:
                name = CLASS_NAMES[predicted.item()]
                full_label, role, authorized = self.get_full_label(name)
            else:
                name = "Unknown"
                full_label = "Unknown (Guest)"
                role = "Guest"
                authorized = False
            
            result = {
                "name": name.capitalize(),
                "role": role,
                "full_label": full_label,
                "authorized": authorized,
                "confidence": round(confidence * 100, 2),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "detection_score": round(float(prob), 2),
                "is_spoof": False,
                "liveness_score": round(spoof_score * 100, 2)
            }
            results.append(result)
            
            # Log to database
            if save_log:
                log_access(name, role, authorized, confidence)
        
        return results
    
    def process_frame(self, frame):
        """Process video frame and return annotated frame"""
        if not self._initialized:
            self.initialize()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Detect faces using selected backend
        faces = self.detect_faces(pil_image)
        
        results = []
        
        for x1, y1, x2, y2, prob in faces:
            # Add minimal padding for classification only
            w, h = x2 - x1, y2 - y1
            pad = int(max(w, h) * 0.05)
            x1_crop = max(0, x1 - pad)
            y1_crop = max(0, y1 - pad)
            x2_crop = min(frame.shape[1], x2 + pad)
            y2_crop = min(frame.shape[0], y2 + pad)
            
            # Classify (use padded coords for better classification)
            face = pil_image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
            face_tensor = self.transform(face).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(face_tensor).logits
                probs_cls = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs_cls, 1)
            
            confidence = confidence.item()
            
            if confidence >= CONFIDENCE_THRESHOLD:
                name = CLASS_NAMES[predicted.item()]
                full_label, role, authorized = self.get_full_label(name)
            else:
                full_label = "Unknown (Guest)"
                authorized = False
            
            # Draw
            color = (0, 255, 0) if authorized else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{full_label} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            results.append({
                "full_label": full_label,
                "authorized": authorized,
                "confidence": confidence
            })
        
        return frame, results


# Singleton instance
face_model = FaceRecognitionModel()
