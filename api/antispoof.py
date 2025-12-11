"""
============================================================
ANTI-SPOOFING MODULE - Silent Face Anti-Spoofing
============================================================
Deteksi apakah wajah berasal dari foto/layar HP atau wajah asli.

Menggunakan:
1. MiniFASNet Deep Learning Model (Primary)
2. Texture Analysis (LBP) - Backup/ensemble
3. Color Analysis - Backup/ensemble
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "resources", "anti_spoof_models")


# ============================================================
# MiniFASNet Model Architecture
# ============================================================

class Conv2d_cd(nn.Module):
    """Central Difference Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if self.theta == 0:
            return out_normal
        
        kernel_diff = self.conv.weight.sum(2).sum(2)
        out_diff = F.conv2d(input=x, weight=kernel_diff[:, :, None, None], 
                           bias=self.conv.bias, stride=self.conv.stride, padding=0)
        return out_normal - self.theta * out_diff


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention)


class MiniFASNetV2(nn.Module):
    """MiniFASNet V2 for Anti-Spoofing"""
    def __init__(self, conv_type='Conv2d_cd', num_classes=3):
        super().__init__()
        
        Conv2d = Conv2d_cd if conv_type == 'Conv2d_cd' else nn.Conv2d
        
        self.conv1 = nn.Sequential(
            Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            Conv2d(128, 196, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            Conv2d(196, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.sa = SpatialAttention()
        
        self.downsample = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 10 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        attention = self.sa(x)
        x = x * attention
        
        x = self.downsample(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MiniFASNetV1SE(nn.Module):
    """MiniFASNet V1 with Squeeze-and-Excitation"""
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 196, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # SE block
        self.se_fc1 = nn.Linear(128, 32)
        self.se_fc2 = nn.Linear(32, 128)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 10 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # SE attention
        b, c, _, _ = x.size()
        se = F.adaptive_avg_pool2d(x, 1).view(b, c)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se)).view(b, c, 1, 1)
        x = x * se
        
        x = self.downsample(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# Anti-Spoofing Class
# ============================================================

class AntiSpoofing:
    """Anti-spoofing detector menggunakan Silent Face Anti-Spoofing"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.model_configs = [
            # (model_path, input_size, model_class)
            ("2.7_80x80_MiniFASNetV2.pth", 80, MiniFASNetV2),
            ("4_0_0_80x80_MiniFASNetV1SE.pth", 80, MiniFASNetV1SE),
        ]
        self._initialized = False
        
        # Fallback thresholds for traditional methods
        self.lbp_threshold = 0.35
        self.color_threshold = 0.4
        
    def initialize(self):
        """Load anti-spoofing models"""
        if self._initialized:
            return
            
        print("[AntiSpoof] Initializing Silent Face Anti-Spoofing...")
        
        for model_name, input_size, model_class in self.model_configs:
            model_path = os.path.join(MODEL_DIR, model_name)
            
            if os.path.exists(model_path):
                try:
                    model = model_class(num_classes=3)
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # Handle different state dict formats
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    
                    # Try to load, ignore mismatched keys
                    try:
                        model.load_state_dict(state_dict, strict=False)
                    except Exception as e:
                        print(f"[AntiSpoof] Warning loading {model_name}: {e}")
                        continue
                    
                    model.to(self.device)
                    model.eval()
                    self.models.append((model, input_size))
                    print(f"[AntiSpoof] Loaded {model_name}")
                except Exception as e:
                    print(f"[AntiSpoof] Failed to load {model_name}: {e}")
            else:
                print(f"[AntiSpoof] Model not found: {model_path}")
        
        if not self.models:
            print("[AntiSpoof] No deep learning models loaded, using traditional methods only")
        else:
            print(f"[AntiSpoof] Loaded {len(self.models)} models")
        
        self._initialized = True
        
    def check_liveness(self, face_img, face_id=None):
        """
        Main function: Cek apakah wajah real atau spoof
        Menggunakan ensemble dari deep learning + traditional methods
        
        Returns:
            tuple: (is_real, confidence, reason)
        """
        # Initialize models if not done
        if not self._initialized:
            self.initialize()
        
        if face_img is None:
            return False, 0.0, "Invalid image"
        
        # Convert to numpy if PIL Image
        if hasattr(face_img, 'convert'):
            face_np = np.array(face_img)
        else:
            face_np = face_img
        
        if face_np.size == 0:
            return False, 0.0, "Empty image"
            
        # Ensure BGR format for OpenCV
        if len(face_np.shape) == 3 and face_np.shape[2] == 3:
            face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        else:
            face_bgr = face_np
        
        scores = []
        reasons = []
        
        # ============================================================
        # 1. Deep Learning Anti-Spoofing (Primary - highest weight)
        # ============================================================
        if self.models:
            dl_score, dl_reason = self._check_deep_learning(face_bgr)
            scores.append(('deep_learning', dl_score, 0.6))  # 60% weight
            if dl_score < 0.5:
                reasons.append(dl_reason)
        
        # ============================================================
        # 2. Traditional Methods (Backup/Ensemble)
        # ============================================================
        
        # Texture Analysis (LBP-based)
        texture_score, texture_reason = self._check_texture(face_bgr)
        scores.append(('texture', texture_score, 0.15))
        if texture_score < 0.5:
            reasons.append(texture_reason)
        
        # Color Distribution Analysis
        color_score, color_reason = self._check_color_distribution(face_bgr)
        scores.append(('color', color_score, 0.1))
        if color_score < 0.5:
            reasons.append(color_reason)
        
        # Frequency Analysis (Moire pattern detection)
        freq_score, freq_reason = self._check_frequency(face_bgr)
        scores.append(('frequency', freq_score, 0.1))
        if freq_score < 0.5:
            reasons.append(freq_reason)
        
        # Edge Analysis
        edge_score, edge_reason = self._check_edges(face_bgr)
        scores.append(('edge', edge_score, 0.05))
        if edge_score < 0.5:
            reasons.append(edge_reason)
        
        # ============================================================
        # Calculate Final Score (Weighted Average)
        # ============================================================
        total_weight = sum(w for _, _, w in scores)
        final_score = sum(s * w for _, s, w in scores) / total_weight if total_weight > 0 else 0.5
        
        # Determine if real
        is_real = final_score >= 0.5
        
        if is_real:
            reason = "Real face detected"
        else:
            reason = "; ".join(reasons[:2]) if reasons else "Suspected spoof"
        
        return is_real, final_score, reason
    
    def _check_deep_learning(self, face_bgr):
        """
        Deep learning anti-spoofing menggunakan MiniFASNet
        Returns: (score, reason)
        """
        if not self.models:
            return 0.5, "No model"
        
        predictions = []
        
        for model, input_size in self.models:
            try:
                # Preprocess image
                img = cv2.resize(face_bgr, (input_size, input_size))
                img = img.astype(np.float32) / 255.0
                
                # Normalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img - mean) / std
                
                # Convert to tensor (B, C, H, W)
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                # Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = F.softmax(output, dim=1)
                    
                    # Class 1 is typically "real" in Silent Face Anti-Spoofing
                    # Class 0, 2 are different types of spoof
                    real_prob = probs[0, 1].item()
                    predictions.append(real_prob)
                    
            except Exception as e:
                print(f"[AntiSpoof] Inference error: {e}")
                continue
        
        if not predictions:
            return 0.5, "Inference failed"
        
        # Average predictions from all models
        avg_score = sum(predictions) / len(predictions)
        
        if avg_score < 0.5:
            return avg_score, "DL: Spoof detected"
        return avg_score, "OK"
    
    def _check_texture(self, face_bgr):
        """
        Analisis tekstur menggunakan Local Binary Pattern (LBP)
        Foto dari layar cenderung punya tekstur yang lebih smooth/uniform
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent analysis
        gray = cv2.resize(gray, (128, 128))
        
        # Calculate LBP
        lbp = self._compute_lbp(gray)
        
        # Calculate histogram variance
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        
        # Real faces have more texture variation
        variance = np.var(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalize scores
        variance_score = min(variance * 1000, 1.0)  # Scale variance
        entropy_score = entropy / 8.0  # Max entropy for 256 bins is 8
        
        score = (variance_score + entropy_score) / 2
        
        if score < self.lbp_threshold:
            return score, "Texture too smooth (possible screen)"
        return score, "OK"
    
    def _compute_lbp(self, gray):
        """Compute Local Binary Pattern"""
        rows, cols = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        return lbp
    
    def _check_color_distribution(self, face_bgr):
        """
        Analisis distribusi warna
        Layar HP cenderung punya warna yang lebih saturated/artificial
        """
        # Convert to HSV
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Check saturation distribution
        sat_mean = np.mean(s)
        sat_std = np.std(s)
        
        # Check value (brightness) distribution
        val_mean = np.mean(v)
        val_std = np.std(v)
        
        # Screens often have very uniform brightness or oversaturated colors
        # Real skin has natural variation
        
        # Score based on natural variation
        sat_score = min(sat_std / 50.0, 1.0)  # Expect std around 30-50
        val_score = min(val_std / 40.0, 1.0)  # Expect std around 30-40
        
        # Check for unnatural saturation (too high = screen)
        if sat_mean > 150:
            sat_score *= 0.5
        
        score = (sat_score + val_score) / 2
        
        if score < self.color_threshold:
            return score, "Unnatural color distribution"
        return score, "OK"
    
    def _check_reflection(self, face_bgr):
        """
        Deteksi refleksi/glare dari layar
        Layar HP sering punya hotspot/refleksi terang
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        
        # Find very bright spots (potential screen glare)
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
        
        # Find very dark spots
        _, dark_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
        
        # Screens often have extreme contrast (very bright + very dark areas)
        extreme_ratio = bright_ratio + dark_ratio
        
        # Also check for rectangular bright regions (screen reflection)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_score = 1.0
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box.astype(np.int32))
                cnt_area = cv2.contourArea(cnt)
                if box_area > 0 and cnt_area / box_area > 0.8:  # Very rectangular
                    rectangular_score *= 0.7
        
        # Score: less extreme = more likely real
        score = (1.0 - min(extreme_ratio * 5, 1.0)) * rectangular_score
        
        if score < self.reflection_threshold:
            return score, "Screen reflection detected"
        return max(score, 0.3), "OK"  # Don't penalize too much
    
    def _check_frequency(self, face_bgr):
        """
        Analisis frekuensi untuk deteksi Moire pattern
        Foto dari layar sering punya pola Moire
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        
        # Apply FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Log transform for better visualization
        magnitude = np.log1p(magnitude)
        
        # Check for periodic patterns (Moire)
        # Real faces should have more random frequency distribution
        center = magnitude.shape[0] // 2
        
        # Analyze high frequency components
        high_freq_region = magnitude.copy()
        high_freq_region[center-10:center+10, center-10:center+10] = 0
        
        # Check for peaks in high frequency (Moire indicator)
        high_freq_std = np.std(high_freq_region)
        high_freq_max = np.max(high_freq_region)
        
        # Moire patterns create distinct peaks
        peak_ratio = high_freq_max / (high_freq_std + 1e-10)
        
        # Score: lower peak ratio = more natural
        score = 1.0 - min(peak_ratio / 20.0, 0.8)
        
        if score < 0.4:
            return score, "Moire pattern detected"
        return score, "OK"
    
    def _check_edges(self, face_bgr):
        """
        Analisis edge sharpness
        Foto dari layar punya edge characteristics berbeda
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        
        # Compute Laplacian (edge detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        
        # Compute Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_var = sobel_mag.var()
        
        # Real faces have natural edge variation
        # Screens might have very sharp or very blurry edges
        
        # Normalize scores
        lap_score = min(lap_var / 500.0, 1.0)
        sobel_score = min(sobel_var / 2000.0, 1.0)
        
        score = (lap_score + sobel_score) / 2
        
        # Penalize if too sharp (screen) or too blurry (low quality spoof)
        if lap_var > 2000 or lap_var < 50:
            score *= 0.7
        
        if score < 0.3:
            return score, "Unnatural edge pattern"
        return score, "OK"


# Singleton instance
anti_spoof = AntiSpoofing()
