"""Partie B : CNN radar, fusion par concaténation, fusion tardive."""
import torch
import torch.nn as nn


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class RadarCNN(nn.Module):
    """Entrée (B,1,H,W) -> 128 features -> 64 logits."""
    def __init__(self, n_beams=64):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64), _block(64, 128),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(128, n_beams)

    def forward(self, x):
        return self.head(self.features(x))


class FusionNet(nn.Module):
    """Fusion par concaténation : features caméra (512, pré-calculées par A) + features radar (128)."""
    def __init__(self, radar_cnn, cam_dim=512, n_beams=64):
        super().__init__()
        self.radar = radar_cnn
        self.head = nn.Sequential(nn.Linear(cam_dim + 128, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, n_beams))

    def forward(self, cam_feat, radar_map):
        return self.head(torch.cat([cam_feat, self.radar.features(radar_map)], dim=1))


def late_fusion(cam_logits, radar_logits, w_cam=0.5):
    """Fusion tardive : moyenne pondérée des probabilités. Retourne des 'scores' (N,64)."""
    cam_logits, radar_logits = torch.as_tensor(cam_logits), torch.as_tensor(radar_logits)
    return w_cam * cam_logits.softmax(1) + (1 - w_cam) * radar_logits.softmax(1)