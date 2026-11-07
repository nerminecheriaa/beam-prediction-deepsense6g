"""
Personne A : CameraResNet18 (baseline caméra)
Personne B : ajoutera ici le CNN radar et le modèle de fusion.
"""
import torch
import torch.nn as nn
from torchvision import models

N_BEAMS = 64


class CameraResNet18(nn.Module):
    """ResNet18 pré-entraîné sur ImageNet, tête remplacée par 64 classes (faisceaux).

    image (3x224x224) -> backbone ResNet18 -> vecteur de features (512) -> Linear -> 64 logits
    Le vecteur de 512 features est celui que la collègue utilisera pour la fusion.
    """

    def __init__(self, n_beams=N_BEAMS, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feat_dim = backbone.fc.in_features          # 512
        backbone.fc = nn.Identity()                       # on retire la tête ImageNet (1000 classes)
        self.backbone = backbone
        self.head = nn.Linear(self.feat_dim, n_beams)     # nouvelle tête : 64 faisceaux

    def forward(self, x, return_features=False):
        feats = self.backbone(x)                          # (B, 512)
        logits = self.head(feats)                         # (B, 64)
        return (logits, feats) if return_features else logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)