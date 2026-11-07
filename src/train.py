"""
Boucle d'entraînement générique : CrossEntropy, AdamW, cosinus, précision mixte (AMP),
sélection du meilleur modèle sur la VALIDATION (le test n'est jamais regardé ici).
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .dataset import make_loader, to_model_input
from .evaluate import topk_accuracy

DEFAULT_CFG = dict(
    epochs=20,          # nombre de passages sur le jeu d'entraînement
    batch_size=64,
    lr=1e-4,            # petit taux : on affine, on ne veut pas détruire le pré-entraînement
    weight_decay=1e-4,  # régularisation L2 (AdamW)
    seed=42,
    augment=False,      # A2 : augmentation photométrique + cutout (train seulement)
    soft_sigma=0.0,     # A2 : 0 = étiquettes classiques ; >0 = étiquettes douces (gaussienne, en faisceaux)
)
# Avec les valeurs par défaut, l'entraînement est équivalent à la v1 / A1 (étiquettes classiques, sans augmentation).


def soft_targets(y, n_beams=64, sigma=1.0):
    """Cible douce : cloche gaussienne centrée sur le bon faisceau (somme = 1)."""
    beams = torch.arange(n_beams, device=y.device, dtype=torch.float32).view(1, -1)
    t = torch.exp(-0.5 * ((beams - y.view(-1, 1).float()) / sigma) ** 2)
    return t / t.sum(dim=1, keepdim=True)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, images, device, batch_size=128, return_features=False):
    """Logits (et features 512) pour un tenseur d'images uint8, dans le même ordre."""
    model.eval()
    logits_l, feats_l = [], []
    for i in range(0, len(images), batch_size):
        xb = to_model_input(images[i:i + batch_size], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            lo, fe = model(xb, return_features=True)
        logits_l.append(lo.float().cpu())
        feats_l.append(fe.float().cpu())
    logits = torch.cat(logits_l)
    return (logits, torch.cat(feats_l)) if return_features else logits


def train_model(model, x_tr, y_tr, x_va, y_va, device, ckpt_path, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    set_seed(cfg["seed"])
 
    loader = make_loader(x_tr, y_tr, cfg["batch_size"], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
 
    history, best_top1 = [], -1.0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tot_loss, correct, n = time.time(), 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = to_model_input(xb, device, augment=cfg["augment"]), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(xb)
            if cfg["soft_sigma"] > 0:
                target = soft_targets(yb, logits.size(1), cfg["soft_sigma"])
                loss = -(target * logits.float().log_softmax(dim=1)).sum(dim=1).mean()
            else:
                loss = criterion(logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        lr_used = opt.param_groups[0]["lr"]
        sched.step()
 
        val_logits = predict(model, x_va, device)
        val_loss = criterion(val_logits, y_va).item()
        acc = topk_accuracy(val_logits, y_va)
 
        rec = dict(epoch=ep, lr=lr_used, train_loss=tot_loss / n, train_top1=correct / n,
                   val_loss=val_loss, val_top1=acc["top1"], val_top3=acc["top3"],
                   val_top5=acc["top5"], seconds=time.time() - t0)
        history.append(rec)
 
        star = ""
        if acc["top1"] > best_top1:
            best_top1 = acc["top1"]
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "cfg": cfg, "val": acc}, ckpt_path)
            star = "  <- meilleur (sauvegardé)"
        print(f"ep {ep:2d} | train loss {rec['train_loss']:.3f} top1 {rec['train_top1']:.3f} | "
              f"val loss {val_loss:.3f} top1 {acc['top1']:.3f} top3 {acc['top3']:.3f} "
              f"top5 {acc['top5']:.3f} | {rec['seconds']:.1f}s{star}")
 
    with open(os.path.splitext(ckpt_path)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)
    return history