"""Métriques d'évaluation de la prédiction de faisceau.
Partie A : précision Top-K. Partie B : overhead vs balayage exhaustif.
"""
import torch

N_BEAMS = 64


def topk_accuracy(logits, labels, ks=(1, 3, 5)):
    """Précision Top-K.

    logits : tableau (N, 64) des scores du modèle pour chaque faisceau
    labels : tableau (N,) des vrais faisceaux (0..63)
    Retourne un dict {"top1": ..., "top3": ..., "top5": ...}
    """
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels).view(-1, 1)
    # indices des K meilleurs faisceaux pour chaque échantillon
    topk_idx = logits.topk(max(ks), dim=1).indices      # (N, Kmax)
    correct = topk_idx.eq(labels)                        # (N, Kmax) booléens
    return {f"top{k}": correct[:, :k].any(dim=1).float().mean().item() for k in ks}


# ---- Partie B  : overhead ----