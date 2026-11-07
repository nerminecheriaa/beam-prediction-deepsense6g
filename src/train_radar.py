"""Partie B : boucle d'entraînement générique (CNN radar ou fusion)."""
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.evaluate import topk_accuracy


def _loader(tensors, bs, shuffle):
    return DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def predict(model, tensors, device="cpu", bs=128):
    """tensors = (entrées..., labels). Retourne les logits (N,64)."""
    model.eval().to(device)
    out = []
    for *xs, _ in _loader(tensors, bs, False):
        out.append(model(*[x.float().to(device) for x in xs]).cpu())
    return torch.cat(out)


def fit(model, train_t, val_t, epochs=30, lr=1e-3, bs=32, device="cpu"):
    """train_t / val_t = (entrées..., labels). Garde le meilleur état selon le Top-1 de validation."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for *xs, y in _loader(train_t, bs, True):
            opt.zero_grad()
            loss = lossf(model(*[x.float().to(device) for x in xs]), y.long().to(device))
            loss.backward()
            opt.step()
        acc = topk_accuracy(predict(model, val_t, device), val_t[-1])
        print(f"epoch {ep+1:02d}  loss {loss.item():.3f}  val top1 {acc['top1']:.3f}  top3 {acc['top3']:.3f}")
        if acc["top1"] > best:
            best, best_state = acc["top1"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model