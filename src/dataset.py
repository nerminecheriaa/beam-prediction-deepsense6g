"""
Étape 2.1 : images caméra du scénario 32.
  - redimensionnement UNE SEULE FOIS en 224x224,
  - stockage en mémoire sous forme de tenseur uint8 (environ 0,5 Go pour 3 235 images),
  - sauvegarde du cache sur Drive pour ne plus jamais relire les JPEG,
  - découpage train/val/test d'après splits.csv,
  - normalisation ImageNet faite sur le GPU, par batch.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image

IMG_SIZE = (224, 224)                    # (hauteur, largeur) : entrée standard des CNN ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def full_path(root, rel_path):
    """'./unit1/camera_data/image_1.jpg' -> '<root>/unit1/camera_data/image_1.jpg'"""
    return os.path.join(root, rel_path.replace("./", "", 1))


def _load_and_resize(path, size=IMG_SIZE):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)            # (H, W, 3)


def build_image_cache(splits_csv, data_root, cache_path, size=IMG_SIZE, workers=8):
    """Construit (ou recharge) le cache des images redimensionnées.

    Retourne un dict :
        images : uint8 (N, 3, H, W)   index : (N,)   labels : (N,)  [0..63]
        split  : liste de 'train' / 'val' / 'test'
    Les lignes sont triées par 'index' : l'ordre est le même pour toute l'équipe.
    """
    df = pd.read_csv(splits_csv).sort_values("index").reset_index(drop=True)

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        same_rows = cache["index"].tolist() == df["index"].tolist()
        same_size = tuple(cache["images"].shape[2:]) == tuple(size)
        if same_rows and same_size:
            print(f"Cache rechargé depuis {cache_path} : {tuple(cache['images'].shape)}")
            return cache
        print("Cache incompatible avec splits.csv : reconstruction...")

    paths = [full_path(data_root, p) for p in df["unit1_rgb"]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images manquantes, ex. : {missing[0]}")

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        arrays = list(ex.map(lambda p: _load_and_resize(p, size), paths))
    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    print(f"{len(paths)} images redimensionnées en {time.time() - t0:.1f} s")

    cache = {
        "images": images,
        "index": torch.tensor(df["index"].values),
        "labels": torch.tensor(df["beam_label"].values, dtype=torch.long),
        "split": df["split"].tolist(),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Cache sauvegardé : {cache_path} ({os.path.getsize(cache_path) / 1e6:.0f} Mo)")
    return cache


def get_split(cache, split):
    """Images uint8, labels et index d'un ensemble ('train', 'val' ou 'test')."""
    mask = torch.tensor([s == split for s in cache["split"]])
    return cache["images"][mask], cache["labels"][mask], cache["index"][mask]


def make_loader(images, labels, batch_size=64, shuffle=False):
    """Tout est déjà en mémoire : pas besoin de workers."""
    ds = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=0, pin_memory=True)


def to_model_input(x_uint8, device):
    """uint8 (B,3,H,W) -> float normalisé ImageNet, directement sur le GPU."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = x_uint8.to(device, non_blocking=True).float().div_(255.0)
    return (x - mean) / std