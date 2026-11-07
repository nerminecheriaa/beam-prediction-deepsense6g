"""Partie B : prétraitement radar -> cartes distance-angle (FFT)."""
import os
import numpy as np
from tqdm import tqdm


def full_path(root, p):
    """'./unit1/radar_data/radar_data_1.npy' -> '<root>/unit1/radar_data/radar_data_1.npy'"""
    return os.path.join(root, p.replace("./", "", 1))


def radar_to_range_angle(cube, n_angle=64, remove_static=True):
    """cube (n_rx, n_samples, n_chirps) -> carte (n_angle, n_samples) normalisée dans [0, 1].

    0) (remove_static) on retire la moyenne sur les chirps : supprime les échos fixes
       (fuite émetteur, murs...) qui écrasent la cible en mouvement
    1) FFT sur les samples  -> distance
    2) FFT sur les antennes -> angle (zero-padding à n_angle points)
    3) moyenne du module sur les chirps, log, puis normalisation (écrêtage au 99,5e percentile)
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Cube radar attendu en 3D (antennes, samples, chirps), reçu {cube.shape}")
    if remove_static:
        cube = cube - cube.mean(axis=2, keepdims=True)
    x = np.fft.fft(cube, axis=1)
    x = np.fft.fftshift(np.fft.fft(x, n=n_angle, axis=0), axes=0)
    ra = np.log1p(np.abs(x).mean(axis=2))
    lo, hi = ra.min(), np.percentile(ra, 99.5)
    ra = np.clip((ra - lo) / (hi - lo + 1e-8), 0, 1)
    return ra.astype(np.float32)


def build_radar_cache(df, root, out_path, remove_static=True):
    """Calcule la carte de chaque ligne de df (dans l'ordre de splits.csv) et sauvegarde (N,1,H,W) en float16."""
    maps = [radar_to_range_angle(np.load(full_path(root, p)), remove_static=remove_static) for p in tqdm(df["unit1_radar"])]
    maps = np.stack(maps)[:, None].astype(np.float16)
    np.save(out_path, maps)
    return maps