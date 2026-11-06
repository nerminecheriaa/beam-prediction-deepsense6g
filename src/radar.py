"""Partie B : prétraitement radar -> cartes distance-angle (FFT)."""
import os
import numpy as np
from tqdm import tqdm


def full_path(root, p):
    """'./unit1/radar_data/radar_data_1.npy' -> '<root>/unit1/radar_data/radar_data_1.npy'"""
    return os.path.join(root, p.replace("./", "", 1))


def radar_to_range_angle(cube, n_angle=64):
    """cube (n_rx, n_samples, n_chirps) -> carte (n_angle, n_samples) normalisée dans [0, 1].

    1) FFT sur les samples  -> distance
    2) FFT sur les antennes -> angle (zero-padding à n_angle points)
    3) moyenne du module sur les chirps, puis log et normalisation
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Cube radar attendu en 3D (antennes, samples, chirps), reçu {cube.shape}")
    x = np.fft.fft(cube, axis=1)
    x = np.fft.fftshift(np.fft.fft(x, n=n_angle, axis=0), axes=0)
    ra = np.log1p(np.abs(x).mean(axis=2))
    ra = (ra - ra.min()) / (ra.max() - ra.min() + 1e-8)
    return ra.astype(np.float32)


def build_radar_cache(df, root, out_path):
    """Calcule la carte de chaque ligne de df (dans l'ordre de splits.csv) et sauvegarde (N,1,H,W) en float16."""
    maps = [radar_to_range_angle(np.load(full_path(root, p))) for p in tqdm(df["unit1_radar"])]
    maps = np.stack(maps)[:, None].astype(np.float16)
    np.save(out_path, maps)
    return maps