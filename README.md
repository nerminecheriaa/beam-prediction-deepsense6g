# Prédiction de faisceau mmWave à partir de capteurs (DeepSense 6G, scénario 32)

Projet 04 du module Futurs Réseaux Mobiles (RT5, INSAT, 2026/2027).

Le but est de prédire, parmi les 64 faisceaux du codebook à 60 GHz, celui qui donne la plus forte puissance reçue, à partir de la caméra et du radar de la station de base, sans faire le balayage complet des 64 faisceaux. On compare une baseline caméra seule, un modèle radar seul et deux façons de fusionner caméra et radar, en Top-1, Top-3 et Top-5.

Binôme :
- Partie A (Nermine Cheriaa) : données, labels, découpage, baseline caméra, métrique Top-K
- Partie B (Mariem Logtari) : prétraitement radar, CNN radar, fusion, overhead

## Contenu du dépôt

```
notebooks/
  A_exploration.ipynb     étape 1 : données, labels, split par séquence, plancher "faisceau le plus fréquent"
  A_train_camera.ipynb    étape 2 : première baseline caméra (une graine)
  A_camera_seeds.ipynb    caméra sur 5 graines, variantes contre le sur-apprentissage, test final
  B_radar_fusion.ipynb    radar, fusion par concaténation, fusion tardive, overhead
src/
  dataset.py       cache des images 224x224, découpage, normalisation, augmentation photométrique
  models.py        CameraResNet18 (ResNet18 ImageNet, tête 64 faisceaux)
  train.py         boucle d'entraînement caméra (sélection sur la validation)
  evaluate.py      topk_accuracy
  radar.py         cube radar -> carte distance-angle (FFT)
  radar_models.py  RadarCNN, FusionNet, late_fusion
  train_radar.py   entraînement radar / fusion
  overhead.py      faisceaux testés et gain d'overhead
splits.csv         découpage train / val / test utilisé partout
requirements.txt
```

## Données

Scénario 32 de DeepSense 6G (enregistré de jour, avec caméra, radar, LiDAR et GPS). L'archive `scenario32.zip` (3,8 Go) se télécharge depuis le site DeepSense 6G ; le lien utilisé est dans la cellule 1 de `A_exploration.ipynb`.

Les notebooks supposent l'arborescence suivante sur Google Drive :

```
MyDrive/beam_data/scenario32.zip       archive d'origine
MyDrive/beam_data/camera/              sorties de la partie A
MyDrive/beam_data_B/                   sorties de la partie B
```

L'archive est décompressée dans `/content/data` au début de chaque session (lecture plus rapide que depuis Drive).

Quelques faits utiles sur le scénario :
- 3 235 échantillons répartis en 15 séquences (un passage du véhicule par séquence), environ 10,9 Hz ;
- images caméra 960x540 ; cubes radar (4 antennes, 256 échantillons, 250 chirps) ;
- le label est l'indice du faisceau de puissance maximale (0 à 63). Il vaut `unit1_beam - 1` dans le CSV d'origine, qui numérote de 1 à 64. Seuls les faisceaux 1 à 61 apparaissent, et le faisceau 16 représente 13,3 % des échantillons.

## Environnement

Tout a été exécuté sur Google Colab avec un GPU T4. Les bibliothèques de `requirements.txt` sont déjà installées sur Colab. En local, il faut PyTorch 2.x.

```
pip install -r requirements.txt
```

Les trois premières cellules de chaque notebook montent Drive, décompressent les données et clonent ce dépôt :

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/data
!unzip -q -n /content/drive/MyDrive/beam_data/scenario32.zip -d /content/data

!rm -rf /content/repo
!git clone -q https://github.com/nerminecheriaa/beam-prediction-deepsense6g.git /content/repo
import sys; sys.path.insert(0, "/content/repo")
```

Après une modification de `src/`, il faut redémarrer la session Colab avant de relancer ces cellules, sinon Python garde l'ancienne version des modules.

## Relancer les expériences

Les notebooks s'exécutent dans cet ordre. Chaque étape lit les fichiers produits par la précédente.

### 1. Données et découpage : `A_exploration.ipynb`

- calcule les labels à partir des 64 puissances (`argmax`) et vérifie l'écart de 1 avec `unit1_beam` ;
- vérifie la synchronisation (pas de fichier manquant, pas de temps médian 0,092 s) ;
- découpe **par séquence** et non par échantillon : à 10 Hz, deux images voisines sont presque identiques, un découpage aléatoire mettrait les mêmes scènes en train et en test ;
- écrit `splits.csv` (colonnes `index, seq_index, split, beam_label` et chemins des capteurs).

| Ensemble | Séquences | Échantillons |
|---|---|---|
| train | 1, 2, 5, 6, 7, 8, 10, 11, 12, 16 | 2 265 |
| val | 4, 13, 17 | 490 |
| test | 3, 14 | 480 |

`splits.csv` est déjà dans le dépôt : cette étape n'est à relancer que pour vérifier.

### 2. Baseline caméra : `A_train_camera.ipynb`

- les images sont redimensionnées une seule fois en 224x224 et gardées dans un cache (`camera/cam_cache_224.pt`, 487 Mo, environ 30 s à construire, rechargé ensuite en quelques secondes) ;
- ResNet18 pré-entraîné sur ImageNet, dernière couche remplacée par 64 sorties, réseau entier affiné ;
- AdamW (lr 1e-4, weight decay 1e-4), décroissance cosinus, 20 époques, batch 64, précision mixte ; environ 4 s par époque ;
- l'époque conservée est celle du meilleur Top-1 sur la validation ; le test n'intervient qu'à la fin.

Ce notebook correspond à la première version (une seule graine, 42). Les résultats de référence viennent de l'étape suivante.

### 3. Caméra sur 5 graines et réduction du sur-apprentissage : `A_camera_seeds.ipynb`

Toutes les méthodes du projet sont évaluées sur les mêmes graines (0 à 4) et rapportées en moyenne ± écart-type. Avec seulement deux séquences de test, une exécution isolée n'est pas fiable : le Top-1 de la caméra varie de 11 % à 20 % selon la graine.

Le notebook contient trois étapes :

- **A1** : la configuration de l'étape 2 sur 5 graines (`camera/seeds_baseline/`) ;
- **A2** : trois variantes, comparées uniquement sur la validation :
  - `aug` : variations de luminosité, contraste et saturation, et un carré gris placé au hasard (cutout) ;
  - `soft` : étiquettes douces, c'est-à-dire une gaussienne d'écart-type 1 faisceau autour du bon faisceau ;
  - `aug_soft` : les deux.

  Aucune transformation géométrique n'est utilisée : décaler ou retourner l'image déplacerait le véhicule, et donc le faisceau.
- **A3** : la variante retenue (`aug_soft`, meilleure somme Top-1 + Top-3 + Top-5 sur la validation, critère fixé avant les essais) est évaluée une fois sur le test. Ses sorties sont copiées dans `camera/seeds_final/`.

Durée : environ 10 min pour A1 et 30 min pour A2. Les graines déjà calculées sont sautées si Colab se déconnecte.

Les options correspondantes sont dans `src/train.py` :

```python
train_model(model, x_tr, y_tr, x_val, y_val, device, ckpt,
            cfg={"seed": s, "augment": True, "soft_sigma": 1.0})   # aug_soft
```

Avec les valeurs par défaut (`augment=False`, `soft_sigma=0`), l'entraînement est celui de l'étape 2.

### 4. Radar et fusion : `B_radar_fusion.ipynb`

<!-- Partie B : à compléter par Mériem (ordre des cellules, durées, fichiers produits) -->

- carte distance-angle 64x256 : FFT distance, FFT Doppler avec suppression des composantes de vitesse nulle (échos fixes), FFT angle sur les 4 antennes ; cache `beam_data_B/radar_maps_dop.npy` ;
- CNN radar, puis deux fusions avec la caméra :
  - concaténation des features caméra (ResNet18 ImageNet non affiné) et radar ;
  - fusion tardive : `w * softmax(caméra) + (1 - w) * softmax(radar)`, avec `w` choisi sur la validation ;
- la caméra est lue dans `camera/seeds_final/camera_outputs_seed{s}.pt` ; l'alignement avec le radar se fait sur la colonne `index` de `splits.csv`.

## Fichiers échangés entre les deux parties

`camera/seeds_final/camera_outputs_seed{s}.pt` (un fichier par graine) contient, pour `train`, `val` et `test` :

| Clé | Forme | Contenu |
|---|---|---|
| `logits` | N x 64 | scores de la caméra, avant softmax |
| `features` | N x 512 | sortie du ResNet18 avant la dernière couche |
| `labels` | N | faisceau optimal (0 à 63) |
| `index` | N | colonne `index` de `splits.csv`, lignes triées par `index` |

```python
co = torch.load(".../camera_outputs_seed0.pt", weights_only=False)
df = pd.read_csv("splits.csv")
m = (df["split"] == "test").values
assert (df["index"].values[m] == co["test"]["index"].numpy()).all()
```

## Résultats

Test (480 échantillons, séquences 3 et 14), moyenne ± écart-type sur 5 graines, en %.

| Méthode | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Faisceau le plus fréquent du train | 13,1 | 36,3 | 42,3 |
| Caméra, sans régularisation (A1) | 16,1 ± 3,7 | 33,8 ± 4,8 | 45,9 ± 4,1 |
| Caméra, `aug_soft` (finale) | 16,2 ± 2,9 | 38,0 ± 2,4 | 49,3 ± 2,3 |
| Radar seul | à compléter | | |
| Fusion par concaténation | à compléter | | |
| Fusion tardive | à compléter | | |

Comparaison des variantes caméra sur la validation (5 graines) :

| Variante | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| A1 | 26,0 ± 2,1 | 49,7 ± 3,3 | 60,6 ± 3,2 |
| `aug` | 31,2 ± 2,1 | 57,4 ± 4,1 | 69,6 ± 3,5 |
| `soft` | 23,9 ± 3,9 | 53,3 ± 5,2 | 68,0 ± 3,6 |
| `aug_soft` | 28,3 ± 2,7 | 57,9 ± 3,7 | 72,6 ± 3,0 |

Ce qu'on en retient pour la caméra :
- l'augmentation apporte l'essentiel du gain en validation ; `aug` et `aug_soft` sont à égalité ;
- sur le test, `aug_soft` gagne en Top-3 et Top-5 par rapport à A1, pas en Top-1, et les résultats varient deux fois moins d'une graine à l'autre ;
- le gain est plus faible sur le test que sur la validation : le test ne compte que deux séquences, dont l'une représente 78 % des échantillons.

**Overhead.** Prédire le Top-K revient à tester K faisceaux au lieu de 64, soit 1 - K/64 de mesures en moins (98,4 % pour K = 1, 95,3 % pour K = 3, 92,2 % pour K = 5). Le nombre de faisceaux à tester pour trouver le bon dans 50 % des cas est calculé dans `B_radar_fusion.ipynb` (à compléter).

## Limites

- Un seul scénario, 15 séquences au total : 10 pour l'entraînement, 2 pour le test. Les écarts de quelques points entre méthodes restent dans la variabilité.
- La caméra ne sait pas quel véhicule porte l'émetteur quand plusieurs objets sont visibles.
- Le radar n'a que 4 antennes, donc une résolution angulaire faible.
- Les chemins sont ceux de Colab et de notre Drive (`/content/...`, `MyDrive/beam_data/...`) ; ils sont à adapter pour une autre installation.

## Référence

A. Alkhateeb et al., "DeepSense 6G: A large-scale real-world multi-modal sensing and communication dataset," IEEE Communications Magazine, vol. 61, no. 9, pp. 122-128, 2023.
