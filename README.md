# BraTS – Unsupervised Anomaly Detection (3D)

Dieses Repository enthält mehrere Pipelines für unüberwachtes Anomalie-Detektion auf dem BraTS-Datensatz (Hirntumore) auf Patch-Basis in 3D. Implementiert sind u. a. Autoencoder (AE), VQ-VAE, f-AnoGAN (GAN), Diffusion (DDPM) sowie feature-basierte Verfahren (Anatomix+KNN, VQ-VAE-Encoder+KNN).

## Installation

```bash
# ins Repo wechseln
cd Brats

# Python-Abhängigkeiten installieren (idealerweise in einer virtuellen Umgebung)
pip install -r requirements.txt
```

Hinweise:
- Für schnellere Laufzeiten wird eine GPU empfohlen. Die Abhängigkeit `faiss-gpu` benötigt eine passende CUDA-Umgebung.
- Getestet wurde unter Linux und Python 3.x.

## Datensatz

Standardmäßig wird der BraTS-Datensatz relativ zum Projekt-Root erwartet:

```
./datasets/BraTS2025-GLI-PRE-Challenge-TrainingData/
  ├─ <SubjectID_1>/
  │    ├─ *-t1c.nii.gz
  │    ├─ *-seg.nii.gz (oder *seg.nii.gz)
  │    └─ ...
  ├─ <SubjectID_2>/
  └─ ...
```

Der Pfad kann mit `--dataset_path` überschrieben werden (absolut oder relativ zum Projekt-Root).

## Gemeinsame Konzepte

- Patches: Es werden 3D-Patches aus den Volumina extrahiert. Normale Patches stammen aus gesundem Gewebe; anomale Patches werden über Segmentationslabels definiert.
- Anomalie-Labels: Es werden die BraTS-Labels verwendet: `1 = NCR/NET`, `2 = ED`, `4 = ET`. Mit `--anomaly_labels` kann die Auswahl gesteuert werden.
- Splits: Es wird auf Subjekt-Ebene gesplittet, um Leakage zu vermeiden (train/val/test nach Subjekten).
- Ergebnisse: Die meisten Skripte legen automatisch einen eindeutigen Ordner unter `results/` an, z. B. `results/results_ae_brats`, `results/results_vqvae_brats` usw. Die wichtigsten Artefakte sind Metriken (`evaluation_results.txt`), Trainingskurven und ggf. Modellgewichte (`best_*.pth`).

Wichtig: In den meisten Skripten wird `--output_dir` intern durch einen automatisch erzeugten Unterordner unter `results/` ersetzt. Sie finden den tatsächlichen Pfad im Konsolen-Log. Das Diffusionsskript (`diffusion_brats.py`) verwendet den angegebenen `--output_dir` direkt.

## Start der Modelle

Alle Befehle werden aus dem Projekt-Root ausgeführt.

### 1) Autoencoder (AE)

Script: `src/ae_brats.py`

```bash
python src/ae_brats.py \
  --num_subjects 100 \
  --patch_size 32 \
  --patches_per_volume 50 \
  --batch_size 8 \
  --num_epochs 100 \
  --learning_rate 5e-5 \
  --latent_dim 256 \
  --dataset_path datasets/BraTS2025-GLI-PRE-Challenge-TrainingData \
  --max_normal_to_anomaly_ratio 3 \
  --min_tumor_ratio_in_patch 0.05 \
  --anomaly_labels 1 2 4 \
  -v
```

Sinnvolle Argumente (mit Defaults):
- `--num_subjects` (None): Anzahl genutzter Subjekte (None = alle)
- `--patch_size` (32): Kantenlänge des 3D-Patches
- `--patches_per_volume` (50): Anzahl Patches pro Volumen
- `--batch_size` (8)
- `--num_epochs` (100)
- `--learning_rate` (5e-5)
- `--latent_dim` (256)
- `--dataset_path` (`datasets/BraTS2025-...`)
- `--max_normal_to_anomaly_ratio` (3): balanciert Normale vs. Anomale Patches
- `--min_tumor_ratio_in_patch` (0.05): Mindesttumoranteil, damit ein Patch als anomal gilt
- `--anomaly_labels` (`1 2 4`): Auswahl der Tumorlabels
- `-v/--verbose`: detailliertere Logs

Ergebnisse u. a.: `best_autoencoder_3d.pth`, `evaluation_results.txt`, PR-/ROC-Kurven, Rekonstruktionsfehler-Histogramm, Latent-Visualisierung.

### 2) VQ-VAE

Script: `src/vqvae_brats.py`

```bash
python src/vqvae_brats.py \
  --num_subjects 100 \
  --patch_size 32 \
  --patches_per_volume 50 \
  --batch_size 8 \
  --num_epochs 100 \
  --learning_rate 5e-5 \
  --embedding_dim 128 \
  --codebook_size 512 \
  --commitment_beta 0.25 \
  --dataset_path datasets/BraTS2025-GLI-PRE-Challenge-TrainingData \
  --max_normal_to_anomaly_ratio 3 \
  --min_tumor_ratio_in_patch 0.05 \
  --anomaly_labels 1 2 4 \
  -v
```

Zusätzliche Argumente:
- `--embedding_dim` (128)
- `--codebook_size` (512)
- `--commitment_beta` (0.25)

Ergebnisse u. a.: `best_vqvae_3d.pth`, `evaluation_results.txt`, Trainingskurven, PR-/ROC-Kurven etc.

### 3) VQ-VAE Encoder + KNN

Script: `src/vqvae_encoder_knn_brats.py`

```bash
python src/vqvae_encoder_knn_brats.py \
  --num_subjects 100 \
  --patch_size 32 \
  --patches_per_volume 50 \
  --embedding_dim 128 \
  --k_neighbors 7 \
  --threshold_percentile 95 \
  --feature_pooling mean \
  --pretrained_weights path/to/best_vqvae_3d.pth \
  --dataset_path datasets/BraTS2025-GLI-PRE-Challenge-TrainingData \
  --max_normal_to_anomaly_ratio 3 \
  --min_tumor_ratio_in_patch 0.05 \
  --anomaly_labels 1 2 4 \
  -v
```

Zusätzliche Argumente:
- `--k_neighbors` (7): K in KNN
- `--threshold_percentile` (95.0): Perzentil der NORMAL-Scores (valid), bestimmt Schwellwert
- `--feature_pooling` (`mean` | `mean_max_std`): Pooling über Encoder-Featuremap
- `--pretrained_weights` (leer): optional Pfad zu VQ-VAE-Gewichten. Wenn leer, wird automatisch unter `results/results_vqvae_brats*` nach dem neuesten `best_vqvae_3d.pth` gesucht.

### 4) Anatomix + KNN

Script: `src/anatomixknn_brats.py`

```bash
python src/anatomixknn_brats.py \
  --num_subjects 100 \
  --patch_size 32 \
  --patches_per_volume 50 \
  --k_neighbors 7 \
  --threshold_percentile 90 \
  --select_topk_channels 0 \
  --channel_selection_mode supervised \
  --dataset_path datasets/BraTS2025-GLI-PRE-Challenge-TrainingData \
  --max_normal_to_anomaly_ratio 3 \
  --min_tumor_ratio_in_patch 0.05 \
  --anomaly_labels 1 4 \
  -v
```

Zusätzliche Argumente:
- `--select_topk_channels` (0): wähle Top‑K Anatomix-Basis-Kanäle (0 = alle)
- `--channel_selection_mode` (`supervised` | `unsupervised` | `pca`)

### 5) f-AnoGAN (GAN)

Script: `src/gan_brats.py`

```bash
python src/gan_brats.py \
  --num_subjects 100 \
  --patch_size 32 \
  --patches_per_volume 50 \
  --batch_size 8 \
  --latent_dim 128 \
  --g_lr 2e-4 \
  --d_lr 2e-4 \
  --e_lr 1e-4 \
  --gan_epochs 80 \
  --encoder_epochs 40 \
  --dataset_path datasets/BraTS2025-GLI-PRE-Challenge-TrainingData \
  --max_normal_to_anomaly_ratio 3 \
  --min_tumor_ratio_in_patch 0.05 \
  --anomaly_labels 1 2 4 \
  -v
```

Zusätzliche Argumente:
- `--alpha_residual` (0.9): Gewichtung Residual- vs. Feature-Verlust bei der Anomalie-Score-Berechnung
- `--g_lr`, `--d_lr`, `--e_lr`: Lernraten Generator/Discriminator/Encoder
- `--gan_epochs` (80), `--encoder_epochs` (40)

### 6) Diffusion (DDPM)

Script: `src/diffusion_brats.py`

```bash
python src/diffusion_brats.py \
  --num_subjects 100 \
  --patch_size 32 \
  --patches_per_volume 50 \
  --batch_size 8 \
  --train_epochs 100 \
  --lr 2e-4 \
  --num_diffusion_steps 200 \
  --score_timesteps_eval 10 \
  --dataset_path datasets/BraTS2025-GLI-PRE-Challenge-TrainingData \
  --max_normal_to_anomaly_ratio 3 \
  --min_tumor_ratio_in_patch 0.05 \
  --anomaly_labels 1 2 4 \
  -v \
  --output_dir diffusion_brats_results
```

Hinweis: Dieses Skript nutzt `--output_dir` direkt (kein automatisches `results/`-Unterverzeichnis).

## Tipps

- Für schnelle Tests die Anzahl `--num_subjects` reduzieren und ggf. `--patches_per_volume` verkleinern.
- Prüfen Sie die Konsole: Der tatsächlich verwendete Ergebnisordner wird immer ausgegeben.
- Die Ergebnisse (PR/ROC, Fehlerhistos, Latentplots) helfen bei der Modellauswahl und Threshold-Wahl.

## Beispiele (kurz)

- Nur ET (4) als Anomalie, 25 Patches/Volumen, kleiner Datensatz:

```bash
python src/ae_brats.py --num_subjects 20 --patches_per_volume 25 --anomaly_labels 4 -v
```

- VQ‑VAE mit größerem Codebook und stärkere Verpflichtung:

```bash
python src/vqvae_brats.py --codebook_size 1024 --commitment_beta 0.5 -v
```

- VQ‑VAE‑Encoder+KNN mit automatischem Finden der besten VQ‑VAE‑Gewichte:

```bash
python src/vqvae_encoder_knn_brats.py -v
```

Viel Erfolg!
