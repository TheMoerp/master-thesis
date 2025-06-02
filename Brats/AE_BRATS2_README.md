# 3D Autoencoder für Anomaly Detection auf BraTS Dataset

## Übersicht

Das `ae_brats2.py` Programm implementiert einen 3D Convolutional Autoencoder für Anomalieerkennung auf dem BraTS Dataset. Es extrahiert normale Patches (ohne Tumorinformationen) zum Training und evaluiert die Fähigkeit des Modells, Anomalien zu erkennen.

## Features

- ✅ 3D Patch-Extraktion aus BraTS-Volumina
- ✅ GPU-Beschleunigung für alle rechenintensiven Operationen
- ✅ Echte Fortschrittsbalken (keine Treppenstufen)
- ✅ Umfassende Evaluierungsmetriken
- ✅ Multiple Visualisierungsoptionen
- ✅ Konfigurierbare Parameter über Kommandozeilenargumente

## Installation

```bash
pip install -r requirements.txt
```

## Grundlegende Verwendung

```bash
python ae_brats2.py --dataset_path datasets/BraTS2025-GLI-PRE-Challenge-TrainingData
```

## Wichtige Parameter

### Dataset-Parameter
- `--dataset_path`: Pfad zum BraTS Dataset (Standard: `datasets/BraTS2025-GLI-PRE-Challenge-TrainingData`)
- `--num_subjects`: Anzahl der zu verwendenden Subjekte (0 = alle)
- `--patch_size`: Größe der 3D-Patches (Standard: 32)
- `--patches_per_volume`: Anzahl der Patches pro Volumen (Standard: 20)
- `--train_ratio`: Verhältnis der Daten für Training (Standard: 0.8)

### Modell-Parameter
- `--latent_dim`: Latente Dimension des Autoencoders (Standard: 128)
- `--learning_rate`: Lernrate (Standard: 1e-3)
- `--batch_size`: Batch-Größe (Standard: 8)
- `--epochs`: Anzahl der Trainingsepochen (Standard: 50)

### Output-Parameter
- `--output_dir`: Output-Verzeichnis (Standard: `ae_brats2_results`)
- `--save_model`: Speichere trainiertes Modell
- `--visualize`: Erstelle Visualisierungen (Standard: True)

## Beispiele

### Schneller Test mit wenigen Subjekten
```bash
python ae_brats2.py \
    --num_subjects 10 \
    --epochs 10 \
    --batch_size 4 \
    --output_dir test_run
```

### Vollständiges Training
```bash
python ae_brats2.py \
    --num_subjects 0 \
    --epochs 100 \
    --batch_size 16 \
    --latent_dim 256 \
    --save_model \
    --output_dir full_training_results
```

### Training mit angepassten Patch-Parametern
```bash
python ae_brats2.py \
    --patch_size 64 \
    --patches_per_volume 50 \
    --epochs 50 \
    --output_dir large_patches_results
```

## Output

Das Programm erstellt folgende Outputs im angegebenen Output-Verzeichnis:

### Metriken
- ROC AUC
- Average Precision
- Accuracy
- Precision
- Recall
- F1 Score
- Optimaler Threshold

### Visualisierungen
- `training_curves.png`: Trainings- und Validierungsverluste
- `confusion_matrix.png`: Konfusionsmatrix
- `roc_curve.png`: ROC-Kurve
- `precision_recall_curve.png`: Precision-Recall-Kurve
- `reconstruction_error_histogram.png`: Histogramm der Rekonstruktionsfehler
- `latent_space_tsne.png`: t-SNE-Projektion des latenten Raums
- `latent_space_pca.png`: PCA-Projektion des latenten Raums
- `sample_patches.png`: Beispiel-3D-Patches
- `anomaly_detection_summary.png`: Zusammenfassungsplot aller Ergebnisse

### Modell und Logs
- `best_autoencoder_brats.pth`: Trainiertes Modell (wenn `--save_model` verwendet)
- `ae_brats2.log`: Detaillierte Logs

## Dataset-Struktur

Das Programm erwartet die folgende Struktur für das BraTS Dataset:

```
datasets/BraTS2025-GLI-PRE-Challenge-TrainingData/
├── BraTS-GLI-01449-000/
│   ├── BraTS-GLI-01449-000-t1n.nii.gz
│   ├── BraTS-GLI-01449-000-t1c.nii.gz
│   ├── BraTS-GLI-01449-000-t2w.nii.gz
│   ├── BraTS-GLI-01449-000-t2f.nii.gz
│   └── BraTS-GLI-01449-000-seg.nii.gz
├── BraTS-GLI-01450-000/
│   └── ...
└── ...
```

## Funktionsweise

1. **Datenverarbeitung**: Lädt BraTS-Volumina und extrahiert normale Patches (ohne Tumor) für Training
2. **Training**: Trainiert einen 3D Convolutional Autoencoder nur auf normalen Patches
3. **Anomalieerkennung**: Nutzt Rekonstruktionsfehler zur Erkennung von Anomalien (Tumor-Patches)
4. **Evaluation**: Berechnet umfassende Metriken und erstellt Visualisierungen

## Hardware-Anforderungen

- **GPU**: Empfohlen für akzeptable Trainingszeiten
- **RAM**: Mindestens 16GB (mehr bei größeren Batch-Größen)
- **Speicher**: Mehrere GB für Dataset und Outputs

## Tipps zur Optimierung

1. **Batch-Größe**: Reduzieren bei Speicherproblemen
2. **Patch-Größe**: Kleinere Patches für schnelleres Training
3. **Anzahl Subjekte**: Für Tests zunächst wenige Subjekte verwenden
4. **GPU-Speicher**: Bei CUDA-Fehlern Batch-Größe reduzieren

## Troubleshooting

### Häufige Probleme

1. **CUDA out of memory**: Reduziere `--batch_size`
2. **Zu wenige normale Patches**: Erhöhe `--patches_per_volume`
3. **Lange Trainingszeit**: Reduziere `--num_subjects` oder `--epochs`
4. **Dataset nicht gefunden**: Überprüfe `--dataset_path`

### Logs analysieren

Das Programm erstellt detaillierte Logs in `ae_brats2.log`. Bei Problemen zuerst die Logs überprüfen. 