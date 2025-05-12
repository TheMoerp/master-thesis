# 3D Anomaly Detection für Rippenfrakturen

Dieses Projekt implementiert eine 3D Anomaly Detection für Rippenfrakturen mit Hilfe eines Autoencoders basierend auf dem RibFrac Dataset 2020. Die Implementation verwendet MONAI, eine spezialisierte Deep Learning-Bibliothek für medizinische Bildverarbeitung.

## Projektübersicht

Der Ansatz basiert auf der Idee, einen Autoencoder nur mit normalen (frakturfreien) CT-Scans zu trainieren. Nach dem Training kann der Autoencoder normale Strukturen gut rekonstruieren, während er bei abnormalen Strukturen (wie Frakturen) höhere Rekonstruktionsfehler erzeugt. Diese Rekonstruktionsfehler werden als Anomalien-Scores verwendet.

## Funktionen

- 3D Autoencoder-Architektur mit Skip-Connections
- Datenvorbereitung für RibFrac-Dataset 2020
- Training des Autoencoders auf normalen (frakturfreien) Rippenscans
- Evaluation des Modells mit ROC-Kurve, AUC und Precision-Recall-Kurve
- Visualisierung der Rekonstruktionen und Fehler-Maps

## Installation

1. Repository klonen:

```bash
git clone <repository-url>
cd ripfrac-anomaly-detection
```

2. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

## Datenstruktur

Das Projekt erwartet die folgende Datenstruktur für das RibFrac-Dataset:

```
ribfrac-dataset/
├── imagesTr/                  # Training CT-Scans
│   ├── RibFrac1_0000.nii.gz
│   ├── RibFrac2_0000.nii.gz
│   └── ...
├── labelsTr/                  # Training-Annotationen (Segmentierungsmasken)
│   ├── RibFrac1.nii.gz
│   ├── RibFrac2.nii.gz
│   └── ...
```

Das RibFrac-Dataset kann von der offiziellen Webseite heruntergeladen werden: [RibFrac Challenge](https://ribfrac.grand-challenge.org/)

## Verwendung

### Daten vorbereiten

Das Skript `data_preparation.py` enthält Funktionen zur Vorbereitung der RibFrac-Daten. Es lädt die Bilder und Label-Dateien, filtert normale (frakturfreie) und anormale (mit Frakturen) Fälle und bereitet sie für das Training vor.

### Training

Zum Trainieren des Autoencoders:

```bash
python train.py --data_dir ./ribfrac-dataset --batch_size 4 --epochs 100
```

Optionen:
- `--data_dir`: Pfad zum RibFrac-Dataset
- `--checkpoint_dir`: Verzeichnis zum Speichern der Checkpoints (Standard: "./checkpoints")
- `--batch_size`: Batch-Größe (Standard: 4)
- `--epochs`: Anzahl der Trainings-Epochen (Standard: 100)
- `--lr`: Lernrate (Standard: 1e-4)
- `--weight_decay`: Gewichtsabnahme für den Optimizer (Standard: 1e-5)
- `--cache_rate`: Cache-Rate für den MONAI CacheDataset (Standard: 1.0)
- `--threshold_percentile`: Perzentil für die Anomalie-Schwelle (Standard: 95)
- `--seed`: Random Seed für Reproduzierbarkeit (Standard: 42)
- `--no_cuda`: Disable CUDA

### Evaluation

Zur Evaluation des trainierten Modells:

```bash
python evaluate.py --data_dir ./ribfrac-dataset --checkpoint_path ./checkpoints/best_model.pth
```

Optionen:
- `--data_dir`: Pfad zum RibFrac-Dataset
- `--checkpoint_path`: Pfad zum Modell-Checkpoint (Standard: "./checkpoints/best_model.pth")
- `--output_dir`: Verzeichnis zum Speichern der Evaluationsergebnisse (Standard: "./evaluation_results")
- `--batch_size`: Batch-Größe (Standard: 4)
- `--cache_rate`: Cache-Rate für den MONAI CacheDataset (Standard: 1.0)
- `--seed`: Random Seed für Reproduzierbarkeit (Standard: 42)
- `--no_cuda`: Disable CUDA

## Code-Übersicht

- `data_preparation.py`: Funktionen zur Datenvorbereitung
- `model.py`: Definition des 3D Autoencoder-Modells und des Anomaliedetektors
- `train.py`: Skript zum Trainieren des Modells
- `evaluate.py`: Skript zur Evaluation des trainierten Modells
- `requirements.txt`: Liste der benötigten Python-Pakete

## Methodischer Hintergrund

Dieser Ansatz zur Anomalieerkennung basiert auf dem Rekonstruktionsprinzip:

1. Der Autoencoder wird ausschließlich mit normalen (frakturfreien) CT-Scans trainiert.
2. Während des Trainings lernt der Autoencoder, normale anatomische Strukturen der Rippen zu rekonstruieren.
3. Für die Anomalieerkennung werden sowohl normale als auch anormale (mit Frakturen) Scans durch den trainierten Autoencoder geleitet.
4. Bei anormalen Strukturen (wie Frakturen) erzeugt der Autoencoder höhere Rekonstruktionsfehler, da er solche Strukturen während des Trainings nicht gesehen hat.
5. Diese Rekonstruktionsfehler werden als Anomalien-Scores verwendet, um zwischen normalen und anormalen Scans zu unterscheiden.

## Limitationen

- Der Ansatz erkennt möglicherweise nicht alle Arten von Frakturen mit gleicher Genauigkeit.
- Die Leistung hängt stark von der Qualität und Repräsentativität der Trainingsdaten ab.
- Der 3D Autoencoder benötigt beträchtliche Rechenressourcen, insbesondere GPU-Speicher. 