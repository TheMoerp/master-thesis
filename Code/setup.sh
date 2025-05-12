#!/bin/bash

# Farben für die Ausgabe
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting setup of RibFrac environment...${NC}"

# Prüfen ob Python installiert ist
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 ist nicht installiert. Bitte installieren Sie Python3 und versuchen Sie es erneut.${NC}"
    exit 1
fi

# Prüfen ob pip installiert ist
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip3 ist nicht installiert. Bitte installieren Sie pip3 und versuchen Sie es erneut.${NC}"
    exit 1
fi

# Prüfen ob venv Modul verfügbar ist
python3 -c "import venv" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Python venv Modul ist nicht installiert. Bitte installieren Sie python3-venv und versuchen Sie es erneut.${NC}"
    exit 1
fi

# Name der virtuellen Umgebung
VENV_NAME="ribfrac_env"

# Prüfen ob die Umgebung bereits existiert
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Virtual environment '$VENV_NAME' exists already. Do you want to remove it and create a new one? (y/n)${NC}"
    read -r answer
    if [ "$answer" != "${answer#[Yy]}" ]; then
        echo -e "${BLUE}Removing existing environment...${NC}"
        rm -rf $VENV_NAME
    else
        echo -e "${BLUE}Using existing environment...${NC}"
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    echo -e "${BLUE}Creating virtual environment '${VENV_NAME}'...${NC}"
    # Virtuelle Umgebung erstellen
    python3 -m venv $VENV_NAME
fi

# Virtuelle Umgebung aktivieren
source "${VENV_NAME}/bin/activate"

echo -e "${BLUE}Installing required packages...${NC}"

# Pip upgraden
pip install --upgrade pip

# Requirements installieren
pip install -r requirements.txt

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${BLUE}To activate the environment, run:${NC}"
echo -e "${GREEN}source ${VENV_NAME}/bin/activate${NC}"

# Optional: Dataset-Struktur überprüfen
echo -e "${YELLOW}Would you like to test the dataset loading? This might require significant memory. (y/n)${NC}"
read -r answer
if [ "$answer" != "${answer#[Yy]}" ]; then
    echo -e "${BLUE}Testing dataset structure...${NC}"
    echo -e "${YELLOW}Note: If this fails due to memory issues, you can still use the environment and test the dataset later with smaller batch sizes.${NC}"
    # Setze Umgebungsvariable für reduzierte Parallelität
    export MONAI_DATA_LOADER_NUM_WORKERS=1
    # Führe Test mit reduziertem Batch-Size aus
    python3 -c "
import sys
sys.path.append('.')
from data_preparation import prepare_ribfrac_dataset
try:
    train_loader, val_loader, test_loader = prepare_ribfrac_dataset(batch_size=1, cache_rate=0.0)
    print('Dataset loading successful!')
    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of validation batches: {len(val_loader)}')
    print(f'Number of testing batches: {len(test_loader)}')
except Exception as e:
    print(f'Error loading dataset: {str(e)}')
"
fi

# Deaktiviere die virtuelle Umgebung
deactivate

echo -e "${GREEN}Setup process completed!${NC}"
echo -e "${BLUE}You can now activate the environment with:${NC}"
echo -e "${GREEN}source ${VENV_NAME}/bin/activate${NC}"
echo -e "${YELLOW}For training, consider adjusting the batch size and number of workers in data_preparation.py if you experience memory issues.${NC}" 