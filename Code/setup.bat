@echo off
REM setup.bat - Windows setup script for RibFrac Anomaly Detection

echo Starting setup of RibFrac environment...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo pip is not installed or not in PATH. Please install pip and try again.
    exit /b 1
)

REM Check if venv module is available
python -c "import venv" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python venv module is not available. Please install it and try again.
    exit /b 1
)

REM Name of the virtual environment
set VENV_NAME=ribfrac_env

REM Check if the environment already exists
if exist %VENV_NAME% (
    echo Virtual environment '%VENV_NAME%' exists already. Do you want to remove it and create a new one? (y/n)
    set /p answer=
    if /i "%answer%"=="y" (
        echo Removing existing environment...
        rmdir /s /q %VENV_NAME%
    ) else (
        echo Using existing environment...
    )
)

if not exist %VENV_NAME% (
    echo Creating virtual environment '%VENV_NAME%'...
    python -m venv %VENV_NAME%
)

REM Activate virtual environment
call %VENV_NAME%\Scripts\activate.bat

echo Installing required packages...

REM Upgrade pip
pip install --upgrade pip

REM Check if CUDA is available
python -c "import torch; print('CUDA is available' if torch.cuda.is_available() else 'CUDA is not available')"

REM Install requirements
pip install -r requirements.txt

echo Setup completed successfully!
echo To activate the environment, run:
echo %VENV_NAME%\Scripts\activate.bat

REM Optional: Dataset structure check
echo Would you like to test the dataset loading? This might require significant memory. (y/n)
set /p answer=
if /i "%answer%"=="y" (
    echo Testing dataset structure...
    echo Note: If this fails due to memory issues, you can still use the environment and test the dataset later with smaller batch sizes.
    
    REM Set environment variable for reduced parallelism
    set MONAI_DATA_LOADER_NUM_WORKERS=1
    
    REM Run test with reduced batch size
    python -c "import sys; sys.path.append('.'); from data_preparation import prepare_ribfrac_dataset; try: train_loader, val_loader, test_loader = prepare_ribfrac_dataset(batch_size=1, cache_rate=0.0); print('Dataset loading successful!'); print(f'Number of training batches: {len(train_loader)}'); print(f'Number of validation batches: {len(val_loader)}'); print(f'Number of testing batches: {len(test_loader)}'); except Exception as e: print(f'Error loading dataset: {str(e)}')"
)

REM Deactivate virtual environment
call %VENV_NAME%\Scripts\deactivate.bat

echo Setup process completed!
echo You can now activate the environment with:
echo %VENV_NAME%\Scripts\activate.bat
echo For training, consider adjusting the batch size and number of workers if you experience memory issues. 