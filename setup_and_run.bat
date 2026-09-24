@echo off
echo ========================================
echo    SPROUT Setup and Training Script
echo ========================================
echo.

:: Step 1: Create conda environment
echo [Step 1] Creating conda environment 'sprout'...
call conda create -n sprout python=3.9 -y
if errorlevel 1 (
    echo Error creating environment!
    pause
    exit /b 1
)

:: Step 2: Activate environment
echo.
echo [Step 2] Activating environment...
call conda activate sprout

:: Step 3: Install packages
echo.
echo [Step 3] Installing required packages...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib seaborn scikit-learn Pillow tqdm timm

if errorlevel 1 (
    echo Error installing packages!
    pause
    exit /b 1
)

:: Step 4: Run training
echo.
echo [Step 4] Starting SPROUT Training...
echo ========================================
cd /d C:\Users\HP\SPROUT-main
python experiments/train.py --data_dir ./data/plantvillage --output_dir ./results --backbone resnet50 --n_way 5 --k_shot 5 --n_episodes 100 --num_epochs 10

echo.
echo ========================================
echo Training Complete!
echo Check results in: C:\Users\HP\SPROUT-main\results
echo ========================================
pause
