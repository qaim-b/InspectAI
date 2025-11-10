@echo off
ECHO ========================================
ECHO InspectAI - Quick Setup for Windows
ECHO ========================================
ECHO.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    ECHO [ERROR] Anaconda/Miniconda not found!
    ECHO Please install Anaconda from https://www.anaconda.com/
    pause
    exit /b 1
)

ECHO [1/6] Creating conda environment...
call conda create -n InspectAI python=3.11 -y
if %errorlevel% neq 0 (
    ECHO [ERROR] Failed to create environment
    pause
    exit /b 1
)

ECHO [2/6] Activating environment...
call conda activate InspectAI

ECHO [3/6] Installing PyTorch with CUDA support...
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

ECHO [4/6] Installing other dependencies...
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0
pip install numpy==1.24.3 scikit-learn==1.3.0 Pillow==10.0.0
pip install matplotlib==3.7.2 seaborn==0.12.2
pip install opencv-python==4.8.0.76 albumentations==1.3.1
pip install python-multipart==0.0.6 jinja2==3.1.2 aiofiles==23.2.1
pip install tqdm==4.66.1 pydantic==2.5.0 python-dotenv==1.0.0
pip install tensorboard==2.14.0

ECHO [5/6] Creating directory structure...
if not exist "data\train\good" mkdir data\train\good
if not exist "data\train\defect" mkdir data\train\defect
if not exist "data\val\good" mkdir data\val\good
if not exist "data\val\defect" mkdir data\val\defect
if not exist "data\test\good" mkdir data\test\good
if not exist "data\test\defect" mkdir data\test\defect
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "static" mkdir static
if not exist "templates" mkdir templates

ECHO [6/6] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if %errorlevel% neq 0 (
    ECHO [WARNING] Verification failed, but setup might still work
)

ECHO.
ECHO ========================================
ECHO Setup Complete!
ECHO ========================================
ECHO.
ECHO Next steps:
ECHO 1. Generate sample data: python src/prepare_data.py --action sample --samples 100
ECHO 2. Train model: python src/train.py
ECHO 3. Run API: cd api ^& python app.py
ECHO.
ECHO For detailed instructions, see USAGE_GUIDE.md
ECHO.
pause
