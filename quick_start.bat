@echo off
echo ============================================================
echo InspectAI - Quick Start
echo ============================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Creating conda environment...
echo ============================================================
call conda create -n InspectAI python=3.11 -y
if errorlevel 1 (
    echo Failed to create conda environment
    echo Make sure Anaconda/Miniconda is installed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Activating environment...
echo ============================================================
call conda activate InspectAI

echo.
echo ============================================================
echo Installing dependencies...
echo ============================================================
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Running setup...
echo ============================================================
python setup.py

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To get started:
echo 1. conda activate InspectAI
echo 2. python src/prepare_data.py --action sample --samples 50
echo 3. python src/train.py
echo 4. python api/app.py
echo.
pause
