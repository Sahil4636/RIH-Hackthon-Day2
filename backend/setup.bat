@echo off
REM ============================================================
REM  ShelfVision — Windows Setup Script
REM  Run this ONCE to set up your Python environment
REM ============================================================

echo.
echo  ╔══════════════════════════════════════╗
echo  ║   ShelfVision Backend Setup          ║
echo  ╚══════════════════════════════════════╝
echo.

REM ── 1. Check Python ─────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause & exit /b 1
)
echo [OK] Python found

REM ── 2. Create virtual environment ───────────────────────────
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)
echo [OK] Virtual environment ready

REM ── 3. Activate venv ────────────────────────────────────────
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM ── 4. Upgrade pip ──────────────────────────────────────────
python -m pip install --upgrade pip --quiet

REM ── 5. Install PyTorch with CUDA 12.1 ───────────────────────
echo.
echo [INFO] Installing PyTorch with CUDA 12.1 support...
echo        (This may take a few minutes — ~2GB download)
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM ── 6. Install remaining requirements ───────────────────────
echo.
echo [INFO] Installing remaining dependencies...
pip install -r requirements.txt

REM ── 7. Install pydantic-settings (needed for config.py) ─────
pip install pydantic-settings

REM ── 8. Verify CUDA ──────────────────────────────────────────
echo.
echo [INFO] Verifying GPU setup...
python -c "import torch; print('[GPU]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND — will use CPU')"

REM ── 9. Create required directories ──────────────────────────
if not exist "data\uploads"    mkdir data\uploads
if not exist "data\annotated"  mkdir data\annotated
if not exist "data\planograms" mkdir data\planograms
if not exist "models"          mkdir models

echo.
echo  ╔══════════════════════════════════════╗
echo  ║   Setup complete!                    ║
echo  ║                                      ║
echo  ║   To start the server run:           ║
echo  ║     run_server.bat                   ║
echo  ║                                      ║
echo  ║   API docs at:                       ║
echo  ║     http://localhost:8000/docs       ║
echo  ╚══════════════════════════════════════╝
echo.
pause
