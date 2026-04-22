@echo off
REM Start ShelfVision API server
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist "..\..\venv\Scripts\activate.bat" (
    call "..\..\venv\Scripts\activate.bat"
) else (
    echo [WARN] Virtual environment not found. Run setup.bat first.
)

echo.
echo  Starting ShelfVision API...
echo  API docs -> http://localhost:8000/docs
echo  Press Ctrl+C to stop
echo.
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app-dir "%~dp0"