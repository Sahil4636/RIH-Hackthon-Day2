echo off
cd /d "%~dp0"
if not exist "frontend" (
  echo [ERROR] frontend folder not found.
  exit /b 1
)
echo Starting ShelfVision frontend at http://localhost:3000
python -m http.server 3000 -d frontend
