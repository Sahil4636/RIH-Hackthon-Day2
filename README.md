# ShelfVision

Shelf visibility intelligence for retail stores. The system analyzes shelf images from a camera or phone to detect:

- Out-of-stock or low-stock shelf zones
- Incorrect product placement against a planogram
- Visibility problems caused by missing or weak detections

It combines those signals into alerts, scan history, and a shelf health score that is easy to demo in a hackathon setting.

## Quick Start

```bash
# 1. Open the backend folder
cd shelf-vision/backend

# 2. One-time setup
setup.bat

# 3. Optional sanity check
venv\Scripts\activate
python test_pipeline.py

# 4. Start the API
run_server.bat
# API: http://localhost:8000
# Docs: http://localhost:8000/docs

# 5. In a second terminal, start the frontend demo
cd ..
start_frontend.bat
# Frontend: http://localhost:3000
```

## Demo Flow

1. Open the dashboard in the browser.
2. Choose a shelf ID.
3. Load or edit the planogram in the built-in editor.
4. Upload a shelf image.
5. Review the annotated output, alerts, compliance table, and scan history.

## Project Structure

```text
shelf-vision/
|-- backend/
|   |-- main.py
|   |-- config.py
|   |-- database.py
|   |-- requirements.txt
|   |-- test_pipeline.py
|   |-- api/
|   |   |-- routes.py
|   |   `-- schemas.py
|   |-- ml/
|   |   |-- detector.py
|   |   |-- planogram.py
|   |   `-- scorer.py
|   `-- data/
|       |-- uploads/
|       |-- annotated/
|       `-- planograms/
|-- frontend/
|   `-- index.html
`-- start_frontend.bat
```

## Core Features

- FastAPI backend for image upload, analysis, planograms, alerts, history, and stats
- YOLO-based detection pipeline for shelf objects and empty zones
- Planogram compliance checker using grid-based shelf zones
- Shelf health scoring with occupancy, compliance, and visibility components
- Frontend dashboard with:
  - Image upload and live analysis
  - Planogram editor
  - Annotated shelf view
  - Alerts, shelf history, and aggregate stats

## Suggested Pitch

"ShelfVision gives supermarkets and MSMEs a low-cost shelf audit system. A staff member snaps one photo, and the system identifies stock gaps, misplaced items, and shelf visibility issues in seconds."
