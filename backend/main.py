"""
main.py — FastAPI application entrypoint
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from api.routes import router

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== ShelfVision API starting up ===")
    await init_db()
    logger.info("Database initialised ✅")
    # ML models are loaded at import time via singletons in ml/detector.py
    logger.info("ML model ready ✅")
    yield
    logger.info("=== ShelfVision API shutting down ===")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShelfVision API",
    description=(
        "Shelf Visibility Intelligence — detect out-of-stock zones, "
        "planogram violations, and compute shelf health scores from images."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow React dev server ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api")


# ── Dev entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
