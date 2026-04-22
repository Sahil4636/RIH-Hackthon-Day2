"""
config.py — Centralised app settings loaded from .env
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "ShelfVision"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Paths
    MODEL_DIR: Path = Path("./models")
    DATA_DIR: Path = Path("./data")
    UPLOAD_DIR: Path = Path("./data/uploads")

    # ML
    YOLO_MODEL: str = "yolov8n.pt"
    SKU110K_MODEL: str = "sku110k_dense.pt"
    DEVICE: str = "cuda"
    CONFIDENCE_THRESHOLD: float = 0.45
    IOU_THRESHOLD: float = 0.5
    INPUT_SIZE: int = 640
    DENSE_CONFIDENCE_THRESHOLD: float = 0.18
    DENSE_IOU_THRESHOLD: float = 0.35
    ENABLE_DENSE_TILING: bool = True
    DENSE_TILE_SIZE: int = 960
    DENSE_TILE_OVERLAP: float = 0.25

    # Thresholds
    SHELF_HEALTH_WARN: int = 70
    SHELF_HEALTH_CRITICAL: int = 50
    EMPTY_ZONE_THRESHOLD: float = 0.30

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/shelf_vision.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        """Create all required directories if they don't exist."""
        for d in [self.MODEL_DIR, self.DATA_DIR, self.UPLOAD_DIR,
                  self.DATA_DIR / "planograms", self.DATA_DIR / "annotated"]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
