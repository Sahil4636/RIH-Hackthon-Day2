"""
database.py — SQLAlchemy async setup + ORM models
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────────

class ScanResult(Base):
    __tablename__ = "scan_results"

    id             = Column(Integer, primary_key=True, index=True)
    shelf_id       = Column(String, index=True, nullable=False)
    image_path     = Column(String, nullable=False)
    annotated_path = Column(String, nullable=True)
    health_score   = Column(Float, nullable=False)
    oos_count      = Column(Integer, default=0)      # out-of-stock zones found
    violation_count= Column(Integer, default=0)      # planogram violations
    total_zones    = Column(Integer, default=0)
    scan_meta      = Column(Text, nullable=True)     # JSON blob of full result
    created_at     = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id          = Column(Integer, primary_key=True, index=True)
    shelf_id    = Column(String, index=True, nullable=False)
    scan_id     = Column(Integer, nullable=True)
    severity    = Column(String, nullable=False)     # critical | warning | info
    alert_type  = Column(String, nullable=False)     # oos | planogram | visibility
    message     = Column(Text, nullable=False)
    resolved    = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class Product(Base):
    __tablename__ = "products"

    id          = Column(Integer, primary_key=True, index=True)
    sku         = Column(String, unique=True, index=True, nullable=False)
    name        = Column(String, nullable=False)
    category    = Column(String, nullable=True)
    image_path  = Column(String, nullable=True)      # reference image for matching
    created_at  = Column(DateTime, default=datetime.utcnow)


# ── Helpers ────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
