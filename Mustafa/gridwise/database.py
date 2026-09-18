from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False) if engine else None


def initialize_database() -> None:
    if engine is None:
        return
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def close_database() -> None:
    if engine is not None:
        engine.dispose()