"""Модуль для работы с базой данных PostgreSQL (sync + async)."""

import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


def _build_url(driver: str = "psycopg2") -> str:
    user = os.environ.get("POSTGRES_USER", "cbr_user")
    pwd = os.environ.get("POSTGRES_PASSWORD", "cbr_password")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "cbr_news")
    if driver == "asyncpg":
        return f"postgresql+asyncpg://{user}:{pwd}@{host}:{port}/{db}"
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def get_database_url() -> str:
    """Sync URL (psycopg2) — Celery worker, парсеры, data_loader."""
    return _build_url("psycopg2")


def get_async_database_url() -> str:
    """Async URL (asyncpg) — FastAPI."""
    return _build_url("asyncpg")


# ---- sync engine (Celery worker, парсеры, data_loader) ----

engine = create_engine(
    get_database_url(),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---- async engine (FastAPI) ----

async_engine = create_async_engine(
    get_async_database_url(),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


# ---- context managers / dependencies ----

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """Sync FastAPI dependency — для обратной совместимости."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async FastAPI dependency — для новых task-роутов."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def init_db():
    """Создание таблиц через sync engine."""
    from cbr_news import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


async def async_init_db():
    """Создание таблиц через async engine."""
    from cbr_news import models  # noqa: F401
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
