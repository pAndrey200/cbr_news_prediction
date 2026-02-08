"""Модуль для работы с базой данных PostgreSQL."""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


def get_database_url() -> str:
    db_user = os.environ.get("POSTGRES_USER", "cbr_user")
    db_password = os.environ.get("POSTGRES_PASSWORD", "cbr_password")
    db_host = os.environ.get("POSTGRES_HOST", "localhost")
    db_port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "cbr_news")

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


engine = create_engine(
    get_database_url(),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from cbr_news import models
    Base.metadata.create_all(bind=engine)
