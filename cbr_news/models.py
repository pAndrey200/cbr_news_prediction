"""Модели базы данных для хранения данных Банка России."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Date, Index
from sqlalchemy.dialects.postgresql import JSONB

from cbr_news.database import Base


class News(Base):
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    link = Column(String(512), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    news_type = Column(String(50), nullable=False, default="press_release")

    key_rate = Column(Float, nullable=True)
    inflation = Column(Float, nullable=True)
    usd_rate = Column(Float, nullable=True)
    eur_rate = Column(Float, nullable=True)
    cny_rate = Column(Float, nullable=True)
    ruonia = Column(Float, nullable=True)

    prediction = Column(String(10), nullable=True)
    prediction_probs = Column(JSONB, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_news_date_type", "date", "news_type"),
        Index("idx_news_created_at", "created_at"),
    )

    def __repr__(self):
        return f"<News(id={self.id}, date={self.date}, title={self.title[:50]}...)>"


class KeyRate(Base):
    __tablename__ = "key_rates"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False, index=True)
    rate = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<KeyRate(date={self.date}, rate={self.rate})>"


class CurrencyRate(Base):
    __tablename__ = "currency_rates"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    currency_code = Column(String(3), nullable=False, index=True)
    rate = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_currency_date_code", "date", "currency_code", unique=True),
    )

    def __repr__(self):
        return f"<CurrencyRate(date={self.date}, currency={self.currency_code}, rate={self.rate})>"


class Inflation(Base):
    __tablename__ = "inflation"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False, index=True)
    value = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Inflation(date={self.date}, value={self.value})>"


class Ruonia(Base):
    __tablename__ = "ruonia"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False, index=True)
    rate = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Ruonia(date={self.date}, rate={self.rate})>"


class PreciousMetal(Base):
    __tablename__ = "precious_metals"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    metal_type = Column(String(20), nullable=False, index=True)
    price = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_metal_date_type", "date", "metal_type", unique=True),
    )

    def __repr__(self):
        return f"<PreciousMetal(date={self.date}, metal={self.metal_type}, price={self.price})>"


class Reserve(Base):
    __tablename__ = "reserves"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False, index=True)
    reserves_corset = Column(Float, nullable=True)
    reserves_avg = Column(Float, nullable=True)
    reserves_accounts = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Reserve(date={self.date}, avg={self.reserves_avg})>"
