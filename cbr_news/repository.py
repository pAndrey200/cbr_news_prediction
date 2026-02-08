import logging
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import desc, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from cbr_news.models import (
    CurrencyRate,
    Inflation,
    KeyRate,
    News,
    PreciousMetal,
    Reserve,
    Ruonia,
)

logger = logging.getLogger(__name__)


class NewsRepository:
    @staticmethod
    def create_or_update(session: Session, news_data: dict) -> News:
        stmt = insert(News).values(**news_data)
        stmt = stmt.on_conflict_do_update(
            index_elements=["link"],
            set_={
                "title": stmt.excluded.title,
                "content": stmt.excluded.content,
                "date": stmt.excluded.date,
                "news_type": stmt.excluded.news_type,
                "key_rate": stmt.excluded.key_rate,
                "inflation": stmt.excluded.inflation,
                "usd_rate": stmt.excluded.usd_rate,
                "eur_rate": stmt.excluded.eur_rate,
                "cny_rate": stmt.excluded.cny_rate,
                "ruonia": stmt.excluded.ruonia,
                "prediction": stmt.excluded.prediction,
                "prediction_probs": stmt.excluded.prediction_probs,
                "updated_at": datetime.utcnow(),
            },
        )
        session.execute(stmt)
        session.commit()

        return session.query(News).filter_by(link=news_data["link"]).first()

    @staticmethod
    def get_latest_news(
        session: Session, limit: int = 10, news_type: Optional[str] = None
    ) -> List[News]:
        query = session.query(News).order_by(desc(News.date), desc(News.created_at))

        if news_type:
            query = query.filter(News.news_type == news_type)

        return query.limit(limit).all()

    @staticmethod
    def get_news_by_date_range(
        session: Session, start_date: date, end_date: date
    ) -> List[News]:
                return (
            session.query(News)
            .filter(News.date >= start_date, News.date <= end_date)
            .order_by(desc(News.date))
            .all()
        )

    @staticmethod
    def get_by_link(session: Session, link: str) -> Optional[News]:
                return session.query(News).filter_by(link=link).first()

    @staticmethod
    def count(session: Session) -> int:
        """Подсчет общего количества новостей."""
        return session.query(func.count(News.id)).scalar()


class KeyRateRepository:
    """Репозиторий для работы с ключевой ставкой."""

    @staticmethod
    def create_or_update(session: Session, rate_date: date, rate: float) -> KeyRate:
        """Создание или обновление ключевой ставки."""
        stmt = insert(KeyRate).values(date=rate_date, rate=rate)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date"],
            set_={"rate": stmt.excluded.rate, "updated_at": datetime.utcnow()},
        )
        session.execute(stmt)
        session.commit()
        return session.query(KeyRate).filter_by(date=rate_date).first()

    @staticmethod
    def get_latest(session: Session) -> Optional[KeyRate]:
        """Получение последней ключевой ставки."""
        return session.query(KeyRate).order_by(desc(KeyRate.date)).first()

    @staticmethod
    def get_by_date(session: Session, rate_date: date) -> Optional[KeyRate]:
        """Получение ключевой ставки на дату."""
        return session.query(KeyRate).filter_by(date=rate_date).first()

    @staticmethod
    def get_all(session: Session) -> List[KeyRate]:
        """Получение всех записей ключевой ставки."""
        return session.query(KeyRate).order_by(KeyRate.date).all()


class CurrencyRateRepository:
    """Репозиторий для работы с курсами валют."""

    @staticmethod
    def create_or_update(
        session: Session, rate_date: date, currency_code: str, rate: float
    ) -> CurrencyRate:
        """Создание или обновление курса валюты."""
        stmt = insert(CurrencyRate).values(
            date=rate_date, currency_code=currency_code, rate=rate
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "currency_code"],
            set_={"rate": stmt.excluded.rate, "updated_at": datetime.utcnow()},
        )
        session.execute(stmt)
        session.commit()
        return (
            session.query(CurrencyRate)
            .filter_by(date=rate_date, currency_code=currency_code)
            .first()
        )

    @staticmethod
    def get_latest(session: Session, currency_code: str) -> Optional[CurrencyRate]:
        """Получение последнего курса валюты."""
        return (
            session.query(CurrencyRate)
            .filter_by(currency_code=currency_code)
            .order_by(desc(CurrencyRate.date))
            .first()
        )

    @staticmethod
    def get_by_date(
        session: Session, rate_date: date, currency_code: str
    ) -> Optional[CurrencyRate]:
        """Получение курса валюты на дату."""
        return (
            session.query(CurrencyRate)
            .filter_by(date=rate_date, currency_code=currency_code)
            .first()
        )


class InflationRepository:
    """Репозиторий для работы с инфляцией."""

    @staticmethod
    def create_or_update(session: Session, inflation_date: date, value: float) -> Inflation:
        """Создание или обновление инфляции."""
        stmt = insert(Inflation).values(date=inflation_date, value=value)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date"],
            set_={"value": stmt.excluded.value, "updated_at": datetime.utcnow()},
        )
        session.execute(stmt)
        session.commit()
        return session.query(Inflation).filter_by(date=inflation_date).first()

    @staticmethod
    def get_latest(session: Session) -> Optional[Inflation]:
        """Получение последней инфляции."""
        return session.query(Inflation).order_by(desc(Inflation.date)).first()


class RuoniaRepository:
    """Репозиторий для работы со ставкой RUONIA."""

    @staticmethod
    def create_or_update(session: Session, ruonia_date: date, rate: float) -> Ruonia:
        """Создание или обновление ставки RUONIA."""
        stmt = insert(Ruonia).values(date=ruonia_date, rate=rate)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date"],
            set_={"rate": stmt.excluded.rate, "updated_at": datetime.utcnow()},
        )
        session.execute(stmt)
        session.commit()
        return session.query(Ruonia).filter_by(date=ruonia_date).first()

    @staticmethod
    def get_latest(session: Session) -> Optional[Ruonia]:
        """Получение последней ставки RUONIA."""
        return session.query(Ruonia).order_by(desc(Ruonia.date)).first()


class PreciousMetalRepository:
    """Репозиторий для работы с драгметаллами."""

    @staticmethod
    def create_or_update(
        session: Session, metal_date: date, metal_type: str, price: float
    ) -> PreciousMetal:
        """Создание или обновление цены драгметалла."""
        stmt = insert(PreciousMetal).values(
            date=metal_date, metal_type=metal_type, price=price
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "metal_type"],
            set_={"price": stmt.excluded.price, "updated_at": datetime.utcnow()},
        )
        session.execute(stmt)
        session.commit()
        return (
            session.query(PreciousMetal)
            .filter_by(date=metal_date, metal_type=metal_type)
            .first()
        )

    @staticmethod
    def get_latest(session: Session, metal_type: str) -> Optional[PreciousMetal]:
        """Получение последней цены драгметалла."""
        return (
            session.query(PreciousMetal)
            .filter_by(metal_type=metal_type)
            .order_by(desc(PreciousMetal.date))
            .first()
        )


class ReserveRepository:
    """Репозиторий для работы с резервами."""

    @staticmethod
    def create_or_update(
        session: Session,
        reserve_date: date,
        reserves_corset: Optional[float] = None,
        reserves_avg: Optional[float] = None,
        reserves_accounts: Optional[float] = None,
    ) -> Reserve:
        """Создание или обновление резервов."""
        stmt = insert(Reserve).values(
            date=reserve_date,
            reserves_corset=reserves_corset,
            reserves_avg=reserves_avg,
            reserves_accounts=reserves_accounts,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["date"],
            set_={
                "reserves_corset": stmt.excluded.reserves_corset,
                "reserves_avg": stmt.excluded.reserves_avg,
                "reserves_accounts": stmt.excluded.reserves_accounts,
                "updated_at": datetime.utcnow(),
            },
        )
        session.execute(stmt)
        session.commit()
        return session.query(Reserve).filter_by(date=reserve_date).first()

    @staticmethod
    def get_latest(session: Session) -> Optional[Reserve]:
        """Получение последних резервов."""
        return session.query(Reserve).order_by(desc(Reserve.date)).first()
