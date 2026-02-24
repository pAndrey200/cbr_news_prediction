"""Модули для работы с базой данных."""

__all__ = [
    # Database connection
    "Base",
    "engine",
    "async_engine",
    "SessionLocal",
    "AsyncSessionLocal",
    "get_db_session",
    "get_db",
    "get_async_db",
    "init_db",
    "async_init_db",
    # Models
    "News",
    "KeyRate",
    "CurrencyRate",
    "Inflation",
    "Ruonia",
    "PreciousMetal",
    "Reserve",
    "Task",
    "TaskStatus",
    "TaskType",
    # Repositories
    "NewsRepository",
    "KeyRateRepository",
    "CurrencyRateRepository",
    "InflationRepository",
    "RuoniaRepository",
    "PreciousMetalRepository",
    "ReserveRepository",
    "TaskRepositorySync",
    "TaskRepositoryAsync",
]


def __getattr__(name: str):
    if name in ("Base", "engine", "async_engine", "SessionLocal", "AsyncSessionLocal",
                "get_db_session", "get_db", "get_async_db", "init_db", "async_init_db"):
        from cbr_news.database.db import (
            AsyncSessionLocal,
            Base,
            SessionLocal,
            async_engine,
            async_init_db,
            engine,
            get_async_db,
            get_db,
            get_db_session,
            init_db,
        )
        return {
            "Base": Base,
            "engine": engine,
            "async_engine": async_engine,
            "SessionLocal": SessionLocal,
            "AsyncSessionLocal": AsyncSessionLocal,
            "get_db_session": get_db_session,
            "get_db": get_db,
            "get_async_db": get_async_db,
            "init_db": init_db,
            "async_init_db": async_init_db,
        }[name]

    if name in ("News", "KeyRate", "CurrencyRate", "Inflation", "Ruonia",
                "PreciousMetal", "Reserve", "Task", "TaskStatus", "TaskType"):
        from cbr_news.database.models import (
            CurrencyRate,
            Inflation,
            KeyRate,
            News,
            PreciousMetal,
            Reserve,
            Ruonia,
            Task,
            TaskStatus,
            TaskType,
        )
        return {
            "News": News,
            "KeyRate": KeyRate,
            "CurrencyRate": CurrencyRate,
            "Inflation": Inflation,
            "Ruonia": Ruonia,
            "PreciousMetal": PreciousMetal,
            "Reserve": Reserve,
            "Task": Task,
            "TaskStatus": TaskStatus,
            "TaskType": TaskType,
        }[name]

    if name in ("NewsRepository", "KeyRateRepository", "CurrencyRateRepository",
                "InflationRepository", "RuoniaRepository", "PreciousMetalRepository", "ReserveRepository"):
        from cbr_news.database.repository import (
            CurrencyRateRepository,
            InflationRepository,
            KeyRateRepository,
            NewsRepository,
            PreciousMetalRepository,
            ReserveRepository,
            RuoniaRepository,
        )
        return {
            "NewsRepository": NewsRepository,
            "KeyRateRepository": KeyRateRepository,
            "CurrencyRateRepository": CurrencyRateRepository,
            "InflationRepository": InflationRepository,
            "RuoniaRepository": RuoniaRepository,
            "PreciousMetalRepository": PreciousMetalRepository,
            "ReserveRepository": ReserveRepository,
        }[name]

    if name in ("TaskRepositorySync", "TaskRepositoryAsync"):
        from cbr_news.database.task_repository import TaskRepositoryAsync, TaskRepositorySync
        return {"TaskRepositorySync": TaskRepositorySync, "TaskRepositoryAsync": TaskRepositoryAsync}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
