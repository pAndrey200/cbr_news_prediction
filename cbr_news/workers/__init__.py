"""Модули для фоновых задач (Celery)."""

__all__ = [
    "celery_app",
    "run_training",
    "run_prediction",
]


def __getattr__(name: str):
    if name == "celery_app":
        from cbr_news.workers.celery_app import celery_app
        return celery_app
    if name in ("run_training", "run_prediction"):
        from cbr_news.workers.tasks import run_prediction, run_training
        return {"run_training": run_training, "run_prediction": run_prediction}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
