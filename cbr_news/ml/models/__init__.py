"""ML модели для предсказания экономических показателей."""

__all__ = [
    "CBRNewsModel",
    "CBRNewsMultiTaskModel",
]


def __getattr__(name: str):
    if name == "CBRNewsModel":
        from cbr_news.ml.models.base_model import CBRNewsModel
        return CBRNewsModel
    if name == "CBRNewsMultiTaskModel":
        from cbr_news.ml.models.multitask_model import CBRNewsMultiTaskModel
        return CBRNewsMultiTaskModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
