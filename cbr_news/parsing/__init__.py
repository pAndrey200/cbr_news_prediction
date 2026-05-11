"""Модули для парсинга данных Банка России."""

__all__ = [
    "CBRDataParser",
    "CBRNewsParser",
    "CBRNewsDataLoader",
]


def __getattr__(name: str):
    if name == "CBRDataParser":
        from cbr_news.parsing.parser import CBRDataParser
        return CBRDataParser
    if name == "CBRNewsParser":
        from cbr_news.parsing.news_parser import CBRNewsParser
        return CBRNewsParser
    if name == "CBRNewsDataLoader":
        from cbr_news.parsing.data_loader import CBRNewsDataLoader
        return CBRNewsDataLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
