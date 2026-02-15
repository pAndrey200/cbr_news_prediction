__version__ = "0.1.0"
__author__ = "Andrey Potseluev"
__email__ = "ampotseluev@edu.hse.ru"

from cbr_news.model import CBRNewsModel

__all__ = [
    "CBRNewsDataLoader",
    "CBRDataParser",
    "CBRNewsParser",
    "CBRNewsDataset",
    "CBRNewsDataModule",
    "CBRNewsModel",
    "train",
    "setup_logging",
    "log_git_info",
    "save_predictions",
]


def __getattr__(name: str):
    if name == "CBRDataParser":
        from cbr_news.parser import CBRDataParser
        return CBRDataParser
    if name == "CBRNewsParser":
        from cbr_news.news_parser import CBRNewsParser
        return CBRNewsParser
    if name == "CBRNewsDataLoader":
        from cbr_news.data_loader import CBRNewsDataLoader
        return CBRNewsDataLoader
    if name in ("CBRNewsDataModule", "CBRNewsDataset"):
        from cbr_news.dataset import CBRNewsDataModule, CBRNewsDataset
        return CBRNewsDataModule if name == "CBRNewsDataModule" else CBRNewsDataset
    if name == "train":
        from cbr_news.train import train
        return train
    if name in ("setup_logging", "log_git_info", "save_predictions"):
        from cbr_news.utils import log_git_info, save_predictions, setup_logging
        return {"setup_logging": setup_logging, "log_git_info": log_git_info, "save_predictions": save_predictions}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
