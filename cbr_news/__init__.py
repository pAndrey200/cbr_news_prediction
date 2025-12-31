"""
Пакет для классификации новостей Банка России.
Прогнозирование изменений ключевой ставки на основе текстов пресс-релизов.
"""

__version__ = "0.1.0"
__author__ = "Andrey Potseluev"
__email__ = "ampotseluev@edu.hse.ru"

from cbr_news.data_loader import CBRNewsDataLoader
from cbr_news.dataset import CBRNewsDataModule, CBRNewsDataset
from cbr_news.model import CBRNewsModel
from cbr_news.parser import CBRDataParser
from cbr_news.train import train
from cbr_news.utils import log_git_info, save_predictions, setup_logging

__all__ = [
    "CBRNewsDataLoader",
    "CBRDataParser",
    "CBRNewsDataset",
    "CBRNewsDataModule",
    "CBRNewsModel",
    "train",
    "setup_logging",
    "log_git_info",
    "save_predictions",
]
