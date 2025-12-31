import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from dvc.api import DVCFileSystem

from cbr_news.parser import CBRDataParser

logger = logging.getLogger(__name__)


class CBRNewsDataLoader:
    """Загрузчик данных новостей Банка России"""

    def __init__(self, config):
        self.config = config
        self.data_path = Path(config.data.dataset_path)
        self.parser = CBRDataParser(config)

    def download_data(self, force_download: bool = False) -> pd.DataFrame:
        """Скачивание данных через DVC или парсинг"""
        logger.info("Загрузка данных...")

        if self.data_path.exists() and not force_download:
            logger.info(f"Данные уже существуют: {self.data_path}")
            return self._load_local_data()

        try:
            logger.info("Попытка загрузки через DVC...")
            fs = DVCFileSystem()
            fs.get(str(self.data_path), str(self.data_path))
            return self._load_local_data()
        except Exception as e:
            logger.warning(f"DVC загрузка не удалась: {e}")
            return self._collect_data_from_source()

    def _load_local_data(self) -> pd.DataFrame:
        """Загрузка локальных данных"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Загружено {len(df)} записей")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return self._collect_data_from_source()

    def _collect_data_from_source(self) -> pd.DataFrame:
        """Сбор данных из исходных источников"""
        logger.info("Сбор данных из исходных источников...")

        df = self.parser.collect_all_data()

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.data_path, index=False)

        logger.info(f"Собрано и сохранено {len(df)} записей")

        return df

    def split_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Разделение данных по времени"""

        # Извлекаем год из текста (первая дата в тексте)
        def extract_year(text):
            try:
                # Пытаемся найти год в тексте
                for word in str(text).split():
                    if word.isdigit() and len(word) == 4:
                        year = int(word)
                        if 2010 <= year <= 2025:
                            return year
            except:
                pass
            return 2020

        df["year"] = df["cleaned_text"].apply(extract_year)

        train_years = self.config.data.train_years
        test_years = self.config.data.test_years

        train_df = df[df["year"].isin(train_years)]
        test_df = df[df["year"].isin(test_years)]

        if train_df.empty or test_df.empty:
            logger.warning(
                "Не удалось разделить по годам, используется случайное разделение"
            )
            from sklearn.model_selection import train_test_split

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.data.test_size,
                random_state=self.config.seed,
                stratify=df["label"] if "label" in df.columns else None,
            )

        logger.info(f"Train: {len(train_df)} записей")
        logger.info(f"Test: {len(test_df)} записей")

        return train_df, test_df

    def prepare_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка данных для обучения"""
        df["cleaned_text"] = df["cleaned_text"].apply(self._clean_text)
        label_mapping = {label: i for i, label in enumerate(self.config.data.classes)}
        df["label_encoded"] = df["label"].map(label_mapping)
        df = df.dropna(subset=["cleaned_text", "label_encoded"])
        df = df.drop_duplicates(subset=["cleaned_text"])
        logger.info(f"Подготовлено {len(df)} записей для обучения")

        return df

    @staticmethod
    def _clean_text(text: str) -> str:
        """Очистка текста"""
        if not isinstance(text, str):
            return ""

        text = text.lower().strip()

        import re

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?%\-]", "", text)

        return text
