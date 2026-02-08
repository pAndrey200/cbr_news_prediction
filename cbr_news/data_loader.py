import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from dvc.api import DVCFileSystem
from hydra.utils import get_original_cwd

from cbr_news.parser import CBRDataParser

logger = logging.getLogger(__name__)


class CBRNewsDataLoader:
    """Загрузчик данных новостей Банка России"""

    def __init__(self, config):
        self.config = config
        try:
            project_root = Path(get_original_cwd())
        except (ValueError, AttributeError):
            project_root = Path.cwd()
        self.data_path = project_root / config.data.dataset_path
        self.parser = CBRDataParser(config)

        self._setup_dvc_remote()

    def _setup_dvc_remote(self):
        """Настройка DVC remote из конфигурации для Yandex Object Storage"""
        try:
            dvc_config = self.config.data.dvc
            if not dvc_config or not dvc_config.get('enabled', False):
                return
        except (AttributeError, KeyError):
            return
        
        try:
            from dvc.repo import Repo
            
            remote_config = dvc_config.remote
            try:
                project_root = Path(get_original_cwd())
            except (ValueError, AttributeError):
                project_root = Path.cwd()

            dvc_dir = project_root / ".dvc"
            if not dvc_dir.exists():
                logger.info("DVC репозиторий не инициализирован, пропускаем настройку remote")
                logger.info("Для использования DVC выполните: dvc init")
                return

            try:
                dvc_repo = Repo(project_root)
            except Exception as e:
                if "not inside of a DVC repository" in str(e):
                    logger.info("DVC репозиторий не найден, пропускаем настройку remote")
                    logger.info("Для использования DVC выполните: dvc init")
                    return
                raise

            remote_name = remote_config.get('name', 's3remote')
            remote_url = remote_config.get('url')
            
            if remote_url:

                import subprocess

                try:

                    subprocess.run([
                        'dvc', 'remote', 'add', '-f', remote_name, remote_url
                    ], check=True, capture_output=True, text=True)

                    subprocess.run([
                        'dvc', 'remote', 'default', remote_name
                    ], check=True, capture_output=True, text=True)

                    if remote_config.get('region_name'):
                        subprocess.run([
                            'dvc', 'remote', 'modify', remote_name, 'region', remote_config.get('region_name')
                        ], check=True, capture_output=True, text=True)

                    if remote_config.get('endpoint_url'):
                        subprocess.run([
                            'dvc', 'remote', 'modify', remote_name, 'endpointurl', remote_config.get('endpoint_url')
                        ], check=True, capture_output=True, text=True)

                    logger.info(f"DVC remote '{remote_name}' настроен для Yandex Object Storage: {remote_url}")
                    logger.info(f"  Region: {remote_config.get('region_name')}")
                    logger.info(f"  Endpoint: {remote_config.get('endpoint_url')}")

                except subprocess.CalledProcessError as e:
                    logger.warning(f"Не удалось настроить DVC remote через команды: {e}")
                    logger.info("Попробуйте настроить remote вручную:")
                    logger.info(f"  dvc remote add -f {remote_name} {remote_url}")
                    logger.info(f"  dvc remote default {remote_name}")
                    if remote_config.get('region_name'):
                        logger.info(f"  dvc remote modify {remote_name} region {remote_config.get('region_name')}")
                    if remote_config.get('endpoint_url'):
                        logger.info(f"  dvc remote modify {remote_name} endpointurl {remote_config.get('endpoint_url')}")
                
        except ImportError:
            logger.info("DVC не установлен, пропускаем настройку remote")
        except Exception as e:
            if "not inside of a DVC repository" in str(e):
                logger.info("DVC репозиторий не найден, пропускаем настройку remote")
                logger.info("Для использования DVC выполните: dvc init")
            else:
                logger.warning(f"Не удалось настроить DVC remote: {e}")

    def download_data(self, force_download: bool = False) -> pd.DataFrame:
        """Скачивание данных через DVC или парсинг"""
        logger.info("Загрузка данных...")

        if self.data_path.exists() and not force_download:
            logger.info(f"Данные уже существуют: {self.data_path}")
            return self._load_local_data()

        # Попытка загрузки через DVC
        try:
            logger.info("Попытка загрузки через DVC...")
            fs = DVCFileSystem()
            # Проверяем, существует ли файл в DVC перед загрузкой
            try:
                if fs.exists(str(self.data_path)):
                    fs.get(str(self.data_path), str(self.data_path))
                    logger.info("Данные успешно загружены через DVC")
                    return self._load_local_data()
                else:
                    logger.info("Файл не найден в DVC, переходим к сбору данных")
                    return self._collect_data_from_source()
            except (FileNotFoundError, KeyError) as e:
                logger.info(f"Файл не найден в DVC: {e}, переходим к сбору данных")
                return self._collect_data_from_source()
        except ImportError:
            logger.info("DVC не установлен, переходим к сбору данных из исходных источников")
            return self._collect_data_from_source()
        except Exception as e:
            logger.info(f"DVC недоступен ({type(e).__name__}: {e}), переходим к сбору данных из исходных источников")
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

    def download_multitask_data(self, force_download: bool = False) -> pd.DataFrame:
        """Загрузка multi-task данных"""
        logger.info("Загрузка multi-task данных...")

        try:
            project_root = Path(get_original_cwd())
        except (ValueError, AttributeError):
            project_root = Path.cwd()

        multitask_path = project_root / "data" / "cbr_multitask_dataset.csv"

        if multitask_path.exists() and not force_download:
            logger.info(f"Multi-task данные уже существуют: {multitask_path}")
            df = pd.read_csv(multitask_path)
            logger.info(f"Загружено {len(df)} записей")
            return df

        # Если файла нет, создаем его
        logger.info("Multi-task датасет не найден, создаем...")
        df_processed = self.parser.collect_all_data()

        # Создаем multi-task датасет
        df_multitask = self.parser.prepare_multitask_dataset(df_processed=None)

        return df_multitask

    def split_by_time_multitask(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Разделение multi-task данных по времени"""

        def extract_year(date_str):
            try:
                # Формат: DD.MM.YYYY
                parts = str(date_str).split(".")
                if len(parts) == 3:
                    year = int(parts[2])
                    if 2010 <= year <= 2025:
                        return year
            except:
                pass
            return 2020

        df["year"] = df["date"].apply(extract_year)

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
            )

        logger.info(f"Multi-task Train: {len(train_df)} записей")
        logger.info(f"Multi-task Test: {len(test_df)} записей")

        return train_df, test_df
