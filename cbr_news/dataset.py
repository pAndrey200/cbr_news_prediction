import logging
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class CBRNewsDataset(Dataset):
    """Датасет для новостей Банка России"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name: str,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        logger.info(f"Загружен токенизатор: {tokenizer_name}")
        logger.info(f"Размер датасета: {len(self.texts)}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Токенизация
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class CBRNewsDataModule:
    """DataModule для PyTorch Lightning"""

    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tokenizer = None

    def setup(self, stage: str = None):
        """Настройка датасетов"""
        from cbr_news.data_loader import CBRNewsDataLoader

        data_loader = CBRNewsDataLoader(self.config)

        # Загрузка и подготовка данных
        df = data_loader.download_data()
        df = data_loader.prepare_for_training(df)

        # Разделение по времени
        train_df, test_df = data_loader.split_by_time(df)

        # Дополнительное разделение train на train/val
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.config.data.val_size,
            random_state=self.config.seed,
            stratify=train_df["label_encoded"]
            if "label_encoded" in train_df.columns
            else None,
        )

        # Создание датасетов
        self.train_dataset = CBRNewsDataset(
            texts=train_df["cleaned_text"].tolist(),
            labels=train_df["label_encoded"].tolist(),
            tokenizer_name=self.config.data.tokenizer_name,
            max_length=self.config.data.max_length,
        )

        self.val_dataset = CBRNewsDataset(
            texts=val_df["cleaned_text"].tolist(),
            labels=val_df["label_encoded"].tolist(),
            tokenizer_name=self.config.data.tokenizer_name,
            max_length=self.config.data.max_length,
        )

        self.test_dataset = CBRNewsDataset(
            texts=test_df["cleaned_text"].tolist(),
            labels=test_df["label_encoded"].tolist(),
            tokenizer_name=self.config.data.tokenizer_name,
            max_length=self.config.data.max_length,
        )

        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
