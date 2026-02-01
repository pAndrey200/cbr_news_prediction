"""Модуль инференса для предсказания изменений ключевой ставки по тексту новостей."""

import logging
from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from cbr_news.model import CBRNewsModel

logger = logging.getLogger(__name__)


class CBRNewsPredictor:
    """Предсказание направления изменения ключевой ставки по тексту новостей."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = None,
    ):
        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
        if cp.is_dir():
            files = list(cp.rglob("*.ckpt"))
            if not files:
                raise FileNotFoundError(f"В директории нет файлов .ckpt: {checkpoint_path}")
            cp = max(files, key=lambda f: f.stat().st_mtime)
        self.checkpoint_path = cp

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if config_path and Path(config_path).exists():
            cfg = OmegaConf.load(config_path)
            config_dir = Path(config_path).parent
            if OmegaConf.select(cfg, "model") is None and (config_dir / "model.yaml").exists():
                cfg = OmegaConf.merge(cfg, OmegaConf.load(config_dir / "model.yaml"))
            if OmegaConf.select(cfg, "data") is None and (config_dir / "data.yaml").exists():
                cfg = OmegaConf.merge(cfg, OmegaConf.load(config_dir / "data.yaml"))
        else:
            cfg = OmegaConf.create({
                "model": {
                    "backbone": "cointegrated/rubert-tiny2",
                    "num_classes": 3,
                    "dropout": 0.1,
                    "freeze_backbone": False,
                    "num_frozen_layers": 0,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "warmup_steps": 100,
                },
                "data": {
                    "max_length": 512,
                    "tokenizer_name": "cointegrated/rubert-tiny2",
                    "classes": ["down", "same", "up"],
                },
            })

        self.config = cfg
        self.model = CBRNewsModel.load_from_checkpoint(
            str(self.checkpoint_path),
            config=cfg,
            map_location=self.device,
            weights_only=False,
        )
        self.model.to(self.device)
        self.model.eval()

        tokenizer_name = OmegaConf.select(cfg, "data.tokenizer_name") or "cointegrated/rubert-tiny2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.classes = OmegaConf.select(cfg, "data.classes") or ["down", "same", "up"]

        logger.info(f"Модель загружена: {checkpoint_path}, device={self.device}")

    def predict(self, texts: List[str]):
        """
        Предсказание для списка текстов.
        """
        if not texts:
            return []
        return self.model.predict(
            texts,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def predict_single(self, text: str):
        """Предсказание для одного текста."""
        results = self.predict([text])
        return results[0] if results else None
