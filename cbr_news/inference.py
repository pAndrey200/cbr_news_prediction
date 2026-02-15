import logging
from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def _load_composed_config(config_dir: Path, multitask: bool = True):
    model_file = "multitask_model.yaml" if multitask else "model.yaml"
    cfgs = []
    for fname in ["data.yaml", model_file, "train.yaml"]:
        p = config_dir / fname
        if p.exists():
            cfgs.append(OmegaConf.load(p))
    if not cfgs:
        return None
    return OmegaConf.merge(*cfgs)


class CBRNewsPredictor:

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = None,
        multitask: bool = True,
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
        self.multitask = multitask

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        cfg = None
        if config_path and Path(config_path).exists():
            config_dir = Path(config_path).parent
            cfg = _load_composed_config(config_dir, multitask=multitask)

        if cfg is None and _CONFIGS_DIR.exists():
            cfg = _load_composed_config(_CONFIGS_DIR, multitask=multitask)

        if cfg is None:
            cfg = OmegaConf.create({
                "model": {
                    "backbone": "DeepPavlov/rubert-base-cased",
                    "num_classes": 3,
                    "dropout": 0.1,
                    "freeze_backbone": False,
                    "num_frozen_layers": 0,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "warmup_steps": 100,
                    "auxiliary_tasks": ["usd", "inflation", "key_rate"],
                    "auxiliary_weight": 0.3,
                    "main_weight": 1.0,
                },
                "data": {
                    "max_length": 512,
                    "tokenizer_name": "DeepPavlov/rubert-base-cased",
                    "classes": ["down", "same", "up"],
                },
            })

        self.config = cfg

        if multitask:
            from cbr_news.multitask_model import CBRNewsMultiTaskModel
            self.model = CBRNewsMultiTaskModel.load_from_checkpoint(
                str(self.checkpoint_path),
                config=cfg,
                map_location=self.device,
                weights_only=False,
            )
        else:
            from cbr_news.model import CBRNewsModel
            self.model = CBRNewsModel.load_from_checkpoint(
                str(self.checkpoint_path),
                config=cfg,
                map_location=self.device,
                weights_only=False,
            )
        self.model.to(self.device)
        self.model.eval()

        tokenizer_name = OmegaConf.select(cfg, "data.tokenizer_name") or "DeepPavlov/rubert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.classes = OmegaConf.select(cfg, "data.classes") or ["down", "same", "up"]

        logger.info(f"Модель загружена: {checkpoint_path}, multitask={multitask}, device={self.device}")

    def predict(self, texts: List[str]):
        if not texts:
            return []
        return self.model.predict(
            texts,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def predict_single(self, text: str):
        results = self.predict([text])
        return results[0] if results else None
