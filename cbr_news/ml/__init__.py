"""Модули для машинного обучения."""

__all__ = [
    # Datasets
    "CBRNewsDataset",
    "CBRNewsDataModule",
    "CBRNewsMultiTaskDataset",
    "CBRNewsMultiTaskDataModule",
    # Models
    "CBRNewsModel",
    "CBRNewsMultiTaskModel",
    # Training
    "train",
    "train_multitask",
    # Inference
    "CBRNewsPredictor",
    # Utils
    "setup_logging",
    "log_git_info",
    "save_predictions",
]


def __getattr__(name: str):
    if name in ("CBRNewsDataset", "CBRNewsDataModule", "CBRNewsMultiTaskDataset", "CBRNewsMultiTaskDataModule"):
        from cbr_news.ml.dataset import (
            CBRNewsDataModule,
            CBRNewsDataset,
            CBRNewsMultiTaskDataModule,
            CBRNewsMultiTaskDataset,
        )
        return {
            "CBRNewsDataset": CBRNewsDataset,
            "CBRNewsDataModule": CBRNewsDataModule,
            "CBRNewsMultiTaskDataset": CBRNewsMultiTaskDataset,
            "CBRNewsMultiTaskDataModule": CBRNewsMultiTaskDataModule,
        }[name]
    if name == "CBRNewsPredictor":
        from cbr_news.ml.inference import CBRNewsPredictor
        return CBRNewsPredictor
    if name == "CBRNewsModel":
        from cbr_news.ml.models.base_model import CBRNewsModel
        return CBRNewsModel
    if name == "CBRNewsMultiTaskModel":
        from cbr_news.ml.models.multitask_model import CBRNewsMultiTaskModel
        return CBRNewsMultiTaskModel
    if name == "train":
        from cbr_news.ml.train import train
        return train
    if name == "train_multitask":
        from cbr_news.ml.train_multitask import train_multitask
        return train_multitask
    if name in ("setup_logging", "log_git_info", "save_predictions"):
        from cbr_news.ml.utils import log_git_info, save_predictions, setup_logging
        return {"setup_logging": setup_logging, "log_git_info": log_git_info, "save_predictions": save_predictions}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
