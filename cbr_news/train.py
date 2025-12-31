import logging
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from cbr_news.dataset import CBRNewsDataModule
from cbr_news.model import CBRNewsModel
from cbr_news.utils import log_git_info, setup_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    """Основная функция тренировки"""
    setup_logging()
    logger.info("Начало тренировки модели")

    logger.info(f"Конфигурация:\n{OmegaConf.to_yaml(cfg)}")

    log_git_info()

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.training.mlflow.experiment_name,
        tracking_uri=cfg.training.mlflow.tracking_uri,
        log_model=False,  # Отключаем автоматическое логирование модели, чтобы избежать ошибок с boto3
    )

    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    pl.seed_everything(cfg.seed, workers=True)

    data_module = CBRNewsDataModule(cfg)
    data_module.setup()

    model = CBRNewsModel(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.monitor,
        mode=cfg.training.mode,
        save_top_k=1,
        save_last=True,
        filename="cbr-model-{epoch:02d}-{val_loss:.2f}",
        dirpath=Path.cwd() / "checkpoints",
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.monitor,
        patience=cfg.training.patience,
        mode=cfg.training.mode,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    logger.info("Запуск тренировки...")
    try:
        trainer.fit(model, datamodule=data_module)
    except Exception as e:
        # Обрабатываем ошибки finalize MLflow (например, отсутствие boto3 для S3)
        if "boto3" in str(e) or "ModuleNotFoundError" in str(type(e).__name__):
            logger.warning(f"Ошибка при завершении MLflow (возможно, отсутствует boto3): {e}")
            logger.info("Тренировка завершена успешно, но логирование артефактов в MLflow не удалось")
        else:
            raise
    
    logger.info("Запуск тестирования...")
    test_results = trainer.test(model, datamodule=data_module)

    logger.info(f"Результаты теста: {test_results}")

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Лучшая модель сохранена: {best_model_path}")

        if cfg.training.mlflow.log_artifacts:
            try:
                if mlflow_logger.run_id:
                    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, best_model_path)
                    logger.info(f"Модель залогирована в MLflow: {mlflow_logger.run_id}")
            except Exception as e:
                logger.warning(f"Не удалось залогировать модель в MLflow: {e}")

    logger.info("Пример предсказания...")
    sample_texts = [
        "Банк России сохранил ключевую ставку на прежнем уровне",
        "Центробанк принял решение повысить ключевую ставку",
        "Совет директоров Банка России снизил ставку",
    ]

    predictions = model.predict(
        sample_texts, tokenizer=data_module.test_dataset.tokenizer, device=model.device
    )

    for i, pred in enumerate(predictions):
        logger.info(f"Текст: {sample_texts[i]}")
        logger.info(f"Предсказание: {pred['prediction']}")
        logger.info(f"Вероятности: {pred['probabilities']}")

    logger.info("Тренировка завершена успешно!")


if __name__ == "__main__":
    train()
