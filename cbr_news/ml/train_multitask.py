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

from cbr_news.ml.dataset import CBRNewsMultiTaskDataModule
from cbr_news.ml.models.multitask_model import CBRNewsMultiTaskModel
from cbr_news.ml.utils import log_git_info, setup_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="multitask_config", version_base="1.3")
def train_multitask(cfg: DictConfig):
    """Основная функция тренировки multi-task модели"""
    setup_logging()
    logger.info("Начало тренировки multi-task модели")

    # Оптимизация для Tensor Cores (RTX 30xx/40xx)
    import torch
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')  # 'medium' or 'high' для лучшей производительности
        logger.info("Tensor Cores оптимизация включена (float32_matmul_precision='medium')")

    logger.info(f"Конфигурация:\n{OmegaConf.to_yaml(cfg)}")

    log_git_info()

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.training.mlflow.experiment_name + "_multitask",
        tracking_uri=cfg.training.mlflow.tracking_uri,
        log_model=False,
    )

    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    pl.seed_everything(cfg.seed, workers=True)

    data_module = CBRNewsMultiTaskDataModule(cfg)
    data_module.setup()

    model = CBRNewsMultiTaskModel(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.monitor,
        mode=cfg.training.mode,
        save_top_k=1,
        save_last=True,
        filename="cbr-multitask-{epoch:02d}-{val_loss:.2f}",
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
        accelerator=cfg.training.get('accelerator', 'auto'),
        devices=cfg.training.get('devices', 1),
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

    logger.info("Запуск тренировки multi-task модели...")
    try:
        trainer.fit(model, datamodule=data_module)
    except Exception as e:
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
        logger.info(f"\nТекст: {sample_texts[i]}")
        logger.info(f"Предсказание RUONIA: {pred['prediction']}")
        logger.info(f"Вероятности: {pred['probabilities']}")
        logger.info(f"Вспомогательные предсказания:")
        for task, task_pred in pred['auxiliary_predictions'].items():
            logger.info(f"  {task}: {task_pred['prediction']} (вероятности: {task_pred['probabilities']})")

    logger.info("Тренировка multi-task модели завершена успешно!")


if __name__ == "__main__":
    train_multitask()
