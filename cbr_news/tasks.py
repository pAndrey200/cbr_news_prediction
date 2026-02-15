"""Celery-задачи для обучения и предсказания."""

import logging
import os
import traceback
from pathlib import Path
from uuid import UUID

from cbr_news.celery_app import celery_app
from cbr_news.database import SessionLocal
from cbr_news.models import TaskStatus
from cbr_news.task_repository import TaskRepositorySync

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_checkpoint(path: str) -> str:
    """Находит .ckpt файл в директории или возвращает путь к файлу.
    Возвращает None если чекпоинт не найден."""
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file() and p.suffix == ".ckpt":
        return str(p)
    if p.is_dir():
        files = list(p.rglob("*.ckpt"))
        if not files:
            return None
        return str(max(files, key=lambda f: f.stat().st_mtime))
    return None


def _default_checkpoint_path() -> str:
    """Возвращает путь к чекпоинту из переменных окружения."""
    env_path = os.environ.get("CHECKPOINT_PATH")
    if env_path:
        resolved = _resolve_checkpoint(env_path)
        if resolved:
            return resolved
    # Фолбэк — проверяем стандартные директории
    for candidate in (_PROJECT_ROOT / "checkpoints", _PROJECT_ROOT / "outputs"):
        resolved = _resolve_checkpoint(str(candidate))
        if resolved:
            return resolved
    return None


@celery_app.task(bind=True, name="cbr_news.tasks.run_training")
def run_training(self, task_id: str, params: dict):
    """
    Запуск обучения модели как Celery-задача.

    Использует Hydra Compose API вместо @hydra.main декоратора,
    чтобы избежать проблем с глобальным состоянием и сменой рабочей директории.
    """
    session = SessionLocal()
    task_uuid = UUID(task_id)
    try:
        TaskRepositorySync.update_status(
            session, task_uuid, TaskStatus.running.value,
            celery_task_id=self.request.id,
        )

        import torch
        import pytorch_lightning as pl
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
        from pytorch_lightning.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )
        from pytorch_lightning.loggers import MLFlowLogger

        from cbr_news.dataset import CBRNewsMultiTaskDataModule
        from cbr_news.multitask_model import CBRNewsMultiTaskModel

        config_dir = str(_PROJECT_ROOT / "configs")
        overrides = params.get("overrides", [])

        # Очищаем глобальное состояние Hydra перед инициализацией
        GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="multitask_config", overrides=overrides)

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")

        pl.seed_everything(cfg.seed, workers=True)

        logger.info("Конфигурация:\n%s", OmegaConf.to_yaml(cfg))

        # Данные
        data_module = CBRNewsMultiTaskDataModule(cfg)
        data_module.setup()

        # Модель
        model = CBRNewsMultiTaskModel(cfg)

        # Директория для вывода
        output_dir = _PROJECT_ROOT / "outputs" / "celery_training" / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.training.monitor,
            mode=cfg.training.mode,
            save_top_k=1,
            save_last=True,
            filename="cbr-multitask-{epoch:02d}-{val_loss:.2f}",
            dirpath=str(output_dir / "checkpoints"),
        )

        early_stopping = EarlyStopping(
            monitor=cfg.training.monitor,
            patience=cfg.training.patience,
            mode=cfg.training.mode,
            verbose=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.training.mlflow.experiment_name + "_celery",
            tracking_uri=cfg.training.mlflow.tracking_uri,
            log_model=False,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=cfg.training.get("accelerator", "auto"),
            devices=cfg.training.get("devices", 1),
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            log_every_n_steps=cfg.training.log_every_n_steps,
            val_check_interval=cfg.training.val_check_interval,
            deterministic=True,
            default_root_dir=str(output_dir),
        )

        logger.info("Запуск обучения (task_id=%s)...", task_id)
        trainer.fit(model, datamodule=data_module)

        logger.info("Запуск тестирования...")
        test_results = trainer.test(model, datamodule=data_module)

        best_path = checkpoint_callback.best_model_path

        result_payload = {
            "best_checkpoint": best_path,
            "test_results": test_results,
            "output_dir": str(output_dir),
        }

        TaskRepositorySync.update_status(
            session, task_uuid, TaskStatus.completed.value,
            result=result_payload,
        )
        logger.info("Обучение завершено (task_id=%s)", task_id)
        return result_payload

    except Exception as e:
        logger.exception("Ошибка обучения (task_id=%s): %s", task_id, e)
        TaskRepositorySync.update_status(
            session, task_uuid, TaskStatus.failed.value,
            error=traceback.format_exc(),
        )
        raise
    finally:
        session.close()


@celery_app.task(bind=True, name="cbr_news.tasks.run_prediction")
def run_prediction(self, task_id: str, params: dict):
    """
    Запуск батч-предсказания как Celery-задача.
    """
    session = SessionLocal()
    task_uuid = UUID(task_id)
    try:
        TaskRepositorySync.update_status(
            session, task_uuid, TaskStatus.running.value,
            celery_task_id=self.request.id,
        )

        texts = params.get("texts", [])

        # Разрешаем путь к чекпоинту: из параметров → CHECKPOINT_PATH → стандартные директории
        checkpoint_path = params.get("checkpoint_path")
        if checkpoint_path:
            checkpoint_path = _resolve_checkpoint(checkpoint_path)
        if not checkpoint_path:
            checkpoint_path = _default_checkpoint_path()

        if not checkpoint_path:
            raise FileNotFoundError(
                "Чекпоинт модели не найден. Запустите обучение через /tasks/train"
            )

        config_path = params.get("config_path")
        logger.info("Используется чекпоинт: %s", checkpoint_path)

        from cbr_news.inference import CBRNewsPredictor

        predictor = CBRNewsPredictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            multitask=True,
        )

        results = predictor.predict(texts)

        # Конвертируем в JSON-совместимый формат
        serializable = []
        for r in results:
            item = {
                "text": r.get("text", ""),
                "prediction": r.get("prediction", ""),
                "probabilities": {
                    k: float(v) for k, v in r.get("probabilities", {}).items()
                },
            }
            if "auxiliary_predictions" in r:
                item["auxiliary_predictions"] = r["auxiliary_predictions"]
            serializable.append(item)

        result_payload = {"predictions": serializable}

        TaskRepositorySync.update_status(
            session, task_uuid, TaskStatus.completed.value,
            result=result_payload,
        )
        logger.info("Предсказание завершено (task_id=%s)", task_id)
        return result_payload

    except Exception as e:
        logger.exception("Ошибка предсказания (task_id=%s): %s", task_id, e)
        TaskRepositorySync.update_status(
            session, task_uuid, TaskStatus.failed.value,
            error=traceback.format_exc(),
        )
        raise
    finally:
        session.close()
