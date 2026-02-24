import json
import logging
import subprocess
import sys
from pathlib import Path

import mlflow


def setup_logging(level: int = logging.INFO):
    """Настройка логирования"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log"),
        ],
    )


def log_git_info():
    """Логирование информации о git коммите"""
    try:
        # Получение текущего коммита
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )

        # Получение текущей ветки
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )

        # Логирование в MLflow
        mlflow.set_tag("git_commit", commit_hash)
        mlflow.set_tag("git_branch", branch)

        logging.info(f"Git commit: {commit_hash}")
        logging.info(f"Git branch: {branch}")

        return commit_hash, branch

    except Exception as e:
        logging.warning(f"Не удалось получить информацию о git: {e}")
        return None, None


def save_predictions(predictions: list, filepath: Path):
    """Сохранение предсказаний в JSON"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logging.info(f"Предсказания сохранены в {filepath}")
