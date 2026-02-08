#!/usr/bin/env python3

import sys
from pathlib import Path

import fire

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cbr_news.train_multitask import train_multitask


def main():
    print("Запуск тренировки multi-task модели для предсказания ключевой ставки...")
    print("=" * 80)
    print("Эта модель использует подход multi-task learning:")
    print("- Общий encoder (RuBERT) для извлечения признаков из новостей")
    print("- Вспомогательные задачи: предсказание USD, EUR, CNY, инфляции, RUONIA")
    print("- Основная задача: предсказание изменения ключевой ставки ЦБ")
    print("=" * 80)
    print()

    train_multitask()


if __name__ == "__main__":
    fire.Fire(main)
