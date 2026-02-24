#!/usr/bin/env python3

import sys
from pathlib import Path

import fire

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cbr_news.ml.train import train


def main():
    print("Запуск тренировки модели классификации новостей Банка России...")
    print("=" * 60)

    train()


if __name__ == "__main__":
    fire.Fire(main)
