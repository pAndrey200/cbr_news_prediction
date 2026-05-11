# CBR Key Rate Predictor

Двухэтапный конвейер прогнозирования решений Банка России по ключевой ставке на основе новостных текстов и макроэкономических данных.

**Результаты:** Stage 1 macro-F1 = 0.649 (тест 2025), Stage 2 macro-F1 = 0.830 (12 заседаний июль 2024 — декабрь 2025).
д
## Архитектура

```
                     ┌─────────────────────────────────────────┐
Новости за N дней    │  Stage 1: Article Scorer                │
до заседания ──────► │  RuBERT (rubert-base-cased)             │──► P(cut/hold/hike)
                     │  + табличные признаки (RUONIA-тренд)    │    на статью
                     └─────────────────────────────────────────┘
                                        │
                                        ▼ агрегация по окну
                     ┌─────────────────────────────────────────┐
                     │  Stage 2: Rate Predictor                │
Макроэкономика ────► │  GradientBoosting (sklearn)             │──► cut / hold / hike
+ структура ────────►│  BERT_w-эмбеддинги + табличные +        │    по заседанию
  заседания          │  структурные признаки заседания         │
                     └─────────────────────────────────────────┘
```

**Stage 1** обучается на слабой разметке: каждая статья в N-дневном окне перед заседанием наследует метку решения по ставке.

**Stage 2** принимает на вход агрегированные эмбеддинги из Stage 1 + макроэкономические признаки + признаки инерции заседаний (streak, days_since_change, hold_streak и др.).

## Структура проекта

```
cbr_news/
├── ml/
│   ├── inference.py            # двухэтапный инференс (Stage 1 + 2)
│   ├── train_joint.py          # модель и DataModule для Stage 1
│   ├── sweep.py                # поиск гиперпараметров Stage 1
│   ├── sweep_meeting.py        # поиск гиперпараметров Stage 2
│   ├── feature_engineering.py  # табличные признаки (RUONIA, курсы, инфляция)
│   ├── prepare_news_dataset.py # подготовка датасета статей
│   ├── prepare_meeting_dataset.py  # подготовка датасета заседаний
│   ├── dataset.py              # Dataset / DataModule
│   ├── models/                 # классы моделей PyTorch Lightning
│   └── utils.py
├── parsing/
│   ├── news_parser.py          # парсер cbr.ru
│   ├── sync_data.py            # периодическая синхронизация данных
│   └── data_loader.py
├── workers/
│   ├── celery_app.py           # Celery app (Redis broker)
│   └── tasks.py                # задачи обучения и инференса
└── database/
    ├── models.py               # SQLAlchemy модели
    ├── task_repository.py      # CRUD задач (async + sync)
    └── db.py

api/                            # FastAPI приложение
bot/                            # Telegram-бот
configs/                        # Hydra конфигурации
k8s/                            # Kubernetes манифесты
checkpoints/
└── sweep/run_002/              # Stage 1: best-v24.ckpt (production)
data/                           # CSV и Parquet файлы
sweep_results/                  # результаты экспериментов
```

## Быстрый старт (Docker Compose)

### Предварительные требования

- Docker + Docker Compose
- Чекпоинт Stage 1: `checkpoints/sweep/run_002/best-v24.ckpt`
- Данные: `data/` (CSV + Parquet файлы)

### Запуск

```bash
# Скопировать и настроить переменные окружения
cp .env.example .env
# Отредактировать .env: TELEGRAM_BOT_TOKEN, POSTGRES_PASSWORD

# Собрать и запустить все сервисы
docker compose up -d --build

# Проверить статус
docker compose ps
curl http://localhost:8000/health
```

### Переменные окружения (.env)

```env
POSTGRES_USER=cbr_user
POSTGRES_PASSWORD=cbr_password
POSTGRES_DB=cbr_news
TELEGRAM_BOT_TOKEN=<токен бота>
```

### Сервисы

| Сервис | Порт | Описание |
|--------|------|----------|
| `api` | 8000 | FastAPI — REST API |
| `celery-worker` | — | фоновые задачи обучения/инференса |
| `parser` | — | периодический парсинг cbr.ru (каждые 6 ч) |
| `bot` | — | Telegram-бот |
| `postgres` | 5432 | база задач и новостей |
| `redis` | 6379 | брокер Celery |

### API эндпоинты

```bash
# Статус
GET  /health

# Классификация произвольного текста (Stage 1)
POST /predict
{"texts": ["Банк России сохранил ключевую ставку на уровне 21%"]}

# Прогноз по последним новостям из БД (Stage 1)
GET  /predict_news?limit=5

# Прогноз решения по ставке (Stage 2, полный конвейер)
GET  /predict_key_rate?window_days=14

# Последние новости из БД
GET  /news?limit=10

# Макроэкономические индикаторы
GET  /indicators

# Управление задачами (требует HTTP Basic auth)
POST /tasks/train
POST /tasks/predict
GET  /tasks/{task_id}
GET  /tasks
```

Документация Swagger (Basic auth `admin:cbr-admin`):
```
http://localhost:8000/docs
```

### Логи и остановка

```bash
docker compose logs -f api
docker compose logs -f celery-worker
docker compose down          # остановить
docker compose down -v       # остановить + удалить тома
```

---

## Обучение

### 1. Установка зависимостей

```bash
poetry install
poetry shell
```

### 2. Подготовка данных

```bash
# Подготовка датасета статей (Stage 1)
python cbr_news/ml/prepare_news_dataset.py

# Подготовка датасета заседаний (Stage 2)
PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py
```

### 3. Stage 1 — Article Scorer (RuBERT)

Поиск гиперпараметров (полная сетка, ~24 конфигурации):
```bash
python cbr_news/ml/sweep.py
```

Быстрая проверка (2 конфигурации):
```bash
python cbr_news/ml/sweep.py --max-runs 2
```

Только подготовка данных без обучения:
```bash
python cbr_news/ml/sweep.py --prepare-only
```

Возобновление с определённого шага:
```bash
python cbr_news/ml/sweep.py --resume-from 5 --skip-prepare
```

Чекпоинты сохраняются в `checkpoints/sweep/run_NNN/`.

### 4. Stage 2 — Rate Predictor (GradientBoosting)

Поиск лучшей конфигурации признаков и классификатора:
```bash
PYTHONPATH=. python cbr_news/ml/sweep_meeting.py \
    --checkpoint checkpoints/sweep/run_002/best-v24.ckpt
```

Результаты сохраняются в `sweep_results/sweep_meeting_results.csv`.

Stage 2 также обучается автоматически при первом вызове `/predict_key_rate` (кэшируется в `data/rate_model_cache_v2_*.pkl`).

### 5. Обучение через Celery API

Запустить задачу обучения Stage 1 через API:
```bash
curl -X POST http://localhost:8000/tasks/train \
  -H "Content-Type: application/json" \
  -u admin:cbr-admin \
  -d '{"overrides": ["training.max_epochs=5"]}'

# Проверить статус
curl http://localhost:8000/tasks/<task_id> -u admin:cbr-admin
```

---

## Kubernetes (Minikube)

### Предварительные требования

```bash
# Установить minikube и kubectl
minikube start --memory=8192 --cpus=4
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

### Деплой

```bash
# Первый запуск (сборка образа + применение всех манифестов)
bash k8s/deploy-minikube.sh --rebuild

# Повторный деплой без пересборки образа
bash k8s/deploy-minikube.sh
```

Скрипт автоматически:
1. Стартует Minikube если не запущен
2. Монтирует `./checkpoints` и `./data` в кластер
3. Собирает Docker-образ и загружает в Minikube
4. Применяет все манифесты (`namespace`, `configmap`, `secrets`, `postgres`, `redis`, `api`, `celery-worker`, `bot`, `ingress`)

### Статус и доступ

```bash
kubectl get pods -n cbr-news
kubectl logs -f deploy/api -n cbr-news

# URL API
minikube service api -n cbr-news --url
```

### Структура манифестов

```
k8s/
├── namespace.yaml        # namespace: cbr-news
├── configmap.yaml        # переменные окружения (CHECKPOINT_PATH и др.)
├── secrets.yaml          # POSTGRES_PASSWORD, TELEGRAM_BOT_TOKEN
├── storage.yaml          # PersistentVolume для checkpoints и data
├── postgres.yaml         # StatefulSet + Service
├── redis.yaml            # Deployment + Service
├── api.yaml              # Deployment + Service (port 8000)
├── celery-worker.yaml    # Deployment
├── bot.yaml              # Deployment
├── cronjob-parser.yaml   # CronJob парсинга (каждые 6 ч)
├── ingress.yaml          # NGINX Ingress (публичные /health /predict*, защищённые /tasks /docs)
└── deploy-minikube.sh    # скрипт полного деплоя
```

### Ingress

Публичные эндпоинты (без авторизации):
- `/health`, `/predict`, `/predict_news`, `/predict_key_rate`

Защищённые (HTTP Basic auth `admin:cbr-admin`):
- `/docs`, `/tasks`

### Ручной запуск парсера

```bash
kubectl create job --from=cronjob/cbr-parser parse-now -n cbr-news
kubectl logs -f job/parse-now -n cbr-news
```

---

## Данные

| Файл | Описание |
|------|----------|
| `data/key-rates-cbr.csv` | история ключевой ставки ЦБ |
| `data/ruonia-cbr.csv` | ставка RUONIA (метка для Stage 1) |
| `data/cbr-press-releases.csv` | пресс-релизы заседаний |
| `data/cbr_multitask_dataset.csv` | размеченный датасет статей |
| `data/meeting_dataset.parquet` | датасет заседаний (Stage 2) |
| `data/tabular_features.parquet` | табличные признаки заседаний |
| `data/news_collection.parquet` | коллекция новостей |
| `data/cur-{usd,eur,cny}-cbr.csv` | курсы валют |
| `data/inflation-cbr.csv` | данные об инфляции |
| `data/brent-oil.csv` | цена нефти Brent |

Внешний датасет новостей: [Russian Financial News](https://www.kaggle.com/datasets/kkhubiev/russian-financial-news) (91 955 статей, Apache 2.0).

---

## Контакты

**Андрей Поцелуев** — ampotseluev@edu.hse.ru
