import logging
import os
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import sys

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from cbr_news.database.db import get_db, init_db, async_init_db
from cbr_news.ml.inference import CBRNewsPredictor
from cbr_news.database.repository import NewsRepository
from api.tasks_router import router as tasks_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CBR News Rate Prediction API",
    description="Предсказание направления изменения ключевой ставки по новостям Банка России",
    version="0.1.0",
)

app.include_router(tasks_router)

predictor: CBRNewsPredictor | None = None
USE_DATABASE = os.environ.get("USE_DATABASE", "true").lower() == "true"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def resolve_checkpoint_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return str(p)
    if p.is_dir():
        files = list(p.rglob("*.ckpt"))
        if not files:
            return None
        return str(max(files, key=lambda f: f.stat().st_mtime))
    return None


def fetch_latest_news_from_db(db: Session, limit: int = 5) -> list[dict]:
    try:
        news_list = NewsRepository.get_latest_news(db, limit=limit)
        result = []
        for news in news_list:
            result.append({
                "title": news.title,
                "text": news.content or news.title,
                "date": news.date.isoformat() if news.date else "",
                "link": news.link,
            })
        return result
    except Exception as e:
        logger.error("Ошибка при получении новостей из БД: %s", e)
        return []


def fetch_latest_news_from_web(limit: int = 5) -> list[dict]:
    base_url = "https://www.cbr.ru/news/eventandpress/"
    events_data: list = []
    try:
        resp = httpx.get(
            base_url,
            params={
                "page": 0,
                "IsEng": "false",
                "type": "100",
                "dateFrom": "",
                "dateTo": "",
                "Tid": "",
                "vol": "",
                "phrase": "",
            },
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        events_data = [e for e in resp.json() if e.get("TBLType") == "events"]
    except Exception as e:
        logger.warning("Ошибка загрузки списка новостей: %s", e)
        return []

    result = []
    for event in events_data[:limit]:
        doc_htm = event.get("doc_htm")
        if not doc_htm:
            continue
        title = event.get("name_doc", "")
        url = f"https://www.cbr.ru/press/event/?id={doc_htm}"
        try:
            r = httpx.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            content = (
                soup.select_one(".landing-text")
                or soup.select_one(".news-content")
                or soup.select_one("article")
                or soup.find("main")
            )
            text = content.get_text(separator=" ", strip=True) if content else ""
        except Exception as e:
            logger.warning("Ошибка загрузки новости %s: %s", doc_htm, e)
            text = ""
        result.append({"title": title, "text": text or title})
    return result


@app.on_event("startup")
async def startup():
    global predictor

    if USE_DATABASE:
        try:
            await async_init_db()
            logger.info("База данных инициализирована успешно")
        except Exception as e:
            logger.error("Не удалось инициализировать БД: %s", e)

    raw = os.environ.get("CHECKPOINT_PATH")
    checkpoint = resolve_checkpoint_file(raw) if raw else None
    if not checkpoint:
        for p in (
            _project_root / "checkpoints",
            _project_root / "outputs",
            Path("/app/checkpoints"),
        ):
            if p.exists():
                checkpoint = resolve_checkpoint_file(p)
                if checkpoint:
                    break
    if not checkpoint:
        logger.warning(
            "Чекпоинт не найден. API будет работать без предсказаний."
        )
        predictor = None
        return
    config_path = os.environ.get("CONFIG_PATH") or str(_project_root / "configs" / "multitask_config.yaml")
    try:
        predictor = CBRNewsPredictor(
            checkpoint_path=checkpoint,
            config_path=config_path if Path(config_path).exists() else None,
            multitask=True,
        )
        logger.info("Модель загружена успешно")
    except Exception as e:
        logger.exception("Не удалось загрузить модель: %s", e)
        predictor = None


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    predictions: list[dict]


class PredictNewsResponse(BaseModel):
    news: list[dict]
    predictions: list[dict]
    summary: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/news")
def get_news(limit: int = 5, db: Session = Depends(get_db)):
    """Вернуть последние новости ЦБ без предсказаний (для бота)."""
    limit = max(1, min(limit, 20))
    if USE_DATABASE:
        news = fetch_latest_news_from_db(db, limit=limit)
        if not news:
            news = fetch_latest_news_from_web(limit=limit)
    else:
        news = fetch_latest_news_from_web(limit=limit)
    return {"news": news}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Модель не загружена")
    try:
        results = predictor.predict(request.texts)
        return PredictResponse(predictions=results)
    except Exception as e:
        logger.exception("Ошибка предсказания: %s", e)
        raise HTTPException(500, str(e))


@app.get("/predict_news", response_model=PredictNewsResponse)
@app.post("/predict_news", response_model=PredictNewsResponse)
def predict_news(limit: int = 5, db: Session = Depends(get_db)):
    if predictor is None:
        raise HTTPException(503, "Модель не загружена")
    limit = max(1, min(limit, 20))

    if USE_DATABASE:
        news = fetch_latest_news_from_db(db, limit=limit)
        if not news:
            logger.warning("Нет новостей в БД, пробуем загрузить с сайта")
            news = fetch_latest_news_from_web(limit=limit)
    else:
        news = fetch_latest_news_from_web(limit=limit)

    if not news:
        raise HTTPException(502, "Не удалось загрузить новости")

    texts = [n["text"] or n["title"] for n in news]
    try:
        results = predictor.predict(texts)
    except Exception as e:
        logger.exception("Ошибка предсказания: %s", e)
        raise HTTPException(500, str(e))
    # Считаем итог по большинству
    pred_labels = [r["prediction"] for r in results]
    summary = {
        "down": pred_labels.count("down"),
        "same": pred_labels.count("same"),
        "up": pred_labels.count("up"),
    }
    summary["recommendation"] = max(summary, key=summary.get)
    return PredictNewsResponse(news=news, predictions=results, summary=summary)
