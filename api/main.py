import logging
import os
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Добавляем корень проекта в path
import sys
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from cbr_news.inference import CBRNewsPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CBR News Rate Prediction API",
    description="Предсказание направления изменения ключевой ставки по новостям Банка России",
    version="0.1.0",
)

predictor: CBRNewsPredictor | None = None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def resolve_checkpoint_file(path: str | Path) -> str | None:
    """
    Возвращает путь к файлу .ckpt.
    """
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


def fetch_latest_news(limit: int = 5) -> list[dict]:
    """
    Загружает последние новости типа events с сайта ЦБ РФ.
    Возвращает список {title, text} для limit новостей.
    """
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
def startup():
    global predictor
    raw = os.environ.get("CHECKPOINT_PATH")
    checkpoint = resolve_checkpoint_file(raw) if raw else None
    if not checkpoint:
        # Пробуем типичные пути (директории Hydra: outputs/дата/время/checkpoints/)
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
            "Чекпоинт не найден.
        )
        predictor = None
        return
    config_path = os.environ.get("CONFIG_PATH") or str(_project_root / "configs" / "config.yaml")
    try:
        predictor = CBRNewsPredictor(
            checkpoint_path=checkpoint,
            config_path=config_path if Path(config_path).exists() else None,
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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Предсказание направления ставки по списку текстов (down/same/up)."""
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
def predict_news(limit: int = 5):
    """
    Загружает последние новости ЦБ, предсказывает по ним направление ставки,
    возвращает новости и предсказания. limit — число новостей (по умолчанию 5).
    """
    if predictor is None:
        raise HTTPException(503, "Модель не загружена")
    limit = max(1, min(limit, 20))
    news = fetch_latest_news(limit=limit)
    if not news:
        raise HTTPException(502, "Не удалось загрузить новости с сайта ЦБ")
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
