import logging
import os

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

API_BASE = os.environ.get("API_URL", "http://localhost:8000")


def _api_get(path: str, params: dict = None) -> dict:
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{API_BASE.rstrip('/')}{path}", params=params or {})
        r.raise_for_status()
        return r.json()


def _api_post(path: str, json: dict) -> dict:
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{API_BASE.rstrip('/')}{path}", json=json)
        r.raise_for_status()
        return r.json()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот для предсказания направления ключевой ставки ЦБ по новостям.\n\n"
        "Команды:\n"
        "/predict_news — предсказание по последним новостям с сайта ЦБ\n"
        "/predict <текст> — предсказание по введённому тексту\n\n"
        "Или просто отправьте текст новости — я верну предсказание (down/same/up)."
    )


async def cmd_predict_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    limit = 5
    if context.args and context.args[0].isdigit():
        limit = max(1, min(int(context.args[0]), 20))
    try:
        data = _api_get("/predict_news", params={"limit": limit})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            await update.message.reply_text("Сервис предсказаний временно недоступен (модель не загружена).")
        elif e.response.status_code == 502:
            await update.message.reply_text("Не удалось загрузить новости с сайта ЦБ. Попробуйте позже.")
        else:
            await update.message.reply_text(f"Ошибка API: {e.response.status_code}")
        return
    except Exception as e:
        logger.exception("Ошибка запроса к API: %s", e)
        await update.message.reply_text("Не удалось связаться с сервисом предсказаний. Попробуйте позже.")
        return

    summary = data.get("summary", {})
    rec = summary.get("recommendation", "same")
    labels = {"down": "⬇️ Снижение", "same": "➡️ Без изменений", "up": "⬆️ Повышение"}
    rec_ru = labels.get(rec, rec)

    lines = [
        f"Итог по последним {len(data.get('predictions', []))} новостям:",
        f"Рекомендация: {rec_ru}",
        "",
        "По каждой новости:",
    ]
    for i, (news_item, pred) in enumerate(
        zip(data.get("news", [])[:10], data.get("predictions", [])[:10]), 1
    ):
        title = (news_item.get("title") or "")[:60]
        if len((news_item.get("title") or "")) > 60:
            title += "..."
        p = pred.get("prediction", "?")
        lines.append(f"{i}. {title} → {labels.get(p, p)}")
    await update.message.reply_text("\n".join(lines))


async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args).strip() if context.args else ""
    if not text:
        await update.message.reply_text(
            "Использование: /predict <текст новости>\nИли отправьте текст сообщением."
        )
        return
    await _do_predict(update, text)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return
    await _do_predict(update, text)


async def _do_predict(update: Update, text: str) -> None:
    try:
        data = _api_post("/predict", json={"texts": [text]})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            await update.message.reply_text("Сервис предсказаний временно недоступен (модель не загружена).")
        else:
            await update.message.reply_text(f"Ошибка API: {e.response.status_code}")
        return
    except Exception as e:
        logger.exception("Ошибка запроса к API: %s", e)
        await update.message.reply_text("Не удалось связаться с сервисом предсказаний.")
        return

    preds = data.get("predictions", [])
    if not preds:
        await update.message.reply_text("Нет ответа от модели.")
        return
    p = preds[0]
    pred = p.get("prediction", "?")
    probs = p.get("probabilities", {})
    labels = {"down": "⬇️ Снижение", "same": "➡️ Без изменений", "up": "⬆️ Повышение"}
    pred_ru = labels.get(pred, pred)
    prob_str = ", ".join(f"{labels.get(k, k)}: {v:.0%}" for k, v in probs.items())
    await update.message.reply_text(f"Предсказание: {pred_ru}\nВероятности: {prob_str}")


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Установите переменную окружения TELEGRAM_BOT_TOKEN")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("predict_news", cmd_predict_news))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
