import asyncio
import logging
import os
from html import escape

import httpx
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

API_BASE = os.environ.get("API_URL", "http://localhost:8000")

LABELS = {"down": "⬇️", "same": "➡️", "up": "⬆️"}
LABELS_RU = {"down": "⬇️ Снижение RUONIA", "same": "➡️ Без изменений", "up": "⬆️ Повышение RUONIA"}
RATE_LABELS = {"cut": "⬇️ Снижение ставки", "hold": "➡️ Без изменений", "hike": "⬆️ Повышение ставки"}
STATUS_ICONS = {"pending": "🕐", "running": "⚙️", "completed": "✅", "failed": "❌"}

# Постоянная нижняя клавиатура
REPLY_KB = ReplyKeyboardMarkup(
    [[KeyboardButton("📊 Анализ")]],
    resize_keyboard=True,
    is_persistent=True,
)

# Инлайн-кнопка под результатом анализа
KB_REFRESH = InlineKeyboardMarkup(
    [[InlineKeyboardButton("🔄 Обновить", callback_data="full_predict")]]
)


async def _api_get(path: str, params: dict = None, timeout: float = 15.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(f"{API_BASE.rstrip('/')}{path}", params=params or {})
        r.raise_for_status()
        return r.json()


async def _api_post(path: str, json: dict, timeout: float = 15.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{API_BASE.rstrip('/')}{path}", json=json)
        r.raise_for_status()
        return r.json()


async def _poll_task(task_id: str, max_wait: float = 60.0, interval: float = 2.0) -> dict:
    elapsed = 0.0
    while elapsed < max_wait:
        data = await _api_get(f"/tasks/{task_id}")
        if data["status"] in ("completed", "failed"):
            return data
        await asyncio.sleep(interval)
        elapsed += interval
    return await _api_get(f"/tasks/{task_id}")


def _fmt_date(iso: str) -> str:
    try:
        y, m, d = iso[:10].split("-")
        return f"{d}.{m}.{y}"
    except Exception:
        return iso[:10]


async def _run_full_predict(message) -> None:
    """Полный анализ: прогноз ставки + индикаторы + новости с RUONIA-сигналами."""
    rate_data, ind, news_resp = await asyncio.gather(
        _api_get("/predict_key_rate", timeout=120.0),
        _api_get("/indicators", timeout=15.0),
        _api_get("/news", params={"limit": 10}, timeout=15.0),
        return_exceptions=True,
    )

    if isinstance(rate_data, httpx.HTTPStatusError) and rate_data.response.status_code == 503:
        await message.reply_text("Модель не загружена. Попробуйте позже.")
        return
    if isinstance(rate_data, Exception):
        logger.exception("Ошибка predict_key_rate: %s", rate_data)
        await message.reply_text("Не удалось получить прогноз. Попробуйте позже.")
        return

    # RUONIA-сигналы по последним новостям через Celery
    news_list = news_resp.get("news", []) if isinstance(news_resp, dict) else []
    ruonia_labels: list[str] = []
    if news_list:
        texts = [n.get("text") or n.get("title") or "" for n in news_list]
        texts = [t for t in texts if t]
        try:
            task = await _api_post("/tasks/predict", json={"texts": texts})
            result = await _poll_task(task["id"], max_wait=60.0)
            if result["status"] == "completed":
                preds = (result.get("result") or {}).get("predictions", [])
                ruonia_labels = [p.get("prediction", "same") for p in preds]
        except Exception as e:
            logger.warning("RUONIA predictions failed: %s", e)

    pred = rate_data.get("prediction", "?")
    current = rate_data.get("current_rate", 0)
    n_articles = rate_data.get("n_articles", 0)
    pred_ru = RATE_LABELS.get(pred, pred)

    lines = [
        "📊 <b>АНАЛИТИКА ЦБ РФ</b>",
        "",
        f"🏦 <b>Прогноз ставки:</b> {escape(pred_ru)}",
        f"Текущая ставка: <b>{current:.1f}%</b>",
        f"<i>Проанализировано статей: {n_articles}</i>",
    ]

    if isinstance(ind, dict) and ind:
        lines.append("")
        lines.append("📈 <b>Рыночные данные</b>")
        for key, label, fmt in (
            ("USD",       "USD",        lambda v: f"{v:.2f} ₽"),
            ("EUR",       "EUR",        lambda v: f"{v:.2f} ₽"),
            ("CNY",       "CNY",        lambda v: f"{v:.2f} ₽"),
            ("ruonia",    "RUONIA",     lambda v: f"{v:.2f}%"),
            ("inflation", "Инфляция",   lambda v: f"{v:.1f}%"),
            ("brent",     "Brent",      lambda v: f"${v:.1f}"),
        ):
            if key in ind:
                lines.append(
                    f"{label}   <b>{fmt(ind[key]['value'])}</b>"
                    f"  <i>({_fmt_date(ind[key]['date'])})</i>"
                )

    if news_list:
        lines.append("")
        lines.append("📰 <b>Последние новости ЦБ</b>")
        for i, item in enumerate(news_list):
            title = (item.get("title") or "").strip()
            link = (item.get("link") or "").strip()
            if not title:
                continue
            icon = LABELS.get(ruonia_labels[i], "") if i < len(ruonia_labels) else ""
            if link:
                lines.append(f'{icon} <a href="{escape(link)}">{escape(title)}</a>')
            else:
                lines.append(f"{icon} {escape(title)}")

    await message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=KB_REFRESH,
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот для анализа монетарной политики ЦБ РФ.\n\n"
        "Нажмите <b>📊 Анализ</b> для полного отчёта:\n"
        "прогноз ставки, рыночные данные, сигналы по новостям.\n\n"
        "Или отправьте текст новости — оценю её сигнал.",
        parse_mode=ParseMode.HTML,
        reply_markup=REPLY_KB,
    )


async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args).strip() if context.args else ""
    if text:
        await _do_predict(update, text)
        return
    await update.message.reply_text("🔄 Анализирую...")
    await _run_full_predict(update.message)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.data == "full_predict":
        await query.message.reply_text("🔄 Анализирую...")
        await _run_full_predict(query.message)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return
    if text == "📊 Анализ":
        await update.message.reply_text("🔄 Анализирую...")
        await _run_full_predict(update.message)
        return
    await _do_predict(update, text)


async def _do_predict(update: Update, text: str) -> None:
    try:
        task = await _api_post("/tasks/predict", json={"texts": [text]})
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(
            "Сервис предсказаний временно недоступен." if e.response.status_code == 503
            else f"Ошибка API: {e.response.status_code}"
        )
        return
    except Exception as e:
        logger.exception("Ошибка создания задачи: %s", e)
        await update.message.reply_text("Не удалось поставить задачу в очередь.")
        return

    await update.message.reply_text(f"⏳ Задача {task['id']}\nЖду результат...")
    result = await _poll_task(task["id"])

    if result["status"] == "failed":
        await update.message.reply_text(f"❌ Ошибка:\n{(result.get('error') or '')[:300]}")
        return
    preds = (result.get("result") or {}).get("predictions", [])
    if not preds:
        await update.message.reply_text("Нет результатов от модели.")
        return

    pred_ru = LABELS_RU.get(preds[0].get("prediction", "?"), preds[0].get("prediction", "?"))
    await update.message.reply_text(
        f"Оценка новости: {pred_ru}",
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("📊 Полный анализ", callback_data="full_predict")]]
        ),
    )


async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    overrides = list(context.args) if context.args else []
    try:
        task = await _api_post("/tasks/train", json={"overrides": overrides})
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"Ошибка запуска обучения: {e.response.status_code}")
        return
    except Exception as e:
        logger.exception("Ошибка создания задачи обучения: %s", e)
        await update.message.reply_text("Не удалось запустить обучение.")
        return
    overrides_str = f"\nПараметры: {', '.join(overrides)}" if overrides else ""
    await update.message.reply_text(
        f"🚀 Обучение запущено!{overrides_str}\nID: {task['id']}\n\n/status {task['id']}"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Использование: /status <task_id>")
        return
    task_id = context.args[0].strip()
    try:
        task = await _api_get(f"/tasks/{task_id}")
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(
            f"Задача {task_id} не найдена." if e.response.status_code == 404
            else f"Ошибка API: {e.response.status_code}"
        )
        return
    except Exception as e:
        logger.exception("Ошибка запроса статуса: %s", e)
        await update.message.reply_text("Не удалось получить статус задачи.")
        return

    icon = STATUS_ICONS.get(task["status"], "🔄")
    task_type = "Обучение" if task["task_type"] == "train" else "Предсказание"
    lines = [f"{icon} {task_type} | {task['status']}", f"ID: {task['id']}"]
    for key, label in (("created_at", "Создана"), ("started_at", "Начата"), ("completed_at", "Завершена")):
        if task.get(key):
            lines.append(f"{label}: {task[key][:19].replace('T', ' ')}")
    if task["status"] == "completed" and task["task_type"] == "train":
        result = task.get("result") or {}
        if result.get("best_checkpoint"):
            lines.append(f"\nЧекпоинт: {result['best_checkpoint']}")
        for m in (result.get("test_results") or [])[:1]:
            if m.get("test_acc"):
                lines.append(f"Accuracy: {float(m['test_acc']):.3f}")
            if m.get("test_f1"):
                lines.append(f"F1: {float(m['test_f1']):.3f}")
    elif task["status"] == "failed":
        lines.append(f"\n❌ {(task.get('error') or '')[:300]}")
    await update.message.reply_text("\n".join(lines))


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Установите переменную окружения TELEGRAM_BOT_TOKEN")

    request = HTTPXRequest(
        connection_pool_size=4,
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )
    app = Application.builder().token(token).request(request).get_updates_request(request).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(CommandHandler("train", cmd_train))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES, bootstrap_retries=-1)


if __name__ == "__main__":
    main()
