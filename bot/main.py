import asyncio
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

LABELS = {"down": "⬇️ Снижение", "same": "➡️ Без изменений", "up": "⬆️ Повышение"}
STATUS_ICONS = {
    "pending": "🕐",
    "running": "⚙️",
    "completed": "✅",
    "failed": "❌",
}


async def _api_get(path: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{API_BASE.rstrip('/')}{path}", params=params or {})
        r.raise_for_status()
        return r.json()


async def _api_post(path: str, json: dict) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(f"{API_BASE.rstrip('/')}{path}", json=json)
        r.raise_for_status()
        return r.json()


async def _poll_task(task_id: str, max_wait: float = 60.0, interval: float = 2.0) -> dict:
    """Опрашивает статус задачи до завершения или таймаута."""
    elapsed = 0.0
    while elapsed < max_wait:
        data = await _api_get(f"/tasks/{task_id}")
        if data["status"] in ("completed", "failed"):
            return data
        await asyncio.sleep(interval)
        elapsed += interval
    return await _api_get(f"/tasks/{task_id}")


def _format_prediction_result(task: dict) -> str:
    """Форматирует результат задачи предсказания."""
    if task["status"] == "failed":
        return f"❌ Ошибка при выполнении задачи:\n{task.get('error', 'неизвестная ошибка')[:300]}"

    if task["status"] != "completed":
        icon = STATUS_ICONS.get(task["status"], "🔄")
        return f"{icon} Задача ещё выполняется (ID: {task['id']})\nПроверьте статус: /status {task['id']}"

    result = task.get("result") or {}
    preds = result.get("predictions", [])
    if not preds:
        return "Нет результатов от модели."

    lines = []
    for i, p in enumerate(preds, 1):
        pred = p.get("prediction", "?")
        probs = p.get("probabilities", {})
        pred_ru = LABELS.get(pred, pred)
        prob_str = ", ".join(f"{LABELS.get(k, k)}: {v:.0%}" for k, v in probs.items())
        text_preview = (p.get("text") or "")[:60]
        if len(p.get("text") or "") > 60:
            text_preview += "..."

        if len(preds) == 1:
            lines.append(f"Предсказание: {pred_ru}")
            lines.append(f"Вероятности: {prob_str}")
            aux = p.get("auxiliary_predictions", {})
            if aux:
                lines.append("\nВспомогательные предсказания:")
                for task_name, aux_pred in aux.items():
                    aux_label = LABELS.get(aux_pred.get("prediction", "?"), aux_pred.get("prediction", "?"))
                    lines.append(f"  {task_name.upper()}: {aux_label}")
        else:
            lines.append(f"{i}. {text_preview} → {LABELS.get(pred, pred)}")

    return "\n".join(lines)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот для предсказания направления RUONIA по новостям ЦБ.\n\n"
        "Команды:\n"
        "/predict_news [N] — предсказание по последним N новостям ЦБ\n"
        "/predict <текст> — предсказание по тексту\n"
        "/train [override=val ...] — запустить обучение модели\n"
        "/status <task_id> — проверить статус задачи\n\n"
        "Или просто отправьте текст новости — верну предсказание."
    )


async def cmd_predict_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    limit = 5
    if context.args and context.args[0].isdigit():
        limit = max(1, min(int(context.args[0]), 20))

    await update.message.reply_text("🔄 Получаю последние новости...")

    try:
        news_data = await _api_get("/news", params={"limit": limit})
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"Ошибка получения новостей: {e.response.status_code}")
        return
    except Exception as e:
        logger.exception("Ошибка запроса новостей: %s", e)
        await update.message.reply_text("Не удалось получить новости. Попробуйте позже.")
        return

    news = news_data.get("news", [])
    if not news:
        await update.message.reply_text("Нет доступных новостей.")
        return

    texts = [n.get("text") or n.get("title") or "" for n in news]
    texts = [t for t in texts if t]

    try:
        task = await _api_post("/tasks/predict", json={"texts": texts})
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"Ошибка при создании задачи: {e.response.status_code}")
        return
    except Exception as e:
        logger.exception("Ошибка создания задачи: %s", e)
        await update.message.reply_text("Не удалось поставить задачу в очередь.")
        return

    task_id = task["id"]
    await update.message.reply_text(f"⏳ Задача в очереди (ID: {task_id})\nЖду результат...")

    result = await _poll_task(task_id)

    if result["status"] == "completed":
        preds_list = (result.get("result") or {}).get("predictions", [])
        pred_labels = [p.get("prediction", "?") for p in preds_list]
        summary = {k: pred_labels.count(k) for k in ("down", "same", "up")}
        rec = max(summary, key=summary.get)

        lines = [
            f"Итог по {len(pred_labels)} новостям:",
            f"Рекомендация: {LABELS.get(rec, rec)}",
            "",
            "По каждой новости:",
        ]
        for i, (news_item, pred) in enumerate(zip(news[:10], preds_list[:10]), 1):
            title = (news_item.get("title") or "")[:60]
            if len(news_item.get("title") or "") > 60:
                title += "..."
            p = pred.get("prediction", "?")
            lines.append(f"{i}. {title} → {LABELS.get(p, p)}")
        await update.message.reply_text("\n".join(lines))
    else:
        await update.message.reply_text(_format_prediction_result(result))


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
        task = await _api_post("/tasks/predict", json={"texts": [text]})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            await update.message.reply_text("Сервис предсказаний временно недоступен.")
        else:
            await update.message.reply_text(f"Ошибка API: {e.response.status_code}")
        return
    except Exception as e:
        logger.exception("Ошибка создания задачи: %s", e)
        await update.message.reply_text("Не удалось поставить задачу в очередь.")
        return

    task_id = task["id"]
    await update.message.reply_text(f"⏳ Задача в очереди (ID: {task_id})\nЖду результат...")

    result = await _poll_task(task_id)
    await update.message.reply_text(_format_prediction_result(result))


async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Запустить обучение модели через очередь."""
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

    task_id = task["id"]
    overrides_str = f"\nПараметры: {', '.join(overrides)}" if overrides else ""
    await update.message.reply_text(
        f"🚀 Обучение запущено!{overrides_str}\n"
        f"ID задачи: {task_id}\n\n"
        f"Проверяйте статус командой:\n/status {task_id}"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Проверить статус задачи по ID."""
    if not context.args:
        await update.message.reply_text("Использование: /status <task_id>")
        return

    task_id = context.args[0].strip()
    try:
        task = await _api_get(f"/tasks/{task_id}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            await update.message.reply_text(f"Задача {task_id} не найдена.")
        else:
            await update.message.reply_text(f"Ошибка API: {e.response.status_code}")
        return
    except Exception as e:
        logger.exception("Ошибка запроса статуса: %s", e)
        await update.message.reply_text("Не удалось получить статус задачи.")
        return

    icon = STATUS_ICONS.get(task["status"], "🔄")
    task_type = "Обучение" if task["task_type"] == "train" else "Предсказание"
    lines = [
        f"{icon} {task_type} | Статус: {task['status']}",
        f"ID: {task['id']}",
    ]
    if task.get("created_at"):
        lines.append(f"Создана: {task['created_at'][:19].replace('T', ' ')}")
    if task.get("started_at"):
        lines.append(f"Начата: {task['started_at'][:19].replace('T', ' ')}")
    if task.get("completed_at"):
        lines.append(f"Завершена: {task['completed_at'][:19].replace('T', ' ')}")

    if task["status"] == "completed":
        if task["task_type"] == "predict":
            lines.append("")
            result_text = _format_prediction_result(task)
            lines.append(result_text)
        elif task["task_type"] == "train":
            result = task.get("result") or {}
            checkpoint = result.get("best_checkpoint", "")
            if checkpoint:
                lines.append(f"\nЛучший чекпоинт: {checkpoint}")
            test_results = result.get("test_results", [])
            if test_results and isinstance(test_results, list) and test_results:
                metrics = test_results[0]
                acc = metrics.get("test_acc", "")
                f1 = metrics.get("test_f1", "")
                if acc:
                    lines.append(f"Accuracy: {float(acc):.3f}")
                if f1:
                    lines.append(f"F1: {float(f1):.3f}")
    elif task["status"] == "failed":
        error = (task.get("error") or "")[:300]
        lines.append(f"\n❌ Ошибка:\n{error}")

    await update.message.reply_text("\n".join(lines))


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Установите переменную окружения TELEGRAM_BOT_TOKEN")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("predict_news", cmd_predict_news))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(CommandHandler("train", cmd_train))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Бот запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
