"""Periodic sync of CBR news and economic indicators to the database.

Usage modes:
    # Long-running service (Docker Compose):
    python -m cbr_news.parsing.sync_data --interval-hours 6

    # One-shot (k8s CronJob):
    python -m cbr_news.parsing.sync_data --once

    # One-shot with fewer months (faster):
    python -m cbr_news.parsing.sync_data --once --months-back 1

k8s CronJob example:
    spec:
      schedule: "0 */6 * * *"
      jobTemplate:
        spec:
          template:
            spec:
              containers:
              - name: sync
                image: vkr-api:latest
                command: ["python", "-m", "cbr_news.parsing.sync_data", "--once"]
                envFrom: [secretRef: {name: cbr-secrets}]
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"


def sync_news(months_back: int = 2) -> int:
    """Parse CBR news for the last `months_back` months.

    Uses CBRNewsParser._parse_month which skips already-saved links,
    so calling repeatedly is safe (idempotent).
    Returns count of newly saved articles.
    """
    from cbr_news.parsing.news_parser import CBRNewsParser

    parser = CBRNewsParser(config=None, use_database=True)
    now = datetime.now()
    new_count = 0

    for offset in range(months_back - 1, -1, -1):
        month = now.month - offset
        year = now.year
        while month <= 0:
            month += 12
            year -= 1
        logger.info("Syncing news %d-%02d …", year, month)
        try:
            articles = parser._parse_month(year, month)
            new_count += len(articles)
        except Exception as e:
            logger.error("Failed to sync news for %d-%02d: %s", year, month, e)

    logger.info("News sync done: %d new articles saved", new_count)
    return new_count


def sync_indicators() -> None:
    """Fetch all economic indicators from CBR and save to DB.

    All _save_*_to_db methods use create_or_update, so this is idempotent.
    """
    from cbr_news.parsing.parser import CBRDataParser

    parser = CBRDataParser(config=None, use_database=True)

    steps = [
        ("key rates",  parser.get_key_rate),
        ("inflation",  parser.get_inflation),
        ("USD rate",   parser.get_usd_rate),
        ("EUR rate",   parser.get_eur_rate),
        ("CNY rate",   parser.get_cny_rate),
        ("RUONIA",     parser.get_ruonia),
        ("Brent oil",  parser.get_brent_oil),
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            logger.error("Failed to sync %s: %s", name, e, exc_info=True)

    logger.info("Indicators sync done")


def invalidate_caches() -> None:
    """Remove cached sklearn rate model — rebuilt on next /predict_key_rate call.
    Also remove stale tabular_features parquet so inference uses fresh data.
    """
    for pattern in ("rate_model_cache_*.pkl",):
        for f in _DATA_DIR.glob(pattern):
            try:
                f.unlink()
                logger.info("Removed cache: %s", f.name)
            except Exception as e:
                logger.warning("Could not remove %s: %s", f.name, e)


def run_once(months_back: int = 2) -> None:
    start = datetime.now()
    logger.info("=== Data sync started at %s ===", start.isoformat())

    sync_news(months_back=months_back)
    sync_indicators()

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=== Data sync finished in %.0fs ===", elapsed)


def main() -> None:
    ap = argparse.ArgumentParser(description="CBR data sync service")
    ap.add_argument(
        "--once", action="store_true",
        help="Run once and exit (use this for k8s CronJob)",
    )
    ap.add_argument(
        "--interval-hours", type=float, default=6.0,
        help="Sync interval in hours when running as a daemon (default: 6)",
    )
    ap.add_argument(
        "--months-back", type=int, default=2,
        help="How many months of news to re-check on each sync (default: 2)",
    )
    args = ap.parse_args()

    if args.once:
        run_once(months_back=args.months_back)
        return

    # Daemon mode — run immediately, then sleep and repeat
    while True:
        run_once(months_back=args.months_back)
        sleep_sec = int(args.interval_hours * 3600)
        logger.info("Next sync in %.1fh …", args.interval_hours)
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
