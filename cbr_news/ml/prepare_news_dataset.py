"""
Prepare combined training dataset by labeling news_collection.parquet
with RUONIA direction (and auxiliary indicators) loaded from the database,
then merging with the existing cbr_multitask_dataset.csv.

Usage:
    python cbr_news/ml/prepare_news_dataset.py
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parents[2] / "data"
NEWS_PARQUET = DATA_DIR / "news_collection.parquet"
EXISTING_CSV = DATA_DIR / "cbr_multitask_dataset.csv"
OUTPUT_CSV = DATA_DIR / "cbr_combined_dataset.csv"

HORIZON_DAYS = 2      # days forward for label (config: data.prediction_horizon)
# 0.02  → too small: catches daily RUONIA fluctuations unrelated to policy
# 0.10  → recommended: meaningful intraday/next-day policy signal
# 0.25  → strict: corresponds to minimum key-rate step
RUONIA_THRESHOLD = 0.10  # config: data.ruonia_threshold (overridden by --threshold)


# ─── helpers ────────────────────────────────────────────────────────────────

def _load_series_from_db():
    """Return {name: pd.Series(value, index=DatetimeIndex)} for each indicator."""
    from cbr_news.database.db import get_db_session
    from cbr_news.database.repository import (
        RuoniaRepository,
        KeyRateRepository,
        CurrencyRateRepository,
    )

    with get_db_session() as session:
        def _to_series(records, date_attr, val_attr):
            rows = [(getattr(r, date_attr), getattr(r, val_attr)) for r in records]
            df = pd.DataFrame(rows, columns=["date", "value"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna().drop_duplicates("date").sort_values("date")
            return df.set_index("date")["value"]

        ruonia_ts = _to_series(RuoniaRepository.get_all(session), "date", "rate")
        key_rate_ts = _to_series(KeyRateRepository.get_all(session), "date", "rate")
        usd_ts = _to_series(
            CurrencyRateRepository.get_all_by_currency(session, "USD"), "date", "rate"
        )
        eur_ts = _to_series(
            CurrencyRateRepository.get_all_by_currency(session, "EUR"), "date", "rate"
        )

    logger.info(
        "Loaded from DB: RUONIA=%d, key_rate=%d, USD=%d, EUR=%d",
        len(ruonia_ts), len(key_rate_ts), len(usd_ts), len(eur_ts),
    )
    return {
        "ruonia": ruonia_ts,
        "key_rate": key_rate_ts,
        "usd": usd_ts,
        "eur": eur_ts,
    }


def _next_business_day_value(date, ts, n_bdays: int):
    """
    Return the time-series value n_bdays business days after date.

    Uses business-day offset so that Friday+2BD=Tuesday, avoiding the
    weekend collapse where asof(Sunday)==asof(Friday) → delta always 0.
    """
    from pandas.tseries.offsets import BDay
    future = date + BDay(n_bdays)
    if future > ts.index.max():
        return None, None
    cur = ts.asof(date)
    fut = ts.asof(future)
    return cur, fut


def _make_ruonia_label(date, ruonia_ts):
    cur, fut = _next_business_day_value(date, ruonia_ts, HORIZON_DAYS)
    if cur is None or pd.isna(cur) or pd.isna(fut):
        return None
    delta = fut - cur
    if delta > RUONIA_THRESHOLD:
        return "up"
    if delta < -RUONIA_THRESHOLD:
        return "down"
    return "same"


def _make_ruonia_delta(date, ruonia_ts):
    """Return raw float RUONIA change (fut - cur) over HORIZON_DAYS business days, or NaN."""
    import math
    cur, fut = _next_business_day_value(date, ruonia_ts, HORIZON_DAYS)
    if cur is None or pd.isna(cur) or pd.isna(fut):
        return math.nan
    return float(fut - cur)


def _make_aux_label(date, ts):
    future = date + pd.Timedelta(days=HORIZON_DAYS)
    if future > ts.index.max():
        return None
    cur = ts.asof(date)
    fut = ts.asof(future)
    if pd.isna(cur) or pd.isna(fut):
        return None
    return 1 if fut > cur else 0


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?%\-]", "", text)
    return text


def _build_text(row) -> str:
    """Combine title + body; skip 'no title' placeholders."""
    title = str(row["title"]).strip()
    body = str(row["body"]).strip()
    if title.lower() in ("no title", "nan", ""):
        return body
    return f"{title}. {body}"


# ─── main ────────────────────────────────────────────────────────────────────

def prepare(threshold: float = RUONIA_THRESHOLD, cbr_only: bool = False,
            output_path=None):
    global RUONIA_THRESHOLD
    RUONIA_THRESHOLD = threshold
    _output = Path(output_path) if output_path is not None else OUTPUT_CSV
    logger.info("RUONIA threshold: %.4f  cbr_only: %s", RUONIA_THRESHOLD, cbr_only)

    # 1. Load indicator time series
    series = _load_series_from_db()

    # 2. Load raw news
    logger.info("Loading %s", NEWS_PARQUET)
    news = pd.read_parquet(NEWS_PARQUET)
    logger.info("Raw news rows: %d", len(news))

    # 3. Parse dates
    news["parsed_date"] = pd.to_datetime(news["date"], format="mixed", errors="coerce")
    n_bad = news["parsed_date"].isna().sum()
    if n_bad:
        logger.warning("Dropping %d rows with unparseable dates", n_bad)
    news = news[news["parsed_date"].notna()].copy()

    # 4. Build cleaned text and filter very short bodies
    news["raw_text"] = news.apply(_build_text, axis=1)
    news = news[news["raw_text"].str.len() >= 50].copy()
    news["cleaned_text"] = news["raw_text"].apply(_clean_text)
    logger.info("After length filter: %d rows", len(news))

    # 5. Label RUONIA (main task)
    logger.info("Labeling RUONIA (horizon=%d days, threshold=%.2f)…", HORIZON_DAYS, RUONIA_THRESHOLD)
    news["target"] = news["parsed_date"].apply(
        lambda d: _make_ruonia_label(d, series["ruonia"])
    )
    before = len(news)
    news = news[news["target"].notna()].copy()
    logger.info("Dropped %d rows with no RUONIA label (future beyond data range)", before - len(news))

    # 6. Auxiliary labels
    for task, ts_name in [("key_rate", "key_rate"), ("usd", "usd"), ("eur", "eur")]:
        ts = series[ts_name]
        news[f"{task}_label"] = news["parsed_date"].apply(lambda d, _ts=ts: _make_aux_label(d, _ts))

    # 7. Format date as DD.MM.YYYY to match existing dataset
    news["date"] = news["parsed_date"].dt.strftime("%d.%m.%Y")

    # 8. Keep only relevant columns, mark source
    news_out = news[["date", "cleaned_text", "target", "key_rate_label", "usd_label", "eur_label"]].copy()
    news_out.insert(0, "source", "news_collection")
    news_out["news_type"] = "news"  # external news — never a key rate press release

    logger.info("New news rows after labeling: %d", len(news_out))
    logger.info("Target distribution:\n%s", news_out["target"].value_counts())

    # 9. Load existing multitask dataset and RE-LABEL with the current threshold.
    # The CSV may have been generated with a different threshold (e.g. 0.02 from
    # data.yaml). Using stale labels would mix thresholds across sources.
    logger.info("Loading existing dataset: %s", EXISTING_CSV)
    existing = pd.read_csv(EXISTING_CSV)
    existing.insert(0, "source", "cbr_press_releases")

    # Parse dates in DD.MM.YYYY format used by the existing dataset
    existing["_parsed_date"] = pd.to_datetime(
        existing["date"], format="%d.%m.%Y", errors="coerce"
    )
    n_bad = existing["_parsed_date"].isna().sum()
    if n_bad:
        logger.warning("Existing dataset: %d unparseable dates — will drop", n_bad)
        existing = existing[existing["_parsed_date"].notna()].copy()

    logger.info(
        "Re-labeling CBR press releases with threshold=%.2f (horizon=%d BD)…",
        RUONIA_THRESHOLD, HORIZON_DAYS,
    )
    existing["target"] = existing["_parsed_date"].apply(
        lambda d: _make_ruonia_label(d, series["ruonia"])
    )
    existing = existing.drop(columns=["_parsed_date"])

    # Re-compute auxiliary labels to match the same horizon
    for task, ts_name in [("key_rate", "key_rate"), ("usd", "usd"), ("eur", "eur")]:
        ts = series[ts_name]
        existing[f"{task}_label"] = existing["date"].apply(
            lambda d, _ts=ts: _make_aux_label(
                pd.to_datetime(d, format="%d.%m.%Y", errors="coerce"), _ts
            )
        )

    # Drop rows where RUONIA label could not be computed
    before_ex = len(existing)
    existing = existing[existing["target"].notna()].copy()
    logger.info(
        "Existing dataset after re-labeling: %d rows (dropped %d)",
        len(existing), before_ex - len(existing),
    )

    # Keep common columns; existing has extra cols (text, ruonia, inflation, …) not needed for training
    keep_cols = ["source", "date", "cleaned_text", "target", "key_rate_label", "usd_label", "eur_label", "news_type"]
    existing_out = existing.reindex(columns=keep_cols)
    # news_type is set by the parser (e.g. "press_release", "press", "events");
    # fill missing for older CSV files that predate this column
    existing_out["news_type"] = existing_out["news_type"].fillna("press")

    logger.info("Existing rows: %d  target dist:\n%s",
                len(existing_out), existing_out["target"].value_counts().to_string())

    # 10. Combine and deduplicate
    if cbr_only:
        logger.info("cbr_only=True: using only CBR press releases (no news_collection)")
        combined = existing_out.copy()
    else:
        combined = pd.concat([existing_out, news_out], ignore_index=True)

    # Diagnostic: log duplicate breakdown before dropping
    dups_mask = combined.duplicated(subset=["cleaned_text"], keep=False)
    if dups_mask.any():
        dup_df = combined[dups_mask]
        dup_pr = int((dup_df["news_type"] == "press_release").sum())
        dup_cross = int(
            dup_df.duplicated(subset=["cleaned_text"], keep=False).sum() -
            combined[dups_mask & (combined["source"] == "cbr_press_releases")]
            .duplicated(subset=["cleaned_text"], keep=False).sum()
        )
        logger.info(
            "Duplicates found: %d total rows involved, %d are press_release",
            dups_mask.sum(), dup_pr,
        )
        if dup_pr > 0:
            logger.warning(
                "%d press_release rows appear in duplicates — sorting to preserve them", dup_pr
            )

    # Sort so press_release rows come FIRST within each cleaned_text group,
    # ensuring they survive drop_duplicates(keep='first') over other types.
    _pr_priority = (combined["news_type"] == "press_release").astype(int)
    combined = (
        combined.assign(_pr_priority=_pr_priority)
        .sort_values("_pr_priority", ascending=False, kind="stable")
        .drop(columns=["_pr_priority"])
        .reset_index(drop=True)
    )

    before = len(combined)
    combined = combined.drop_duplicates(subset=["cleaned_text"])
    logger.info("Dropped %d duplicates; combined rows: %d", before - len(combined), len(combined))

    # 11. Stats by year
    def _year(d):
        try:
            parts = str(d).split(".")
            if len(parts) == 3:
                return int(parts[2])
        except Exception:
            pass
        return None

    combined["year"] = combined["date"].apply(_year)
    logger.info(
        "Combined target distribution:\n%s",
        combined["target"].value_counts()
    )
    logger.info(
        "Samples per year:\n%s",
        combined["year"].value_counts().sort_index().to_string()
    )
    logger.info(
        "Samples per source:\n%s",
        combined["source"].value_counts().to_string()
    )

    combined = combined.drop(columns=["year"])

    # 12. Add raw RUONIA delta (regression target for auxiliary head)
    ruonia_ts = series["ruonia"]
    combined["_parsed_date"] = pd.to_datetime(combined["date"], format="%d.%m.%Y", errors="coerce")
    combined["ruonia_delta"] = combined["_parsed_date"].apply(
        lambda d: _make_ruonia_delta(d, ruonia_ts) if pd.notna(d) else float("nan")
    )
    combined = combined.drop(columns=["_parsed_date"])
    n_nan = combined["ruonia_delta"].isna().sum()
    if n_nan:
        logger.info("ruonia_delta: %d NaN rows (future beyond data) → filling with 0.0", n_nan)
        combined["ruonia_delta"] = combined["ruonia_delta"].fillna(0.0)
    logger.info("ruonia_delta range: [%.4f, %.4f]",
                combined["ruonia_delta"].min(), combined["ruonia_delta"].max())

    # 13. Mark key rate press releases — used by cbr_weight in train_joint.py.
    # Priority:
    #   A) news_type == "press_release"  — exact DB type set by the CBR press releases parser
    #   B) fallback: CBR source + date matches a Board of Directors meeting date
    if "news_type" in combined.columns and (combined["news_type"] == "press_release").any():
        combined["is_key_rate_pr"] = (combined["news_type"] == "press_release").astype(int)
        n_kpr = int(combined["is_key_rate_pr"].sum())
        logger.info("is_key_rate_pr: %d articles (news_type='press_release')", n_kpr)
    else:
        logger.info("news_type='press_release' not found — falling back to meeting-date matching")
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parents[2]))
            from cbr_news.ml.prepare_meeting_dataset import CBR_MEETING_DATES
            _meeting_dates = {pd.Timestamp(d).strftime("%d.%m.%Y") for d in CBR_MEETING_DATES}
            combined["is_key_rate_pr"] = (
                (combined["source"] == "cbr_press_releases") &
                (combined["date"].isin(_meeting_dates))
            ).astype(int)
            n_kpr = int(combined["is_key_rate_pr"].sum())
            logger.info("is_key_rate_pr: %d articles (CBR + meeting date)", n_kpr)
        except Exception as _e:
            logger.warning("Could not compute is_key_rate_pr: %s — column set to 0", _e)
            combined["is_key_rate_pr"] = 0

    # 14. Save
    combined.to_csv(_output, index=False)
    logger.info("Saved combined dataset → %s (%d rows)", _output, len(combined))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=RUONIA_THRESHOLD,
                    help="RUONIA change threshold for up/down labels (default 0.10)")
    ap.add_argument("--cbr-only", action="store_true",
                    help="Use only CBR press releases (skip news_collection)")
    ap.add_argument("--output", type=str, default=str(OUTPUT_CSV),
                    help="Output CSV path")
    args = ap.parse_args()

    OUTPUT_CSV = Path(args.output)
    prepare(threshold=args.threshold, cbr_only=args.cbr_only)
