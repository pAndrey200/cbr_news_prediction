"""
Build a meeting-centric dataset for key rate direction prediction.

One training example per CBR Board of Directors meeting:
  Input  : all articles published in the N days BEFORE the meeting date
  Target : "hike" / "hold" / "cut"

Meeting dates are taken from the official CBR calendar (cbr.ru/dkp/cal_mp/).
Label is determined by comparing the key rate N days before vs. N days after
the meeting (to account for the implementation lag of ~3 business days).

Data coverage:
  2014–2025 → ~92 meetings (excluding 2026 future sessions)
  Hike: ~25   Hold: ~45   Cut: ~22 (approx)

Temporal split (default):
  train: 2014 – 2021   (~60 meetings)
  val  : 2022 – 2023   (~19 meetings, includes extraordinary sessions)
  test : 2024 – 2025   (~16 meetings)

Usage:
    PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py
    PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py --window-days 21
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cbr_news.ml.feature_engineering import (
    build_tabular_features,
    extract_text_features,
    get_feature_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parents[2] / "data"
CSV_PATH    = DATA_DIR / "cbr_combined_dataset_t025.csv"
OUTPUT_PATH = DATA_DIR / "meeting_dataset.parquet"

LABEL_MAP   = {"cut": 0, "hold": 1, "hike": 2}
CLASS_NAMES = ["cut", "hold", "hike"]

# ─── Official CBR key rate meeting calendar ────────────────────────────────────
# Source: https://cbr.ru/dkp/cal_mp/  (tabs t1–t12, years 2014–2025)
# Includes both scheduled and extraordinary (внеплановые) sessions.
# 2026 sessions excluded — outcomes not yet known as of 2026-03-01.
CBR_MEETING_DATES = [
    # 2014 (t1)
    "2014-02-14", "2014-03-03",  # extraordinary (Crimea)
    "2014-03-14", "2014-04-25", "2014-06-16", "2014-07-25",
    "2014-09-12", "2014-10-31", "2014-12-11",
    "2014-12-15",  # extraordinary (ruble crash, +6.5pp)
    # 2015 (t2)
    "2015-01-30", "2015-03-13", "2015-04-30", "2015-06-15",
    "2015-07-31", "2015-09-11", "2015-10-30", "2015-12-11",
    # 2016 (t3)
    "2016-01-29", "2016-03-18", "2016-04-29", "2016-06-10",
    "2016-07-29", "2016-09-16", "2016-10-28", "2016-12-16",
    # 2017 (t4)
    "2017-02-03", "2017-03-24", "2017-04-28", "2017-06-16",
    "2017-07-28", "2017-09-15", "2017-10-27", "2017-12-15",
    # 2018 (t5)
    "2018-02-09", "2018-03-23", "2018-04-27", "2018-06-15",
    "2018-07-27", "2018-09-14", "2018-10-26", "2018-12-14",
    # 2019 (t6)
    "2019-02-08", "2019-03-22", "2019-04-26", "2019-06-14",
    "2019-07-26", "2019-09-06", "2019-10-25", "2019-12-13",
    # 2020 (t7)
    "2020-02-07", "2020-03-20", "2020-04-24", "2020-06-19",
    "2020-07-24", "2020-09-18", "2020-10-23", "2020-12-18",
    # 2021 (t8)
    "2021-02-12", "2021-03-19", "2021-04-23", "2021-06-11",
    "2021-07-23", "2021-09-10", "2021-10-22", "2021-12-17",
    # 2022 (t9)
    "2022-02-11",
    "2022-02-28",  # extraordinary (war shock, +10.5pp)
    "2022-04-08",  # extraordinary
    "2022-04-29", "2022-05-26",  # extraordinary
    "2022-06-10", "2022-07-22", "2022-09-16", "2022-10-28", "2022-12-16",
    # 2023 (t10)
    "2023-02-10", "2023-03-17", "2023-04-28", "2023-06-09",
    "2023-07-21",
    "2023-08-15",  # extraordinary (ruble defence, emergency hike)
    "2023-09-15", "2023-10-27", "2023-12-15",
    # 2024 (t11)
    "2024-02-16", "2024-03-22", "2024-04-26", "2024-06-07",
    "2024-07-26", "2024-09-13", "2024-10-25", "2024-12-20",
    # 2025 (t12)
    "2025-02-14", "2025-03-21", "2025-04-25", "2025-06-06",
    "2025-07-25", "2025-09-12", "2025-10-24", "2025-12-19",
]


def _parse_date(d) -> pd.Timestamp:
    try:
        p = str(d).split(".")
        if len(p) == 3:
            return pd.Timestamp(int(p[2]), int(p[1]), int(p[0]))
        return pd.to_datetime(d)
    except Exception:
        return pd.NaT


def _get_key_rate_series(df_feat: pd.DataFrame, dates: pd.Series) -> pd.Series:
    """Return a date-indexed Series of key_rate values."""
    sr = pd.Series(
        df_feat["feat_key_rate"].values,
        index=pd.to_datetime(dates),
        name="key_rate",
    ).sort_index()
    # Multiple articles per day → deduplicate (keep last non-NaN per day)
    sr = sr[~sr.index.duplicated(keep="last")]
    # Forward-fill weekends / holidays
    full_idx = pd.date_range(sr.index.min(), sr.index.max(), freq="D")
    sr = sr.reindex(full_idx).ffill()
    return sr


def _meeting_label(
    meeting_date: pd.Timestamp,
    key_rate_sr: pd.Series,
    lag_days: int = 7,
    threshold: float = 0.05,
) -> str:
    """
    Compare key rate lag_days before vs lag_days after meeting.
    threshold: minimum pp change to call hike/cut (avoids float noise).
    """
    before = meeting_date - pd.Timedelta(days=lag_days)
    after  = meeting_date + pd.Timedelta(days=lag_days)

    rate_before = key_rate_sr.asof(before) if before >= key_rate_sr.index[0] else None
    rate_after  = key_rate_sr.asof(after)  if after  <= key_rate_sr.index[-1] else None

    if rate_before is None or rate_after is None or pd.isna(rate_before) or pd.isna(rate_after):
        return None

    delta = rate_after - rate_before
    if delta > threshold:
        return "hike"
    if delta < -threshold:
        return "cut"
    return "hold"


def build_meeting_dataset(
    window_days: int  = 14,
    max_articles: int = 0,   # 0 = store ALL articles (no cap); >0 caps per meeting
    csv_path: Path    = CSV_PATH,
) -> pd.DataFrame:
    logger.info("Loading article dataset: %s", csv_path)
    df_raw = pd.read_csv(csv_path)
    df_raw["parsed_date"] = df_raw["date"].apply(_parse_date)
    df_raw = df_raw[df_raw["parsed_date"].notna()].copy()
    df_raw = df_raw.sort_values("parsed_date").reset_index(drop=True)
    logger.info("Total articles: %d  range: %s – %s",
                len(df_raw), df_raw["parsed_date"].min().date(),
                df_raw["parsed_date"].max().date())

    # ── Build tabular features ────────────────────────────────────────────────
    logger.info("Computing tabular features…")
    df_feat = build_tabular_features(df_raw, date_col="date")
    if "cleaned_text" in df_raw.columns:
        df_feat = extract_text_features(df_feat, text_col="cleaned_text")

    feat_cols = get_feature_columns(df_feat)
    X_tab     = df_feat[feat_cols].values.astype(np.float32)

    # ── Per-article hawkishness score (keyword-based, no trained model needed) ──
    # Score = hawkish_count + dovish_count: articles that mention rate-relevant
    # language get ranked higher in the window, ensuring top-N articles are the
    # most informative ones regardless of how many we cap at training time.
    hawk_col  = "feat_hawkish_count" if "feat_hawkish_count" in df_feat.columns else None
    dove_col  = "feat_dovish_count"  if "feat_dovish_count"  in df_feat.columns else None
    if hawk_col and dove_col:
        df_raw = df_raw.copy()
        df_raw["_hawk_score"] = (
            df_feat[hawk_col].values.astype(float)
            + df_feat[dove_col].values.astype(float)
        )
        logger.info("Article hawkishness scores computed (hawk+dove keyword counts)")
    else:
        df_raw["_hawk_score"] = 0.0

    # Key rate time series (for label determination)
    key_rate_sr = _get_key_rate_series(df_feat, df_raw["parsed_date"])
    logger.info("Key rate series: %d days  (%.1f%% – %.1f%%)",
                len(key_rate_sr), key_rate_sr.min(), key_rate_sr.max())

    # ── Process each meeting ──────────────────────────────────────────────────
    meeting_dates = [pd.Timestamp(d) for d in CBR_MEETING_DATES]

    rows = []
    label_counts = {"hike": 0, "hold": 0, "cut": 0, "unknown": 0}

    for meeting_date in meeting_dates:
        # Determine label
        direction = _meeting_label(meeting_date, key_rate_sr)
        if direction is None:
            label_counts["unknown"] += 1
            logger.warning("No key rate data for meeting %s — skipping", meeting_date.date())
            continue
        label_counts[direction] += 1

        # Collect articles in (meeting_date - window_days, meeting_date)
        # Strictly before meeting to avoid look-ahead
        window_start = meeting_date - pd.Timedelta(days=window_days)
        mask = (df_raw["parsed_date"] >= window_start) & (df_raw["parsed_date"] < meeting_date)
        window_df = df_raw[mask].copy()

        # Sort: CBR press releases first → hawkishness score desc → recency desc
        # This ensures that when max_articles caps the list, the most rate-relevant
        # articles are retained regardless of publication date.
        if "source" in window_df.columns:
            window_df = window_df.assign(
                _is_cbr=(window_df["source"] == "cbr_press_releases").astype(int)
            ).sort_values(
                ["_is_cbr", "_hawk_score", "parsed_date"],
                ascending=[False, False, False],
            )

        texts         = window_df["cleaned_text"].fillna("").tolist()
        article_dates = window_df["parsed_date"].dt.strftime("%Y-%m-%d").tolist()
        if max_articles > 0:
            texts         = texts[:max_articles]
            article_dates = article_dates[:max_articles]
        if not texts:
            texts         = [""]
            article_dates = [""]

        # Tabular features at meeting date (use row closest to meeting date)
        if len(window_df) > 0:
            date_idx = (df_raw["parsed_date"] - meeting_date).abs().idxmin()
        else:
            # No articles in window: use closest row overall
            date_idx = (df_raw["parsed_date"] - meeting_date).abs().idxmin()
        tab = X_tab[date_idx]

        rows.append({
            "meeting_date":  meeting_date,
            "label":         LABEL_MAP[direction],
            "direction":     direction,
            "n_articles":    len(window_df),
            "texts":         texts,
            "article_dates": article_dates,
            "tabular":       tab,
        })

    result = pd.DataFrame(rows)
    result["year"] = result["meeting_date"].dt.year

    logger.info("\nLabel distribution: hike=%d  hold=%d  cut=%d  (skipped=%d)",
                label_counts["hike"], label_counts["hold"],
                label_counts["cut"],  label_counts["unknown"])

    for name, start, end in [
        ("Train 2014–2022", 2014, 2022),
        ("Val   2023",      2023, 2023),
        ("Test  2024–2025", 2024, 2025),
    ]:
        sub = result[result["year"].between(start, end)]
        dist = sub["direction"].value_counts().to_dict()
        logger.info("  %s: %d meetings  %s", name, len(sub), dist)

    logger.info("\nArticles per meeting: min=%d  median=%.0f  max=%d",
                result["n_articles"].min(),
                result["n_articles"].median(),
                result["n_articles"].max())

    result = result.drop(columns=["year"])
    return result


def main():
    ap = argparse.ArgumentParser(description="Build meeting-centric dataset")
    ap.add_argument("--window-days",  type=int, default=14)
    ap.add_argument("--max-articles", type=int, default=0,
                    help="Cap articles per meeting (0=store all, default)")
    ap.add_argument("--csv",          type=str, default=str(CSV_PATH))
    ap.add_argument("--output",       type=str, default=str(OUTPUT_PATH))
    args = ap.parse_args()

    df = build_meeting_dataset(
        window_days  = args.window_days,
        max_articles = args.max_articles,
        csv_path     = Path(args.csv),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Saved → %s  (%d meetings)", out, len(df))


if __name__ == "__main__":
    main()
