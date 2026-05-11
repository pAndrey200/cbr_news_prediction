"""
Build tabular economic features for each news item date.

Loads time series from the DB (RUONIA, key_rate, USD, EUR, CNY, Brent, inflation)
and constructs lag features, rolling statistics, calendar features, and
keyword-based text features extracted from CBR press-release language.

Usage (standalone):
    python cbr_news/ml/feature_engineering.py  # prints feature matrix info

Import API:
    from cbr_news.ml.feature_engineering import build_tabular_features, extract_text_features
    df_feat = build_tabular_features(df, date_col="date")
    df_feat = extract_text_features(df_feat, text_col="cleaned_text")
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── CBR keyword vocabulary ───────────────────────────────────────────────────
# Hawkish language → model predicts rate increase / RUONIA up
_HAWKISH = [
    r"повыш", r"ужесточ", r"ужесточен",
    r"проинфляц", r"инфляц[ионный]+", r"инфляц[ия]+",
    r"ускорен", r"рост.*инфляц", r"давлен.*инфляц",
    r"перегрев", r"перегрет",
    r"повышательн", r"проинфляционн",
    r"выше.*прогноз", r"превыш.*ожидан",
]

# Dovish language → model predicts rate decrease / RUONIA down
_DOVISH = [
    r"снижен", r"снизит", r"снизил", r"сниж",
    r"смягчен", r"смягчит",
    r"замедлен", r"дезинфляц",
    r"ниже.*прогноз", r"ниже.*ожидан",
    r"уменьш", r"сокращен",
    r"дефляц",
]

# Neutral / maintenance language
_NEUTRAL = [
    r"сохран[ить]+", r"сохранен", r"сохранит",
    r"неизменн", r"стабилиз",
    r"ключевую ставку на уровне",
    r"нейтральн",
    r"диапазон.*цел",
]

# Uncertainty / risk language
_UNCERTAINTY = [
    r"риск", r"неопределенност", r"волатильн",
    r"геополит", r"санкц", r"внешн.*условия",
]

# Forward guidance signals
_FORWARD_UP   = [r"допускает повыш", r"не исключает повыш", r"может повыс"]
_FORWARD_DOWN = [r"допускает снижен", r"не исключает снижен", r"может снизить"]


def _count_matches(text: str, patterns: list) -> int:
    text = text.lower()
    return sum(1 for p in patterns if re.search(p, text))


def extract_text_features(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Add keyword-based text features derived from CBR press-release language.

    Features added (all prefixed with ``text_``):
      - text_hawkish_count    : count of hawkish keywords
      - text_dovish_count     : count of dovish keywords
      - text_neutral_count    : count of neutral/hold keywords
      - text_uncertainty_count: count of risk/uncertainty keywords
      - text_forward_up       : forward guidance for increase
      - text_forward_down     : forward guidance for decrease
      - text_hawk_dove_ratio  : (hawkish - dovish) / (hawkish + dovish + 1)
      - text_len              : character length of text
      - text_word_count       : number of words
      - text_has_key_rate_mention : 1 if key rate is explicitly mentioned
      - text_has_inflation_mention: 1 if inflation explicitly mentioned
    """
    if text_col not in df.columns:
        logger.warning(f"Column '{text_col}' not found; skipping text features")
        return df

    df = df.copy()
    texts = df[text_col].fillna("").tolist()

    df["text_hawkish_count"]     = [_count_matches(t, _HAWKISH)     for t in texts]
    df["text_dovish_count"]      = [_count_matches(t, _DOVISH)      for t in texts]
    df["text_neutral_count"]     = [_count_matches(t, _NEUTRAL)     for t in texts]
    df["text_uncertainty_count"] = [_count_matches(t, _UNCERTAINTY) for t in texts]
    df["text_forward_up"]        = [_count_matches(t, _FORWARD_UP)  for t in texts]
    df["text_forward_down"]      = [_count_matches(t, _FORWARD_DOWN) for t in texts]

    h = df["text_hawkish_count"].values.astype(float)
    d = df["text_dovish_count"].values.astype(float)
    df["text_hawk_dove_ratio"] = (h - d) / (h + d + 1.0)

    df["text_len"]        = [len(t) for t in texts]
    df["text_word_count"] = [len(t.split()) for t in texts]
    df["text_has_key_rate_mention"] = [
        1 if re.search(r"ключев.*ставк|ставк.*ключев", t.lower()) else 0 for t in texts
    ]
    df["text_has_inflation_mention"] = [
        1 if re.search(r"инфляц", t.lower()) else 0 for t in texts
    ]

    text_cols = [c for c in df.columns if c.startswith("text_")]
    logger.info(f"Added {len(text_cols)} text features")
    return df

DATA_DIR = Path(__file__).parents[2] / "data"


# ─── DB loading ───────────────────────────────────────────────────────────────

def load_series_from_db() -> dict:
    """Return {name: pd.Series(float, index=DatetimeIndex)} for all indicators."""
    from cbr_news.database.db import get_db_session
    from cbr_news.database.repository import (
        CurrencyRateRepository,
        InflationRepository,
        KeyRateRepository,
        OilPriceRepository,
        RuoniaRepository,
    )

    def _to_series(records, date_attr, val_attr):
        rows = [(getattr(r, date_attr), getattr(r, val_attr)) for r in records]
        df = pd.DataFrame(rows, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna().drop_duplicates("date").sort_values("date")
        return df.set_index("date")["value"]

    with get_db_session() as session:
        series = {
            "ruonia":    _to_series(RuoniaRepository.get_all(session),    "date", "rate"),
            "key_rate":  _to_series(KeyRateRepository.get_all(session),   "date", "rate"),
            "usd":       _to_series(CurrencyRateRepository.get_all_by_currency(session, "USD"), "date", "rate"),
            "eur":       _to_series(CurrencyRateRepository.get_all_by_currency(session, "EUR"), "date", "rate"),
            "cny":       _to_series(CurrencyRateRepository.get_all_by_currency(session, "CNY"), "date", "rate"),
            "brent":     _to_series(OilPriceRepository.get_all(session),  "date", "price"),
            "inflation": _to_series(InflationRepository.get_all(session), "date", "value"),
        }
    logger.info("Loaded from DB: " + ", ".join(f"{k}={len(v)}" for k, v in series.items()))
    return series


def _make_daily(ts: pd.Series, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.Series:
    """Reindex ts to a full calendar-day range and forward-fill gaps."""
    if ts.empty or ts.index.min() is pd.NaT:
        date_range = pd.date_range(start=min_date, end=max_date + pd.Timedelta(days=1), freq="D")
        return pd.Series(np.nan, index=date_range)
    date_range = pd.date_range(
        start=min(ts.index.min(), min_date),
        end=max(ts.index.max(), max_date) + pd.Timedelta(days=1),
        freq="D",
    )
    # Normalise timezone
    if ts.index.tz is not None:
        ts = ts.copy()
        ts.index = ts.index.tz_localize(None)
    ts_daily = ts.reindex(date_range).ffill()
    return ts_daily


# ─── feature builder ──────────────────────────────────────────────────────────

def build_tabular_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    For every row in *df* compute economic features based on the news date.

    Parameters
    ----------
    df       : DataFrame containing at least *date_col* (DD.MM.YYYY or ISO format).
    date_col : Name of the date column.

    Returns
    -------
    DataFrame with all original columns **plus** ``feat_*`` columns.
    No rows are dropped; NaN for dates outside DB coverage.
    """
    df = df.copy()

    # ── parse dates ───────────────────────────────────────────────────────────
    dates = pd.to_datetime(df[date_col], format="%d.%m.%Y", errors="coerce")
    dates = dates.fillna(pd.to_datetime(df[date_col], errors="coerce"))
    df["_dt"] = dates

    min_date = dates.dropna().min()
    max_date = dates.dropna().max()

    # ── calendar features ─────────────────────────────────────────────────────
    df["feat_year"]         = df["_dt"].dt.year
    df["feat_month"]        = df["_dt"].dt.month
    df["feat_quarter"]      = df["_dt"].dt.quarter
    df["feat_is_month_end"] = df["_dt"].dt.is_month_end.astype(float)
    # feat_dow intentionally excluded: day-of-week gets ~45% importance because
    # RUONIA is not published on weekends, so a 2-day horizon from Friday lands
    # on Monday — this is a data artifact, not a real signal.
    # Instead: explicit flag for whether the prediction window crosses a weekend.
    dow = df["_dt"].dt.dayofweek
    # With BDay(2) labeling:
    #   Thursday (dow=3) + 2 BDay = Monday  → crosses Sat+Sun ✓
    #   Friday   (dow=4) + 2 BDay = Tuesday → crosses Sat+Sun ✓
    #   Sat/Sun  (dow=5/6): published on weekend (rare but flag anyway)
    df["feat_horizon_crosses_weekend"] = (
        (dow >= 3) & (dow <= 6)
    ).astype(float)

    # ── load economic time series ─────────────────────────────────────────────
    series = load_series_from_db()

    # Vectorised lag lookup: shift dates by N days, then map to daily series
    def _lookup(ts_daily: pd.Series, shifted_dates: pd.Series) -> np.ndarray:
        """Map shifted_dates to ts_daily values (NaN if out of range)."""
        # Normalise to midnight Timestamps (no tz)
        sd = shifted_dates.dt.floor("D")
        return sd.map(ts_daily).values.astype(float)

    LAGS_DENSE  = [1, 2, 5, 10, 30]   # RUONIA, USD, EUR, brent
    LAGS_SPARSE = [1, 2, 5]            # key_rate, CNY
    WINDOWS     = [7, 30]

    for name, ts in series.items():
        ts_daily = _make_daily(ts, min_date, max_date)
        feat = f"feat_{name}"

        # current value
        df[feat] = _lookup(ts_daily, df["_dt"])

        if name == "inflation":
            # Inflation is monthly; current value is sufficient
            continue

        lags = LAGS_DENSE if name in ("ruonia", "usd", "eur", "brent") else LAGS_SPARSE

        # lag values
        for lag in lags:
            shifted = df["_dt"] - pd.Timedelta(days=lag)
            df[f"{feat}_lag{lag}"] = _lookup(ts_daily, shifted)

        # directional changes from lag
        df[f"{feat}_change1"]  = df[feat] - df[f"{feat}_lag1"]
        if 5 in lags:
            df[f"{feat}_change5"]  = df[feat] - df[f"{feat}_lag5"]
        if 30 in lags:
            df[f"{feat}_change30"] = df[feat] - df[f"{feat}_lag30"]

        # rolling mean / std
        for win in WINDOWS:
            ts_roll_mean = ts_daily.rolling(win, min_periods=1).mean()
            ts_roll_std  = ts_daily.rolling(win, min_periods=2).std()
            df[f"{feat}_roll{win}mean"] = _lookup(ts_roll_mean, df["_dt"])
            df[f"{feat}_roll{win}std"]  = _lookup(ts_roll_std,  df["_dt"])

    # ── days since last key-rate change ───────────────────────────────────────
    key_ts = series["key_rate"]
    changes = key_ts[key_ts.diff().abs().fillna(0) > 1e-4]
    if len(changes) > 0:
        change_dates = np.array(changes.index, dtype="datetime64[ns]")

        def _days_since_change(d):
            if pd.isna(d):
                return np.nan
            d_ns = np.datetime64(d, "ns")
            past = change_dates[change_dates <= d_ns]
            return (d - pd.Timestamp(past[-1])).days if len(past) > 0 else np.nan

        df["feat_days_since_rate_change"] = df["_dt"].apply(_days_since_change)
    else:
        df["feat_days_since_rate_change"] = np.nan

    df = df.drop(columns=["_dt"])
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    logger.info(f"Built {len(feat_cols)} tabular features for {len(df)} rows")
    return df


# ─── feature column names ─────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of all feature column names (feat_* + text_* + sentiment_*)."""
    return [
        c for c in df.columns
        if c.startswith("feat_") or c.startswith("text_") or c.startswith("sentiment_")
    ]


# ─── standalone run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_path = DATA_DIR / "cbr_combined_dataset.csv"
    logger.info(f"Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset rows: {len(df)}")

    df_feat = build_tabular_features(df, date_col="date")
    feat_cols = get_feature_columns(df_feat)

    logger.info(f"Feature columns ({len(feat_cols)}):")
    for c in feat_cols:
        logger.info(f"  {c:45s}  null={df_feat[c].isna().sum()}")

    out = DATA_DIR / "tabular_features.parquet"
    df_feat[["date", "target"] + feat_cols].to_parquet(out, index=False)
    logger.info(f"Saved → {out}")
