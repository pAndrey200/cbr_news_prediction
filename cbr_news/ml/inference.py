"""
Key Rate Inference Service — two-stage pipeline.

Stage 1 — Article Scorer (daily RUONIA model):
  Runs each article through JointBertTabularModel.
  Relevance score = max(P(up), P(down))  — how much the article signals a rate move.
  Hawkishness     = P(up) − P(down)      — positive means rate-hike signal.

Stage 2 — Rate Predictor (sklearn on meeting features):
  Fits LogReg / SVC on training meetings from meeting_dataset.parquet.
  Predicts hike / hold / cut for the current window with class probabilities.

Output: ranked article list + net signal + predicted direction + confidence.

Usage:
    # Predict for a historical meeting (uses pre-built dataset):
    PYTHONPATH=. python cbr_news/ml/inference.py \\
        --meeting-date 2024-12-20 \\
        --daily-model checkpoints/joint/best.ckpt

    # Predict from a custom date range in the raw news CSV:
    PYTHONPATH=. python cbr_news/ml/inference.py \\
        --news-csv data/cbr_combined_dataset_t015.csv \\
        --from-date 2025-12-06 --to-date 2025-12-19 \\
        --daily-model checkpoints/joint/best.ckpt

    # Run without daily model (BERT CLS + tabular only):
    PYTHONPATH=. python cbr_news/ml/inference.py --meeting-date 2024-12-20
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Ensure project root is importable when run as a script
_PROJECT_ROOT = str(Path(__file__).parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cbr_news.ml.prepare_meeting_dataset import (
    _parse_date,
    build_meeting_dataset,
)
from cbr_news.ml.feature_engineering import (
    build_tabular_features,
    extract_text_features,
    get_feature_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parents[2] / "data"
MEETING_DS  = DATA_DIR / "meeting_dataset.parquet"
CSV_PATH    = DATA_DIR / "cbr_combined_dataset_t025.csv"

BERT_NAME    = "DeepPavlov/rubert-base-cased"
MAX_LENGTH   = 512   # Stage 1 (RUONIA daily model)
MAX_LENGTH_STAGE2 = 256  # Stage 2 (meeting model) — must match sweep_meeting.py training
MAX_ARTICLES = 20

# RUONIA direction labels (daily model)
RUONIA_LABELS = {0: "down", 1: "same", 2: "up"}
RUONIA_LABELS_RU = {"down": "Снижение", "same": "Без изменений", "up": "Повышение"}

# Key rate direction labels (meeting model)
CLASS_NAMES = ["cut", "hold", "hike"]
LABEL_MAP   = {0: "cut", 1: "hold", 2: "hike"}

TRAIN_END = pd.Timestamp("2024-06-30")

CURRENT_KEY_RATE = 21.0  # fallback if DB is unavailable


def get_current_key_rate() -> float:
    """Fetch the latest key rate from the database. Falls back to hardcoded default."""
    try:
        from cbr_news.database.db import get_db_session
        from cbr_news.database.repository import KeyRateRepository
        with get_db_session() as session:
            record = KeyRateRepository.get_latest(session)
            if record is not None:
                logger.info("Current key rate from DB: %.2f%% (date=%s)", record.rate, record.date)
                return float(record.rate)
    except Exception as e:
        logger.warning("Could not fetch key rate from DB: %s — using default %.1f%%", e, CURRENT_KEY_RATE)
    return CURRENT_KEY_RATE


# ─── Stage 1: article scoring via daily RUONIA model ─────────────────────────

def _load_tab_default(n_tab: int, device: torch.device) -> torch.Tensor:
    """
    Compute tabular features for today from the DB (always fresh).
    Falls back to latest row of precomputed parquet, then to zeros.
    """
    from cbr_news.ml.feature_engineering import build_tabular_features, get_feature_columns
    try:
        today_str = pd.Timestamp.now().strftime("%d.%m.%Y")
        df_today = pd.DataFrame([{"date": today_str}])
        df_feat = build_tabular_features(df_today, date_col="date")
        feat_cols = get_feature_columns(df_feat)
        row = df_feat[feat_cols].iloc[0].values.astype(np.float32)
        row = np.append(row, 0.0)
        if len(row) < n_tab:
            row = np.concatenate([row, np.zeros(n_tab - len(row), dtype=np.float32)])
        if not np.isnan(row).all():
            logger.info("Tab default: live DB features for %s (%d features)", today_str, n_tab)
            return torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
    except Exception as e:
        logger.warning("Failed to compute live tab features: %s", e)

    # ── fallback: latest row of precomputed parquet ────────────────────────────
    for parquet_path in (
        DATA_DIR / "tabular_features.parquet",
        DATA_DIR / "tabular_features_t025.parquet",
    ):
        if not parquet_path.exists():
            continue
        try:
            df = pd.read_parquet(parquet_path)
            feat_cols = get_feature_columns(df)
            row = df[feat_cols].iloc[-1].values.astype(np.float32)
            row = np.append(row, 0.0)
            if len(row) == n_tab:
                logger.info("Tab default: parquet fallback from %s (date=%s)",
                            parquet_path.name, df["date"].iloc[-1] if "date" in df.columns else "?")
                return torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
        except Exception as e:
            logger.warning("Parquet fallback failed (%s): %s", parquet_path.name, e)

    logger.warning("Using zeros as tabular default")
    return torch.zeros(1, n_tab, device=device)


def load_daily_model(ckpt_path: str):
    """
    Load JointBertTabularModel from checkpoint.
    Returns (model, tokenizer, n_tabular, tab_default, device).
    tab_default: (1, n_tab) tensor with real features from latest date in parquet.
    """
    from cbr_news.ml.train_joint import JointBertTabularModel

    logger.info("Loading daily RUONIA model: %s", ckpt_path)
    model = JointBertTabularModel.load_from_checkpoint(
        ckpt_path, map_location="cpu", strict=False
    )
    model.eval()

    n_tab = (
        model.hparams.get("n_tabular", 90)
        if hasattr(model.hparams, "get")
        else getattr(model.hparams, "n_tabular", 90)
    )
    tok_name  = getattr(model.hparams, "backbone", BERT_NAME) or BERT_NAME
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    tab_default = _load_tab_default(n_tab, device)

    logger.info("  backbone=%s  n_tabular=%d  device=%s", tok_name, n_tab, device)
    return model, tokenizer, n_tab, tab_default, device


def score_articles(
    texts: List[str],
    model,
    tokenizer,
    n_tab: int,
    device,
    max_length: int = MAX_LENGTH,
    tab_default: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Run each article through the daily model.
    Returns (N, 3) float32 — P(down), P(same), P(up) per article.
    tab_default: (1, n_tab) tensor — real tabular features for current date.
    """
    if tab_default is None:
        tab_default = torch.zeros(1, n_tab, device=device)

    probs_list = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                str(text),
                truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            logits = model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
                tab_default,
            )                                                        # (1, 3)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]    # (3,)
            probs_list.append(probs)

    return np.array(probs_list, dtype=np.float32)                    # (N, 3)


def aggregate_probs(probs: np.ndarray) -> np.ndarray:
    """
    Aggregate per-article probabilities into 13 meeting-level features:
      [0-2]  mean P(down/same/up)
      [3-5]  max  P(down/same/up)
      [6-8]  std  P(down/same/up)
      [9]    net_hawkishness = mean P(up) − mean P(down)
      [10-12] fraction of articles whose argmax is down / same / up
    """
    mean_p      = probs.mean(0)
    max_p       = probs.max(0)
    std_p       = probs.std(0)
    net_hawk    = float(mean_p[2] - mean_p[0])
    argmax_frac = np.bincount(probs.argmax(1), minlength=3) / len(probs)
    return np.concatenate([mean_p, max_p, std_p, [net_hawk], argmax_frac])


def extract_soft_feats_for_all(
    df_meetings: pd.DataFrame,
    model,
    tokenizer,
    n_tab: int,
    device,
    max_articles: int,
    max_length: int,
) -> np.ndarray:
    """
    Compute 13-dim daily-model features for every row in df_meetings.
    Reuses an already-loaded model to avoid double loading.
    """
    logger.info("Computing daily-model features for %d meetings …", len(df_meetings))
    all_feats = []
    with torch.no_grad():
        for i, (_, row) in enumerate(df_meetings.iterrows()):
            texts = [str(t) for t in list(row["texts"])[:max_articles]] or [""]
            probs = score_articles(texts, model, tokenizer, n_tab, device, max_length)
            all_feats.append(aggregate_probs(probs))
            if (i + 1) % 20 == 0 or (i + 1) == len(df_meetings):
                logger.info("  … %d / %d meetings processed", i + 1, len(df_meetings))

    out = np.array(all_feats, dtype=np.float32)
    logger.info("Daily-model features shape: %s", out.shape)
    return out


# ─── BERT CLS embeddings ──────────────────────────────────────────────────────

def _bert_cls_for_texts(
    texts: List[str],
    tokenizer,
    bert,
    device,
    max_length: int,
    max_articles: int,
) -> np.ndarray:
    """Mean-pool CLS embeddings over a list of article texts. Returns (H,)."""
    texts_enc = [str(t) for t in texts[:max_articles]] or [""]
    enc = tokenizer(
        texts_enc,
        truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )
    enc  = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = bert(**enc)
    cls = out.last_hidden_state[:, 0, :].cpu().numpy()     # (n, H)
    return cls.mean(0)                                      # (H,)


def precompute_bert_cls_all(
    df: pd.DataFrame,
    bert_name_or_model,
    max_length: int,
    max_articles: int,
    tokenizer=None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute mean BERT CLS per meeting. Returns (N, H) float32.

    Pass an already-loaded model + tokenizer to avoid re-downloading from HF.
    If bert_name_or_model is a string, loads a fresh model (legacy behaviour).
    """
    logger.info("Computing BERT embeddings for %d meetings …", len(df))
    if isinstance(bert_name_or_model, str):
        tokenizer = AutoTokenizer.from_pretrained(bert_name_or_model)
        bert      = AutoModel.from_pretrained(bert_name_or_model).eval()
        device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert      = bert.to(device)
        own_bert  = True
    else:
        bert     = bert_name_or_model   # already-loaded JointBertTabularModel.bert
        device   = device or next(bert.parameters()).device
        own_bert = False

    H    = bert.config.hidden_size
    embs = np.zeros((len(df), H), dtype=np.float32)
    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            texts = [str(t) for t in list(row["texts"])[:max_articles]] or [""]
            enc   = tokenizer(
                texts,
                truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            enc     = {k: v.to(device) for k, v in enc.items()}
            out     = bert(**enc)
            cls     = out.last_hidden_state[:, 0, :].cpu().numpy()
            embs[i] = cls.mean(0)
            if (i + 1) % 20 == 0 or (i + 1) == len(df):
                logger.info("  … %d / %d meetings", i + 1, len(df))

    if own_bert:
        del bert
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return embs


# ─── Stage 2: sklearn classifier ─────────────────────────────────────────────

def _build_meet_feats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute meeting-level structural features for all rows in df.
    Returns (meet_feats, mf_means, mf_std) — normalised on train split.
    """
    from cbr_news.ml.sweep_meeting import add_meeting_features, extract_meeting_feats
    train_mask = (df["meeting_date"] <= TRAIN_END).values
    df2 = add_meeting_features(df.sort_values("meeting_date").reset_index(drop=True).copy())
    mf = extract_meeting_feats(df2).astype(np.float32)
    mf_means = mf[train_mask].mean(0)
    mf_std   = mf[train_mask].std(0) + 1e-6
    return (mf - mf_means) / mf_std, mf_means, mf_std


def compute_current_meet_feats(
    df_meetings: pd.DataFrame,
    n_articles: int,
    current_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """
    Compute meeting-level features for the CURRENT (upcoming) prediction window.
    Uses historical meeting outcomes to compute streak / days_since_change etc.
    Returns raw (unnormalised) feature vector of shape (6,):
      [meet_streak, meet_days_since_change, n_articles, meet_hold_streak, meet_month, meet_year]
    """
    from cbr_news.ml.sweep_meeting import add_meeting_features, extract_meeting_feats
    if current_date is None:
        current_date = pd.Timestamp.now()

    df = df_meetings.sort_values("meeting_date").reset_index(drop=True).copy()
    df2 = add_meeting_features(df)
    mf = extract_meeting_feats(df2)

    # Start with last historical meeting's features, then override time-dependent ones
    current_mf = mf[-1].copy().astype(np.float32)

    # Recompute days_since_change: days from last non-hold meeting to current_date
    directions = df["direction"].tolist()
    last_change_date = df["meeting_date"].iloc[0]  # fallback
    for i in range(len(directions) - 1, -1, -1):
        if directions[i] != "hold":
            last_change_date = df["meeting_date"].iloc[i]
            break
    days_since = float((current_date - last_change_date).days)

    # cols order from extract_meeting_feats:
    # [meet_streak, meet_days_since_change, n_articles, meet_hold_streak, meet_month, meet_year]
    current_mf[1] = days_since
    current_mf[2] = float(n_articles)
    current_mf[4] = float(current_date.month)
    current_mf[5] = float(current_date.year)

    return current_mf


def fit_sklearn(
    df: pd.DataFrame,
    embs: np.ndarray,
    soft_feats: Optional[np.ndarray] = None,
    pca_dim: int = 150,
):
    """
    Fit Stage-2 GradientBoosting on meeting features (best config from sweep).
    Feature set: PCA(BERT) + tab + meet
    Returns (pipeline, col_means, col_std, pca, mf_means, mf_std).
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    dates      = df["meeting_date"]
    train_mask = (dates <= TRAIN_END).values

    # ── Tabular features (normalise on train) ──────────────────────────────
    X_tab = np.stack(df["tabular"].values).astype(np.float32)
    col_means = np.nanmean(X_tab[train_mask], axis=0)
    col_means[np.isnan(col_means)] = 0.0
    nan_idx = np.where(np.isnan(X_tab))
    if nan_idx[0].size:
        X_tab[nan_idx] = np.take(col_means, nan_idx[1])
    col_std    = np.nanstd(X_tab[train_mask], axis=0) + 1e-6
    X_tab_norm = (X_tab - col_means) / col_std

    # ── PCA on BERT embeddings (fit on train only) ─────────────────────────
    n_comp = min(pca_dim, embs.shape[1], int(train_mask.sum()) - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(embs[train_mask])
    embs_pca = pca.transform(embs)
    logger.info("PCA: %d → %d components (%.1f%% var)",
                embs.shape[1], n_comp, pca.explained_variance_ratio_.sum() * 100)

    # ── Meeting-level structural features ──────────────────────────────────
    meet_feats, mf_means, mf_std = _build_meet_feats(df)

    # Assemble: bert_pca + tab + meet
    X_all = np.concatenate([embs_pca, X_tab_norm, meet_feats], axis=1)
    logger.info("Stage-2 feature matrix: %s  (bert_pca=%d, tab=%d, meet=%d)",
                X_all.shape, embs_pca.shape[1], X_tab_norm.shape[1], meet_feats.shape[1])

    y = df["label"].values

    # ── GradientBoosting (best config from sweep: cv_f1=0.7345) ───────────
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        min_samples_leaf=2, subsample=0.9, random_state=42,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_all[train_mask], y[train_mask])
    logger.info("Stage-2 GBoost fitted on %d meetings (up to %s)", train_mask.sum(), TRAIN_END.date())

    return pipe, col_means, col_std, pca, mf_means, mf_std


def predict_window(
    texts: List[str],
    tabular: np.ndarray,
    col_means: np.ndarray,
    col_std: np.ndarray,
    bert_name: str,
    max_length: int,
    max_articles: int,
    pipeline,
    daily_model=None,
    daily_tokenizer=None,
    daily_n_tab: int = 90,
    daily_device=None,
    use_soft_feats: bool = False,
    pca=None,
    meet_feats: np.ndarray | None = None,
) -> dict:
    """
    Build features for ONE meeting window and predict direction.
    Returns dict with keys: pred_label, probs (3,), article_probs (N, 3) or None.
    pca: fitted sklearn PCA (applied to BERT emb if provided)
    meet_feats: (1, K) normalised meeting-level features to append
    """
    # ── BERT CLS mean ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert      = AutoModel.from_pretrained(bert_name).eval()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert      = bert.to(device)

    texts_use = [str(t) for t in texts[:max_articles]] or [""]
    enc = tokenizer(
        texts_use,
        truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )
    with torch.no_grad():
        out = bert(**{k: v.to(device) for k, v in enc.items()})
    emb = out.last_hidden_state[:, 0, :].cpu().numpy().mean(0, keepdims=True)  # (1, H)

    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Apply PCA to BERT embeddings ───────────────────────────────────────
    if pca is not None:
        emb = pca.transform(emb)   # (1, n_comp)

    # ── Normalise tabular ──────────────────────────────────────────────────
    tab_arr = np.array(tabular, dtype=np.float32).reshape(1, -1)
    nan_idx = np.where(np.isnan(tab_arr))
    if nan_idx[0].size:
        tab_arr[nan_idx] = np.take(col_means, nan_idx[1])
    tab_norm = (tab_arr - col_means) / col_std                        # (1, n_tab)

    X = np.concatenate([emb, tab_norm], axis=1)

    # ── Meeting-level structural features ──────────────────────────────────
    if meet_feats is not None:
        X = np.concatenate([X, meet_feats.reshape(1, -1)], axis=1)

    # ── Daily model soft-label features (legacy) ───────────────────────────
    article_probs = None
    if daily_model is not None and use_soft_feats:
        article_probs = score_articles(
            texts_use, daily_model, daily_tokenizer,
            daily_n_tab, daily_device, max_length,
        )
        soft = aggregate_probs(article_probs).reshape(1, -1)
        X    = np.concatenate([X, soft], axis=1)

    # ── Predict ───────────────────────────────────────────────────────────
    pred_idx = int(pipeline.predict(X)[0])
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = pipeline.predict_proba(X)[0]
    elif hasattr(clf, "decision_function"):
        df_scores = pipeline.decision_function(X)[0]
        exp_s = np.exp(df_scores - df_scores.max())
        probs = exp_s / exp_s.sum()
    else:
        probs = np.eye(3)[pred_idx]

    return {
        "pred_idx":    pred_idx,
        "pred_label":  LABEL_MAP[pred_idx],
        "probs":       probs,
        "article_probs": article_probs,
    }


# ─── Query window from raw CSV ────────────────────────────────────────────────

def load_window_from_csv(
    csv_path: Path,
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
):
    """
    Load articles in [from_date, to_date) from the raw CSV and compute tabular features.
    Returns (texts, tabular_vector, df_window).
    """
    logger.info("Loading raw CSV: %s", csv_path)
    df_raw = pd.read_csv(csv_path)
    df_raw["parsed_date"] = df_raw["date"].apply(_parse_date)
    df_raw = (
        df_raw[df_raw["parsed_date"].notna()]
        .sort_values("parsed_date")
        .reset_index(drop=True)
    )

    mask   = (df_raw["parsed_date"] >= from_date) & (df_raw["parsed_date"] < to_date)
    window = df_raw[mask].copy()
    logger.info("Articles in window: %d  (%s – %s)", len(window), from_date.date(), to_date.date())

    # Tabular features — computed on the full dataset; pick row closest to meeting date
    df_feat   = build_tabular_features(df_raw, date_col="date")
    if "cleaned_text" in df_raw.columns:
        df_feat = extract_text_features(df_feat, text_col="cleaned_text")
    feat_cols = get_feature_columns(df_feat)
    X_tab     = df_feat[feat_cols].values.astype(np.float32)

    date_idx = (df_raw["parsed_date"] - to_date).abs().idxmin()
    tabular  = X_tab[date_idx]

    texts = window["cleaned_text"].fillna("").tolist() or [""]
    return texts, tabular, window

_DIRECTION_RU = {
    "hike": "↑  ПОВЫШЕНИЕ",
    "hold": "→  БЕЗ ИЗМЕНЕНИЙ",
    "cut":  "↓  СНИЖЕНИЕ",
}


def print_report(
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
    texts: List[str],
    result: dict,
    top_n: int = 5,
    current_key_rate: float = CURRENT_KEY_RATE,
    true_label: Optional[str] = None,
):
    article_probs = result["article_probs"]
    probs         = result["probs"]
    pred_label    = result["pred_label"]

    print()
    print("=" * 70)
    print("  ПРОГНОЗ КЛЮЧЕВОЙ СТАВКИ ЦБ РФ")
    print(f"  Окно новостей: {from_date.date()} – {to_date.date()}")
    print(f"  Статей в окне: {len(texts)}")
    print("=" * 70)

    if article_probs is not None:
        relevance = np.maximum(article_probs[:, 0], article_probs[:, 2])  # max(P↓, P↑)
        hawk      = article_probs[:, 2] - article_probs[:, 0]              # P↑ − P↓

        print(f"\nТоп-{top_n} важных статей (по релевантности):")
        print(f"{'#':<4}  {'Релев.':>7}  {'Сигнал':>8}  Текст (первые 60 символов)")
        print("-" * 70)

        top_idx = np.argsort(relevance)[::-1][:top_n]
        for rank, idx in enumerate(top_idx, 1):
            rel_pct  = relevance[idx] * 100
            hv       = hawk[idx]
            signal   = f"{'+'if hv > 0 else ''}{hv:.2f}"
            snippet  = str(texts[idx])[:60].replace("\n", " ").strip()
            print(f"{rank:<4}  {rel_pct:>6.1f}%  {signal:>8}  {snippet}…")

        net_hawk    = float(hawk.mean())
        n_hike_arts = int((article_probs.argmax(1) == 2).sum())
        n_cut_arts  = int((article_probs.argmax(1) == 0).sum())
        n_hold_arts = int((article_probs.argmax(1) == 1).sum())

        if abs(net_hawk) < 0.05:
            signal_desc = "нейтральный"
        elif net_hawk > 0:
            signal_desc = f"проинфляционный (→ повышение)  +{net_hawk:.3f}"
        else:
            signal_desc = f"дезинфляционный (→ снижение)  {net_hawk:.3f}"

        print()
        print(f"  Суммарный сигнал: {signal_desc}")
        print(f"  Распределение: ↑ повышение={n_hike_arts}  → нейтрально={n_hold_arts}  ↓ снижение={n_cut_arts}")

    print()
    print("─" * 70)
    print("  ПРОГНОЗИРУЕМОЕ РЕШЕНИЕ ПО КЛЮЧЕВОЙ СТАВКЕ:")
    print(f"    {_DIRECTION_RU[pred_label]}")
    print()
    conf = probs[result["pred_idx"]] * 100
    print(f"  Уверенность: {conf:.1f}%")
    print(f"  Вероятности:  снижение={probs[0]:.1%}  без изм.={probs[1]:.1%}  повышение={probs[2]:.1%}")

    # Expected rate level
    if pred_label == "hike":
        expected = current_key_rate + 1.0
        note     = f"+1 пп (типичный шаг; возможно ±0.5–2 пп)"
    elif pred_label == "cut":
        expected = current_key_rate - 0.5
        note     = f"−0.5 пп (типичный первый шаг снижения)"
    else:
        expected = current_key_rate
        note     = "ставка остаётся без изменений"

    print(f"\n  Текущая ставка: {current_key_rate:.1f}%")
    print(f"  Ожидаемый уровень: {expected:.1f}%  ({note})")

    if true_label is not None:
        status = "ВЕРНО ✓" if true_label == pred_label else "НЕВЕРНО ✗"
        print(f"\n  Фактическое решение: {_DIRECTION_RU[true_label]}  [{status}]")

    print("=" * 70)
    print()


# ─── Unified predictor class (used by API & bot) ─────────────────────────────

class CBRNewsPredictor:
    """
    Wraps both prediction stages:
      - predict()         : per-article RUONIA direction (backward-compatible)
      - predict_ruonia()  : same, returns structured dicts
      - predict_key_rate(): meeting-level key-rate prediction (hike/hold/cut)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,   # unused, kept for backward compat
        multitask: bool = False,          # unused, kept for backward compat
        meeting_dataset: str | Path | None = None,
        rate_model_path: str | Path | None = None,
    ):
        self.ckpt_path = checkpoint_path
        self._rate_model_path = Path(rate_model_path) if rate_model_path else None
        self._model = None
        self._tokenizer = None
        self._n_tab = None
        self._tab_default = None
        self._device = None
        self._meeting_ds_path = Path(meeting_dataset) if meeting_dataset else MEETING_DS
        self._sklearn_pipeline = None
        self._col_means = None
        self._col_std = None
        self._pca = None
        self._mf_means = None
        self._mf_std = None
        self._soft_feats_all = None
        self._bert_embs = None
        self._df_meetings = None
        self._rate_model_ready = False

    def _ensure_daily_model(self):
        if self._model is not None:
            return
        self._model, self._tokenizer, self._n_tab, self._tab_default, self._device = (
            load_daily_model(self.ckpt_path)
        )

    def _ensure_rate_model(self):
        """Lazy-load the sklearn rate-prediction pipeline (with disk cache)."""
        if self._rate_model_ready:
            return
        self._ensure_daily_model()

        if not self._meeting_ds_path.exists():
            raise FileNotFoundError(
                f"Meeting dataset not found: {self._meeting_ds_path}. "
                "Run: PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py"
            )

        self._df_meetings = pd.read_parquet(self._meeting_ds_path)
        self._df_meetings["meeting_date"] = pd.to_datetime(
            self._df_meetings["meeting_date"]
        )

        # ── Disk cache: skip costly compute if checkpoint hasn't changed ──────
        import pickle, hashlib

        # Explicit rate model path (e.g. from CHECKPOINT_DIR) takes priority
        explicit_path = self._rate_model_path
        if explicit_path and explicit_path.exists():
            logger.info("Loading rate model from explicit path: %s", explicit_path)
            try:
                with open(explicit_path, "rb") as f:
                    cache = pickle.load(f)
                self._sklearn_pipeline = cache["pipeline"]
                self._col_means        = cache["col_means"]
                self._col_std          = cache["col_std"]
                self._pca              = cache.get("pca")
                self._mf_means         = cache.get("mf_means")
                self._mf_std           = cache.get("mf_std")
                self._soft_feats_all   = cache["soft_feats"]
                self._bert_embs        = cache["bert_embs"]
                self._rate_model_ready = True
                logger.info("Rate model loaded from %s", explicit_path.name)
                return
            except Exception as e:
                logger.warning("Explicit rate model load failed (%s), recomputing …", e)

        # v2: GBoost + PCA(150) + meeting features (invalidates old LogReg cache)
        CACHE_VERSION = "v2"
        ckpt_hash = hashlib.md5(self.ckpt_path.encode()).hexdigest()[:8]
        cache_path = DATA_DIR / f"rate_model_cache_{CACHE_VERSION}_{ckpt_hash}.pkl"

        if cache_path.exists():
            logger.info("Loading rate model cache from %s", cache_path)
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                self._sklearn_pipeline = cache["pipeline"]
                self._col_means        = cache["col_means"]
                self._col_std          = cache["col_std"]
                self._pca              = cache.get("pca")
                self._mf_means         = cache.get("mf_means")
                self._mf_std           = cache.get("mf_std")
                self._soft_feats_all   = cache["soft_feats"]
                self._bert_embs        = cache["bert_embs"]
                self._rate_model_ready = True
                logger.info("Rate model loaded from cache (ckpt=%s)", ckpt_hash)
                return
            except Exception as e:
                logger.warning("Cache load failed (%s), recomputing …", e)

        # Daily-model soft features — reuse already-loaded model
        self._soft_feats_all = extract_soft_feats_for_all(
            self._df_meetings,
            model=self._model,
            tokenizer=self._tokenizer,
            n_tab=self._n_tab,
            device=self._device,
            max_articles=MAX_ARTICLES,
            max_length=MAX_LENGTH,
        )

        # BERT embeddings — reuse already-loaded BERT (no HF download)
        self._bert_embs = precompute_bert_cls_all(
            self._df_meetings,
            bert_name_or_model=self._model.bert,
            max_length=MAX_LENGTH,
            max_articles=MAX_ARTICLES,
            tokenizer=self._tokenizer,
            device=self._device,
        )

        # Fit sklearn on train split (GBoost + PCA + meeting features)
        (
            self._sklearn_pipeline,
            self._col_means,
            self._col_std,
            self._pca,
            self._mf_means,
            self._mf_std,
        ) = fit_sklearn(self._df_meetings, self._bert_embs)

        # Save cache to DATA_DIR; also copy to checkpoint dir if writable
        payload = {
            "pipeline":   self._sklearn_pipeline,
            "col_means":  self._col_means,
            "col_std":    self._col_std,
            "pca":        self._pca,
            "mf_means":   self._mf_means,
            "mf_std":     self._mf_std,
            "soft_feats": self._soft_feats_all,
            "bert_embs":  self._bert_embs,
        }
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)
            logger.info("Rate model cache saved → %s", cache_path)
        except Exception as e:
            logger.warning("Could not save rate model cache: %s", e)

        # Mirror to checkpoint dir so both stages live together
        ckpt_dir = Path(self.ckpt_path).parent
        mirror_path = ckpt_dir / cache_path.name
        if ckpt_dir != DATA_DIR and mirror_path != cache_path:
            try:
                with open(mirror_path, "wb") as f:
                    pickle.dump(payload, f)
                logger.info("Rate model cache mirrored → %s", mirror_path)
            except Exception as e:
                logger.debug("Could not mirror rate model to checkpoint dir: %s", e)

        self._rate_model_ready = True

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Backward-compatible: per-article RUONIA direction prediction.
        Returns list of dicts with keys: text, prediction, probabilities.
        """
        return self.predict_ruonia(texts)

    def predict_ruonia(self, texts: list[str]) -> list[dict]:
        """
        Predict RUONIA direction (down/same/up) for each article.
        Returns list of dicts.
        """
        self._ensure_daily_model()
        probs = score_articles(
            texts, self._model, self._tokenizer,
            self._n_tab, self._device, MAX_LENGTH,
            tab_default=self._tab_default,
        )
        results = []
        for i, text in enumerate(texts):
            p = probs[i]
            pred_idx = int(p.argmax())
            results.append({
                "text": str(text)[:200],
                "prediction": RUONIA_LABELS[pred_idx],
                "probabilities": {
                    "down": float(p[0]),
                    "same": float(p[1]),
                    "up":   float(p[2]),
                },
            })
        return results

    def predict_key_rate(
        self,
        texts: list[str] | None = None,
        tabular: np.ndarray | None = None,
        current_rate: float | None = None,
    ) -> dict:
        """
        Predict CBR key-rate decision (hike/hold/cut) for the next meeting.

        If texts is None, fetches latest articles from the raw CSV.
        Returns dict with: prediction, probabilities, signal, articles_summary.
        """
        self._ensure_rate_model()

        if current_rate is None:
            current_rate = get_current_key_rate()

        # If no texts provided, load from CSV (last 21 days)
        if texts is None:
            today = pd.Timestamp.now()
            from_date = today - pd.Timedelta(days=21)
            texts, tabular, _ = load_window_from_csv(
                CSV_PATH, from_date, today,
            )

        # If no tabular features, compute from DB for today
        if tabular is None:
            try:
                tab_len = len(self._col_means)
            except Exception:
                tab_len = 90
            try:
                from cbr_news.ml.feature_engineering import build_tabular_features, get_feature_columns
                today_str = pd.Timestamp.now().strftime("%d.%m.%Y")
                df_today = pd.DataFrame([{"date": today_str}])
                df_feat = build_tabular_features(df_today, date_col="date")
                feat_cols = get_feature_columns(df_feat)
                tabular = df_feat[feat_cols].iloc[0].values.astype(np.float32)
                if len(tabular) != tab_len:
                    logger.warning(
                        "Live tab features length mismatch (%d vs %d) — using training means",
                        len(tabular), tab_len,
                    )
                    tabular = self._col_means.copy()
                else:
                    logger.info("Key-rate window tabular: live DB features for %s", today_str)
            except Exception as e:
                logger.warning("Could not get live tabular features: %s — using training means", e)
                tabular = self._col_means.copy() if self._col_means is not None else np.zeros(tab_len, dtype=np.float32)

        # Compute current window's meeting-level features
        current_meet_raw = compute_current_meet_feats(
            self._df_meetings, n_articles=len(texts),
        )
        if self._mf_means is not None and self._mf_std is not None:
            current_meet = ((current_meet_raw - self._mf_means) / self._mf_std).reshape(1, -1)
        else:
            current_meet = None

        result = predict_window(
            texts=texts,
            tabular=tabular,
            col_means=self._col_means,
            col_std=self._col_std,
            bert_name=BERT_NAME,
            max_length=MAX_LENGTH_STAGE2,
            max_articles=MAX_ARTICLES,
            pipeline=self._sklearn_pipeline,
            daily_model=self._model,
            daily_tokenizer=self._tokenizer,
            daily_n_tab=self._n_tab,
            daily_device=self._device,
            use_soft_feats=False,
            pca=self._pca,
            meet_feats=current_meet,
        )

        probs = result["probs"]
        pred_label = result["pred_label"]
        article_probs = result.get("article_probs")

        # Build articles summary
        signal = {}
        if article_probs is not None:
            hawk = article_probs[:, 2] - article_probs[:, 0]
            net_hawk = float(hawk.mean())
            n_total = len(article_probs)
            argmax_labels = article_probs.argmax(1)
            signal = {
                "net_hawkishness": round(net_hawk, 4),
                "tone": (
                    "hawkish" if net_hawk > 0.05
                    else "dovish" if net_hawk < -0.05
                    else "neutral"
                ),
                "articles_hike": int((argmax_labels == 2).sum()),
                "articles_hold": int((argmax_labels == 1).sum()),
                "articles_cut":  int((argmax_labels == 0).sum()),
                "total_articles": n_total,
            }

        # Expected rate
        if pred_label == "hike":
            expected = current_rate + 1.0
        elif pred_label == "cut":
            expected = current_rate - 0.5
        else:
            expected = current_rate

        return {
            "prediction": pred_label,
            "probabilities": {
                "cut":  float(probs[0]),
                "hold": float(probs[1]),
                "hike": float(probs[2]),
            },
            "current_rate": current_rate,
            "expected_rate": expected,
            "signal": signal,
            "n_articles": len(texts),
        }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Key rate inference service")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--meeting-date", type=str, default="",
                     help="Historical CBR meeting date YYYY-MM-DD (uses pre-built dataset)")
    grp.add_argument("--to-date",      type=str, default="",
                     help="Meeting date for custom window (YYYY-MM-DD)")
    ap.add_argument("--from-date",       type=str, default="",
                    help="Start of article window (YYYY-MM-DD); default = to-date − window-days")
    ap.add_argument("--window-days",     type=int, default=14)
    ap.add_argument("--news-csv",        type=str, default=str(CSV_PATH))
    ap.add_argument("--meeting-dataset", type=str, default=str(MEETING_DS))
    ap.add_argument("--daily-model",     type=str, default="",
                    help="Path to JointBertTabularModel .ckpt for Stage 1 article scoring")
    ap.add_argument("--max-articles",    type=int, default=MAX_ARTICLES)
    ap.add_argument("--max-length",      type=int, default=MAX_LENGTH)
    ap.add_argument("--top-n",           type=int, default=5)
    ap.add_argument("--current-rate",    type=float, default=CURRENT_KEY_RATE)
    args = ap.parse_args()

    # ── Validate args ──────────────────────────────────────────────────────
    if not args.meeting_date and not args.to_date:
        ap.error("Provide --meeting-date OR --to-date.")

    # ── Load meeting dataset for sklearn training ──────────────────────────
    ds_path = Path(args.meeting_dataset)
    if not ds_path.exists():
        logger.error("Meeting dataset not found: %s", ds_path)
        logger.error("Run: PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py")
        return

    logger.info("Loading meeting dataset: %s", ds_path)
    df_meetings = pd.read_parquet(ds_path)
    df_meetings["meeting_date"] = pd.to_datetime(df_meetings["meeting_date"])

    # ── Determine query window ─────────────────────────────────────────────
    if args.meeting_date:
        meeting_date = pd.Timestamp(args.meeting_date)
        row_mask     = df_meetings["meeting_date"] == meeting_date
        if not row_mask.any():
            logger.error("Meeting date %s not found in dataset.", args.meeting_date)
            available = sorted(df_meetings["meeting_date"].dt.date.tolist())
            logger.error("Available: %s", available)
            return

        row_data   = df_meetings[row_mask].iloc[0]
        texts      = list(row_data["texts"])
        tabular    = np.array(row_data["tabular"], dtype=np.float32)
        from_date  = meeting_date - pd.Timedelta(days=14)
        to_date    = meeting_date
        true_label = row_data["direction"]
    else:
        to_date   = pd.Timestamp(args.to_date)
        from_date = (
            pd.Timestamp(args.from_date)
            if args.from_date
            else to_date - pd.Timedelta(days=args.window_days)
        )
        texts, tabular, _ = load_window_from_csv(
            Path(args.news_csv), from_date, to_date
        )
        true_label = None

    daily_model     = None
    daily_tokenizer = None
    daily_n_tab     = 90
    daily_device    = None
    soft_feats_all  = None

    if args.daily_model:
        daily_model, daily_tokenizer, daily_n_tab, daily_device = load_daily_model(
            args.daily_model
        )
        soft_feats_all = extract_soft_feats_for_all(
            df_meetings,
            model        = daily_model,
            tokenizer    = daily_tokenizer,
            n_tab        = daily_n_tab,
            device       = daily_device,
            max_articles = args.max_articles,
            max_length   = args.max_length,
        )

    # ── BERT embeddings for all training meetings ──────────────────────────
    bert_embs = precompute_bert_cls_all(
        df_meetings, BERT_NAME, args.max_length, args.max_articles
    )

    # ── Fit sklearn on training split ─────────────────────────────────────
    pipeline, col_means, col_std = fit_sklearn(
        df_meetings, bert_embs, soft_feats=soft_feats_all
    )

    # ── Predict for the query window ───────────────────────────────────────
    result = predict_window(
        texts           = texts,
        tabular         = tabular,
        col_means       = col_means,
        col_std         = col_std,
        bert_name       = BERT_NAME,
        max_length      = args.max_length,
        max_articles    = args.max_articles,
        pipeline        = pipeline,
        daily_model     = daily_model,
        daily_tokenizer = daily_tokenizer,
        daily_n_tab     = daily_n_tab,
        daily_device    = daily_device,
        use_soft_feats  = soft_feats_all is not None,
    )

    # ── Print human-readable report ────────────────────────────────────────
    print_report(
        from_date        = from_date,
        to_date          = to_date,
        texts            = texts,
        result           = result,
        top_n            = args.top_n,
        current_key_rate = args.current_rate,
        true_label       = true_label,
    )


if __name__ == "__main__":
    main()
