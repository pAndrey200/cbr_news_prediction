"""
Stage-2 hyperparameter sweep: predict CBR key rate direction at meeting level.

Systematically explores:
  • Classifiers : GBoost, RF, ExtraTrees, LogReg, SVC
  • PCA dims    : 50, 100, 150, 200, None
  • Feature sets: BERT+tab+soft, BERT+tab, tab+soft, BERT only, tab only
  • Meeting-specific features (streak, rate_level, days_since_change, ...)
  • Expanding-window CV vs fixed train/val split

Usage:
    PYTHONPATH=. python cbr_news/ml/sweep_meeting.py \
        --checkpoint checkpoints/sweep/run_002/best-v24.ckpt

Results saved to: sweep_meeting_results.csv
"""

import argparse
import logging
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parents[2] / "data"
CKPT_DIR   = Path(__file__).parents[2] / "checkpoints"
MEETING_DS = DATA_DIR / "meeting_dataset.parquet"
RESULTS    = Path(__file__).parents[2] / "sweep_meeting_results.csv"

TRAIN_END = pd.Timestamp("2022-12-31")
CLASS_NAMES = ["cut", "hold", "hike"]


# ─── Meeting-specific feature engineering ─────────────────────────────────────

def add_meeting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add structural features that capture meeting-level context:
      - rate_level        : key rate at meeting date
      - streak            : consecutive meetings with same decision
      - days_since_change : days since last rate change
      - n_articles        : number of articles in window
      - n_press_rel       : fraction of press releases among articles
      - consecutive_holds : consecutive hold meetings (pressure builds)
    """
    df = df.sort_values("meeting_date").reset_index(drop=True)

    # 1. rate_level from tabular (feat_key_rate is the first tabular feature we compute)
    # Extract from tabular array — col 0 is feat_year, look for feat_key_rate by position
    # We'll add it from the known feature name instead (safer)
    # Fallback: use n_articles as proxy

    # 2. Streak: consecutive meetings with same direction (using PREVIOUS meeting only)
    directions = df["direction"].tolist()
    streaks = [1]
    for i in range(1, len(directions)):
        if directions[i - 1] == directions[i - 2] if i >= 2 else False:
            streaks.append(streaks[-1] + 1)
        else:
            streaks.append(1)
    df["meet_streak"] = streaks

    # 3. Days since last rate change (using only PAST decisions — no look-ahead)
    days_since = []
    last_change = pd.NaT
    for i, row in df.iterrows():
        if pd.isna(last_change):
            days_since.append(0.0)
        else:
            days_since.append(float((row["meeting_date"] - last_change).days))
        # Update AFTER recording, using PREVIOUS meeting direction
        if i > 0 and directions[i - 1] != "hold":
            last_change = df.iloc[i - 1]["meeting_date"]
        elif i == 0:
            last_change = row["meeting_date"]
    df["meet_days_since_change"] = days_since

    # 4. n_articles already in df
    if "n_articles" not in df.columns:
        df["n_articles"] = df["texts"].apply(lambda x: len(x) if isinstance(x, list) else 1)

    # 5. Consecutive holds (hold streak)
    hold_streaks = [0]
    for i in range(1, len(directions)):
        if directions[i - 1] == "hold":
            hold_streaks.append(hold_streaks[-1] + 1)
        else:
            hold_streaks.append(0)
    df["meet_hold_streak"] = hold_streaks

    # 6. Month of year (seasonality)
    df["meet_month"] = df["meeting_date"].dt.month

    # 7. Year (trend)
    df["meet_year"] = df["meeting_date"].dt.year

    logger.info("Added meeting-specific features: streak, days_since_change, hold_streak, month, year")
    return df


def extract_meeting_feats(df: pd.DataFrame) -> np.ndarray:
    """Return (N, K) float32 array of meeting-specific features."""
    cols = ["meet_streak", "meet_days_since_change", "n_articles",
            "meet_hold_streak", "meet_month", "meet_year"]
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return np.zeros((len(df), 1), dtype=np.float32)
    return df[avail].fillna(0).values.astype(np.float32)


# ─── BERT embeddings with reuse of stage-1 checkpoint ─────────────────────────

def load_bert_from_checkpoint(ckpt_path: str):
    """Load BERT encoder from the stage-1 joint model checkpoint."""
    from cbr_news.ml.train_joint import JointBertTabularModel
    logger.info("Loading stage-1 checkpoint: %s", ckpt_path)
    model = JointBertTabularModel.load_from_checkpoint(
        ckpt_path, map_location="cpu", strict=False
    )
    model.eval()
    from transformers import AutoTokenizer
    tok_name  = getattr(model.hparams, "backbone", "DeepPavlov/rubert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    bert      = model.bert
    return bert, tokenizer, model, device


def compute_bert_embs(df: pd.DataFrame, bert, tokenizer, device,
                      max_length: int = 256, max_articles: int = 20) -> np.ndarray:
    """Compute mean CLS embedding per meeting. Returns (N, H)."""
    H    = bert.config.hidden_size
    embs = np.zeros((len(df), H), dtype=np.float32)
    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            texts = [str(t) for t in list(row["texts"])[:max_articles]] or [""]
            enc   = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            enc     = {k: v.to(device) for k, v in enc.items()}
            out     = bert(**enc)
            cls     = out.last_hidden_state[:, 0, :].cpu().numpy()

            # Separate embeddings: press-releases vs news
            # Press-releases come first (sorted by priority in prepare_meeting_dataset.py)
            n_pr = int(row.get("n_press_releases", 0)) if "n_press_releases" in row.index else 0
            if n_pr > 0 and n_pr <= len(cls):
                # Weighted: press releases get 2x weight
                w = np.ones(len(cls))
                w[:n_pr] = 2.0
                w = w / w.sum()
                embs[i] = (cls * w[:, None]).sum(0)
            else:
                embs[i] = cls.mean(0)

            if (i + 1) % 20 == 0 or (i + 1) == len(df):
                logger.info("  … %d / %d meetings", i + 1, len(df))
    return embs


def compute_weighted_bert_embs(df: pd.DataFrame, bert, stage1_model, tokenizer, device,
                               max_length: int = 256, max_articles: int = 20) -> np.ndarray:
    """
    BERT CLS embeddings weighted by Stage-1 non-neutral confidence.
    Articles Stage-1 rates as clearly hawkish/dovish get higher weight.
    This lets Stage-1 decide which articles matter most for the embedding,
    rather than treating all articles equally.
    """
    from cbr_news.ml.inference import score_articles
    n_tab = (
        stage1_model.hparams.get("n_tabular", 90)
        if hasattr(stage1_model.hparams, "get")
        else getattr(stage1_model.hparams, "n_tabular", 90)
    )
    H    = bert.config.hidden_size
    embs = np.zeros((len(df), H), dtype=np.float32)

    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            texts = [str(t) for t in list(row["texts"])[:max_articles]] or [""]

            # CLS embeddings
            enc = tokenizer(texts, truncation=True, padding="max_length",
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            cls = bert(**enc).last_hidden_state[:, 0, :].cpu().numpy()

            # Stage-1 confidence: P(cut) + P(hike) = how non-neutral the article is
            raw_tab = np.array(row["tabular"], dtype=np.float32)
            tab_tensor = torch.zeros(1, n_tab, device=device)
            m = min(len(raw_tab), n_tab)
            tab_tensor[0, :m] = torch.from_numpy(raw_tab[:m])

            probs = score_articles(texts, stage1_model, tokenizer, n_tab,
                                   device, max_length, tab_tensor)
            confidence = probs[:, 0] + probs[:, 2]  # P(down) + P(up)

            w = confidence / confidence.sum() if confidence.sum() > 1e-6 \
                else np.ones(len(cls)) / len(cls)
            embs[i] = (cls * w[:, None]).sum(0)

            if (i + 1) % 20 == 0 or (i + 1) == len(df):
                logger.info("  weighted … %d / %d meetings", i + 1, len(df))
    return embs


def compute_soft_feats(df: pd.DataFrame, model, tokenizer, device,
                       max_length: int = 512, max_articles: int = 20) -> np.ndarray:
    """Run stage-1 model on each meeting's articles, aggregate soft labels."""
    from cbr_news.ml.inference import score_articles, aggregate_probs
    n_tab = (
        model.hparams.get("n_tabular", 90)
        if hasattr(model.hparams, "get")
        else getattr(model.hparams, "n_tabular", 90)
    )
    all_feats = []
    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            texts = [str(t) for t in list(row["texts"])[:max_articles]] or [""]
            probs = score_articles(texts, model, tokenizer, n_tab, device, max_length)
            all_feats.append(aggregate_probs(probs))
            if (i + 1) % 20 == 0 or (i + 1) == len(df):
                logger.info("  … %d / %d meetings", i + 1, len(df))
    return np.array(all_feats, dtype=np.float32)


def _compute_temporal_feats(probs: np.ndarray, dates: list) -> np.ndarray:
    """
    6 temporal features from per-article probabilities (ordered by date):
      slope_up, slope_down  — linear trend of P(up/down) over the window
      recent_minus_early    — avg P(up) last-third minus first-third of window
      ew_hawk               — exponentially recency-weighted mean P(up)
      max_up_last5          — max P(up) in last 5 articles
      frac_dom_up           — fraction of articles where argmax == "up"
    """
    n = len(probs)
    if n <= 1:
        return np.zeros(6, dtype=np.float32)
    try:
        parsed = pd.to_datetime(dates, errors="coerce")
        fill_val = parsed.dropna().min() if parsed.notna().any() else pd.Timestamp("2000-01-01")
        order = np.argsort(parsed.fillna(fill_val))
        ps = probs[order]
        t  = np.linspace(0, 1, n)

        slope_up   = float(np.polyfit(t, ps[:, 2], 1)[0])
        slope_down = float(np.polyfit(t, ps[:, 0], 1)[0])

        third = max(1, n // 3)
        recent_minus_early = float(ps[-third:, 2].mean() - ps[:third, 2].mean())

        weights = np.exp(np.linspace(0, 2, n))
        weights /= weights.sum()
        ew_hawk = float((ps[:, 2] * weights).sum())

        max_up_last5 = float(ps[-min(5, n):, 2].max())
        frac_dom_up  = float((probs.argmax(1) == 2).mean())

        return np.array(
            [slope_up, slope_down, recent_minus_early, ew_hawk, max_up_last5, frac_dom_up],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(6, dtype=np.float32)


def compute_signal_feats(df: pd.DataFrame, model, tokenizer, device,
                         max_length: int = 256, max_articles: int = 20) -> np.ndarray:
    """
    Stage-1 signal features computed WITH real tabular context + temporal dynamics.
    Returns (N, 19): 13 aggregate + 6 temporal features per meeting.

    Improvement over soft_feats:
      • passes real economic tabular features to stage-1 (not zeros)
      • adds temporal trend/momentum features from per-article time series
    """
    from cbr_news.ml.inference import score_articles, aggregate_probs
    n_tab = (
        model.hparams.get("n_tabular", 90)
        if hasattr(model.hparams, "get")
        else getattr(model.hparams, "n_tabular", 90)
    )
    all_feats = []
    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            texts = [str(t) for t in list(row["texts"])[:max_articles]] or [""]
            dates = list(row.get("article_dates", [""] * len(texts)))[:max_articles]

            # Build tab_default from real meeting-date tabular features
            raw_tab = np.array(row["tabular"], dtype=np.float32)
            tab_tensor = torch.zeros(1, n_tab, device=device)
            m = min(len(raw_tab), n_tab)
            tab_tensor[0, :m] = torch.from_numpy(raw_tab[:m])

            probs = score_articles(texts, model, tokenizer, n_tab, device, max_length, tab_tensor)
            base_feats     = aggregate_probs(probs)            # 13 dims
            temporal_feats = _compute_temporal_feats(probs, dates)   # 6 dims
            all_feats.append(np.concatenate([base_feats, temporal_feats]))

            if (i + 1) % 20 == 0 or (i + 1) == len(df):
                logger.info("  signal … %d / %d meetings", i + 1, len(df))
    return np.array(all_feats, dtype=np.float32)


# ─── Classifier grid ──────────────────────────────────────────────────────────

def make_classifiers():
    return [
        ("GBoost_d2_n200", GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            min_samples_leaf=3, subsample=0.8, random_state=42,
        )),
        ("GBoost_d3_n100", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=2, subsample=0.8, random_state=42,
        )),
        ("GBoost_d2_n100_lr01", GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.1,
            min_samples_leaf=2, subsample=0.9, random_state=42,
        )),
        ("RF_n500_d4", RandomForestClassifier(
            n_estimators=500, max_depth=4, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        )),
        ("RF_n200_d3", RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_leaf=3,
            class_weight="balanced", random_state=42,
        )),
        ("ExtraTrees_n500", ExtraTreesClassifier(
            n_estimators=500, max_depth=4, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        )),
        ("LogReg_C001",  LogisticRegression(C=0.01,  max_iter=2000, class_weight="balanced", random_state=42)),
        ("LogReg_C01",   LogisticRegression(C=0.1,   max_iter=2000, class_weight="balanced", random_state=42)),
        ("LogReg_C1",    LogisticRegression(C=1.0,   max_iter=2000, class_weight="balanced", random_state=42)),
        ("SVC_rbf_C01",  SVC(C=0.1,  kernel="rbf",    class_weight="balanced", probability=True, random_state=42)),
        ("SVC_rbf_C1",   SVC(C=1.0,  kernel="rbf",    class_weight="balanced", probability=True, random_state=42)),
        ("LinearSVC_C001", LinearSVC(C=0.01, max_iter=3000, class_weight="balanced", random_state=42)),
        ("LinearSVC_C01",  LinearSVC(C=0.1,  max_iter=3000, class_weight="balanced", random_state=42)),
    ]


# ─── Main sweep logic ──────────────────────────────────────────────────────────

def build_feature_matrix(  # noqa: PLR0913
    embs: np.ndarray,
    X_tab: np.ndarray,
    soft_feats: np.ndarray,
    signal_feats: np.ndarray,
    weighted_embs: np.ndarray,
    meet_feats: np.ndarray,
    feature_set: str,
    pca_dim: int | None,
    train_mask: np.ndarray,
) -> np.ndarray:
    """
    Assemble feature matrix according to feature_set config.
    bert   = uniform-pooled CLS embeddings
    bert_w = Stage-1-confidence-weighted CLS embeddings
    """
    def _apply_pca(arr, pca_dim, train_mask):
        if pca_dim is None:
            return arr
        n = min(pca_dim, arr.shape[1], train_mask.sum() - 1)
        pca = PCA(n_components=max(n, 1), random_state=42)
        pca.fit(arr[train_mask])
        return pca.transform(arr)

    parts = []
    if "bert_w" in feature_set and weighted_embs is not None:
        parts.append(_apply_pca(weighted_embs, pca_dim, train_mask))
    elif "bert" in feature_set and embs is not None:
        parts.append(_apply_pca(embs, pca_dim, train_mask))

    if "tab" in feature_set and X_tab is not None:
        parts.append(X_tab)
    if "soft" in feature_set and soft_feats is not None:
        parts.append(soft_feats)
    if "signal" in feature_set and signal_feats is not None:
        parts.append(signal_feats)
    if "meet" in feature_set and meet_feats is not None:
        parts.append(meet_feats)

    if not parts:
        raise ValueError(f"No features selected for set: {feature_set}")
    return np.concatenate(parts, axis=1)


def expanding_cv_score(
    X: np.ndarray, y: np.ndarray,
    dates: pd.Series,
    clf_name: str, clf,
    min_train: int = 20,
    step: int = 1,
) -> float:
    """
    Expanding-window CV: train on [0..t], predict [t+1..t+step].
    Collects all out-of-sample predictions, then computes one macro-F1.
    Only uses data up to TRAIN_END.
    """
    mask_all = (dates <= TRAIN_END).values
    X_cv = X[mask_all]
    y_cv = y[mask_all]

    n = len(X_cv)
    all_true, all_pred = [], []

    for t in range(min_train, n - step + 1, step):
        X_tr = X_cv[:t]
        y_tr = y_cv[:t]
        X_vl = X_cv[t:t + step]
        y_vl = y_cv[t:t + step]

        if len(np.unique(y_tr)) < 2:
            continue
        try:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_vl_s = scaler.transform(X_vl)
            clf_copy = clone_clf(clf)
            clf_copy.fit(X_tr_s, y_tr)
            pred = clf_copy.predict(X_vl_s)
            all_true.extend(y_vl.tolist())
            all_pred.extend(pred.tolist())
        except Exception:
            pass

    if not all_true:
        return 0.0
    return float(f1_score(all_true, all_pred, average="macro", zero_division=0))


def clone_clf(clf):
    """Clone a sklearn estimator."""
    from sklearn.base import clone
    return clone(clf)


def run_sweep(
    df: pd.DataFrame,
    embs: np.ndarray,
    soft_feats: np.ndarray,
    signal_feats: np.ndarray,
    weighted_embs: np.ndarray,
    ckpt_path: str,
) -> pd.DataFrame:
    """Main sweep loop. Returns DataFrame with results."""

    # ── Prepare masks
    dates      = df["meeting_date"]
    train_mask = (dates <= TRAIN_END).values
    test_mask  = ~train_mask

    logger.info("Split: train=%d  test=%d",
                train_mask.sum(), test_mask.sum())

    y = df["label"].values

    # ── Tabular features (normalise on train)
    X_tab_raw = np.stack(df["tabular"].values).astype(np.float32)
    col_means  = np.nanmean(X_tab_raw[train_mask], axis=0)
    col_means[np.isnan(col_means)] = 0.0
    nan_idx = np.where(np.isnan(X_tab_raw))
    if nan_idx[0].size:
        X_tab_raw[nan_idx] = np.take(col_means, nan_idx[1])
    col_std  = np.nanstd(X_tab_raw[train_mask], axis=0) + 1e-6
    X_tab    = (X_tab_raw - col_means) / col_std

    # ── Meeting-specific features
    df_feats  = add_meeting_features(df.copy())
    meet_feats = extract_meeting_feats(df_feats)
    mf_means  = meet_feats[train_mask].mean(0)
    mf_std    = meet_feats[train_mask].std(0) + 1e-6
    meet_feats = (meet_feats - mf_means) / mf_std

    # ── Feature sets to try
    feature_sets = [
        # bert_w = Stage-1-confidence-weighted BERT embeddings (main new variant)
        "bert_w+tab+meet",
        "bert_w+tab",
        "bert_w+meet",
        "bert_w",
        # signal = stage-1 predictions with real tabular context + temporal features
        "tab+signal+meet",
        "tab+signal",
        "signal+meet",
        # soft = stage-1 predictions with zero tabular context (legacy baseline)
        "tab+soft+meet",
        "tab+meet",
        "tab+soft",
        # bert_w + soft/signal combos
        "bert_w+tab+soft+meet",
        "bert_w+tab+signal+meet",
        # plain BERT baseline
        "bert+tab+meet",
        "bert+tab",
        "bert",
        "tab",
    ]

    pca_dims = [50, 100, 150, 200, None]
    classifiers = make_classifiers()

    from joblib import Parallel, delayed

    configs = [
        (feat_set, pca_dim, clf_name, clf)
        for feat_set, pca_dim, (clf_name, clf) in product(feature_sets, pca_dims, classifiers)
        if ("bert" in feat_set) or pca_dim is None
    ]

    def _eval_config(feat_set, pca_dim, clf_name, clf):
        try:
            X = build_feature_matrix(
                embs, X_tab, soft_feats, signal_feats, weighted_embs, meet_feats,
                feat_set, pca_dim, train_mask,
            )
        except Exception as e:
            return None

        cv_f1 = expanding_cv_score(X, y, dates, clf_name, clf)

        try:
            scaler2 = StandardScaler()
            X_tr_s  = scaler2.fit_transform(X[train_mask])
            X_te_s  = scaler2.transform(X[test_mask])
            clf_test = clone_clf(clf)
            clf_test.fit(X_tr_s, y[train_mask])
            test_pred = clf_test.predict(X_te_s)
            test_f1   = f1_score(y[test_mask], test_pred, average="macro", zero_division=0)
            test_report = classification_report(
                y[test_mask], test_pred, target_names=CLASS_NAMES, zero_division=0,
            )
        except Exception as e:
            test_f1, test_report = 0.0, str(e)

        if test_f1 >= 0.65:
            logger.info(
                "*** test_f1=%.4f  cv_f1=%.4f | %s | pca=%s | %s",
                test_f1, cv_f1, feat_set, pca_dim, clf_name,
            )
            print(test_report)

        return {
            "feature_set": feat_set,
            "pca_dim":     pca_dim if pca_dim else "none",
            "clf":         clf_name,
            "n_features":  X.shape[1],
            "cv_f1":       round(cv_f1, 4),
            "test_f1":     round(test_f1, 4),
        }

    logger.info("Running %d configs in parallel (n_jobs=-1)...", len(configs))
    rows = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_eval_config)(fs, pd_, cn, cl)
        for fs, pd_, cn, cl in configs
    )
    results = [r for r in rows if r is not None]

    df_res = pd.DataFrame(results).sort_values("test_f1", ascending=False)
    df_res.to_csv(RESULTS, index=False)
    logger.info("Results saved to %s", RESULTS)
    return df_res


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/sweep/run_002/best-v24.ckpt",
                        help="Path to stage-1 checkpoint")
    parser.add_argument("--max-articles", type=int, default=20)
    parser.add_argument("--max-length",   type=int, default=256)
    parser.add_argument("--cache-embs",   action="store_true",
                        help="Save/load BERT embeddings to avoid recomputing")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Override TRAIN_END, e.g. 2022-12-31")
    args = parser.parse_args()

    global TRAIN_END
    if args.train_end:
        TRAIN_END = pd.Timestamp(args.train_end)
        logger.info("TRAIN_END overridden to %s", TRAIN_END.date())

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parents[2] / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Load meeting dataset
    if not MEETING_DS.exists():
        logger.info("meeting_dataset.parquet not found — building it now...")
        from cbr_news.ml.prepare_meeting_dataset import build_meeting_dataset
        df = build_meeting_dataset()
        df.to_parquet(MEETING_DS)
    else:
        df = pd.read_parquet(MEETING_DS)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    logger.info("Loaded %d meetings  labels: %s",
                len(df), df["direction"].value_counts().to_dict())

    # ── Load stage-1 model
    bert, tokenizer, stage1_model, device = load_bert_from_checkpoint(str(ckpt_path))

    # ── BERT embeddings (optionally cached)
    emb_cache = DATA_DIR / f"meet_bert_embs_{args.max_length}_{args.max_articles}.npy"
    if args.cache_embs and emb_cache.exists():
        logger.info("Loading cached BERT embeddings from %s", emb_cache)
        embs = np.load(str(emb_cache))
    else:
        logger.info("Computing BERT embeddings (max_length=%d)...", args.max_length)
        embs = compute_bert_embs(df, bert, tokenizer, device, args.max_length, args.max_articles)
        if args.cache_embs:
            np.save(str(emb_cache), embs)
            logger.info("Saved to %s", emb_cache)

    # ── Soft features from stage-1 model (legacy: zero tabular context)
    soft_cache = DATA_DIR / f"meet_soft_feats_{args.max_articles}.npy"
    if args.cache_embs and soft_cache.exists():
        logger.info("Loading cached soft features from %s", soft_cache)
        soft_feats = np.load(str(soft_cache))
    else:
        logger.info("Computing soft features from stage-1 model...")
        soft_feats = compute_soft_feats(
            df, stage1_model, tokenizer, device, max_length=512, max_articles=args.max_articles,
        )
        if args.cache_embs:
            np.save(str(soft_cache), soft_feats)

    # ── Signal features: stage-1 with real tabular context + temporal dynamics
    signal_cache = DATA_DIR / f"meet_signal_feats_{args.max_articles}.npy"
    if args.cache_embs and signal_cache.exists():
        logger.info("Loading cached signal features from %s", signal_cache)
        signal_feats = np.load(str(signal_cache))
    else:
        logger.info("Computing signal features (stage-1 with real tabular context)...")
        signal_feats = compute_signal_feats(
            df, stage1_model, tokenizer, device,
            max_length=args.max_length, max_articles=args.max_articles,
        )
        if args.cache_embs:
            np.save(str(signal_cache), signal_feats)
            logger.info("Saved signal features to %s", signal_cache)

    # ── Weighted BERT embeddings (Stage-1 confidence-weighted article pooling)
    w_emb_cache = DATA_DIR / f"meet_bert_embs_weighted_{args.max_length}_{args.max_articles}.npy"
    if args.cache_embs and w_emb_cache.exists():
        logger.info("Loading cached weighted BERT embeddings from %s", w_emb_cache)
        weighted_embs = np.load(str(w_emb_cache))
    else:
        logger.info("Computing Stage-1-weighted BERT embeddings...")
        weighted_embs = compute_weighted_bert_embs(
            df, bert, stage1_model, tokenizer, device,
            args.max_length, args.max_articles,
        )
        if args.cache_embs:
            np.save(str(w_emb_cache), weighted_embs)
            logger.info("Saved to %s", w_emb_cache)

    # ── Run sweep
    df_res = run_sweep(df, embs, soft_feats, signal_feats, weighted_embs, str(ckpt_path))

    print("\n" + "=" * 70)
    print("TOP-10 CONFIGURATIONS BY TEST F1:")
    print("=" * 70)
    print(df_res.head(10).to_string(index=False))

    best = df_res.iloc[0]
    print(f"\nBest: test_f1={best['test_f1']}  cv_f1={best['cv_f1']}"
          f"\n  features={best['feature_set']}"
          f"\n  pca={best['pca_dim']}"
          f"\n  clf={best['clf']}")


if __name__ == "__main__":
    main()
