"""
Train CatBoost on BERT embeddings + economic tabular features.

Pipeline:
  1. Load BERT embeddings (data/embeddings.npy) + metadata (data/embeddings_meta.parquet)
  2. Build tabular economic features via feature_engineering.build_tabular_features
  3. Reduce embedding dimensionality with PCA (configurable)
  4. Concatenate PCA embeddings + tabular features + sentiment scores
  5. Temporal train/val/test split
  6. Train CatBoostClassifier with class-weight balancing
  7. Evaluate (accuracy, weighted F1, per-class report)
  8. Log to MLflow

Run:
    python cbr_news/ml/train_catboost.py [--pca-components 100] [--iterations 2000]
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score

from cbr_news.ml.feature_engineering import build_tabular_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
CKPT_DIR = Path(__file__).parents[2] / "checkpoints" / "catboost"

LABEL_MAP   = {"down": 0, "same": 1, "up": 2}
CLASS_NAMES = ["down", "same", "up"]

TRAIN_YEARS = list(range(2010, 2023))
VAL_YEARS   = [2023]
TEST_YEARS  = [2024, 2025, 2026]


def _extract_year(date_str: str):
    try:
        parts = str(date_str).split(".")
        if len(parts) == 3:
            return int(parts[2])
        return pd.to_datetime(date_str).year
    except Exception:
        return None


def _class_weights(y_train: np.ndarray) -> list:
    counts = Counter(y_train)
    total  = sum(counts.values())
    n_cls  = len(LABEL_MAP)
    return [total / (n_cls * counts.get(i, 1)) for i in range(n_cls)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-components", type=int, default=100,
                        help="PCA components for embedding reduction (0 = no PCA)")
    parser.add_argument("--iterations",     type=int, default=2000)
    parser.add_argument("--lr",             type=float, default=0.05)
    parser.add_argument("--depth",          type=int, default=6)
    parser.add_argument("--early-stop",     type=int, default=100)
    parser.add_argument("--dataset",        type=str,
                        default=str(DATA_DIR / "cbr_combined_dataset.csv"))
    args = parser.parse_args()

    emb_path  = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "embeddings_meta.parquet"

    # ── 1. Load embeddings & metadata ─────────────────────────────────────────
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Run compute_embeddings.py first to generate embeddings.npy and embeddings_meta.parquet"
        )

    logger.info("Loading embeddings and metadata...")
    embeddings = np.load(emb_path)          # (N, 768)
    meta       = pd.read_parquet(meta_path) # date, target, sentiment_*, embed_idx

    logger.info(f"Embeddings: {embeddings.shape}")
    logger.info(f"Metadata  : {meta.shape}")
    assert len(meta) == len(embeddings), "Size mismatch between metadata and embeddings"

    # ── 2. Build tabular features ─────────────────────────────────────────────
    logger.info("Loading dataset for feature engineering...")
    df = pd.read_csv(args.dataset, usecols=["date", "target"])
    df = df.reset_index(drop=True)
    assert len(df) == len(meta), "Dataset and metadata row count mismatch"

    logger.info("Building tabular economic features...")
    df_feat = build_tabular_features(df, date_col="date")

    # ── 3. Labels ─────────────────────────────────────────────────────────────
    meta = meta.reset_index(drop=True)
    df_feat = df_feat.reset_index(drop=True)

    labels   = meta["target"].map(LABEL_MAP)
    valid    = labels.notna()
    logger.info(f"Valid rows: {valid.sum()} / {len(valid)}")

    meta      = meta[valid].reset_index(drop=True)
    df_feat   = df_feat[valid].reset_index(drop=True)
    embeddings = embeddings[valid.values]
    labels     = labels[valid].astype(int).reset_index(drop=True)

    # ── 4. Temporal split ─────────────────────────────────────────────────────
    meta["year"] = meta["date"].apply(_extract_year)

    train_mask = meta["year"].isin(TRAIN_YEARS).values
    val_mask   = meta["year"].isin(VAL_YEARS).values
    test_mask  = meta["year"].isin(TEST_YEARS).values

    logger.info(f"Train: {train_mask.sum():>6d} rows (years {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]})")
    logger.info(f"Val:   {val_mask.sum():>6d} rows (years {VAL_YEARS})")
    logger.info(f"Test:  {test_mask.sum():>6d} rows (years {TEST_YEARS})")

    # ── 5. Feature matrices ───────────────────────────────────────────────────
    feat_cols   = get_feature_columns(df_feat)
    sent_cols   = [c for c in meta.columns if c.startswith("sentiment_")]

    X_tab  = df_feat[feat_cols].values.astype(np.float32)
    X_sent = meta[sent_cols].values.astype(np.float32) if sent_cols else np.zeros((len(meta), 0))

    logger.info(f"Tabular features : {X_tab.shape[1]}")
    logger.info(f"Sentiment features: {X_sent.shape[1]}")

    # ── 6. PCA on embeddings ──────────────────────────────────────────────────
    n_pca = args.pca_components
    if n_pca > 0:
        logger.info(f"Fitting PCA({n_pca}) on train embeddings...")
        pca = PCA(n_components=n_pca, random_state=42)
        emb_train = pca.fit_transform(embeddings[train_mask])
        emb_val   = pca.transform(embeddings[val_mask])
        emb_test  = pca.transform(embeddings[test_mask])
        explained = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance: {explained:.3f}")
    else:
        logger.info("No PCA – using raw embeddings")
        pca = None
        emb_train = embeddings[train_mask]
        emb_val   = embeddings[val_mask]
        emb_test  = embeddings[test_mask]
        explained = 1.0

    X_train = np.hstack([emb_train, X_tab[train_mask], X_sent[train_mask]])
    X_val   = np.hstack([emb_val,   X_tab[val_mask],   X_sent[val_mask]])
    X_test  = np.hstack([emb_test,  X_tab[test_mask],  X_sent[test_mask]])

    y_train = labels.values[train_mask]
    y_val   = labels.values[val_mask]
    y_test  = labels.values[test_mask]

    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    logger.info(f"Train label dist: {Counter(y_train)}")
    logger.info(f"Val   label dist: {Counter(y_val)}")
    logger.info(f"Test  label dist: {Counter(y_test)}")

    # ── 7. Train CatBoost ─────────────────────────────────────────────────────
    cw = _class_weights(y_train)
    logger.info(f"Class weights: {[f'{w:.3f}' for w in cw]}")

    train_pool = Pool(X_train, y_train)
    val_pool   = Pool(X_val,   y_val)

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Weighted",
        class_weights=cw,
        early_stopping_rounds=args.early_stop,
        random_seed=42,
        verbose=100,
    )

    logger.info("Training CatBoost...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    y_val_pred  = model.predict(X_val).flatten().astype(int)
    y_test_pred = model.predict(X_test).flatten().astype(int)

    val_f1  = f1_score(y_val,  y_val_pred,  average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    val_acc  = accuracy_score(y_val,  y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    logger.info("=" * 60)
    logger.info(f"Val  – F1: {val_f1:.4f}  Acc: {val_acc:.4f}")
    logger.info(f"Test – F1: {test_f1:.4f}  Acc: {test_acc:.4f}")
    logger.info("\nTest classification report:")
    logger.info(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))
    logger.info("=" * 60)

    # Feature importance (top 20)
    n_emb = emb_train.shape[1]
    n_tab = X_tab.shape[1]
    feat_names = (
        [f"emb_pca_{i}" for i in range(n_emb)]
        + feat_cols
        + sent_cols
    )
    importances = model.get_feature_importance()
    top_idx = np.argsort(importances)[::-1][:20]
    logger.info("Top-20 feature importances:")
    for i in top_idx:
        name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        logger.info(f"  {name:45s} {importances[i]:.4f}")

    # ── 9. Save model ─────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CKPT_DIR / "catboost_ruonia.cbm"
    model.save_model(str(model_path))
    logger.info(f"Model saved → {model_path}")

    # Optionally save PCA
    if pca is not None:
        import pickle
        pca_path = CKPT_DIR / "pca.pkl"
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)
        logger.info(f"PCA saved → {pca_path}")

    # ── 10. MLflow logging ────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5050")
        mlflow.set_experiment("cbr_news_catboost")
        with mlflow.start_run(run_name="catboost_bert_tabular"):
            mlflow.log_params({
                "model":              "CatBoostClassifier",
                "embedding_model":    "DeepPavlov/rubert-base-cased",
                "pca_components":     n_pca,
                "pca_explained_var":  round(float(explained), 4),
                "n_tabular_features": X_tab.shape[1],
                "n_sentiment_features": X_sent.shape[1],
                "iterations":         args.iterations,
                "lr":                 args.lr,
                "depth":              args.depth,
                "train_years":        f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
                "val_years":          str(VAL_YEARS),
                "test_years":         str(TEST_YEARS),
            })
            mlflow.log_metrics({
                "val_f1":   val_f1,
                "test_f1":  test_f1,
                "val_acc":  val_acc,
                "test_acc": test_acc,
            })
            mlflow.log_artifact(str(model_path))
            if pca is not None:
                mlflow.log_artifact(str(CKPT_DIR / "pca.pkl"))
        logger.info("MLflow run logged.")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
