"""
LSTM sequence model for RUONIA direction prediction.

Architecture:
  Input  : sequence of K most recent news items before each news item
             each step = [BERT embedding (PCA-reduced) + tabular economic features]
  Model  : LSTM → last hidden state → Dropout → Linear → Softmax
  Target : RUONIA direction (down=0, same=1, up=2)

Temporal split: train=2010-2022, val=2023, test=2024+

Run:
    python cbr_news/ml/train_lstm.py [--seq-len 8] [--hidden 256] [--epochs 30]
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from cbr_news.ml.feature_engineering import build_tabular_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
CKPT_DIR = Path(__file__).parents[2] / "checkpoints" / "lstm"

LABEL_MAP   = {"down": 0, "same": 1, "up": 2}
CLASS_NAMES = ["down", "same", "up"]

TRAIN_YEARS = list(range(2010, 2023))
VAL_YEARS   = [2023]
TEST_YEARS  = [2024, 2025, 2026]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SequenceNewsDataset(Dataset):
    """
    Each item: the target news article at position `i` is predicted using
    a window of `seq_len` preceding news articles (including itself).

    X : float32 tensor (seq_len, input_dim)
    y : long tensor ()
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int):
        """
        features : (N, input_dim) array, sorted chronologically
        labels   : (N,) int array
        seq_len  : context window length (K)
        """
        self.features = features.astype(np.float32)
        self.labels   = labels.astype(np.int64)
        self.seq_len  = seq_len
        # Replace NaN with 0 (mean-impute would be better but needs train stats)
        self.features = np.nan_to_num(self.features, nan=0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = max(0, idx - self.seq_len + 1)
        window = self.features[start: idx + 1]          # (actual_len, D)
        # Left-pad with zeros if window is shorter than seq_len
        if len(window) < self.seq_len:
            pad = np.zeros((self.seq_len - len(window), window.shape[1]), dtype=np.float32)
            window = np.vstack([pad, window])
        x = torch.from_numpy(window)                    # (seq_len, D)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ─── Model ────────────────────────────────────────────────────────────────────

class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_classes: int, dropout: float, lr: float, class_weights):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        cw_tensor = torch.tensor(class_weights, dtype=torch.float)
        self.loss_fn = nn.CrossEntropyLoss(weight=cw_tensor)

        self._val_preds  = []
        self._val_labels = []
        self._test_preds  = []
        self._test_labels = []

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        out, _ = self.lstm(x)
        last    = out[:, -1, :]           # (B, hidden_dim)
        logits  = self.classifier(self.dropout(last))
        return logits

    def _step(self, batch, stage):
        x, y    = batch
        logits  = self(x)
        loss    = self.loss_fn(logits, y)
        preds   = logits.argmax(dim=1)
        acc     = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss, preds.cpu().numpy(), y.cpu().numpy()

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, "val")
        self._val_preds.append(preds)
        self._val_labels.append(labels)
        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            preds  = np.concatenate(self._val_preds)
            labels = np.concatenate(self._val_labels)
            f1 = f1_score(labels, preds, average="weighted", zero_division=0)
            self.log("val_f1", f1, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, "test")
        self._test_preds.append(preds)
        self._test_labels.append(labels)
        return loss

    def on_test_epoch_end(self):
        if self._test_preds:
            preds  = np.concatenate(self._test_preds)
            labels = np.concatenate(self._test_labels)
            f1  = f1_score(labels, preds, average="weighted", zero_division=0)
            acc = accuracy_score(labels, preds)
            self.log("test_f1",  f1)
            self.log("test_acc", acc)
            logger.info(f"\nTest F1:  {f1:.4f}  Acc: {acc:.4f}")
            logger.info(classification_report(labels, preds, target_names=CLASS_NAMES))
        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len",     type=int,   default=8,
                        help="Context window: K preceding news articles")
    parser.add_argument("--pca-components", type=int, default=64,
                        help="PCA components for BERT embeddings (0=no PCA)")
    parser.add_argument("--hidden",      type=int,   default=256)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--dataset",     type=str,
                        default=str(DATA_DIR / "cbr_combined_dataset.csv"))
    args = parser.parse_args()

    emb_path  = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "embeddings_meta.parquet"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Run compute_embeddings.py first.")

    # ── load data ─────────────────────────────────────────────────────────────
    logger.info("Loading embeddings and metadata...")
    embeddings = np.load(emb_path)
    meta       = pd.read_parquet(meta_path)

    df = pd.read_csv(args.dataset, usecols=["date", "target"])
    df = df.reset_index(drop=True)
    assert len(df) == len(meta)

    logger.info("Building tabular features...")
    df_feat = build_tabular_features(df, date_col="date")

    # ── labels ────────────────────────────────────────────────────────────────
    labels   = meta["target"].map(LABEL_MAP)
    valid    = labels.notna()
    embeddings = embeddings[valid.values]
    df_feat    = df_feat[valid].reset_index(drop=True)
    meta       = meta[valid].reset_index(drop=True)
    labels     = labels[valid].astype(int).reset_index(drop=True)

    # ── sort chronologically ──────────────────────────────────────────────────
    dates   = pd.to_datetime(meta["date"], format="%d.%m.%Y", errors="coerce")
    dates   = dates.fillna(pd.to_datetime(meta["date"], errors="coerce"))
    sort_idx = dates.argsort().values
    embeddings = embeddings[sort_idx]
    df_feat    = df_feat.iloc[sort_idx].reset_index(drop=True)
    meta       = meta.iloc[sort_idx].reset_index(drop=True)
    labels     = labels.iloc[sort_idx].reset_index(drop=True)
    dates      = dates.iloc[sort_idx].reset_index(drop=True)

    meta["year"] = dates.dt.year

    # ── PCA on embeddings ─────────────────────────────────────────────────────
    train_mask = meta["year"].isin(TRAIN_YEARS).values
    val_mask   = meta["year"].isin(VAL_YEARS).values
    test_mask  = meta["year"].isin(TEST_YEARS).values

    n_pca = args.pca_components
    if n_pca > 0:
        logger.info(f"Fitting PCA({n_pca}) on train embeddings...")
        pca = PCA(n_components=n_pca, random_state=42)
        emb_pca = np.zeros((len(embeddings), n_pca), dtype=np.float32)
        emb_pca[train_mask] = pca.fit_transform(embeddings[train_mask])
        emb_pca[~train_mask] = pca.transform(embeddings[~train_mask])
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        emb_pca = embeddings.astype(np.float32)

    # ── sentiment features ────────────────────────────────────────────────────
    sent_cols = [c for c in meta.columns if c.startswith("sentiment_")]
    X_sent    = meta[sent_cols].values.astype(np.float32) if sent_cols else np.zeros((len(meta), 0))

    # ── tabular features ──────────────────────────────────────────────────────
    feat_cols = get_feature_columns(df_feat)
    X_tab     = df_feat[feat_cols].values.astype(np.float32)

    # ── combine ───────────────────────────────────────────────────────────────
    X_all = np.hstack([emb_pca, X_tab, X_sent])
    # Mean-impute NaN on training set, apply same offset to val/test
    col_means = np.nanmean(X_all[train_mask], axis=0)
    col_means  = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask   = np.isnan(X_all)
    X_all      = np.where(nan_mask, col_means, X_all)

    y_all = labels.values

    logger.info(f"Feature matrix: {X_all.shape}")
    logger.info(f"Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}")

    input_dim = X_all.shape[1]

    # ── datasets ──────────────────────────────────────────────────────────────
    # For sequence model we keep chronological order within each split
    # but build windows using ALL preceding data (not just within split)
    # so we need indices into the full sorted array

    # Build sequences: train = rows 0..train_end, val starts after, test after
    train_indices = np.where(train_mask)[0]
    val_indices   = np.where(val_mask)[0]
    test_indices  = np.where(test_mask)[0]

    train_ds = SequenceNewsDataset(X_all[:train_indices[-1]+1], y_all[:train_indices[-1]+1], args.seq_len)
    val_ds   = SequenceNewsDataset(X_all[:val_indices[-1]+1],   y_all[:val_indices[-1]+1],   args.seq_len)
    test_ds  = SequenceNewsDataset(X_all[:test_indices[-1]+1],  y_all[:test_indices[-1]+1],  args.seq_len)

    # Subset to only the rows in each split
    from torch.utils.data import Subset
    train_ds  = Subset(train_ds, train_indices.tolist())
    val_subset_indices = val_indices.tolist()
    test_subset_indices = test_indices.tolist()

    # Re-build SequenceDataset for val/test using full array for context
    val_full  = SequenceNewsDataset(X_all, y_all, args.seq_len)
    test_full = SequenceNewsDataset(X_all, y_all, args.seq_len)
    val_ds   = Subset(val_full,  val_subset_indices)
    test_ds  = Subset(test_full, test_subset_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── class weights ─────────────────────────────────────────────────────────
    counts = Counter(y_all[train_mask])
    total  = sum(counts.values())
    n_cls  = len(LABEL_MAP)
    cw = [total / (n_cls * counts.get(i, 1)) for i in range(n_cls)]
    logger.info(f"Class weights: {[f'{w:.3f}' for w in cw]}")

    # ── model ─────────────────────────────────────────────────────────────────
    pl.seed_everything(42, workers=True)
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden,
        num_layers=args.num_layers,
        num_classes=n_cls,
        dropout=args.dropout,
        lr=args.lr,
        class_weights=cw,
    )
    logger.info(f"LSTM input_dim={input_dim}, hidden={args.hidden}, "
                f"layers={args.num_layers}, seq_len={args.seq_len}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="lstm-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(monitor="val_f1", patience=7, mode="max", verbose=True)

    try:
        mlflow_logger = MLFlowLogger(
            experiment_name="cbr_news_lstm",
            tracking_uri="http://127.0.0.1:5050",
            log_model=False,
        )
        use_mlflow = True
    except Exception:
        mlflow_logger = True   # use default TensorBoard logger
        use_mlflow = False

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger if use_mlflow else True,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    logger.info("Starting LSTM training...")
    trainer.fit(model, train_loader, val_loader)

    logger.info("Testing...")
    trainer.test(model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
