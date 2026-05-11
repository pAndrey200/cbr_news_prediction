"""
Joint BERT + Tabular model for RUONIA direction prediction.

Architecture:
  Text tower   : BERT (frozen except top 2 layers) → [CLS] → Dropout → Linear(768→128) → ReLU
  Tabular tower: BatchNorm(N) → Linear(N→64) → ReLU → Dropout
  Fusion       : concat(128+64) → Linear(192→64) → ReLU → Dropout → Linear(64→3)

Why this works better than separate CatBoost on embeddings:
  - BERT is forced to learn task-relevant text representation (gradients flow through text tower)
  - Tabular features provide stable economic context that stabilises training
  - Joint training prevents BERT from being ignored (unlike CatBoost which dropped emb to 0)
  - Strong regularisation: frozen backbone + high dropout + weight decay

Run:
    # Precompute tabular features first (one-time, ~3 min):
    python cbr_news/ml/train_joint.py --precompute-only

    # Then train:
    python cbr_news/ml/train_joint.py
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cbr_news.ml.feature_engineering import (
    build_tabular_features,
    extract_text_features,
    get_feature_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
CKPT_DIR = Path(__file__).parents[2] / "checkpoints" / "joint"

LABEL_MAP   = {"down": 0, "same": 1, "up": 2}
CLASS_NAMES = ["down", "same", "up"]
TOKENIZER   = "DeepPavlov/rubert-base-cased"

TRAIN_YEARS = list(range(2010, 2024))
VAL_YEARS   = [2024]
TEST_YEARS  = [2025, 2026]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class JointDataset(Dataset):
    """
    Each item:
      input_ids       : (max_length,) long
      attention_mask  : (max_length,) long
      tabular         : (n_tab_features,) float32
      label           : () long
      sample_weight   : () float32  – higher for CBR press releases
    """

    def __init__(
        self,
        texts: List[str],
        tabular: np.ndarray,
        labels: List[int],
        tokenizer_name: str,
        max_length: int = 256,
        sample_weights: Optional[np.ndarray] = None,
        ruonia_deltas: Optional[np.ndarray] = None,
    ):
        self.texts          = texts
        self.tabular        = tabular.astype(np.float32)
        self.labels         = labels
        self.max_length     = max_length
        self.tokenizer      = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sample_weights = (
            sample_weights.astype(np.float32)
            if sample_weights is not None
            else np.ones(len(labels), dtype=np.float32)
        )
        self.ruonia_deltas = (
            ruonia_deltas.astype(np.float32) if ruonia_deltas is not None else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "tabular":        torch.from_numpy(self.tabular[idx]),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
            "sample_weight":  torch.tensor(self.sample_weights[idx], dtype=torch.float),
        }
        if self.ruonia_deltas is not None:
            item["ruonia_delta"] = torch.tensor(float(self.ruonia_deltas[idx]), dtype=torch.float)
        return item


from cbr_news.ml.models.joint_model import (
    FocalLoss,
    JointBertTabularModel,
    _TabDeepMLP,
    _TabGRU,
    _TabMLP,
    get_gru_feature_info,
)

# ─── DataModule ───────────────────────────────────────────────────────────────

class JointDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._tab_cache = DATA_DIR / "tabular_features.parquet"

    def _load_or_compute_tabular(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if self._tab_cache.exists():
            logger.info(f"Loading cached tabular features from {self._tab_cache}")
            df_feat = pd.read_parquet(self._tab_cache)
            if len(df_feat) == len(df_raw):
                return df_feat
            logger.warning("Cache size mismatch — recomputing tabular features")

        logger.info("Computing tabular features (this takes ~3-5 min for 100k rows)...")
        df_feat = build_tabular_features(df_raw, date_col="date")
        if "cleaned_text" in df_raw.columns:
            df_feat = extract_text_features(df_feat, text_col="cleaned_text")
        df_feat.to_parquet(self._tab_cache, index=False)
        logger.info(f"Tabular features cached → {self._tab_cache}")
        return df_feat

    def setup(self, stage=None):
        cfg = self.config

        logger.info(f"Loading dataset: {cfg.dataset_path}")
        df_raw = pd.read_csv(cfg.dataset_path)
        df_raw = df_raw.reset_index(drop=True)
        logger.info(f"Rows: {len(df_raw)}")

        # Tabular features
        df_feat = self._load_or_compute_tabular(df_raw)

        # ── is_cbr_source: 1 for key rate press releases, 0 for everything else ──
        # cbr_weight focuses training gradient on the ~100 key rate decision
        # press releases (one per CBR Board of Directors meeting).
        # Priority order:
        #   1. is_key_rate_pr column  — precise: CBR article on meeting date
        #   2. source="cbr_press_releases" — broad fallback (all CBR press)
        #   3. news_type in {"press","events"} — DB format fallback
        if "is_key_rate_pr" in df_raw.columns:
            is_cbr = df_raw["is_key_rate_pr"].fillna(0).astype(float).values
        elif "source" in df_raw.columns:
            is_cbr = (df_raw["source"] == "cbr_press_releases").astype(float).values
        elif "news_type" in df_raw.columns:
            # "press_release" — exact DB type for CBR rate-decision press releases
            is_cbr = (df_raw["news_type"] == "press_release").astype(float).values
        else:
            is_cbr = np.zeros(len(df_raw))

        # Append is_cbr_source to tabular matrix
        feat_cols = get_feature_columns(df_feat)
        X_tab_base = df_feat[feat_cols].values.astype(np.float32)
        X_tab = np.hstack([X_tab_base, is_cbr.reshape(-1, 1)])
        feat_cols = feat_cols + ["is_cbr_source"]
        self.feat_cols = feat_cols

        # ── GRU sequence feature info (used when tab_arch="gru") ─────────────
        gru_info = get_gru_feature_info(feat_cols)
        if gru_info:
            self.gru_feat_idx = gru_info["gru_feat_idx"]
            self.gru_n_steps  = gru_info["n_steps"]
            self.gru_n_series = gru_info["n_series"]
            logger.info(
                f"GRU sequence: {self.gru_n_steps} steps × {self.gru_n_series} series "
                f"= {len(self.gru_feat_idx)} features"
            )
        else:
            self.gru_feat_idx = None
            self.gru_n_steps  = 0
            self.gru_n_series = 0

        # Labels
        labels   = df_raw["target"].map(LABEL_MAP)
        valid    = labels.notna()
        df_raw   = df_raw[valid].reset_index(drop=True)
        X_tab    = X_tab[valid.values]
        is_cbr   = is_cbr[valid.values]
        labels   = labels[valid].astype(int).reset_index(drop=True)
        texts    = df_raw["cleaned_text"].fillna("").tolist()

        # ── RUONIA delta (regression target for auxiliary head) ───────────────
        if "ruonia_delta" in df_raw.columns:
            ruonia_deltas = df_raw["ruonia_delta"].values.astype(np.float32)
        else:
            ruonia_deltas = None

        # ── Sample weights: CBR press releases weighted higher ────────────────
        # Gives more gradient to the articles that actually affect RUONIA.
        cbr_weight      = getattr(cfg, "cbr_sample_weight", 5.0)
        sample_weights  = np.where(is_cbr == 1, cbr_weight, 1.0).astype(np.float32)
        logger.info(f"Sample weights: CBR={cbr_weight:.1f}x, general=1.0  "
                    f"(CBR rows: {int(is_cbr.sum()):,} / {len(is_cbr):,})")

        # ── Temporal split ────────────────────────────────────────────────────
        def _parse_date(d):
            try:
                p = str(d).split(".")
                if len(p) == 3:
                    return pd.Timestamp(f"{p[2]}-{p[1]}-{p[0]}")
                return pd.to_datetime(d)
            except Exception:
                return pd.Timestamp("2020-01-01")

        dates     = df_raw["date"].apply(_parse_date)
        t_end     = pd.Timestamp(getattr(cfg, "train_end", "2024-06-30"))
        val_end   = pd.Timestamp(getattr(cfg, "val_end",   "2024-12-31"))
        train_mask = (dates <= t_end).values
        val_mask   = ((dates > t_end) & (dates <= val_end)).values
        test_mask  = (dates > val_end).values

        logger.info(f"Train: {train_mask.sum():>6d}  Val: {val_mask.sum():>6d}  "
                    f"Test: {test_mask.sum():>6d}")

        # ── Mean-impute NaN using training statistics ─────────────────────────
        col_means = np.nanmean(X_tab[train_mask], axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_idx   = np.where(np.isnan(X_tab))
        X_tab[nan_idx] = np.take(col_means, nan_idx[1])
        self.col_means = col_means

        # ── Normalise RUONIA delta to unit variance (train std) ───────────────
        if ruonia_deltas is not None:
            delta_std = float(np.std(ruonia_deltas[train_mask]))
            delta_std = max(delta_std, 1e-6)
            ruonia_deltas = ruonia_deltas / delta_std
            logger.info(f"ruonia_delta normalised  std={delta_std:.4f}")

        # ── Class weights (balanced) ──────────────────────────────────────────
        y_train = labels.values[train_mask]
        from collections import Counter
        counts = Counter(y_train)
        total  = sum(counts.values())
        n_cls  = len(LABEL_MAP)
        self.class_weights = [total / (n_cls * counts.get(i, 1)) for i in range(n_cls)]
        self.n_tabular     = X_tab.shape[1]

        logger.info(f"n_tabular features : {self.n_tabular}")
        logger.info(f"class_weights      : {[f'{w:.3f}' for w in self.class_weights]}")
        logger.info(f"Train label dist   : {Counter(y_train)}")

        def _make(mask, is_train=False):
            t  = [t for t, m in zip(texts, mask) if m]
            x  = X_tab[mask]
            l  = labels.values[mask].tolist()
            sw = sample_weights[mask] if is_train else None
            rd = ruonia_deltas[mask] if ruonia_deltas is not None else None

            # ── Oversample "up" (class 2) in training ─────────────────────────
            n_up = getattr(cfg, "oversample_up", 1)
            if is_train and n_up > 1:
                up_idx = [i for i, lbl in enumerate(l) if lbl == LABEL_MAP["up"]]
                for _ in range(n_up - 1):
                    t  = t  + [t[i]  for i in up_idx]
                    x  = np.vstack([x,  x[up_idx]])
                    l  = l  + [l[i]  for i in up_idx]
                    if sw is not None:
                        sw = np.concatenate([sw, sw[up_idx]])
                    if rd is not None:
                        rd = np.concatenate([rd, rd[up_idx]])
                logger.info(
                    "Oversampled 'up' ×%d: %d → %d training examples",
                    n_up, train_mask.sum(), len(l),
                )

            return JointDataset(
                texts          = t,
                tabular        = x,
                labels         = l,
                tokenizer_name = cfg.tokenizer_name,
                max_length     = cfg.max_length,
                sample_weights = sw,
                ruonia_deltas  = rd,
            )

        self.train_ds = _make(train_mask, is_train=True)
        self.val_ds   = _make(val_mask)
        self.test_ds  = _make(test_mask)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)


# ─── Config ───────────────────────────────────────────────────────────────────

class Config:
    dataset_path     : str  = str(DATA_DIR / "cbr_combined_dataset.csv")
    tokenizer_name   : str  = TOKENIZER
    max_length       : int  = 256
    batch_size       : int  = 64  # increased from 32; use fp16 to keep VRAM ~same
    # Model
    text_proj_dim    : int  = 128
    tab_proj_dim     : int  = 64
    fusion_hidden_dim: int  = 64
    aux_weight       : float = 0.0   # weight for auxiliary RUONIA-delta regression head
    dropout          : float = 0.4
    lr               : float = 2e-5
    weight_decay     : float = 0.1
    warmup_steps     : int  = 200
    n_trainable_layers: int = 2          # how many BERT encoder layers to unfreeze
    focal_gamma      : float = 0.0       # 0 = weighted CE; 2 = standard focal loss
    label_smoothing  : float = 0.0       # 0 = hard labels; 0.1 = recommended for noisy data
    oversample_up    : int   = 1         # repeat "up" training examples N times (1=off)
    pool_mode        : str   = "cls"     # "cls" | "mean" — how to pool BERT output
    # Training
    max_epochs       : int  = 20
    patience         : int  = 5
    # Source weighting: how many times more important is a CBR press release
    cbr_sample_weight: float = 5.0
    train_end        : str  = "2024-06-30"  # last date in training set
    val_end          : str  = "2024-12-31"  # last date in val; test = everything after


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute-only", action="store_true",
                        help="Only precompute tabular features then exit")
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--n-trainable",  type=int,   default=2,
                        help="Number of BERT encoder layers to unfreeze (top N)")
    parser.add_argument("--dropout",         type=float, default=0.4)
    parser.add_argument("--cbr-weight",      type=float, default=5.0,
                        help="Sample weight multiplier for CBR press releases (default 5)")
    parser.add_argument("--focal-gamma",      type=float, default=0.0,
                        help="Focal loss γ (0=plain CE, 2=standard focal). Fixes class-collapse.")
    parser.add_argument("--label-smoothing",  type=float, default=0.0,
                        help="Label smoothing ε (0=hard labels, 0.1=recommended for noisy data).")
    parser.add_argument("--oversample-up",    type=int,   default=1,
                        help="Repeat 'up' training examples N times (1=off, 2=double, 3=triple).")
    parser.add_argument("--pool-mode",        type=str,   default="cls",
                        choices=["cls", "mean"],
                        help="BERT pooling: 'cls' (default) or 'mean' (mean of non-pad tokens).")
    parser.add_argument("--dataset",         type=str,
                        default=str(DATA_DIR / "cbr_combined_dataset.csv"))
    args = parser.parse_args()

    cfg = Config()
    cfg.dataset_path       = args.dataset
    cfg.batch_size         = args.batch_size
    cfg.lr                 = args.lr
    cfg.max_epochs         = args.epochs
    cfg.patience           = args.patience
    cfg.n_trainable_layers = args.n_trainable
    cfg.dropout            = args.dropout
    cfg.cbr_sample_weight  = args.cbr_weight
    cfg.focal_gamma        = args.focal_gamma
    cfg.label_smoothing    = args.label_smoothing
    cfg.oversample_up      = args.oversample_up
    cfg.pool_mode          = args.pool_mode

    # ── Precompute tabular features if requested ───────────────────────────────
    if args.precompute_only:
        logger.info("Precomputing tabular features only...")
        df_raw = pd.read_csv(cfg.dataset_path)
        df_feat = build_tabular_features(df_raw, date_col="date")
        if "cleaned_text" in df_raw.columns:
            df_feat = extract_text_features(df_feat, text_col="cleaned_text")
        out = DATA_DIR / "tabular_features.parquet"
        df_feat.to_parquet(out, index=False)
        logger.info(f"Saved → {out}")
        return

    # ── Setup data ────────────────────────────────────────────────────────────
    pl.seed_everything(42, workers=True)
    dm = JointDataModule(cfg)
    dm.setup()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = JointBertTabularModel(
        n_tabular          = dm.n_tabular,
        num_classes        = len(LABEL_MAP),
        text_proj_dim      = cfg.text_proj_dim,
        tab_proj_dim       = cfg.tab_proj_dim,
        fusion_hidden_dim  = cfg.fusion_hidden_dim,
        dropout            = cfg.dropout,
        lr                 = cfg.lr,
        weight_decay       = cfg.weight_decay,
        warmup_steps       = cfg.warmup_steps,
        class_weights      = dm.class_weights,
        n_trainable_layers = cfg.n_trainable_layers,
        focal_gamma        = cfg.focal_gamma,
        label_smoothing    = cfg.label_smoothing,
        aux_weight         = cfg.aux_weight,
        backbone           = cfg.tokenizer_name,
        pool_mode          = cfg.pool_mode,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath    = str(CKPT_DIR),
        filename   = "joint-{epoch:02d}-{val_f1:.4f}",
        monitor    = "val_f1",
        mode       = "max",
        save_top_k = 1,
    )
    early_stop_cb = EarlyStopping(
        monitor  = "val_f1",
        patience = cfg.patience,
        mode     = "max",
        verbose  = True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── Logger ────────────────────────────────────────────────────────────────
    try:
        mlflow_logger = MLFlowLogger(
            experiment_name = "cbr_news_joint",
            tracking_uri    = "http://127.0.0.1:5050",
            log_model       = False,
        )
        mlflow_logger.log_hyperparams({
            "backbone":          cfg.tokenizer_name,
            "n_trainable_layers": cfg.n_trainable_layers,
            "n_tabular":         dm.n_tabular,
            "batch_size":        cfg.batch_size,
            "lr":                cfg.lr,
            "dropout":           cfg.dropout,
            "max_epochs":        cfg.max_epochs,
        })
        use_mlflow = True
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")
        mlflow_logger = True
        use_mlflow = False

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs       = cfg.max_epochs,
        accelerator      = "auto",
        devices          = 1,
        precision        = "16-mixed" if torch.cuda.is_available() else "32-true",
        logger           = mlflow_logger if use_mlflow else True,
        callbacks        = [checkpoint_cb, early_stop_cb, lr_monitor],
        gradient_clip_val= 1.0,
        log_every_n_steps= 50,
    )

    logger.info("Training joint BERT + tabular model...")
    trainer.fit(model, dm)

    logger.info("Testing on best checkpoint...")
    trainer.test(model, dm, ckpt_path="best")


if __name__ == "__main__":
    main()
