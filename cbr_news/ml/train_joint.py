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
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

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


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    When γ=0, reduces to weighted cross-entropy.
    With γ=2, examples where the model is already confident (p_t ≈ 1)
    contribute little to the loss; hard/misclassified examples dominate.
    This breaks the class-collapse pattern where the model predicts only
    the majority class ("down"/"same") and ignores "up".
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma
        # Underlying CE with class weights (reduction='none' for per-sample control)
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)              # (B,)
        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce_loss)                        # (B,)
        focal = (1.0 - pt) ** self.gamma * ce_loss      # (B,)
        return focal


# ─── GRU sequence helpers ─────────────────────────────────────────────────────

GRU_SERIES = ["ruonia", "usd", "eur", "brent"]
GRU_LAGS   = [30, 10, 5, 2, 1, 0]   # ordered oldest → newest; 0 = current value


def get_gru_feature_info(feat_cols: List[str]) -> Optional[dict]:
    """
    Locate the lag-structured column indices for the four main time series:
      ruonia / usd / eur / brent  at lags [t-30, t-10, t-5, t-2, t-1, t]

    Returns a dict with:
      gru_feat_idx : flat list of column indices (n_steps × n_series)
      n_steps      : number of complete time steps found
      n_series     : number of series (4)
    or None if the features are not present.
    """
    col_idx  = {c: i for i, c in enumerate(feat_cols)}
    n_series = len(GRU_SERIES)
    idx_seq  = []
    for lag in GRU_LAGS:
        step = []
        for s in GRU_SERIES:
            col = f"feat_{s}" if lag == 0 else f"feat_{s}_lag{lag}"
            if col in col_idx:
                step.append(col_idx[col])
        # Only include a time step if ALL series are present — a partial step
        # would corrupt the (B, n_steps, n_series) reshape in _TabGRU.forward().
        if len(step) == n_series:
            idx_seq.extend(step)
    n_steps = len(idx_seq) // n_series
    if n_steps == 0:
        return None
    return {
        "gru_feat_idx": idx_seq,
        "n_steps":      n_steps,
        "n_series":     n_series,
    }


# ─── Tabular towers ────────────────────────────────────────────────────────────

class _TabMLP(nn.Module):
    """Single-layer projection (fast, default)."""
    def __init__(self, n_in: int, n_out: int, dropout: float):
        super().__init__()
        self.bn  = nn.BatchNorm1d(n_in)
        self.net = nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout / 2))
        self.out_dim = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.bn(x))


class _TabDeepMLP(nn.Module):
    """Two-layer MLP (GELU + skip connection) — more capacity for complex feature interactions."""
    def __init__(self, n_in: int, n_out: int, dropout: float):
        super().__init__()
        hidden = n_out * 4
        self.bn   = nn.BatchNorm1d(n_in)
        self.fc1  = nn.Linear(n_in, hidden)
        self.fc2  = nn.Linear(hidden, n_out)
        self.skip = nn.Linear(n_in, n_out)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout / 2)
        self.out_dim = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        return self.fc2(self.drop(self.act(self.fc1(x)))) + self.skip(x)


class _TabGRU(nn.Module):
    """
    GRU over {ruonia, usd, eur, brent} at lags [t-30, t-10, t-5, t-2, t-1, t]
    + a shallow MLP for all remaining tabular features.

    Captures temporal momentum: consecutive RUONIA moves are autocorrelated,
    and a GRU can detect trend reversals that precede key rate decisions.
    """
    def __init__(
        self, n_in: int, n_out: int, dropout: float,
        gru_feat_idx: List[int], n_steps: int, n_series: int, gru_hidden: int = 32,
    ):
        super().__init__()
        self.register_buffer("gru_idx", torch.tensor(gru_feat_idx, dtype=torch.long))
        rem = sorted(set(range(n_in)) - set(gru_feat_idx))
        self.register_buffer("rem_idx", torch.tensor(rem, dtype=torch.long))
        self.n_steps  = n_steps
        self.n_series = n_series

        n_rem    = len(rem)
        rem_out  = max(n_out // 2, 16)
        self.bn_rem   = nn.BatchNorm1d(n_rem)
        self.rem_proj = nn.Sequential(
            nn.Linear(n_rem, rem_out), nn.GELU(), nn.Dropout(dropout / 2),
        )
        self.gru     = nn.GRU(n_series, gru_hidden, batch_first=True)
        self.out_dim = gru_hidden + rem_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_in = x[:, self.gru_idx].view(x.size(0), self.n_steps, self.n_series)
        _, h   = self.gru(gru_in)             # h: (1, B, gru_hidden)
        h      = h.squeeze(0)                  # (B, gru_hidden)
        rem    = self.rem_proj(self.bn_rem(x[:, self.rem_idx]))
        return torch.cat([h, rem], dim=1)      # (B, gru_hidden + rem_out)


# ─── Model ────────────────────────────────────────────────────────────────────

class JointBertTabularModel(pl.LightningModule):
    """
    Dual-tower model:
      BERT tower  : last 2 encoder layers trainable, rest frozen
      Tabular tower: learnable projection with BatchNorm
    Both towers fused into a shared classification head.
    """

    def __init__(
        self,
        n_tabular: int,
        num_classes: int = 3,
        text_proj_dim: int = 128,
        tab_proj_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.4,
        lr: float = 2e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 200,
        class_weights: Optional[List[float]] = None,
        n_trainable_layers: int = 2,
        focal_gamma: float = 0.0,
        label_smoothing: float = 0.0,
        aux_weight: float = 0.0,
        backbone: str = "DeepPavlov/rubert-base-cased",
        # Tabular tower architecture:
        #   "mlp"      — single Linear (fast, default)
        #   "mlp_deep" — 2-layer MLP with skip connection (more capacity)
        #   "gru"      — GRU over ruonia/usd/eur/brent lag series + MLP for rest
        tab_arch: str = "mlp",
        gru_feat_idx: Optional[List[int]] = None,  # column indices for GRU sequence
        gru_n_steps: int = 6,
        gru_n_series: int = 4,
        gru_hidden: int = 32,
        # "cls"  — use [CLS] token (default, standard BERT usage)
        # "mean" — mean-pool all non-padding token vectors (often +1-2% F1)
        pool_mode: str = "cls",
        # Optional path to a Stage-1 checkpoint; weights are loaded after init
        # so you can change n_trainable_layers / lr / etc. for Stage-2 fine-tuning.
        warmstart_ckpt: str = "",
        # LR scheduler: "linear" (default, standard BERT), "cosine" (smoother decay),
        # "onecycle" (super-convergence: warmup → peak → annealing)
        lr_scheduler: str = "linear",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr           = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.aux_weight   = aux_weight
        self.lr_scheduler_type = lr_scheduler

        # ── Text tower ────────────────────────────────────────────────────────
        self.bert = AutoModel.from_pretrained(backbone)
        # Freeze all backbone parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        # Unfreeze top N encoder layers
        total_layers = len(self.bert.encoder.layer)
        for layer in self.bert.encoder.layer[total_layers - n_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.bert.config.hidden_size  # 768

        # LayerNorm stabilises the CLS/mean vector before projection;
        # GELU matches the activation used inside BERT itself.
        self.text_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, text_proj_dim),
            nn.GELU(),
        )

        # ── Tabular tower (selectable architecture) ───────────────────────────
        if tab_arch == "mlp":
            self.tab_tower = _TabMLP(n_tabular, tab_proj_dim, dropout)
        elif tab_arch == "mlp_deep":
            self.tab_tower = _TabDeepMLP(n_tabular, tab_proj_dim, dropout)
        elif tab_arch == "gru":
            if gru_feat_idx is None:
                logger.warning("tab_arch='gru' but gru_feat_idx not provided — falling back to mlp")
                self.tab_tower = _TabMLP(n_tabular, tab_proj_dim, dropout)
            else:
                self.tab_tower = _TabGRU(
                    n_tabular, tab_proj_dim, dropout,
                    gru_feat_idx, gru_n_steps, gru_n_series, gru_hidden,
                )
        else:
            raise ValueError(f"Unknown tab_arch '{tab_arch}'. Choose: mlp | mlp_deep | gru")

        # ── Fusion head ───────────────────────────────────────────────────────
        fusion_dim = text_proj_dim + self.tab_tower.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        # ── Auxiliary regression head: predict RUONIA delta ───────────────────
        # Shares gradients with both towers → forces encoder to learn magnitude.
        self.reg_head = nn.Linear(fusion_dim, 1)

        # ── Loss ──────────────────────────────────────────────────────────────
        cw = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        if focal_gamma > 0:
            self.loss_fn = FocalLoss(weight=cw, gamma=focal_gamma)
            logger.info(f"Using Focal Loss γ={focal_gamma:.1f}  label_smoothing={label_smoothing}")
        else:
            # label_smoothing replaces hard [1,0,0] with soft [1-ε, ε/2, ε/2].
            # Helps when most training articles have noisy (coincidental) labels.
            self.loss_fn = nn.CrossEntropyLoss(
                weight=cw, label_smoothing=label_smoothing, reduction="none"
            )
            logger.info(f"Using weighted CE  label_smoothing={label_smoothing}")

        # metric accumulators
        self._val_preds  = []; self._val_labels  = []
        self._test_preds = []; self._test_labels = []

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Joint model — trainable params: {n_trainable:,}")

        # ── Warmstart (Stage-2 fine-tuning) ───────────────────────────────────
        # Load weights from a Stage-1 checkpoint. Uses strict=False so any
        # extra/missing keys (e.g. from architecture changes) are silently skipped.
        if warmstart_ckpt:
            state = torch.load(warmstart_ckpt, map_location="cpu")
            # Lightning checkpoints store weights under "state_dict" key
            sd = state.get("state_dict", state)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            if missing:
                logger.info("Warmstart — missing  keys: %d (re-initialised)", len(missing))
            if unexpected:
                logger.info("Warmstart — unexpected keys: %d (ignored)", len(unexpected))
            logger.info("Loaded warmstart weights from %s", warmstart_ckpt)

    def _encode(self, input_ids, attention_mask, tabular):
        """Shared forward: both towers → fused embedding (B, fusion_dim)."""
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden   = bert_out.last_hidden_state              # (B, L, hidden)

        if self.hparams.pool_mode == "mean":
            # Mean-pool over non-padding tokens — often outperforms CLS alone
            mask     = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
            pooled   = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            # Default: [CLS] token at position 0
            pooled   = hidden[:, 0, :]                               # (B, hidden)

        text_emb = self.text_proj(pooled)                 # (B, text_proj_dim)
        tab_emb  = self.tab_tower(tabular)                # (B, tab_tower.out_dim)
        return torch.cat([text_emb, tab_emb], dim=1)      # (B, fusion_dim)

    def forward(self, input_ids, attention_mask, tabular):
        return self.fusion(self._encode(input_ids, attention_mask, tabular))

    def _step(self, batch, stage):
        combined = self._encode(batch["input_ids"], batch["attention_mask"], batch["tabular"])
        logits   = self.fusion(combined)

        # ── Classification loss ───────────────────────────────────────────────
        per_sample_loss = self.loss_fn(logits, batch["label"])   # (B,)
        if stage == "train" and "sample_weight" in batch:
            sw      = batch["sample_weight"].to(per_sample_loss.device)
            ce_loss = (per_sample_loss * sw).sum() / sw.sum()
        else:
            ce_loss = per_sample_loss.mean()

        loss = ce_loss

        # ── Auxiliary regression loss: RUONIA delta magnitude ─────────────────
        if self.aux_weight > 0 and "ruonia_delta" in batch:
            delta_target = batch["ruonia_delta"].to(combined.device).float()  # (B,)
            delta_pred   = self.reg_head(combined).squeeze(1)                 # (B,)
            aux_loss     = torch.nn.functional.mse_loss(delta_pred, delta_target)
            loss         = loss + self.aux_weight * aux_loss
            self.log(f"{stage}_aux_loss", aux_loss, prog_bar=False, on_step=False, on_epoch=True)

        preds = logits.argmax(dim=1)
        acc   = (preds == batch["label"]).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss, preds.cpu().numpy(), batch["label"].cpu().numpy()

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
            p = np.concatenate(self._val_preds)
            l = np.concatenate(self._val_labels)
            f1w = f1_score(l, p, average="weighted", zero_division=0)
            f1m = f1_score(l, p, average="macro",    zero_division=0)
            self.log("val_f1",        f1w, prog_bar=True)
            self.log("val_f1_macro",  f1m, prog_bar=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, "test")
        self._test_preds.append(preds)
        self._test_labels.append(labels)
        return loss

    def on_test_epoch_end(self):
        if self._test_preds:
            p = np.concatenate(self._test_preds)
            l = np.concatenate(self._test_labels)
            f1w = f1_score(l, p, average="weighted", zero_division=0)
            f1m = f1_score(l, p, average="macro",    zero_division=0)
            acc = accuracy_score(l, p)
            self.log("test_f1",       f1w)
            self.log("test_f1_macro", f1m)
            self.log("test_acc",      acc)
            logger.info(f"\nTest – Weighted F1: {f1w:.4f}  Macro F1: {f1m:.4f}  Acc: {acc:.4f}")
            logger.info(classification_report(l, p, target_names=CLASS_NAMES, zero_division=0))
        self._test_preds.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        # Separate LR for backbone (lower) and head (higher)
        backbone_params = [p for p in self.bert.parameters() if p.requires_grad]
        head_params     = (
            list(self.text_proj.parameters())
            + list(self.tab_tower.parameters())
            + list(self.fusion.parameters())
            + list(self.reg_head.parameters())
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr * 0.1},
                {"params": head_params,     "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        total_steps = (
            self.trainer.estimated_stepping_batches if self.trainer else 10_000
        )

        sched_type = self.lr_scheduler_type
        if sched_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        elif sched_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.lr * 0.1, self.lr],   # per param-group
                total_steps=total_steps,
                pct_start=0.1,                      # 10% warmup
                anneal_strategy="cos",
            )
        else:  # "linear" (default)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


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
