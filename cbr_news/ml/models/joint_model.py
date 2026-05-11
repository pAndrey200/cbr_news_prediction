"""
Joint BERT + Tabular model for RUONIA direction prediction.

Architecture:
  Text tower   : BERT (frozen except top 2 layers) → [CLS] → Dropout → Linear(768→128) → ReLU
  Tabular tower: BatchNorm(N) → Linear(N→64) → ReLU → Dropout
  Fusion       : concat(128+64) → Linear(192→64) → ReLU → Dropout → Linear(64→3)
"""

import logging
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModel, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

CLASS_NAMES = ["down", "same", "up"]

GRU_SERIES = ["ruonia", "usd", "eur", "brent"]
GRU_LAGS   = [30, 10, 5, 2, 1, 0]


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


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    When γ=0, reduces to weighted cross-entropy.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return (1.0 - pt) ** self.gamma * ce_loss


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
    """Two-layer MLP (GELU + skip connection)."""
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
        _, h   = self.gru(gru_in)
        h      = h.squeeze(0)
        rem    = self.rem_proj(self.bn_rem(x[:, self.rem_idx]))
        return torch.cat([h, rem], dim=1)


class JointBertTabularModel(pl.LightningModule):
    """
    Dual-tower model:
      BERT tower   : last N encoder layers trainable, rest frozen
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
        tab_arch: str = "mlp",
        gru_feat_idx: Optional[List[int]] = None,
        gru_n_steps: int = 6,
        gru_n_series: int = 4,
        gru_hidden: int = 32,
        pool_mode: str = "cls",
        warmstart_ckpt: str = "",
        lr_scheduler: str = "linear",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr               = lr
        self.weight_decay     = weight_decay
        self.warmup_steps     = warmup_steps
        self.aux_weight       = aux_weight
        self.lr_scheduler_type = lr_scheduler

        # ── Text tower ────────────────────────────────────────────────────────
        self.bert = AutoModel.from_pretrained(backbone)
        for param in self.bert.parameters():
            param.requires_grad = False
        total_layers = len(self.bert.encoder.layer)
        for layer in self.bert.encoder.layer[total_layers - n_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.bert.config.hidden_size

        self.text_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, text_proj_dim),
            nn.GELU(),
        )

        # ── Tabular tower ─────────────────────────────────────────────────────
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

        self.reg_head = nn.Linear(fusion_dim, 1)

        # ── Loss ──────────────────────────────────────────────────────────────
        cw = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        if focal_gamma > 0:
            self.loss_fn = FocalLoss(weight=cw, gamma=focal_gamma)
            logger.info("Using Focal Loss γ=%.1f  label_smoothing=%s", focal_gamma, label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=cw, label_smoothing=label_smoothing, reduction="none"
            )
            logger.info("Using weighted CE  label_smoothing=%s", label_smoothing)

        self._val_preds  = []; self._val_labels  = []
        self._test_preds = []; self._test_labels = []

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Joint model — trainable params: %d", n_trainable)

        if warmstart_ckpt:
            state = torch.load(warmstart_ckpt, map_location="cpu")
            sd = state.get("state_dict", state)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            if missing:
                logger.info("Warmstart — missing  keys: %d (re-initialised)", len(missing))
            if unexpected:
                logger.info("Warmstart — unexpected keys: %d (ignored)", len(unexpected))
            logger.info("Loaded warmstart weights from %s", warmstart_ckpt)

    def _encode(self, input_ids, attention_mask, tabular):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden   = bert_out.last_hidden_state

        if self.hparams.pool_mode == "mean":
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = hidden[:, 0, :]

        text_emb = self.text_proj(pooled)
        tab_emb  = self.tab_tower(tabular)
        return torch.cat([text_emb, tab_emb], dim=1)

    def forward(self, input_ids, attention_mask, tabular):
        return self.fusion(self._encode(input_ids, attention_mask, tabular))

    def _step(self, batch, stage):
        combined = self._encode(batch["input_ids"], batch["attention_mask"], batch["tabular"])
        logits   = self.fusion(combined)

        per_sample_loss = self.loss_fn(logits, batch["label"])
        if stage == "train" and "sample_weight" in batch:
            sw      = batch["sample_weight"].to(per_sample_loss.device)
            ce_loss = (per_sample_loss * sw).sum() / sw.sum()
        else:
            ce_loss = per_sample_loss.mean()

        loss = ce_loss

        if self.aux_weight > 0 and "ruonia_delta" in batch:
            delta_target = batch["ruonia_delta"].to(combined.device).float()
            delta_pred   = self.reg_head(combined).squeeze(1)
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
            p  = np.concatenate(self._val_preds)
            l  = np.concatenate(self._val_labels)
            f1w = f1_score(l, p, average="weighted", zero_division=0)
            f1m = f1_score(l, p, average="macro",    zero_division=0)
            self.log("val_f1",       f1w, prog_bar=True)
            self.log("val_f1_macro", f1m, prog_bar=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, "test")
        self._test_preds.append(preds)
        self._test_labels.append(labels)
        return loss

    def on_test_epoch_end(self):
        if self._test_preds:
            p  = np.concatenate(self._test_preds)
            l  = np.concatenate(self._test_labels)
            f1w = f1_score(l, p, average="weighted", zero_division=0)
            f1m = f1_score(l, p, average="macro",    zero_division=0)
            acc = accuracy_score(l, p)
            self.log("test_f1",       f1w)
            self.log("test_f1_macro", f1m)
            self.log("test_acc",      acc)
            logger.info("Test – Weighted F1: %.4f  Macro F1: %.4f  Acc: %.4f", f1w, f1m, acc)
            logger.info(classification_report(l, p, target_names=CLASS_NAMES, zero_division=0))
        self._test_preds.clear(); self._test_labels.clear()

    def configure_optimizers(self):
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

        if self.lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.lr_scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.lr * 0.1, self.lr],
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos",
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
