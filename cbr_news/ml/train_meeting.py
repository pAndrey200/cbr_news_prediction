"""
Meeting-centric model: predict key rate direction (hike / hold / cut) from
articles published in the N days before each CBR Board of Directors meeting.

Labels  : cut=0  hold=1  hike=2  (3-class)
Dataset : prepare_meeting_dataset.py  →  data/meeting_dataset.parquet
Split   : train 2014–2021 (~60 meetings)  val 2022–2023 (~19)  test 2024–2025 (~16)

Overfitting prevention (small-dataset regime ~60 train examples):
  • Default: BERT fully frozen (n_trainable=0), article CLS vectors pre-computed once
    per setup() call — training only touches the attention-pool + MLP head.
    Set --n-trainable > 0 to fine-tune top BERT layers (slower, higher overfit risk).
  • Strong regularisation defaults: dropout=0.4  weight_decay=0.1  label_smoothing=0.15
  • Class-balanced loss weights (cut/hold/hike may be imbalanced)

Usage:
    # 1. Build meeting dataset (once):
    PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py

    # 2. Train with frozen BERT (recommended for small dataset):
    PYTHONPATH=. python cbr_news/ml/train_meeting.py

    # 3. Fine-tune top BERT layers (optional):
    PYTHONPATH=. python cbr_news/ml/train_meeting.py --n-trainable 2 --lr 1e-5

    # 4. Tabular-only baseline:
    PYTHONPATH=. python cbr_news/ml/train_meeting.py --no-text

    # 5. Multi-seed evaluation:
    PYTHONPATH=. python cbr_news/ml/train_meeting.py --seeds 5
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parents[2] / "data"
CKPT_DIR   = Path(__file__).parents[2] / "checkpoints" / "meeting"
MEETING_DS = DATA_DIR / "meeting_dataset.parquet"
TOKENIZER  = "DeepPavlov/rubert-base-cased"

LABEL_MAP   = {"cut": 0, "hold": 1, "hike": 2}
CLASS_NAMES = ["cut", "hold", "hike"]
NUM_CLASSES = 3

# Temporal split:
#   Train: 2014–2022  (76 meetings — includes 2022 shock for diversity)
#   Val  : 2023       ( 9 meetings — gradual hike cycle 7.5%→16%, representative of test)
#   Test : 2024–2025  (16 meetings — hike to 21%, hold/cut cycle)
#
# Note: 2022 moved to train (not val) because the 2022 extraordinary shock meetings
# (war, +10.5pp, 3 emergency sessions) are anomalous and make val unrepresentative
# of the 2024-2025 test regime.
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END   = pd.Timestamp("2023-12-31")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class MeetingDataset(Dataset):
    """
    Two modes depending on whether BERT embeddings are pre-computed:

    Mode A — cached_cls provided (n_trainable == 0):
      Returns: "cls" (A, H), "article_mask" (A,), "tabular", "label"
      Training only runs: attention-pool → text_proj → fusion → classifier

    Mode B — tokenise on-the-fly (n_trainable > 0):
      Returns: "input_ids" (A, L), "attention_mask" (A, L), "article_mask" (A,), "tabular", "label"
      Training runs full BERT each forward pass.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cached_cls: Optional[np.ndarray]    = None,  # (N, A, H)
        article_masks: Optional[np.ndarray] = None,  # (N, A)
        tokenizer_name: str  = TOKENIZER,
        max_length: int      = 128,
        max_articles: int    = 20,
        use_text: bool       = True,
    ):
        self.df            = df.reset_index(drop=True)
        self.cached_cls    = cached_cls
        self.article_masks = article_masks
        self.max_length    = max_length
        self.max_articles  = max_articles
        self.use_text      = use_text

        if use_text and cached_cls is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        label   = torch.tensor(int(row["label"]), dtype=torch.long)
        tabular = torch.from_numpy(np.array(row["tabular"], dtype=np.float32))

        # ── Mode A: pre-computed embeddings ───────────────────────────────────
        if self.cached_cls is not None:
            return {
                "cls":          torch.from_numpy(self.cached_cls[idx].copy()),
                "article_mask": torch.from_numpy(self.article_masks[idx].astype(np.float32)),
                "tabular":      tabular,
                "label":        label,
            }

        # ── Mode B (tabular-only): dummy tensors ──────────────────────────────
        if not self.use_text:
            return {
                "input_ids":      torch.zeros(self.max_articles, self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_articles, self.max_length, dtype=torch.long),
                "article_mask":   torch.zeros(self.max_articles, dtype=torch.float),
                "tabular":        tabular,
                "label":          label,
            }

        # ── Mode B (fine-tune): tokenise on-the-fly ───────────────────────────
        texts  = list(row["texts"])[:self.max_articles]
        n_real = len(texts)
        while len(texts) < self.max_articles:
            texts.append("")

        input_ids_list, attn_mask_list = [], []
        for text in texts:
            enc = self.tokenizer(
                str(text),
                truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_mask_list.append(enc["attention_mask"].squeeze(0))

        article_mask = torch.zeros(self.max_articles, dtype=torch.float)
        article_mask[:n_real] = 1.0

        return {
            "input_ids":      torch.stack(input_ids_list),   # (A, L)
            "attention_mask": torch.stack(attn_mask_list),   # (A, L)
            "article_mask":   article_mask,                   # (A,)
            "tabular":        tabular,
            "label":          label,
        }


# ─── Model ────────────────────────────────────────────────────────────────────

class MeetingModel(pl.LightningModule):
    """
    Attention-pooled article representations + tabular features → 3-class classifier.

    Text tower (use_cached_emb=True, default):
      cached CLS (B, A, H) → learned attention → pooled (B, H) → text_proj (B, D_t)

    Text tower (use_cached_emb=False, fine-tune mode):
      input_ids (B, A, L) → BERT → CLS (B, A, H) → same attention pool

    Tabular tower:
      LayerNorm → Linear(N → D_tab) → GELU → Dropout

    Fusion:
      concat(D_t, D_tab) → Linear(fusion_hidden) → GELU → Dropout → Linear(3)
    """

    def __init__(
        self,
        n_tabular: int,
        max_articles: int       = 20,
        num_classes: int        = NUM_CLASSES,
        text_proj_dim: int      = 64,
        tab_proj_dim: int       = 32,
        fusion_hidden_dim: int  = 64,
        dropout: float          = 0.4,
        lr: float               = 1e-3,
        weight_decay: float     = 0.1,
        warmup_steps: int       = 10,
        class_weights: Optional[List[float]] = None,
        n_trainable_layers: int = 0,
        label_smoothing: float  = 0.15,
        use_text: bool          = True,
        use_cached_emb: bool    = True,
        emb_dim: int            = 768,
        backbone: str           = TOKENIZER,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.warmup_steps    = warmup_steps
        self.max_articles    = max_articles
        self.use_text        = use_text
        self.use_cached_emb  = use_cached_emb

        # ── Text tower ────────────────────────────────────────────────────────
        if use_text:
            if not use_cached_emb:
                self.bert = AutoModel.from_pretrained(backbone)
                for p in self.bert.parameters():
                    p.requires_grad = False
                total_layers = len(self.bert.encoder.layer)
                for layer in self.bert.encoder.layer[total_layers - n_trainable_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                emb_dim = self.bert.config.hidden_size

            # Learned attention over articles
            self.article_attn = nn.Linear(emb_dim, 1)
            self.text_proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(emb_dim, text_proj_dim),
                nn.GELU(),
            )
        else:
            text_proj_dim = 0

        # ── Tabular tower ─────────────────────────────────────────────────────
        self.tab_norm = nn.LayerNorm(n_tabular)
        self.tab_proj = nn.Sequential(
            nn.Linear(n_tabular, tab_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        fusion_in = text_proj_dim + tab_proj_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        cw = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(
            weight=cw, label_smoothing=label_smoothing, reduction="mean"
        )

        self._val_preds  : list = []
        self._val_labels : list = []
        self._test_preds : list = []
        self._test_labels: list = []

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode_str = "cached_emb" if use_cached_emb else (
            f"fine-tune({n_trainable_layers}L)" if n_trainable_layers > 0 else "tabular-only"
        )
        logger.info("MeetingModel — trainable params: %d  mode: %s", n_params, mode_str)

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _pool_articles(self, cls: torch.Tensor, article_mask: torch.Tensor) -> torch.Tensor:
        """
        cls          : (B, A, H)
        article_mask : (B, A)  — 1 = real article, 0 = padding
        Returns      : (B, H)
        """
        attn_logits  = self.article_attn(cls).squeeze(-1)                    # (B, A)
        attn_logits  = attn_logits.masked_fill(article_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=1)                     # (B, A)
        return (cls * attn_weights.unsqueeze(-1)).sum(dim=1)                  # (B, H)

    def forward(self, batch: dict) -> torch.Tensor:
        tab_emb = self.tab_proj(self.tab_norm(batch["tabular"]))

        if self.use_text:
            if self.use_cached_emb:
                pooled = self._pool_articles(batch["cls"], batch["article_mask"])
            else:
                B, A, L = batch["input_ids"].shape
                flat_ids  = batch["input_ids"].view(B * A, L)
                flat_mask = batch["attention_mask"].view(B * A, L)
                out = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
                cls = out.last_hidden_state[:, 0, :].view(B, A, -1)
                pooled = self._pool_articles(cls, batch["article_mask"])
            text_emb = self.text_proj(pooled)
            combined = torch.cat([text_emb, tab_emb], dim=1)
        else:
            combined = tab_emb

        return self.fusion(combined)

    # ── Training / eval steps ─────────────────────────────────────────────────

    def _step(self, batch: dict, stage: str):
        logits = self(batch)
        loss   = self.loss_fn(logits, batch["label"])
        preds  = logits.argmax(dim=1)
        acc    = (preds == batch["label"]).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True,  on_step=False, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=False, on_step=False, on_epoch=True)
        return loss, preds.cpu().numpy(), batch["label"].cpu().numpy()

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, preds, labels = self._step(batch, "val")
        self._val_preds.append(preds)
        self._val_labels.append(labels)

    def on_validation_epoch_end(self):
        if self._val_preds:
            p  = np.concatenate(self._val_preds)
            l  = np.concatenate(self._val_labels)
            self.log("val_f1_macro",    f1_score(l, p, average="macro",    zero_division=0), prog_bar=True)
            self.log("val_f1_weighted", f1_score(l, p, average="weighted", zero_division=0), prog_bar=False)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        _, preds, labels = self._step(batch, "test")
        self._test_preds.append(preds)
        self._test_labels.append(labels)

    def on_test_epoch_end(self):
        if self._test_preds:
            p  = np.concatenate(self._test_preds)
            l  = np.concatenate(self._test_labels)
            f1  = f1_score(l, p, average="macro",    zero_division=0)
            f1w = f1_score(l, p, average="weighted", zero_division=0)
            self.log("test_f1_macro",    f1)
            self.log("test_f1_weighted", f1w)
            logger.info("\n=== TEST RESULTS ===")
            logger.info("Macro F1 : %.4f   Weighted F1 : %.4f", f1, f1w)
            logger.info("\n%s", classification_report(
                l, p, target_names=CLASS_NAMES, zero_division=0))
        self._test_preds.clear()
        self._test_labels.clear()

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        head_params = (
            list(self.tab_norm.parameters())
            + list(self.tab_proj.parameters())
            + list(self.fusion.parameters())
        )
        if self.use_text:
            head_params += (
                list(self.article_attn.parameters())
                + list(self.text_proj.parameters())
            )
        param_groups = [{"params": head_params, "lr": self.lr}]

        if self.use_text and not self.use_cached_emb:
            backbone_params = [p for p in self.bert.parameters() if p.requires_grad]
            if backbone_params:
                param_groups.insert(0, {"params": backbone_params, "lr": self.lr * 0.1})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        total_steps = (
            self.trainer.estimated_stepping_batches if self.trainer else 1000
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = self.warmup_steps,
            num_training_steps = total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ─── DataModule ───────────────────────────────────────────────────────────────

class MeetingDataModule(pl.LightningDataModule):

    def __init__(self, df: pd.DataFrame, cfg):
        super().__init__()
        self.df  = df
        self.cfg = cfg

    # ── BERT pre-computation ──────────────────────────────────────────────────

    def _precompute_embeddings(self, df_all: pd.DataFrame):
        """
        Run all articles through frozen BERT once; cache per-article CLS vectors.

        Returns:
          cached_cls   : (N, max_articles, hidden_size) float32
          article_masks: (N, max_articles)              float32
          hidden_size  : int
        """
        cfg = self.cfg
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        bert      = AutoModel.from_pretrained(cfg.tokenizer_name)

        # Optional: load fine-tuned BERT weights from a train_joint.py checkpoint.
        # The daily RUONIA model teaches BERT which language correlates with rate
        # changes, giving much better representations than vanilla RuBERT.
        if getattr(cfg, "bert_checkpoint", ""):
            ckpt_path = cfg.bert_checkpoint
            logger.info("Loading fine-tuned BERT weights from: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            # train_joint.py stores BERT under "bert.*" keys
            bert_state = {k[len("bert."):]: v
                          for k, v in state.items() if k.startswith("bert.")}
            if bert_state:
                missing, unexpected = bert.load_state_dict(bert_state, strict=False)
                logger.info("  BERT weights loaded — missing: %d  unexpected: %d",
                            len(missing), len(unexpected))
            else:
                logger.warning("  No 'bert.*' keys in checkpoint — using base RuBERT weights")
        else:
            logger.info("Pre-computing BERT article embeddings (base RuBERT, frozen) …")

        bert   = bert.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert   = bert.to(device)
        H      = bert.config.hidden_size

        N = len(df_all)
        A = cfg.max_articles
        cached_cls    = np.zeros((N, A, H), dtype=np.float32)
        article_masks = np.zeros((N, A),    dtype=np.float32)

        with torch.no_grad():
            for i, (_, row) in enumerate(df_all.iterrows()):
                texts  = [str(t) for t in list(row["texts"])[:A]]
                n_real = len(texts)
                if n_real == 0:
                    texts  = [""]
                    n_real = 1
                article_masks[i, :n_real] = 1.0

                enc = tokenizer(
                    texts,
                    truncation=True, padding="max_length",
                    max_length=cfg.max_length, return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = bert(**enc)
                cls = out.last_hidden_state[:, 0, :].cpu().numpy()  # (n_real, H)
                cached_cls[i, :n_real] = cls

                if (i + 1) % 20 == 0 or (i + 1) == N:
                    logger.info("  … %d / %d meetings encoded", i + 1, N)

        del bert
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Pre-compute done — shape: %s", cached_cls.shape)
        return cached_cls, article_masks, H

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage=None):
        cfg = self.cfg
        df  = self.df

        dates      = df["meeting_date"]
        train_mask = (dates <= TRAIN_END).values
        val_mask   = ((dates > TRAIN_END) & (dates <= VAL_END)).values
        test_mask  = (dates > VAL_END).values

        logger.info("Split — Train: %d  Val: %d  Test: %d",
                    train_mask.sum(), val_mask.sum(), test_mask.sum())
        for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
            dist = df[mask]["direction"].value_counts().to_dict()
            logger.info("  %s: %s", name, dist)

        # Normalise tabular (fit on train only)
        X_all = np.stack(df["tabular"].values).astype(np.float32)
        col_means = np.nanmean(X_all[train_mask], axis=0)
        col_means[np.isnan(col_means)] = 0.0
        nan_idx = np.where(np.isnan(X_all))
        if nan_idx[0].size:
            X_all[nan_idx] = np.take(col_means, nan_idx[1])
        col_std = np.nanstd(X_all[train_mask], axis=0) + 1e-6
        X_all = (X_all - col_means) / col_std
        self.n_tabular = X_all.shape[1]

        # Class weights (balanced on train)
        y_train = df["label"].values[train_mask]
        counts  = Counter(y_train.tolist())
        total   = sum(counts.values())
        self.class_weights = [
            total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)
        ]
        logger.info("Class weights: cut=%.2f  hold=%.2f  hike=%.2f", *self.class_weights)

        # Pre-compute BERT embeddings when BERT is fully frozen
        use_cached_emb = cfg.use_text and (cfg.n_trainable_layers == 0)
        self.use_cached_emb = use_cached_emb

        if use_cached_emb:
            cached_cls, article_masks, emb_dim = self._precompute_embeddings(df)
            self.emb_dim = emb_dim
        else:
            cached_cls = article_masks = None
            self.emb_dim = 768

        # Build dataset splits
        def _make_ds(mask):
            indices = np.where(mask)[0]
            sub = df.iloc[indices].copy().reset_index(drop=True)
            sub["tabular"] = [X_all[i] for i in indices]
            return MeetingDataset(
                sub,
                cached_cls     = cached_cls[indices]    if cached_cls    is not None else None,
                article_masks  = article_masks[indices] if article_masks is not None else None,
                tokenizer_name = cfg.tokenizer_name,
                max_length     = cfg.max_length,
                max_articles   = cfg.max_articles,
                use_text       = cfg.use_text,
            )

        self.train_ds = _make_ds(train_mask)
        self.val_ds   = _make_ds(val_mask)
        self.test_ds  = _make_ds(test_mask)

    def _dl(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = shuffle,
            num_workers = 0,   # Windows-safe
            pin_memory  = False,
        )

    def train_dataloader(self): return self._dl(self.train_ds, shuffle=True)
    def val_dataloader(self):   return self._dl(self.val_ds,   shuffle=False)
    def test_dataloader(self):  return self._dl(self.test_ds,  shuffle=False)


# ─── Daily LSTM: dataset ──────────────────────────────────────────────────────

class DailySequenceDataset(Dataset):
    """
    Each meeting → (T, 3) daily P(down/same/up) sequence + tabular → label.
    T = window_days steps; padded with zeros when no articles published that day.
    """

    def __init__(
        self,
        seqs:    np.ndarray,   # (N, T, 3) float32
        tabular: np.ndarray,   # (N, D)    float32
        labels:  np.ndarray,   # (N,)      int64
    ):
        self.seqs    = torch.from_numpy(seqs)
        self.tabular = torch.from_numpy(tabular)
        self.labels  = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seq":     self.seqs[idx],
            "tabular": self.tabular[idx],
            "label":   self.labels[idx],
        }


# ─── Daily LSTM: model ────────────────────────────────────────────────────────

class DailyLSTMModel(pl.LightningModule):
    """
    Small LSTM over daily P(down/same/up) sequences from the fine-tuned RUONIA model.

    Architecture:
      LSTM(input=3, hidden=lstm_hidden, layers=lstm_layers)  → last step hidden
      → Dropout → Linear(lstm_hidden)  → GELU
      [concat tabular projection]
      → Linear(fusion_hidden) → GELU → Dropout → Linear(3)

    ~4 k trainable parameters: regularisation must be aggressive for N~66.
    """

    def __init__(
        self,
        n_tabular: int,
        lstm_hidden: int        = 32,
        lstm_layers: int        = 2,
        tab_proj_dim: int       = 32,
        fusion_hidden: int      = 64,
        dropout: float          = 0.5,
        lr: float               = 1e-3,
        weight_decay: float     = 0.1,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float  = 0.15,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr           = lr
        self.weight_decay = weight_decay

        self.lstm = nn.LSTM(
            input_size  = 3,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.GELU(),
        )

        self.tab_norm = nn.LayerNorm(n_tabular)
        self.tab_proj = nn.Sequential(
            nn.Linear(n_tabular, tab_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden + tab_proj_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, NUM_CLASSES),
        )

        cw = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(
            weight=cw, label_smoothing=label_smoothing, reduction="mean"
        )

        self._val_preds:   list = []
        self._val_labels:  list = []
        self._test_preds:  list = []
        self._test_labels: list = []

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("DailyLSTMModel — trainable params: %d", n_params)

    def forward(self, seq: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)          # (B, T, H)
        last   = out[:, -1, :]          # (B, H) — last timestep
        lstm_emb = self.lstm_proj(last)
        tab_emb  = self.tab_proj(self.tab_norm(tabular))
        return self.fusion(torch.cat([lstm_emb, tab_emb], dim=1))

    def _step(self, batch: dict, stage: str):
        logits = self(batch["seq"], batch["tabular"])
        loss   = self.loss_fn(logits, batch["label"])
        preds  = logits.argmax(dim=1)
        acc    = (preds == batch["label"]).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True)
        if stage == "val":
            self._val_preds.extend(preds.cpu().tolist())
            self._val_labels.extend(batch["label"].cpu().tolist())
        elif stage == "test":
            self._test_preds.extend(preds.cpu().tolist())
            self._test_labels.extend(batch["label"].cpu().tolist())
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _): return self._step(batch, "test")

    def on_validation_epoch_end(self):
        if self._val_preds:
            f1 = f1_score(self._val_labels, self._val_preds,
                          average="macro", zero_division=0)
            self.log("val_f1_macro", f1, prog_bar=True)
            self._val_preds.clear()
            self._val_labels.clear()

    def on_test_epoch_end(self):
        if self._test_preds:
            f1m = f1_score(self._test_labels, self._test_preds,
                           average="macro", zero_division=0)
            f1w = f1_score(self._test_labels, self._test_preds,
                           average="weighted", zero_division=0)
            self.log("test_f1_macro",    f1m)
            self.log("test_f1_weighted", f1w)
            logger.info("\n%s", classification_report(
                self._test_labels, self._test_preds,
                target_names=CLASS_NAMES, zero_division=0,
            ))
            self._test_preds.clear()
            self._test_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


# ─── Config ───────────────────────────────────────────────────────────────────

class Config:
    tokenizer_name     : str   = TOKENIZER
    max_length         : int   = 128
    max_articles       : int   = 20
    batch_size         : int   = 8        # ~60 train meetings → ~8 batches/epoch
    text_proj_dim      : int   = 64
    tab_proj_dim       : int   = 32
    fusion_hidden_dim  : int   = 64
    dropout            : float = 0.4
    lr                 : float = 1e-3     # head-only LR (BERT frozen)
    weight_decay       : float = 0.1
    warmup_steps       : int   = 10
    n_trainable_layers : int   = 0        # 0 = fully frozen (recommended)
    label_smoothing    : float = 0.15
    max_epochs         : int   = 100
    patience           : int   = 15
    use_text           : bool  = True
    bert_checkpoint    : str   = ""       # path to train_joint.py checkpoint for transfer


# ─── Multi-seed runner ────────────────────────────────────────────────────────

def run_one_seed(df: pd.DataFrame, cfg: Config, seed: int) -> dict:
    pl.seed_everything(seed, workers=True)

    dm = MeetingDataModule(df, cfg)
    dm.setup()

    model = MeetingModel(
        n_tabular          = dm.n_tabular,
        max_articles       = cfg.max_articles,
        num_classes        = NUM_CLASSES,
        text_proj_dim      = cfg.text_proj_dim,
        tab_proj_dim       = cfg.tab_proj_dim,
        fusion_hidden_dim  = cfg.fusion_hidden_dim,
        dropout            = cfg.dropout,
        lr                 = cfg.lr,
        weight_decay       = cfg.weight_decay,
        warmup_steps       = cfg.warmup_steps,
        class_weights      = dm.class_weights,
        n_trainable_layers = cfg.n_trainable_layers,
        label_smoothing    = cfg.label_smoothing,
        use_text           = cfg.use_text,
        use_cached_emb     = dm.use_cached_emb,
        emb_dim            = dm.emb_dim,
        backbone           = cfg.tokenizer_name,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(CKPT_DIR / f"seed{seed}"),
        filename   = "best",
        monitor    = "val_f1_macro",
        mode       = "max",
        save_top_k = 1,
    )
    early_cb = EarlyStopping(
        monitor  = "val_f1_macro",
        patience = cfg.patience,
        mode     = "max",
        verbose  = False,
    )

    precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    trainer = pl.Trainer(
        max_epochs          = cfg.max_epochs,
        accelerator         = "auto",
        devices             = 1,
        precision           = precision,
        callbacks           = [ckpt_cb, early_cb],
        logger              = False,
        enable_progress_bar = True,
        gradient_clip_val   = 1.0,
        log_every_n_steps   = 1,
    )

    trainer.fit(model, dm)
    results = trainer.test(model, dm, ckpt_path="best", verbose=False)
    tr = results[0] if results else {}
    return {
        "seed":             seed,
        "test_f1_macro":    float(tr.get("test_f1_macro",    0.0)),
        "test_f1_weighted": float(tr.get("test_f1_weighted", 0.0)),
    }


# ─── Daily-model soft-label features ─────────────────────────────────────────

def _extract_daily_model_probs(
    df: pd.DataFrame,
    ckpt_path: str,
    max_articles: int = 20,
    max_length: int   = 256,
) -> np.ndarray:
    """
    For each meeting, run every article in its window through the fine-tuned
    JointBertTabularModel and aggregate the softmax probabilities.

    Tabular inputs are set to zero so only the text tower contributes —
    we want the article-level signal, not the economic features.

    Returns: (N_meetings, 13) float32 array with columns:
      [0-2]  mean P(down/same/up) over articles
      [3-5]  max  P(down/same/up)
      [6-8]  std  P(down/same/up)
      [9]    net hawkishness = mean P(up) − mean P(down)
      [10-12] fraction of articles whose argmax = down / same / up
    """
    import sys as _sys
    _project_root = str(Path(__file__).parents[2])
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    from cbr_news.ml.train_joint import JointBertTabularModel

    logger.info("Loading daily RUONIA model from: %s", ckpt_path)
    model = JointBertTabularModel.load_from_checkpoint(
        ckpt_path, map_location="cpu", strict=False
    )
    model.eval()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    n_tab     = model.hparams.get("n_tabular", 90) if hasattr(model.hparams, "get") \
                else getattr(model.hparams, "n_tabular", 90)
    tok_name  = getattr(model.hparams, "backbone", "DeepPavlov/rubert-base-cased") or \
                "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    logger.info("  backbone=%s  n_tabular=%d", tok_name, n_tab)

    all_feats = []
    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            texts  = [str(t) for t in list(row["texts"])[:max_articles]]
            if not texts:
                texts = [""]

            probs_list = []
            for text in texts:
                enc = tokenizer(
                    text,
                    truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt",
                )
                logits = model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                    torch.zeros(1, n_tab, device=device),
                )                                              # (1, 3)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (3,)
                probs_list.append(probs)

            arr     = np.array(probs_list, dtype=np.float32)  # (n_arts, 3)
            mean_p  = arr.mean(0)
            max_p   = arr.max(0)
            std_p   = arr.std(0)
            net_hawk = float(mean_p[2] - mean_p[0])
            argmax_frac = np.bincount(arr.argmax(1), minlength=3) / len(arr)

            all_feats.append(np.concatenate([mean_p, max_p, std_p, [net_hawk], argmax_frac]))

            if (i + 1) % 20 == 0 or (i + 1) == len(df):
                logger.info("  … %d / %d meetings processed", i + 1, len(df))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = np.array(all_feats, dtype=np.float32)   # (N, 13)
    logger.info("Daily-model features shape: %s", out.shape)
    return out


# ─── Daily LSTM: sequence building ───────────────────────────────────────────

def _build_daily_sequences(
    df: pd.DataFrame,
    ckpt_path: str,
    window_days: int = 14,
    max_length: int  = 256,
) -> np.ndarray:
    """
    For each meeting build a (window_days, 3) time series:
      day[t] = mean P(down/same/up) of articles published on meeting_date - (window_days-1-t) days.
    Days with no articles → zero vector.

    Requires 'article_dates' column in df (added by prepare_meeting_dataset.py).

    Returns: (N_meetings, window_days, 3) float32
    """
    import sys as _sys
    _project_root = str(Path(__file__).parents[2])
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    from cbr_news.ml.train_joint import JointBertTabularModel

    if "article_dates" not in df.columns:
        raise ValueError(
            "Meeting dataset missing 'article_dates' column. "
            "Rebuild with: PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py"
        )

    logger.info("Loading daily RUONIA model for sequence building: %s", ckpt_path)
    model = JointBertTabularModel.load_from_checkpoint(
        ckpt_path, map_location="cpu", strict=False
    )
    model.eval()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = model.to(device)
    n_tab    = (model.hparams.get("n_tabular", 90)
                if hasattr(model.hparams, "get")
                else getattr(model.hparams, "n_tabular", 90))
    tok_name = (getattr(model.hparams, "backbone", "DeepPavlov/rubert-base-cased")
                or "DeepPavlov/rubert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    logger.info("  backbone=%s  n_tabular=%d  window_days=%d", tok_name, n_tab, window_days)

    all_seqs = []
    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            texts  = [str(t) for t in list(row["texts"])]
            dates  = [str(d) for d in list(row["article_dates"])]
            meeting_date = pd.Timestamp(row["meeting_date"])

            # Group article-level probabilities by publication date
            date_probs: dict = {}
            for text, date_str in zip(texts, dates):
                if not text.strip():
                    continue
                enc = tokenizer(
                    text, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt",
                )
                logits = model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                    torch.zeros(1, n_tab, device=device),
                )
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                date_probs.setdefault(date_str, []).append(probs)

            # Build (window_days, 3): day 0 = oldest, day T-1 = day before meeting
            seq = np.zeros((window_days, 3), dtype=np.float32)
            for t in range(window_days):
                day = meeting_date - pd.Timedelta(days=window_days - 1 - t)
                day_str = day.strftime("%Y-%m-%d")
                if day_str in date_probs:
                    seq[t] = np.mean(date_probs[day_str], axis=0)

            all_seqs.append(seq)

            if (i + 1) % 20 == 0 or (i + 1) == len(df):
                logger.info("  … %d / %d meetings sequenced", i + 1, len(df))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = np.array(all_seqs, dtype=np.float32)
    logger.info("Daily sequences shape: %s  (non-zero days: %.1f%% avg)",
                result.shape,
                100.0 * (result.sum(-1) > 0).mean())
    return result


# ─── Daily LSTM: training runner ──────────────────────────────────────────────

def run_lstm_daily(
    df: pd.DataFrame,
    cfg: Config,
    bert_checkpoint: str,
    window_days: int = 14,
    n_seeds: int     = 3,
) -> None:
    """
    Train a DailyLSTMModel over per-meeting daily RUONIA probability sequences.
    Runs n_seeds seeds and reports mean ± std test macro-F1.
    """
    seqs = _build_daily_sequences(
        df, ckpt_path=bert_checkpoint,
        window_days=window_days, max_length=cfg.max_length,
    )

    dates      = df["meeting_date"]
    train_mask = (dates <= TRAIN_END).values
    val_mask   = ((dates > TRAIN_END) & (dates <= VAL_END)).values
    test_mask  = (dates > VAL_END).values

    labels = df["label"].values

    # Normalise tabular (fit on train)
    X_all     = np.stack(df["tabular"].values).astype(np.float32)
    col_means = np.nanmean(X_all[train_mask], axis=0)
    col_means[np.isnan(col_means)] = 0.0
    col_std   = np.nanstd(X_all[train_mask], axis=0) + 1e-6
    X_all     = (X_all - col_means) / col_std
    X_all     = np.nan_to_num(X_all, 0.0)
    n_tabular = X_all.shape[1]

    # Class weights (balanced on train)
    y_train = labels[train_mask]
    counts  = Counter(y_train.tolist())
    total   = sum(counts.values())
    class_weights = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
    logger.info("Class weights: cut=%.2f  hold=%.2f  hike=%.2f", *class_weights)

    dl_kwargs = dict(batch_size=8, num_workers=0, pin_memory=False)

    all_results = []
    for seed_idx in range(n_seeds):
        seed = seed_idx * 42
        pl.seed_everything(seed, workers=True)

        train_ds = DailySequenceDataset(seqs[train_mask], X_all[train_mask], y_train)
        val_ds   = DailySequenceDataset(seqs[val_mask],   X_all[val_mask],   labels[val_mask])
        test_ds  = DailySequenceDataset(seqs[test_mask],  X_all[test_mask],  labels[test_mask])

        train_dl = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
        val_dl   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
        test_dl  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

        model = DailyLSTMModel(
            n_tabular       = n_tabular,
            dropout         = cfg.dropout,
            lr              = cfg.lr,
            weight_decay    = cfg.weight_decay,
            class_weights   = class_weights,
            label_smoothing = cfg.label_smoothing,
        )

        ckpt_dir = CKPT_DIR / f"daily_lstm_seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="best",
            monitor="val_f1_macro", mode="max", save_top_k=1,
        )
        early_cb = EarlyStopping(
            monitor="val_f1_macro", patience=cfg.patience, mode="max", verbose=False,
        )
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs, accelerator="auto", devices=1,
            precision=precision,
            callbacks=[ckpt_cb, early_cb], logger=False,
            enable_progress_bar=True, gradient_clip_val=1.0,
            log_every_n_steps=1,
        )
        trainer.fit(model, train_dl, val_dl)
        res = trainer.test(model, test_dl, ckpt_path="best", verbose=False)
        r   = res[0] if res else {}
        result = {
            "seed":             seed,
            "test_f1_macro":    float(r.get("test_f1_macro",    0.0)),
            "test_f1_weighted": float(r.get("test_f1_weighted", 0.0)),
        }
        all_results.append(result)
        logger.info("Seed %d — test_f1_macro=%.4f  test_f1_weighted=%.4f",
                    seed, result["test_f1_macro"], result["test_f1_weighted"])

    macros    = [r["test_f1_macro"]    for r in all_results]
    weighteds = [r["test_f1_weighted"] for r in all_results]
    logger.info("\n══ DAILY LSTM — FINAL (%d seeds) ══", n_seeds)
    logger.info("test_f1_macro    : %.4f ± %.4f  (min=%.4f  max=%.4f)",
                np.mean(macros), np.std(macros), min(macros), max(macros))
    logger.info("test_f1_weighted : %.4f ± %.4f  (min=%.4f  max=%.4f)",
                np.mean(weighteds), np.std(weighteds), min(weighteds), max(weighteds))


# ─── Sklearn baselines ────────────────────────────────────────────────────────

def run_sklearn_baselines(
    df: pd.DataFrame,
    cfg: Config,
    daily_model_ckpt: str = "",
    pca_dim: int = 0,
) -> None:
    """
    For N~66 training examples sklearn classifiers on pre-computed BERT embeddings
    generalise far better than any neural fine-tuning.

    Features per meeting:
      mean-pooled BERT CLS over real articles  (768-dim, optionally PCA-reduced)
      + normalised tabular features             (N_tab-dim)
      [+ daily-model soft-label probs          (13-dim, optional)]

    pca_dim > 0: apply PCA to the BERT part only, reducing 768 → pca_dim dims.
    This greatly reduces overfitting on the small dataset.

    Tries several L2-regularised + ensemble classifiers; reports val & test macro-F1.
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Force frozen BERT so embeddings are always pre-computed
    cfg.n_trainable_layers = 0
    cfg.use_text           = True

    dm = MeetingDataModule(df, cfg)
    dm.setup()

    dates      = df["meeting_date"]
    train_mask = (dates <= TRAIN_END).values
    val_mask   = ((dates > TRAIN_END) & (dates <= VAL_END)).values
    test_mask  = (dates > VAL_END).values

    # Reconstruct normalised tabular from datasets (already normalised in setup)
    def _features(ds: MeetingDataset) -> np.ndarray:
        rows = []
        for i in range(len(ds)):
            item = ds[i]
            tab  = item["tabular"].numpy()
            if ds.cached_cls is not None:
                cls  = item["cls"].numpy()           # (A, H)
                mask = item["article_mask"].numpy()  # (A,)
                valid = cls[mask == 1]
                emb  = valid.mean(0) if len(valid) > 0 else cls[0]
            else:
                emb = np.zeros(dm.emb_dim, dtype=np.float32)
            rows.append(np.concatenate([emb, tab]))
        return np.array(rows, dtype=np.float32)

    logger.info("Extracting features for sklearn classifiers …")
    X_train = _features(dm.train_ds)
    X_val   = _features(dm.val_ds)
    X_test  = _features(dm.test_ds)

    # Optionally concatenate daily-model soft-label features
    if daily_model_ckpt:
        logger.info("Adding daily-model soft-label features …")
        dm_feats = _extract_daily_model_probs(
            df,
            ckpt_path    = daily_model_ckpt,
            max_articles = cfg.max_articles,
            max_length   = cfg.max_length,
        )
        X_train = np.concatenate([X_train, dm_feats[train_mask]], axis=1)
        X_val   = np.concatenate([X_val,   dm_feats[val_mask]],   axis=1)
        X_test  = np.concatenate([X_test,  dm_feats[test_mask]],  axis=1)

    labels_all = df["label"].values
    y_train    = labels_all[train_mask]
    y_val      = labels_all[val_mask]
    y_test     = labels_all[test_mask]

    # Combine all features for CV (need full dataset arrays)
    X_all_full = np.concatenate([X_train, X_val, X_test], axis=0)
    idx_all    = (np.where(train_mask)[0].tolist()
                  + np.where(val_mask)[0].tolist()
                  + np.where(test_mask)[0].tolist())
    # Map original df indices → position in X_all_full
    orig_to_pos = {orig: pos for pos, orig in enumerate(idx_all)}

    # Optional PCA on BERT part only (fit on train, apply to all)
    if pca_dim > 0:
        n_bert = dm.emb_dim
        effective_dim = min(pca_dim, n_bert, len(X_train) - 1)
        logger.info("Applying PCA to BERT embeddings: %d → %d dims", n_bert, effective_dim)
        pca = PCA(n_components=effective_dim, random_state=0)
        train_pos = [orig_to_pos[i] for i in np.where(train_mask)[0]]
        pca.fit(X_all_full[train_pos, :n_bert])
        X_all_full = np.concatenate([
            pca.transform(X_all_full[:, :n_bert]),
            X_all_full[:, n_bert:],
        ], axis=1)
        logger.info("After PCA: feature dims = %d", X_all_full.shape[1])

    def _X(mask):
        pos = [orig_to_pos[i] for i in np.where(mask)[0]]
        return X_all_full[pos]

    X_train_f = _X(train_mask)
    X_val_f   = _X(val_mask)
    X_test_f  = _X(test_mask)

    logger.info("Feature shapes — train: %s  val: %s  test: %s",
                X_train_f.shape, X_val_f.shape, X_test_f.shape)

    from sklearn.base import clone

    classifiers = [
        ("LogReg C=0.001", LogisticRegression(C=0.001, max_iter=2000, class_weight="balanced", random_state=0)),
        ("LogReg C=0.01",  LogisticRegression(C=0.01,  max_iter=2000, class_weight="balanced", random_state=0)),
        ("LogReg C=0.1",   LogisticRegression(C=0.1,   max_iter=2000, class_weight="balanced", random_state=0)),
        ("LogReg C=1.0",   LogisticRegression(C=1.0,   max_iter=2000, class_weight="balanced", random_state=0)),
        ("SVC C=0.01",     LinearSVC(C=0.01,  max_iter=2000, class_weight="balanced", random_state=0)),
        ("SVC C=0.1",      LinearSVC(C=0.1,   max_iter=2000, class_weight="balanced", random_state=0)),
        ("GBoost d2 n50",  GradientBoostingClassifier(n_estimators=50,  max_depth=2, random_state=0)),
        ("GBoost d3 n100", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)),
    ]

    # ── Expanding-window time-series CV on training data ──────────────────────
    # With only 9–19 val meetings a single val fold is unreliable (regime shifts).
    # Instead: for each year Y in [cv_start..train_end_year], train on [all years < Y],
    # validate on Y.  Average macro-F1 across folds → stable model selection signal.
    meeting_years = pd.to_datetime(df["meeting_date"]).dt.year.values
    train_end_year = pd.to_datetime(str(TRAIN_END)).year   # e.g. 2022
    cv_val_years   = list(range(2019, train_end_year + 1)) # [2019..2022]

    logger.info("\n── Expanding-window CV folds: val years %s ──", cv_val_years)
    clf_cv_f1: dict = {name: [] for name, _ in classifiers}

    for val_yr in cv_val_years:
        cv_train_idx = np.where(
            train_mask & (meeting_years < val_yr)
        )[0]
        cv_val_idx = np.where(
            train_mask & (meeting_years == val_yr)
        )[0]
        if len(cv_val_idx) < 2 or len(cv_train_idx) < 5:
            logger.info("  year %d: skipped (train=%d val=%d)", val_yr, len(cv_train_idx), len(cv_val_idx))
            continue
        Xc_tr = X_all_full[[orig_to_pos[i] for i in cv_train_idx]]
        Xc_va = X_all_full[[orig_to_pos[i] for i in cv_val_idx]]
        yc_tr = labels_all[cv_train_idx]
        yc_va = labels_all[cv_val_idx]
        for name, clf in classifiers:
            try:
                pipe_cv = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])
                pipe_cv.fit(Xc_tr, yc_tr)
                f1 = f1_score(yc_va, pipe_cv.predict(Xc_va), average="macro", zero_division=0)
            except Exception:
                f1 = 0.0
            clf_cv_f1[name].append(f1)

    cv_mean = {name: float(np.mean(scores)) if scores else 0.0
               for name, scores in clf_cv_f1.items()}

    # ── Train final models on full train set, evaluate val + test ─────────────
    trained_pipes  = {}
    val_f1_scores  = {}
    test_f1_scores = {}

    logger.info("\n%s", "─" * 72)
    logger.info("%-20s  cv_macro  val_macro  test_macro", "Classifier")
    logger.info("%s", "─" * 72)

    for name, clf in classifiers:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train_f, y_train)
        val_f1  = f1_score(y_val,  pipe.predict(X_val_f),  average="macro", zero_division=0)
        test_f1 = f1_score(y_test, pipe.predict(X_test_f), average="macro", zero_division=0)
        logger.info("%-20s   %.4f    %.4f     %.4f", name, cv_mean[name], val_f1, test_f1)
        trained_pipes[name]  = pipe
        val_f1_scores[name]  = val_f1
        test_f1_scores[name] = test_f1

    logger.info("%s", "─" * 72)

    # Select by CV score (more reliable than single val fold)
    best_name = max(cv_mean, key=cv_mean.get)
    logger.info("Best by CV macro-F1 : %s (cv=%.4f  val=%.4f  test=%.4f)",
                best_name, cv_mean[best_name],
                val_f1_scores[best_name], test_f1_scores[best_name])

    p_test = trained_pipes[best_name].predict(X_test_f)
    logger.info("\n=== BEST MODEL (CV-selected) — TEST REPORT ===")
    logger.info("\n%s", classification_report(y_test, p_test, target_names=CLASS_NAMES, zero_division=0))

    # ── Soft-vote ensemble of top CV classifiers ──────────────────────────────
    # Only include classifiers with CV score above the median CV score to avoid
    # degradation from weak classifiers that overfit a single val fold.
    cv_threshold = float(np.median(list(cv_mean.values())))
    top_names    = [n for n in trained_pipes if cv_mean[n] >= cv_threshold]
    logger.info("── Ensemble (top-%d by CV, threshold=%.3f): %s ──",
                len(top_names), cv_threshold, top_names)

    def _to_proba(pipe, X):
        clf_obj = pipe.named_steps["clf"]
        Xs      = pipe.named_steps["scaler"].transform(X)
        if hasattr(clf_obj, "predict_proba"):
            return clf_obj.predict_proba(Xs)
        scores  = clf_obj.decision_function(Xs)
        if scores.ndim == 1:            # binary fallback
            scores = np.column_stack([-scores, scores])
        exp     = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    prob_list = []
    for name in top_names:
        probs = _to_proba(trained_pipes[name], X_test_f)
        if probs.shape[1] == NUM_CLASSES:
            prob_list.append(probs)

    if prob_list:
        ens_probs = np.mean(prob_list, axis=0)
        p_ens     = ens_probs.argmax(axis=1)
        ens_f1    = f1_score(y_test, p_ens, average="macro", zero_division=0)
        logger.info("Ensemble test_macro=%.4f", ens_f1)
        logger.info("\n%s", classification_report(
            y_test, p_ens, target_names=CLASS_NAMES, zero_division=0))


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Meeting-centric 3-class key rate model")
    ap.add_argument("--dataset",         type=str,   default=str(MEETING_DS))
    ap.add_argument("--max-articles",    type=int,   default=20)
    ap.add_argument("--max-length",      type=int,   default=128)
    ap.add_argument("--n-trainable",     type=int,   default=0,
                    help="BERT encoder layers to unfreeze (0=fully frozen, recommended)")
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--dropout",         type=float, default=0.4)
    ap.add_argument("--weight-decay",    type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.15)
    ap.add_argument("--text-proj-dim",   type=int,   default=64)
    ap.add_argument("--tab-proj-dim",    type=int,   default=32)
    ap.add_argument("--fusion-hidden",   type=int,   default=64)
    ap.add_argument("--epochs",          type=int,   default=100)
    ap.add_argument("--patience",        type=int,   default=15)
    ap.add_argument("--batch-size",      type=int,   default=8)
    ap.add_argument("--seeds",           type=int,   default=3,
                    help="Number of random seeds for multi-seed evaluation")
    ap.add_argument("--no-text",         action="store_true",
                    help="Tabular-only baseline (disables BERT tower)")
    ap.add_argument("--sklearn",         action="store_true",
                    help="Run sklearn classifiers on mean BERT emb + tabular (best for N~66)")
    ap.add_argument("--pca-dim",         type=int, default=0,
                    help="Apply PCA to BERT embeddings before sklearn (0=off). "
                         "Recommended: 50 for N~66 to reduce overfitting.")
    ap.add_argument("--bert-checkpoint", type=str, default="",
                    help="Path to train_joint.py checkpoint — loads fine-tuned BERT weights "
                         "for better embeddings (transfer from daily RUONIA task)")
    ap.add_argument("--daily-model", type=str, default="",
                    help="Path to JointBertTabularModel checkpoint — append per-article "
                         "P(down/same/up) aggregations as extra features for --sklearn mode")
    ap.add_argument("--lstm-daily",      action="store_true",
                    help="Train a small LSTM over day-by-day RUONIA probability sequences "
                         "(requires --bert-checkpoint and meeting dataset with article_dates)")
    ap.add_argument("--window-days",     type=int, default=14,
                    help="Number of days before each meeting to build the daily sequence (default 14)")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        logger.error("Meeting dataset not found: %s", ds_path)
        logger.error("Run first: PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py")
        return

    logger.info("Loading meeting dataset: %s", ds_path)
    df = pd.read_parquet(ds_path)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    logger.info("Loaded %d meetings", len(df))
    logger.info("Direction distribution:\n%s", df["direction"].value_counts().to_string())

    cfg = Config()
    cfg.max_articles       = args.max_articles
    cfg.max_length         = args.max_length
    cfg.n_trainable_layers = args.n_trainable
    cfg.lr                 = args.lr
    cfg.dropout            = args.dropout
    cfg.weight_decay       = args.weight_decay
    cfg.label_smoothing    = args.label_smoothing
    cfg.text_proj_dim      = args.text_proj_dim
    cfg.tab_proj_dim       = args.tab_proj_dim
    cfg.fusion_hidden_dim  = args.fusion_hidden
    cfg.max_epochs         = args.epochs
    cfg.patience           = args.patience
    cfg.batch_size         = args.batch_size
    cfg.use_text           = not args.no_text
    cfg.bert_checkpoint    = args.bert_checkpoint

    if args.lstm_daily:
        if not args.bert_checkpoint:
            logger.error("--lstm-daily requires --bert-checkpoint pointing to a train_joint.py ckpt")
            return
        run_lstm_daily(
            df,
            cfg,
            bert_checkpoint = args.bert_checkpoint,
            window_days     = args.window_days,
            n_seeds         = args.seeds,
        )
        return

    if args.sklearn:
        run_sklearn_baselines(df, cfg, daily_model_ckpt=args.daily_model, pca_dim=args.pca_dim)
        return

    mode       = "text+tabular" if cfg.use_text else "tabular-only"
    frozen_str = f"(BERT frozen)" if cfg.n_trainable_layers == 0 else f"(n_trainable={cfg.n_trainable_layers})"
    logger.info(
        "\nConfig: %s %s  lr=%.0e  dropout=%.2f  wd=%.2f  label_smooth=%.2f  seeds=%d",
        mode, frozen_str, cfg.lr, cfg.dropout, cfg.weight_decay, cfg.label_smoothing, args.seeds,
    )

    all_results = []
    for seed_idx in range(args.seeds):
        seed = seed_idx * 42
        logger.info("\n─── Seed %d / %d  (seed=%d) ───", seed_idx + 1, args.seeds, seed)
        result = run_one_seed(df, cfg, seed=seed)
        all_results.append(result)
        logger.info(
            "Seed %d — test_f1_macro=%.4f  test_f1_weighted=%.4f",
            seed, result["test_f1_macro"], result["test_f1_weighted"],
        )

    macros    = [r["test_f1_macro"]    for r in all_results]
    weighteds = [r["test_f1_weighted"] for r in all_results]
    logger.info("\n══ FINAL RESULTS (%d seeds) ══", args.seeds)
    logger.info(
        "test_f1_macro    : %.4f ± %.4f  (min=%.4f  max=%.4f)",
        np.mean(macros), np.std(macros), min(macros), max(macros),
    )
    logger.info(
        "test_f1_weighted : %.4f ± %.4f  (min=%.4f  max=%.4f)",
        np.mean(weighteds), np.std(weighteds), min(weighteds), max(weighteds),
    )


if __name__ == "__main__":
    main()
