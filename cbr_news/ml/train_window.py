"""
Window-aggregated BERT+Tabular model for RUONIA direction prediction.

Problem with per-article approach:
  Each article is treated independently: article₁ → prediction₁, article₂ → prediction₂.
  But RUONIA moves because of the AGGREGATE news environment, not a single article.
  This introduces massive label noise (100k articles → same ~3500 unique dates).

This model instead:
  1. Groups all articles published in the past W days before each prediction date.
  2. Encodes each article's [CLS] token with BERT.
  3. Aggregates article embeddings via attention pooling.
  4. Fuses with tabular economic features → predicts direction.

This reduces dataset from ~100k rows to ~3500 unique dates, but each sample has
richer, more reliable context (multiple articles + tabular).

Architecture:
  Text tower:   {art₁, ..., artₙ} → BERT(each) → attention_pool → Linear(768→128)
  Tabular tower: BatchNorm → Linear(N→64) → ReLU
  Fusion:        concat(192) → Linear(192→64) → ReLU → Linear(64→3)

Run:
    python cbr_news/ml/train_window.py --window-days 7 --max-articles 5
    python cbr_news/ml/train_window.py --window-days 14 --max-articles 8 --n-trainable 4
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from cbr_news.ml.feature_engineering import (
    build_tabular_features,
    extract_text_features,
    get_feature_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
CKPT_DIR = Path(__file__).parents[2] / "checkpoints" / "window"

LABEL_MAP   = {"down": 0, "same": 1, "up": 2}
CLASS_NAMES = ["down", "same", "up"]
TOKENIZER   = "DeepPavlov/rubert-base-cased"

TRAIN_YEARS = list(range(2010, 2023))
VAL_YEARS   = [2023]
TEST_YEARS  = [2024, 2025, 2026]


# ─── Date-grouped dataset ─────────────────────────────────────────────────────

def _parse_year(d) -> int:
    try:
        p = str(d).split(".")
        return int(p[2]) if len(p) == 3 else pd.to_datetime(d).year
    except Exception:
        return 2020


def _parse_date(d) -> Optional[pd.Timestamp]:
    try:
        p = str(d).split(".")
        if len(p) == 3:
            return pd.Timestamp(int(p[2]), int(p[1]), int(p[0]))
        return pd.to_datetime(d)
    except Exception:
        return None


def build_date_grouped_dataset(
    df_raw: pd.DataFrame,
    df_feat: pd.DataFrame,
    window_days: int = 7,
    max_articles: int = 5,
) -> pd.DataFrame:
    """
    Group articles by prediction date and create one row per unique date.

    For each unique date D in the dataset:
      - target  : majority vote of target labels on date D (most consistent label)
      - texts   : up to max_articles texts from [D - window_days, D]
      - tabular : economic features for date D (first article's features)

    Returns a DataFrame with columns:
      date, target, year, texts (list), tabular_idx (index into df_feat)
    """
    logger.info("Building date-grouped dataset (window=%d days, max_articles=%d)…",
                window_days, max_articles)

    df = df_raw.copy()
    df["parsed_date"] = df["date"].apply(_parse_date)
    df = df[df["parsed_date"].notna()].copy()
    df["year"] = df["parsed_date"].apply(lambda d: d.year)

    # Map labels
    df["label"] = df["target"].map(LABEL_MAP)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    # Add feat index
    df["feat_idx"] = np.arange(len(df))

    # Sort by date
    df = df.sort_values("parsed_date").reset_index(drop=True)

    rows = []
    unique_dates = df["parsed_date"].unique()

    for pred_date in unique_dates:
        # All articles within the window up to and including pred_date
        window_start = pred_date - pd.Timedelta(days=window_days)
        mask = (df["parsed_date"] > window_start) & (df["parsed_date"] <= pred_date)
        window_df = df[mask]

        if len(window_df) == 0:
            continue

        # Target: label of articles ON this date (not the whole window)
        # Use majority vote to get the most consistent label for pred_date
        on_date = df[df["parsed_date"] == pred_date]
        if len(on_date) == 0:
            continue

        label_counts = Counter(on_date["label"].tolist())
        target_label = label_counts.most_common(1)[0][0]

        # Texts: sample from window, prioritise CBR press releases, newest first
        cbr_mask = window_df.get("source", pd.Series([""] * len(window_df))) == "cbr_press_releases"
        cbr_texts = window_df[cbr_mask]["cleaned_text"].fillna("").tolist()
        other_texts = window_df[~cbr_mask]["cleaned_text"].fillna("").tolist()
        # CBR first, then recency-ordered others; cap at max_articles
        texts = (cbr_texts + other_texts)[:max_articles]
        if len(texts) == 0:
            texts = [""]

        # Tabular features: use the row on pred_date with the most data (first)
        tab_idx = on_date["feat_idx"].iloc[0]
        year = on_date["year"].iloc[0]

        rows.append({
            "parsed_date": pred_date,
            "year":        year,
            "label":       target_label,
            "texts":       texts,       # list of strings
            "tab_idx":     tab_idx,     # index into df_feat
        })

    result = pd.DataFrame(rows)
    logger.info("Date-grouped: %d unique dates  (from %d articles)", len(result), len(df))

    # Log class distribution per split
    for split, years in [("Train", TRAIN_YEARS), ("Val", VAL_YEARS), ("Test", TEST_YEARS)]:
        sub = result[result["year"].isin(years)]
        dist = Counter(sub["label"].tolist())
        logger.info("  %s (%d dates): %s", split, len(sub),
                    {CLASS_NAMES[k]: v for k, v in sorted(dist.items())})

    return result


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Each item: up to max_articles texts for one prediction date.
    Articles are padded to max_articles with empty strings.
    """

    def __init__(
        self,
        grouped_df: pd.DataFrame,
        tabular: np.ndarray,       # full tabular matrix indexed by tab_idx
        tokenizer_name: str,
        max_length: int = 128,     # shorter than single-article (multiple articles per date)
        max_articles: int = 5,
    ):
        self.df          = grouped_df.reset_index(drop=True)
        self.tabular     = tabular.astype(np.float32)
        self.max_articles = max_articles
        self.max_length  = max_length
        self.tokenizer   = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        texts  = list(row["texts"])[:self.max_articles]
        # Pad with empty string if fewer articles than max
        while len(texts) < self.max_articles:
            texts.append("")
        n_real = min(len(row["texts"]), self.max_articles)

        # Tokenise all articles
        input_ids_list, attn_mask_list = [], []
        for text in texts:
            enc = self.tokenizer(
                str(text),
                truncation  = True,
                padding     = "max_length",
                max_length  = self.max_length,
                return_tensors = "pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_mask_list.append(enc["attention_mask"].squeeze(0))

        # Stack: (max_articles, max_length)
        input_ids   = torch.stack(input_ids_list)
        attention_mask = torch.stack(attn_mask_list)

        # Article mask: 1 for real articles, 0 for padding
        article_mask = torch.zeros(self.max_articles, dtype=torch.float)
        article_mask[:n_real] = 1.0

        tab_idx = int(row["tab_idx"])
        return {
            "input_ids":     input_ids,       # (A, L)
            "attention_mask": attention_mask,  # (A, L)
            "article_mask":  article_mask,     # (A,)  — 1 for real, 0 for pad
            "tabular":       torch.from_numpy(self.tabular[tab_idx]),
            "label":         torch.tensor(int(row["label"]), dtype=torch.long),
        }


# ─── Model ────────────────────────────────────────────────────────────────────

class WindowBertModel(pl.LightningModule):
    """
    Attention-pooled BERT over a window of articles + tabular fusion.

    For each prediction date:
      1. Encode each article through BERT → CLS embedding (B, A, 768)
      2. Attention-pool article embeddings (weighted by a learned scalar per article)
         → (B, 768)
      3. Project → (B, 128)
      4. Tabular tower → (B, 64)
      5. Fusion → (B, 3)
    """

    def __init__(
        self,
        n_tabular: int,
        max_articles: int = 5,
        num_classes: int = 3,
        text_proj_dim: int = 128,
        tab_proj_dim: int = 64,
        dropout: float = 0.3,
        lr: float = 2e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        class_weights: Optional[List[float]] = None,
        n_trainable_layers: int = 4,
        label_smoothing: float = 0.0,
        backbone: str = "DeepPavlov/rubert-base-cased",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.warmup_steps   = warmup_steps
        self.max_articles   = max_articles

        # ── BERT backbone ─────────────────────────────────────────────────────
        self.bert = AutoModel.from_pretrained(backbone)
        for param in self.bert.parameters():
            param.requires_grad = False
        total_layers = len(self.bert.encoder.layer)
        for layer in self.bert.encoder.layer[total_layers - n_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.bert.config.hidden_size  # 768

        # ── Attention pooling over articles ───────────────────────────────────
        # Learns which articles in the window are most relevant
        self.article_attn = nn.Linear(hidden_size, 1)

        # ── Text projection ───────────────────────────────────────────────────
        self.text_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, text_proj_dim),
            nn.ReLU(),
        )

        # ── Tabular tower ─────────────────────────────────────────────────────
        self.tab_bn   = nn.BatchNorm1d(n_tabular)
        self.tab_proj = nn.Sequential(
            nn.Linear(n_tabular, tab_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # ── Fusion head ───────────────────────────────────────────────────────
        fusion_dim = text_proj_dim + tab_proj_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        cw = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(
            weight=cw, label_smoothing=label_smoothing, reduction="mean"
        )

        self._val_preds  = []; self._val_labels  = []
        self._test_preds = []; self._test_labels = []

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Window model — trainable params: %d", n_train)

    def _encode_window(self, input_ids, attention_mask, article_mask):
        """
        input_ids      : (B, A, L)
        attention_mask : (B, A, L)
        article_mask   : (B, A)   — 1 for real, 0 for padding articles
        Returns        : (B, 768)  — attention-pooled CLS embeddings
        """
        B, A, L = input_ids.shape

        # Flatten batch × articles for BERT forward pass
        input_ids_flat      = input_ids.view(B * A, L)
        attention_mask_flat = attention_mask.view(B * A, L)

        bert_out  = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        cls_flat  = bert_out.last_hidden_state[:, 0, :]   # (B*A, 768)
        cls       = cls_flat.view(B, A, -1)                # (B, A, 768)

        # Attention weights — mask padding articles
        attn_scores = self.article_attn(cls).squeeze(-1)   # (B, A)
        attn_scores = attn_scores.masked_fill(article_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)      # (B, A)

        # Weighted sum
        pooled = (attn_weights.unsqueeze(-1) * cls).sum(dim=1)  # (B, 768)
        return pooled

    def forward(self, input_ids, attention_mask, article_mask, tabular):
        pooled   = self._encode_window(input_ids, attention_mask, article_mask)
        text_emb = self.text_proj(pooled)

        tab_norm = self.tab_bn(tabular)
        tab_emb  = self.tab_proj(tab_norm)

        combined = torch.cat([text_emb, tab_emb], dim=1)
        return self.fusion(combined)

    def _step(self, batch, stage):
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["article_mask"], batch["tabular"],
        )
        loss  = self.loss_fn(logits, batch["label"])
        preds = logits.argmax(dim=1)
        acc   = (preds == batch["label"]).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss, preds.cpu().numpy(), batch["label"].cpu().numpy()

    def training_step(self, batch, _):
        loss, _, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, _):
        loss, preds, labels = self._step(batch, "val")
        self._val_preds.append(preds); self._val_labels.append(labels)
        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            p = np.concatenate(self._val_preds)
            l = np.concatenate(self._val_labels)
            f1w = f1_score(l, p, average="weighted", zero_division=0)
            f1m = f1_score(l, p, average="macro",    zero_division=0)
            self.log("val_f1",       f1w, prog_bar=True)
            self.log("val_f1_macro", f1m, prog_bar=True)
            logger.info(
                "Val  — weighted F1: %.4f  macro F1: %.4f\n%s",
                f1w, f1m,
                classification_report(l, p, target_names=CLASS_NAMES, zero_division=0),
            )
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, _):
        loss, preds, labels = self._step(batch, "test")
        self._test_preds.append(preds); self._test_labels.append(labels)
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
            logger.info(
                "Test — weighted F1: %.4f  macro F1: %.4f  acc: %.4f\n%s",
                f1w, f1m, acc,
                classification_report(l, p, target_names=CLASS_NAMES, zero_division=0),
            )
        self._test_preds.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        backbone_params = [p for p in self.bert.parameters() if p.requires_grad]
        head_params = (
            list(self.article_attn.parameters())
            + list(self.text_proj.parameters())
            + list(self.tab_bn.parameters())
            + list(self.tab_proj.parameters())
            + list(self.fusion.parameters())
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr * 0.1},
                {"params": head_params,     "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps  = self.warmup_steps,
            num_training_steps = self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Window-aggregated BERT+Tabular model")
    parser.add_argument("--dataset",       type=str, default=str(DATA_DIR / "cbr_combined_dataset_t015.csv"))
    parser.add_argument("--window-days",   type=int,   default=7,    help="Look-back window in days")
    parser.add_argument("--max-articles",  type=int,   default=5,    help="Max articles per window")
    parser.add_argument("--max-length",    type=int,   default=128,  help="Token length per article")
    parser.add_argument("--batch-size",    type=int,   default=32,   help="Smaller batch: A×L tokens per item")
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--n-trainable",   type=int,   default=4)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--patience",      type=int,   default=5)
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42, workers=True)

    # ── Load and preprocess data ───────────────────────────────────────────────
    logger.info("Loading %s …", args.dataset)
    df_raw = pd.read_csv(args.dataset)

    tab_cache = DATA_DIR / f"tabular_features_{Path(args.dataset).stem.split('_')[-1]}.parquet"
    if tab_cache.exists() and len(pd.read_parquet(tab_cache, columns=["date"])) == len(df_raw):
        logger.info("Loading tabular cache: %s", tab_cache)
        df_feat = pd.read_parquet(tab_cache)
    else:
        logger.info("Computing tabular features…")
        df_feat = build_tabular_features(df_raw, date_col="date")
        if "cleaned_text" in df_raw.columns:
            df_feat = extract_text_features(df_feat, text_col="cleaned_text")
        df_feat.to_parquet(tab_cache, index=False)

    # ── Build date-grouped dataset ─────────────────────────────────────────────
    grouped = build_date_grouped_dataset(
        df_raw, df_feat,
        window_days  = args.window_days,
        max_articles = args.max_articles,
    )

    # ── Tabular features matrix ────────────────────────────────────────────────
    feat_cols = get_feature_columns(df_feat)
    is_cbr = (df_raw["source"] == "cbr_press_releases").astype(float).values \
        if "source" in df_raw.columns else np.zeros(len(df_raw))
    X_tab = np.hstack([
        df_feat[feat_cols].values.astype(np.float32),
        is_cbr.reshape(-1, 1),
    ])

    # ── Temporal split ─────────────────────────────────────────────────────────
    train_mask = grouped["year"].isin(TRAIN_YEARS).values
    val_mask   = grouped["year"].isin(VAL_YEARS).values
    test_mask  = grouped["year"].isin(TEST_YEARS).values

    # Mean-impute NaN using training stats
    train_tab_idxs = grouped[train_mask]["tab_idx"].values
    col_means = np.nanmean(X_tab[train_tab_idxs], axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_idx = np.where(np.isnan(X_tab))
    X_tab[nan_idx] = np.take(col_means, nan_idx[1])

    # ── Class weights ──────────────────────────────────────────────────────────
    y_train = grouped[train_mask]["label"].tolist()
    counts  = Counter(y_train)
    total   = sum(counts.values())
    n_cls   = len(LABEL_MAP)
    class_weights = [total / (n_cls * counts.get(i, 1)) for i in range(n_cls)]
    logger.info("Train label dist: %s", {CLASS_NAMES[k]: v for k, v in counts.items()})
    logger.info("Class weights   : %s", [f"{w:.3f}" for w in class_weights])

    def _make_ds(mask):
        return WindowDataset(
            grouped_df   = grouped[mask],
            tabular      = X_tab,
            tokenizer_name = TOKENIZER,
            max_length   = args.max_length,
            max_articles = args.max_articles,
        )

    train_ds = _make_ds(train_mask)
    val_ds   = _make_ds(val_mask)
    test_ds  = _make_ds(test_mask)

    logger.info("Train: %d  Val: %d  Test: %d", len(train_ds), len(val_ds), len(test_ds))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WindowBertModel(
        n_tabular          = X_tab.shape[1],
        max_articles       = args.max_articles,
        num_classes        = n_cls,
        dropout            = args.dropout,
        lr                 = args.lr,
        weight_decay       = 0.1,
        warmup_steps       = 100,
        class_weights      = class_weights,
        n_trainable_layers = args.n_trainable,
        label_smoothing    = args.label_smoothing,
        backbone           = TOKENIZER,
    )

    # ── Callbacks + logger ────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(CKPT_DIR),
        filename   = "window-{epoch:02d}-{val_f1:.4f}",
        monitor    = "val_f1",
        mode       = "max",
        save_top_k = 1,
    )
    early_cb = EarlyStopping(monitor="val_f1", patience=args.patience, mode="max", verbose=True)

    try:
        mlflow_logger = MLFlowLogger(
            experiment_name = "cbr_news_window",
            tracking_uri    = "http://127.0.0.1:5050",
            log_model       = False,
        )
        mlflow_logger.log_hyperparams(vars(args))
        use_mlflow = True
    except Exception as e:
        logger.warning("MLflow unavailable: %s", e)
        use_mlflow = False

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs          = args.epochs,
        accelerator         = "auto",
        devices             = 1,
        precision           = "16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks           = [ckpt_cb, early_cb],
        logger              = mlflow_logger if use_mlflow else True,
        gradient_clip_val   = 1.0,
        log_every_n_steps   = 10,
        enable_progress_bar = True,
    )

    logger.info("Training window-aggregated model…")
    trainer.fit(model, train_dl, val_dl)

    logger.info("Testing on best checkpoint…")
    test_results = trainer.test(model, test_dl, ckpt_path="best", verbose=False)
    if test_results:
        tr = test_results[0]
        logger.info(
            "Final — test_f1=%.4f  test_macro=%.4f  test_acc=%.4f",
            tr.get("test_f1", 0), tr.get("test_f1_macro", 0), tr.get("test_acc", 0),
        )


if __name__ == "__main__":
    main()
