"""
Article-level scorer for key rate direction (weak supervision).

Each article published in the N-day window before a CBR meeting inherits
the meeting's decision as a weak label:
  cut=0  hold=1  hike=2

This gives ~3 000 training examples instead of 60, enabling proper BERT
fine-tuning. The trained model outputs P(cut/hold/hike) per article and
is saved in JointBertTabularModel format so it can be plugged directly
into:
  train_meeting.py --sklearn --daily-model <ckpt>
  inference.py     --daily-model <ckpt>

Temporal split mirrors train_meeting.py:
  train : articles from 2014–2021 meetings  (~2 400 articles)
  val   : articles from 2022–2023 meetings  (~ 700 articles)
  test  : articles from 2024–2025 meetings  (~ 600 articles, eval only)

Usage:
    PYTHONPATH=. python cbr_news/ml/train_article_scorer.py
    PYTHONPATH=. python cbr_news/ml/train_article_scorer.py --n-trainable 4 --lr 2e-5 --epochs 10
"""

import argparse
import logging
import sys
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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

_PROJECT_ROOT = str(Path(__file__).parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cbr_news.ml.train_joint import JointBertTabularModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parents[2] / "data"
CKPT_DIR   = Path(__file__).parents[2] / "checkpoints" / "article_scorer"
MEETING_DS = DATA_DIR / "meeting_dataset.parquet"

TOKENIZER   = "DeepPavlov/rubert-base-cased"
CLASS_NAMES = ["cut", "hold", "hike"]
NUM_CLASSES = 3

TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END   = pd.Timestamp("2023-12-31")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ArticleScorerDataset(Dataset):
    """
    Flat list of (text, label) pairs — one per article.
    Label is the meeting decision that the article precedes (weak supervision).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name: str = TOKENIZER,
        max_length: int = 256,
    ):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

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
        # Dummy tabular (zeros) — scorer uses text tower only.
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "tabular":        torch.zeros(1, dtype=torch.float),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
            "sample_weight":  torch.tensor(1.0, dtype=torch.float),
        }


# ─── Expand meeting dataset to per-article rows ───────────────────────────────

def expand_meetings_to_articles(
    df: pd.DataFrame,
    max_articles_per_meeting: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Flatten meeting_dataset.parquet into a per-article dataframe.
    Each article inherits its meeting's label (weak supervision).

    max_articles_per_meeting caps articles per meeting with a random sample
    (reproducible via seed) so that recent high-volume years don't dominate.

    Returns DataFrame with columns: text, label, direction, meeting_date, split.
    """
    rng = np.random.default_rng(seed)
    df  = df.copy()
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])

    rows = []
    for _, row in df.iterrows():
        texts     = list(row["texts"])
        label     = int(row["label"])
        md        = row["meeting_date"]
        direction = row["direction"]

        if md <= TRAIN_END:
            split = "train"
        elif md <= VAL_END:
            split = "val"
        else:
            split = "test"

        # Random sample to cap — avoids always picking the first N articles
        if max_articles_per_meeting > 0 and len(texts) > max_articles_per_meeting:
            idx   = rng.choice(len(texts), max_articles_per_meeting, replace=False)
            texts = [texts[i] for i in sorted(idx)]

        for text in texts:
            rows.append({
                "text":         str(text),
                "label":        label,
                "direction":    direction,
                "meeting_date": md,
                "split":        split,
            })

    result = pd.DataFrame(rows)

    logger.info("Expanded %d meetings → %d article-label pairs  (cap=%d per meeting)",
                len(df), len(result), max_articles_per_meeting)
    for sp in ["train", "val", "test"]:
        sub = result[result["split"] == sp]
        n_meetings = sub["meeting_date"].nunique()
        dist = sub["direction"].value_counts().to_dict()
        logger.info("  %-5s: %4d articles  %2d meetings  avg %.0f/meeting  %s",
                    sp, len(sub), n_meetings, len(sub) / max(n_meetings, 1), dist)
    return result


# ─── Training wrapper ─────────────────────────────────────────────────────────

def train_article_scorer(
    df_articles: pd.DataFrame,
    tokenizer_name: str = TOKENIZER,
    max_length: int = 256,
    batch_size: int = 32,
    lr: float = 2e-5,
    n_trainable: int = 2,
    dropout: float = 0.3,
    weight_decay: float = 0.1,
    max_epochs: int = 10,
    patience: int = 3,
    seed: int = 42,
) -> str:
    """
    Fine-tune JointBertTabularModel with weak article labels.
    Returns path to best checkpoint.
    """
    pl.seed_everything(seed, workers=True)

    train_df = df_articles[df_articles["split"] == "train"]
    val_df   = df_articles[df_articles["split"] == "val"]

    # Class weights (balanced on train)
    counts  = Counter(train_df["label"].tolist())
    total   = sum(counts.values())
    cw      = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
    logger.info("Class weights: cut=%.2f  hold=%.2f  hike=%.2f", *cw)

    train_ds = ArticleScorerDataset(
        train_df["text"].tolist(), train_df["label"].tolist(),
        tokenizer_name, max_length,
    )
    val_ds = ArticleScorerDataset(
        val_df["text"].tolist(), val_df["label"].tolist(),
        tokenizer_name, max_length,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # n_tabular=1 (dummy) to keep JointBertTabularModel happy
    model = JointBertTabularModel(
        n_tabular          = 1,
        num_classes        = NUM_CLASSES,
        text_proj_dim      = 128,
        tab_proj_dim       = 4,          # minimal tab tower (not used in inference)
        fusion_hidden_dim  = 64,
        dropout            = dropout,
        lr                 = lr,
        weight_decay       = weight_decay,
        warmup_steps       = max(50, len(train_dl) // 2),
        class_weights      = cw,
        n_trainable_layers = n_trainable,
        backbone           = tokenizer_name,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(CKPT_DIR),
        filename   = "scorer-{epoch:02d}-{val_f1_macro:.4f}",
        monitor    = "val_f1_macro",
        mode       = "max",
        save_top_k = 1,
    )
    early_cb = EarlyStopping(
        monitor  = "val_f1_macro",
        patience = patience,
        mode     = "max",
    )

    precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    trainer   = pl.Trainer(
        max_epochs          = max_epochs,
        accelerator         = "auto",
        devices             = 1,
        precision           = precision,
        callbacks           = [ckpt_cb, early_cb],
        logger              = False,
        enable_progress_bar = True,
        gradient_clip_val   = 1.0,
        log_every_n_steps   = 10,
    )

    logger.info(
        "Training article scorer:  train=%d  val=%d  n_trainable=%d  lr=%.0e",
        len(train_ds), len(val_ds), n_trainable, lr,
    )
    trainer.fit(model, train_dl, val_dl)

    best_ckpt = ckpt_cb.best_model_path
    logger.info("Best checkpoint: %s", best_ckpt)

    # ── Evaluate on val and test ──────────────────────────────────────────
    _evaluate(best_ckpt, df_articles, tokenizer_name, max_length, batch_size)

    return best_ckpt


def _evaluate(
    ckpt_path: str,
    df_articles: pd.DataFrame,
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
):
    model  = JointBertTabularModel.load_from_checkpoint(
        ckpt_path, map_location="cpu", strict=False
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    for split in ["val", "test"]:
        sub = df_articles[df_articles["split"] == split]
        if sub.empty:
            continue
        ds = ArticleScorerDataset(
            sub["text"].tolist(), sub["label"].tolist(), tokenizer_name, max_length,
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dl:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["tabular"].to(device),
                )
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(batch["label"].tolist())

        f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        logger.info("\n=== %s split (article-level, weak labels) ===", split.upper())
        logger.info("Macro F1 = %.4f", f1m)
        logger.info(
            "\n%s",
            classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0),
        )


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train article-level key rate scorer (weak labels)")
    ap.add_argument("--dataset",      type=str, default=str(MEETING_DS))
    ap.add_argument("--max-articles", type=int, default=50,
                    help="Max articles per meeting sampled (default 50; 0=all)")
    ap.add_argument("--max-length",   type=int, default=256)
    ap.add_argument("--batch-size",   type=int, default=32)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--n-trainable",  type=int, default=2,
                    help="BERT encoder layers to fine-tune (top N)")
    ap.add_argument("--dropout",      type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--epochs",       type=int, default=10)
    ap.add_argument("--patience",     type=int, default=3)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        logger.error("Meeting dataset not found: %s", ds_path)
        logger.error("Run: PYTHONPATH=. python cbr_news/ml/prepare_meeting_dataset.py")
        return

    logger.info("Loading meeting dataset: %s", ds_path)
    df = pd.read_parquet(ds_path)

    df_articles = expand_meetings_to_articles(
        df,
        max_articles_per_meeting = args.max_articles,
        seed                     = args.seed,
    )

    best_ckpt = train_article_scorer(
        df_articles,
        tokenizer_name = TOKENIZER,
        max_length     = args.max_length,
        batch_size     = args.batch_size,
        lr             = args.lr,
        n_trainable    = args.n_trainable,
        dropout        = args.dropout,
        weight_decay   = args.weight_decay,
        max_epochs     = args.epochs,
        patience       = args.patience,
        seed           = args.seed,
    )

    print()
    print("=" * 60)
    print("Article scorer trained. Use it as Stage 1 in Stage 2:")
    print()
    print(f"  PYTHONPATH=. python cbr_news/ml/train_meeting.py \\")
    print(f"      --sklearn --daily-model {best_ckpt}")
    print()
    print(f"  PYTHONPATH=. python cbr_news/ml/inference.py \\")
    print(f"      --meeting-date 2024-12-20 \\")
    print(f"      --daily-model {best_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
