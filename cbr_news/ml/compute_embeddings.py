"""
Compute BERT embeddings and financial sentiment scores for the combined dataset.

Run once; results saved to:
  data/embeddings.npy          – float32 array (N, hidden_size)
  data/embeddings_meta.parquet – date, target, sentiment_pos/neg/neu, embed_idx

Usage:
    python cbr_news/ml/compute_embeddings.py [--batch-size 64] [--no-sentiment]

The script supports resuming: if embeddings.npy already exists and has fewer rows
than the dataset, it continues from where it stopped.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"

EMBEDDING_MODEL = "DeepPavlov/rubert-base-cased"
SENTIMENT_MODEL = "blanchefort/rubert-base-cased-sentiment"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── embedding helpers ────────────────────────────────────────────────────────

def _embed_batch(batch_texts, tokenizer, model, max_length, device):
    enc = tokenizer(
        batch_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()  # (B, H)
    return cls


def compute_embeddings(
    texts, tokenizer, model, batch_size=64, max_length=256, device="cpu", start_idx=0
):
    """Return np.ndarray (N, H) for texts[start_idx:]."""
    model.eval()
    chunks = range(start_idx, len(texts), batch_size)
    all_embeds = []
    for s in tqdm(chunks, desc="Embeddings", total=len(chunks)):
        batch = texts[s: s + batch_size]
        all_embeds.append(_embed_batch(batch, tokenizer, model, max_length, device))
    return np.vstack(all_embeds) if all_embeds else np.empty((0, model.config.hidden_size))


# ─── sentiment helpers ────────────────────────────────────────────────────────

def _parse_sentiment(pred):
    """pred: list[{'label': str, 'score': float}] → dict."""
    if isinstance(pred, list):
        d = {x["label"].lower(): x["score"] for x in pred}
    else:
        d = {pred["label"].lower(): pred["score"]}
    return {
        "sentiment_pos": d.get("positive", d.get("pos", 0.0)),
        "sentiment_neg": d.get("negative", d.get("neg", 0.0)),
        "sentiment_neu": d.get("neutral", d.get("neu", d.get("skip", 0.0))),
    }


def compute_sentiment(texts, sent_pipeline, batch_size=64, start_idx=0):
    """Return DataFrame with sentiment_pos/neg/neu for texts[start_idx:]."""
    results = []
    for s in tqdm(range(start_idx, len(texts), batch_size), desc="Sentiment"):
        batch = texts[s: s + batch_size]
        preds = sent_pipeline(batch, truncation=True, max_length=512)
        for p in preds:
            results.append(_parse_sentiment(p))
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["sentiment_pos", "sentiment_neg", "sentiment_neu"]
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Skip sentiment computation")
    parser.add_argument("--dataset", type=str,
                        default=str(DATA_DIR / "cbr_combined_dataset.csv"))
    args = parser.parse_args()

    emb_path  = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "embeddings_meta.parquet"

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    df = df.reset_index(drop=True)
    texts = df["cleaned_text"].fillna("").tolist()
    n = len(texts)
    logger.info(f"Total rows: {n}")

    # ── Resume support ────────────────────────────────────────────────────────
    start_idx = 0
    existing_embeds = None
    if emb_path.exists():
        existing_embeds = np.load(emb_path)
        start_idx = len(existing_embeds)
        logger.info(f"Resuming from row {start_idx} (already have {start_idx} embeddings)")

    # ── Compute embeddings ────────────────────────────────────────────────────
    if start_idx < n:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} → {DEVICE}")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)

        new_embeds = compute_embeddings(
            texts, tokenizer, model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=DEVICE,
            start_idx=start_idx,
        )

        if existing_embeds is not None:
            all_embeds = np.vstack([existing_embeds, new_embeds])
        else:
            all_embeds = new_embeds

        np.save(emb_path, all_embeds.astype(np.float32))
        logger.info(f"Saved embeddings {all_embeds.shape} → {emb_path}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    else:
        logger.info("Embeddings already complete, skipping.")

    # ── Compute sentiment ─────────────────────────────────────────────────────
    sent_start = 0
    existing_sent = None

    if not args.no_sentiment:
        if meta_path.exists():
            existing_meta = pd.read_parquet(meta_path)
            if "sentiment_pos" in existing_meta.columns:
                sent_start = len(existing_meta)
                existing_sent = existing_meta[["sentiment_pos", "sentiment_neg", "sentiment_neu"]]
                logger.info(f"Resuming sentiment from row {sent_start}")

        if sent_start < n:
            logger.info(f"Loading sentiment model: {SENTIMENT_MODEL}")
            try:
                sent_pipe = pipeline(
                    "text-classification",
                    model=SENTIMENT_MODEL,
                    device=0 if DEVICE == "cuda" else -1,
                    return_all_scores=True,
                )
                new_sent = compute_sentiment(
                    texts, sent_pipe,
                    batch_size=args.batch_size,
                    start_idx=sent_start,
                )
                if existing_sent is not None:
                    all_sent = pd.concat(
                        [existing_sent.reset_index(drop=True), new_sent.reset_index(drop=True)],
                        ignore_index=True,
                    )
                else:
                    all_sent = new_sent
                logger.info(f"Computed {len(all_sent)} sentiment scores")
            except Exception as e:
                logger.warning(f"Sentiment model failed ({e}), skipping sentiment.")
                all_sent = pd.DataFrame(
                    np.zeros((n, 3)),
                    columns=["sentiment_pos", "sentiment_neg", "sentiment_neu"],
                )
        else:
            logger.info("Sentiment already complete, skipping.")
            all_sent = existing_sent.reset_index(drop=True) if existing_sent is not None else None
    else:
        logger.info("--no-sentiment: skipping sentiment computation.")
        all_sent = pd.DataFrame(
            np.zeros((n, 3)),
            columns=["sentiment_pos", "sentiment_neg", "sentiment_neu"],
        )

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta_cols = ["date", "target"]
    if "source" in df.columns:
        meta_cols = ["source"] + meta_cols
    meta = df[meta_cols].copy().reset_index(drop=True)
    meta["embed_idx"] = np.arange(n)

    if all_sent is not None:
        for col in ["sentiment_pos", "sentiment_neg", "sentiment_neu"]:
            meta[col] = all_sent[col].values if col in all_sent.columns else 0.0

    meta.to_parquet(meta_path, index=False)
    logger.info(f"Saved metadata {meta.shape} → {meta_path}")
    logger.info("Done. Summary:")
    logger.info(f"  embeddings : {emb_path}  {np.load(emb_path).shape}")
    logger.info(f"  metadata   : {meta_path}  {meta.shape}")


if __name__ == "__main__":
    main()
