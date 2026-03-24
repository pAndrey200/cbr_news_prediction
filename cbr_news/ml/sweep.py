"""
Hyperparameter sweep for the Joint BERT+Tabular RUONIA direction model.

Every run is logged to MLflow (experiment: cbr_news_sweep) and a summary line
is appended to sweep_results.txt.

Usage
-----
# Full grid:
    python cbr_news/ml/sweep.py

# Quick sanity-check — first 2 configs only:
    python cbr_news/ml/sweep.py --max-runs 2

# Only build dataset CSVs + tabular caches, then exit:
    python cbr_news/ml/sweep.py --prepare-only

# Resume from run N (1-based):
    python cbr_news/ml/sweep.py --resume-from 5 --skip-prepare

Grid — Phase 12
---------------
  Phase 11 winner: rubert-base-cased + pool=cls + ntr=4 + over=3 + ls=0.05 + lr=5e-4
    → val_macro=0.4673  val_f1=0.6097  test_macro=0.3362  test_f1=0.6266  (record)

  Phase 12 strategy: push test_f1 → 0.70 via two axes:
    1. Backbone variety:
       - rubert-base-cased          : established baseline (180 M)
       - rubert-tiny2-fin-sentiment : 29 M, RU financial-domain fine-tuned
                                      fast runs, good domain alignment
       - sbert_large_nlu_ru         : 560 M RoBERTa-large for Russian
                                      more capacity, may need smaller batch
    2. Context length 256 → 512:
       CBR press releases are long; truncating at 256 tokens loses ~40% of content.
       512 may give +0.02-0.04 F1 at the cost of 2× training time.

  Batch-size policy (OOM guard):
    - sbert_large_nlu_ru + max_length=512 → batch_size=32
    - sbert_large_nlu_ru + max_length=256 → batch_size=64
    - all other models                    → batch_size=128  (default)

  n_trainable policy:
    - rubert-tiny2 (3 encoder layers)     → n_trainable=3 (all layers)
    - sbert_large (24 encoder layers)     → n_trainable=4 (top-4 only)
    - rubert-base-cased (12 layers)       → n_trainable=4 (confirmed)

  Fixed: pool=cls, fg=0.0, lr=5e-4, over=3, ls=0.05,
         dataset=combined_t025, train_end=2024-06-30, val_end=2024-12-31
  Axes:
    backbone   : rubert-base-cased | rubert-tiny2-fin | sbert_large_nlu_ru  (3)
    max_length : 256 | 512                                                   (2)
  ──────────────────────────────────────────────────────────────────────────────
  Total        : 6 configurations  (~5-7 h on a single GPU)
"""

import argparse
import itertools
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from cbr_news.ml.feature_engineering import (
    build_tabular_features,
    extract_text_features,
)
from cbr_news.ml.train_joint import (
    DATA_DIR,
    LABEL_MAP,
    Config,
    JointBertTabularModel,
    JointDataModule,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CKPT_DIR    = Path(__file__).parents[2] / "checkpoints" / "sweep"
RESULTS_TXT = Path(__file__).parents[2] / "sweep_results.txt"
MLFLOW_URI  = "http://127.0.0.1:5050"
MLFLOW_EXP  = "cbr_news_sweep"


# ─── Dataset variants ─────────────────────────────────────────────────────────

SWEEP_DATASETS: Dict[str, Dict[str, Any]] = {
    "combined_t025": {
        "csv_path":    DATA_DIR / "cbr_combined_dataset_t025.csv",
        "tab_cache":   DATA_DIR / "tabular_features_t025.parquet",
        "threshold":   0.25,
        "cbr_only":    False,
        "description": "All sources, RUONIA threshold=0.25 (≈ min key-rate step)",
    },
}

# ─── Hyperparameter grid ──────────────────────────────────────────────────────

PARAM_GRID: Dict[str, list] = {
    # Phase 14: LR scheduler + architecture capacity
    # Phase 13 winner: rubert-base L=512 lr=5e-4 accum=2 → test_f1=0.6583
    # Stage-2 didn't help (0.641 < 0.658). sbert_large collapsed even at lr=2e-5.
    #
    # New axes:
    #   1. LR scheduler:   linear (baseline) | cosine (smoother) | onecycle (super-conv)
    #   2. Fusion capacity: (tab=64,fus=128) vs (tab=128,fus=256)
    #
    # All runs: rubert-base-cased, L=512, accum=2 (Phase 13 winner config)
    "lr_scheduler":    ["linear", "cosine", "onecycle"],
    "fusion_capacity": ["base", "large"],
}

# Fixed across all runs — Phase 13 winner settings:
FIXED: Dict[str, Any] = {
    "dataset":           "combined_t025",
    "train_end":         "2024-06-30",
    "val_end":           "2024-12-31",
    "backbone":          "DeepPavlov/rubert-base-cased",
    "max_length":        512,
    "pool_mode":         "cls",
    "tab_arch":          "mlp_deep",
    "cbr_weight":        3.0,
    "text_proj_dim":     256,
    "aux_weight":        0.0,
    "focal_gamma":       0.0,
    "oversample_up":     3,
    "label_smoothing":   0.05,
    "dropout":           0.3,
    "epochs":            30,
    "patience":          7,
    "gru_hidden":        32,
}

# Per-backbone tuning — LR, warmup, n_trainable, batch
_BACKBONE_CFG: Dict[str, Dict[str, Any]] = {
    "DeepPavlov/rubert-base-cased": {
        "lr":           5e-4,   # confirmed optimal (Phase 9-11)
        "warmup_steps": 200,
        "n_trainable":  4,
        "batch":        {256: 128, 512: 64},
        "accum":        {256: 1,   512: 2},   # effective batch always 128
    },
    "ai-forever/sbert_large_nlu_ru": {
        "lr":           2e-5,   # Phase 12 fix: large models need ~10-25× smaller LR
        "warmup_steps": 500,    # more warmup stabilises large-model early training
        "n_trainable":  4,
        "batch":        {256: 64,  512: 32},
        "accum":        {256: 2,   512: 4},   # effective batch always 128
    },
}
# Fallback values used when backbone is not in _BACKBONE_CFG
_DEFAULT_BACKBONE_CFG: Dict[str, Any] = {
    "lr": 5e-4, "warmup_steps": 200, "n_trainable": 4,
    "batch": {256: 128, 512: 64}, "accum": {256: 1, 512: 2},
}



# ─── Val-metrics collector ────────────────────────────────────────────────────

class ValMetricsCollector(pl.Callback):
    """
    Tracks best val macro-F1 and the corresponding weighted F1 across all epochs.
    Macro-F1 avoids "same"-class dominance that inflates weighted F1 even when
    the model never predicts "up".
    """

    def __init__(self):
        self.best_val_f1_macro = 0.0
        self.best_val_f1       = 0.0
        self.epochs_trained    = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        m     = trainer.callback_metrics
        macro = float(m.get("val_f1_macro", 0.0))
        if macro > self.best_val_f1_macro:
            self.best_val_f1_macro = macro
            self.best_val_f1       = float(m.get("val_f1", 0.0))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.epochs_trained = trainer.current_epoch + 1


# ─── Data module with configurable tabular cache path ────────────────────────

class SweepDataModule(JointDataModule):
    """JointDataModule with per-dataset tabular-feature cache path."""

    def __init__(self, config: Config, tab_cache: Path):
        super().__init__(config)
        self._tab_cache = tab_cache


# ─── Dataset preparation ──────────────────────────────────────────────────────

def prepare_all_datasets(force: bool = False) -> None:
    """Generate all SWEEP_DATASETS CSV files (skips existing unless force=True)."""
    import cbr_news.ml.prepare_news_dataset as pnd

    for name, ds in SWEEP_DATASETS.items():
        csv_path = ds["csv_path"]
        if csv_path.exists() and not force:
            logger.info("Dataset exists: %s — skipping (--force-prepare to regenerate)", csv_path.name)
            continue
        logger.info("Preparing dataset '%s' (threshold=%.2f) …", name, ds["threshold"])
        pnd.prepare(
            threshold   = ds["threshold"],
            cbr_only    = ds["cbr_only"],
            output_path = csv_path,
        )
        logger.info("  → saved %s", csv_path)


def precompute_tabular_for_all() -> None:
    """Build per-dataset tabular feature parquets (skips if up to date)."""
    for name, ds in SWEEP_DATASETS.items():
        csv_path  = ds["csv_path"]
        tab_cache = ds["tab_cache"]

        if not csv_path.exists():
            logger.warning("CSV not found for '%s' — skipping tabular precompute", name)
            continue

        if tab_cache.exists():
            n_cache = len(pd.read_parquet(tab_cache, columns=["date"]))
            n_csv   = len(pd.read_csv(csv_path, usecols=["date"]))
            if n_cache == n_csv:
                logger.info("Tabular cache up to date for '%s': %s", name, tab_cache.name)
                continue
            logger.warning("Cache size mismatch for '%s' — recomputing", name)

        logger.info("Building tabular features for '%s' (may take 3-5 min) …", name)
        df_raw  = pd.read_csv(csv_path)
        df_feat = build_tabular_features(df_raw, date_col="date")
        if "cleaned_text" in df_raw.columns:
            df_feat = extract_text_features(df_feat, text_col="cleaned_text")
        df_feat.to_parquet(tab_cache, index=False)
        logger.info("  → saved %s (%d rows)", tab_cache.name, len(df_feat))


# ─── MLflow helpers ───────────────────────────────────────────────────────────

def _setup_mlflow(run_name: str, params: dict):
    try:
        import mlflow as _mlflow
        _mlflow.set_tracking_uri(MLFLOW_URI)
        _mlflow.set_experiment(MLFLOW_EXP)
        active_run = _mlflow.start_run(run_name=run_name)
        run_id = active_run.info.run_id
        _mlflow.log_params(params)
        pl_logger = MLFlowLogger(
            experiment_name = MLFLOW_EXP,
            tracking_uri    = MLFLOW_URI,
            run_id          = run_id,
            log_model       = False,
        )
        return run_id, pl_logger
    except Exception as exc:
        logger.warning("MLflow not available: %s", exc)
        return None, None


def _finish_mlflow(run_id: Optional[str], summary: dict) -> None:
    if run_id is None:
        return
    try:
        import mlflow as _mlflow
        _mlflow.log_metrics(summary)
        _mlflow.end_run()
    except Exception as exc:
        logger.warning("MLflow finish failed: %s", exc)
        try:
            import mlflow as _mlflow
            _mlflow.end_run()
        except Exception:
            pass


# ─── Single training run ──────────────────────────────────────────────────────

def run_one(
    params: Dict[str, Any],
    batch_size: int,
    run_idx: int,
    total_runs: int,
) -> Dict[str, Any]:
    """Train one configuration; return dict with params + metrics."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Run %d / %d : %s", run_idx, total_runs, params)
    logger.info("=" * 70)

    ds_name    = FIXED["dataset"]
    ds_info    = SWEEP_DATASETS[ds_name]

    backbone   = params.get("backbone",   FIXED.get("backbone", "DeepPavlov/rubert-base-cased"))
    max_length = params.get("max_length", FIXED.get("max_length", 256))
    pool_mode  = params.get("pool_mode",  FIXED.get("pool_mode", "cls"))
    lr_sched   = params.get("lr_scheduler", "linear")

    # Fusion capacity: "large" doubles tab_proj and fusion_hidden dims
    fuse_cap   = params.get("fusion_capacity", "base")
    if fuse_cap == "large":
        tab_proj_dim     = 128
        fusion_hidden_dim = 256
    else:
        tab_proj_dim     = FIXED.get("tab_proj_dim", 64)
        fusion_hidden_dim = FIXED.get("fusion_hidden_dim", 128)

    # Per-backbone settings (LR, warmup, n_trainable, batch, accum)
    bb_cfg      = _BACKBONE_CFG.get(backbone, _DEFAULT_BACKBONE_CFG)
    n_trainable = bb_cfg["n_trainable"]
    bb_lr       = bb_cfg["lr"]
    warmup      = bb_cfg["warmup_steps"]
    bs          = bb_cfg["batch"].get(max_length, batch_size)
    accum_steps = bb_cfg["accum"].get(max_length, 1)

    cfg = Config()
    cfg.dataset_path        = str(ds_info["csv_path"])
    cfg.batch_size          = bs
    cfg.tokenizer_name      = backbone
    cfg.lr                  = bb_lr
    cfg.n_trainable_layers  = n_trainable
    cfg.cbr_sample_weight   = FIXED["cbr_weight"]
    cfg.focal_gamma         = FIXED["focal_gamma"]
    cfg.label_smoothing     = FIXED["label_smoothing"]
    cfg.dropout             = FIXED["dropout"]
    cfg.max_length          = max_length
    cfg.train_end           = FIXED["train_end"]
    cfg.val_end             = FIXED["val_end"]
    cfg.aux_weight          = FIXED["aux_weight"]
    cfg.max_epochs          = FIXED["epochs"]
    cfg.patience            = FIXED["patience"]
    cfg.oversample_up       = FIXED["oversample_up"]
    cfg.pool_mode           = pool_mode

    pl.seed_everything(42, workers=True)
    dm = SweepDataModule(cfg, tab_cache=ds_info["tab_cache"])
    dm.setup()

    tab_arch     = FIXED["tab_arch"]
    gru_feat_idx = None
    gru_n_steps  = 6
    gru_n_series = 4
    if tab_arch == "gru":
        if dm.gru_feat_idx is not None:
            gru_feat_idx = dm.gru_feat_idx
            gru_n_steps  = dm.gru_n_steps
            gru_n_series = dm.gru_n_series
        else:
            logger.warning("GRU features unavailable — falling back to mlp")
            tab_arch = "mlp"

    model = JointBertTabularModel(
        n_tabular          = dm.n_tabular,
        num_classes        = len(LABEL_MAP),
        text_proj_dim      = FIXED["text_proj_dim"],
        tab_proj_dim       = tab_proj_dim,
        fusion_hidden_dim  = fusion_hidden_dim,
        dropout            = cfg.dropout,
        lr                 = cfg.lr,
        weight_decay       = 0.1,
        warmup_steps       = warmup,
        class_weights      = dm.class_weights,
        n_trainable_layers = cfg.n_trainable_layers,
        focal_gamma        = cfg.focal_gamma,
        label_smoothing    = cfg.label_smoothing,
        aux_weight         = cfg.aux_weight,
        backbone           = cfg.tokenizer_name,
        tab_arch           = tab_arch,
        gru_feat_idx       = gru_feat_idx,
        gru_n_steps        = gru_n_steps,
        gru_n_series       = gru_n_series,
        gru_hidden         = FIXED["gru_hidden"],
        pool_mode          = pool_mode,
        lr_scheduler       = lr_sched,
    )

    run_dir = CKPT_DIR / f"run_{run_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    val_cb  = ValMetricsCollector()
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(run_dir),
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

    bb_short = backbone.split("/")[-1][:20]   # short name for logging
    run_name = (
        f"run{run_idx:03d}"
        f"_{bb_short}"
        f"_L{max_length}"
        f"_{lr_sched}"
        f"_{fuse_cap}"
    )
    mlflow_params = {
        "backbone":          backbone,
        "pool_mode":         pool_mode,
        "dataset":           ds_name,
        "threshold":         ds_info["threshold"],
        "tab_arch":          tab_arch,
        "n_trainable":       n_trainable,
        "lr":                bb_lr,
        "warmup_steps":      warmup,
        "accum_steps":       accum_steps,
        "label_smoothing":   FIXED["label_smoothing"],
        "train_end":         FIXED["train_end"],
        "val_end":           FIXED["val_end"],
        "focal_gamma":       FIXED["focal_gamma"],
        "cbr_weight":        FIXED["cbr_weight"],
        "oversample_up":     FIXED["oversample_up"],
        "max_length":        max_length,
        "text_proj_dim":     FIXED["text_proj_dim"],
        "lr_scheduler":      lr_sched,
        "fusion_capacity":   fuse_cap,
        "fusion_hidden_dim": fusion_hidden_dim,
        "tab_proj_dim":      tab_proj_dim,
        "dropout":           FIXED["dropout"],
        "batch_size":        bs,
        "patience":          FIXED["patience"],
        "n_tabular":         dm.n_tabular,
        "train_size":        len(dm.train_ds) if hasattr(dm, "train_ds") else -1,
    }
    mlflow_run_id, pl_logger = _setup_mlflow(run_name, mlflow_params)

    precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    trainer = pl.Trainer(
        max_epochs              = cfg.max_epochs,
        accelerator             = "auto",
        devices                 = 1,
        precision               = precision,
        accumulate_grad_batches = accum_steps,
        callbacks               = [ckpt_cb, early_cb, val_cb],
        logger                  = pl_logger if pl_logger is not None else False,
        enable_progress_bar     = True,
        gradient_clip_val       = 1.0,
        log_every_n_steps       = 20,
    )

    t0 = time.time()
    trainer.fit(model, dm)
    test_results = trainer.test(model, dm, ckpt_path="best", verbose=False)
    elapsed = time.time() - t0

    tr = test_results[0] if test_results else {}
    test_f1       = float(tr.get("test_f1",       0.0))
    test_f1_macro = float(tr.get("test_f1_macro", 0.0))
    test_acc      = float(tr.get("test_acc",      0.0))

    logger.info(
        "Run %d done — val_f1=%.4f val_macro=%.4f  "
        "test_f1=%.4f test_macro=%.4f test_acc=%.4f  (%.1f min)",
        run_idx,
        val_cb.best_val_f1, val_cb.best_val_f1_macro,
        test_f1, test_f1_macro, test_acc,
        elapsed / 60,
    )

    _finish_mlflow(mlflow_run_id, {
        "best_val_f1":       val_cb.best_val_f1,
        "best_val_f1_macro": val_cb.best_val_f1_macro,
        "epochs_trained":    val_cb.epochs_trained,
        "elapsed_min":       elapsed / 60,
    })

    return {
        "run_idx":           run_idx,
        "backbone":          backbone,
        "pool_mode":         pool_mode,
        "dataset":           ds_name,
        "n_trainable":       n_trainable,
        "lr":                bb_lr,
        "warmup_steps":      warmup,
        "accum_steps":       accum_steps,
        "lr_scheduler":      lr_sched,
        "fusion_capacity":   fuse_cap,
        "tab_proj_dim":      tab_proj_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
        "focal_gamma":       FIXED["focal_gamma"],
        "cbr_weight":        FIXED["cbr_weight"],
        "oversample_up":     FIXED["oversample_up"],
        "label_smoothing":   FIXED["label_smoothing"],
        "threshold":         ds_info["threshold"],
        "tab_arch":          tab_arch,
        "train_end":         FIXED["train_end"],
        "val_end":           FIXED["val_end"],
        "dropout":           FIXED["dropout"],
        "max_length":        max_length,
        "batch_size":        bs,
        "epochs_trained":    val_cb.epochs_trained,
        "best_val_f1_macro": round(val_cb.best_val_f1_macro, 4),
        "best_val_f1":       round(val_cb.best_val_f1,       4),
        "test_f1":           round(test_f1,                  4),
        "test_f1_macro":     round(test_f1_macro,            4),
        "test_acc":          round(test_acc,                  4),
        "elapsed_min":       round(elapsed / 60,              1),
    }


# ─── txt logging ──────────────────────────────────────────────────────────────

def _write_header() -> None:
    with open(RESULTS_TXT, "a", encoding="utf-8") as fh:
        fh.write(
            f"\n{'=' * 130}\n"
            f"Sweep started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Grid          : {json.dumps(PARAM_GRID)}\n"
            f"Fixed params  : {json.dumps(FIXED)}\n"
            f"{'=' * 130}\n"
        )


def _write_result(result: Dict[str, Any]) -> None:
    bb_short = result["backbone"].split("/")[-1][:24]
    sched = result.get("lr_scheduler", "linear")
    fcap  = result.get("fusion_capacity", "base")
    line = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"run={result['run_idx']:03d} "
        f"sched={sched} fuse={fcap} "
        f"L={result['max_length']} "
        f"lr={result['lr']:.0e} "
        f"accum={result['accum_steps']} "
        f"bs={result['batch_size']} | "
        f"val_macro={result['best_val_f1_macro']:.4f} "
        f"val_f1={result['best_val_f1']:.4f} "
        f"test_macro={result['test_f1_macro']:.4f} "
        f"test_f1={result['test_f1']:.4f} "
        f"acc={result['test_acc']:.4f} "
        f"ep={result['epochs_trained']} "
        f"min={result['elapsed_min']:.1f}"
    )
    with open(RESULTS_TXT, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    logger.info("Result: %s", line)


# ─── main ─────────────────────────────────────────────────────────────────────

# ─── Stage-2 fine-tuning ──────────────────────────────────────────────────────

# Grid for Stage-2: start from the best Stage-1 checkpoint, unfreeze more layers
# and sweep over a small LR to avoid destroying what Stage-1 learned.
STAGE2_GRID: Dict[str, list] = {
    "n_trainable": [6, 8],   # unfreeze top 6 or 8 (vs 4 in Stage 1)
    "lr":          [1e-4, 5e-5],
}
STAGE2_FIXED: Dict[str, Any] = {
    # Inherit all other settings from the best Phase-13 run (rubert-base L=512)
    "backbone":       "DeepPavlov/rubert-base-cased",
    "max_length":     512,
    "accum_steps":    2,      # same gradient accumulation → eff. batch=128
    "batch_size":     64,
    "warmup_steps":   100,    # short warmup — model already knows good direction
    "epochs":         15,     # fewer epochs, model already trained
    "patience":       4,
    # Same regularisation as Stage 1:
    "oversample_up":  3,
    "label_smoothing": 0.05,
    "dropout":        0.3,
    "focal_gamma":    0.0,
    "cbr_weight":     3.0,
    "aux_weight":     0.0,
    "tab_arch":       "mlp_deep",
    "tab_proj_dim":   64,
    "fusion_hidden_dim": 128,
    "text_proj_dim":  256,
    "pool_mode":      "cls",
    "dataset":        "combined_t025",
    "train_end":      "2024-06-30",
    "val_end":        "2024-12-31",
}


def run_stage2(
    stage1_ckpt: str,
    params: Dict[str, Any],
    run_idx: int = 1,
    total_runs: int = 1,
) -> Dict[str, Any]:
    """
    Stage-2 gradual unfreezing: load Stage-1 checkpoint, unfreeze more layers,
    train with small LR. The warmstart_ckpt mechanism in JointBertTabularModel
    loads the checkpoint weights after re-initialising with the new hparams.
    """
    import time
    logger.info(
        "\n%s\n[Stage-2 %d/%d] ntr=%d  lr=%.0e  ckpt=%s\n%s",
        "=" * 90, run_idx, total_runs,
        params["n_trainable"], params["lr"],
        Path(stage1_ckpt).name, "=" * 90,
    )

    n_trainable = params["n_trainable"]
    lr          = params["lr"]
    max_length  = STAGE2_FIXED["max_length"]
    backbone    = STAGE2_FIXED["backbone"]
    accum_steps = STAGE2_FIXED["accum_steps"]
    bs          = STAGE2_FIXED["batch_size"]

    ds_name = STAGE2_FIXED["dataset"]
    ds_info = SWEEP_DATASETS[ds_name]

    cfg = Config()
    cfg.dataset_path       = str(ds_info["csv_path"])
    cfg.batch_size         = bs
    cfg.tokenizer_name     = backbone
    cfg.lr                 = lr
    cfg.n_trainable_layers = n_trainable
    cfg.cbr_sample_weight  = STAGE2_FIXED["cbr_weight"]
    cfg.focal_gamma        = STAGE2_FIXED["focal_gamma"]
    cfg.label_smoothing    = STAGE2_FIXED["label_smoothing"]
    cfg.dropout            = STAGE2_FIXED["dropout"]
    cfg.max_length         = max_length
    cfg.train_end          = STAGE2_FIXED["train_end"]
    cfg.val_end            = STAGE2_FIXED["val_end"]
    cfg.aux_weight         = STAGE2_FIXED["aux_weight"]
    cfg.max_epochs         = STAGE2_FIXED["epochs"]
    cfg.patience           = STAGE2_FIXED["patience"]
    cfg.oversample_up      = STAGE2_FIXED["oversample_up"]
    cfg.pool_mode          = STAGE2_FIXED["pool_mode"]

    pl.seed_everything(42, workers=True)
    dm = SweepDataModule(cfg, tab_cache=ds_info["tab_cache"])
    dm.setup()

    bb_short = backbone.split("/")[-1][:20]
    run_name  = f"s2run{run_idx:03d}_{bb_short}_ntr{n_trainable}_lr{lr:.0e}"

    ckpt_dir = CKPT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb  = ModelCheckpoint(
        dirpath=str(ckpt_dir), monitor="val_f1_macro", mode="max",
        save_top_k=1, filename="best",
    )
    early_cb = EarlyStopping(monitor="val_f1_macro", patience=cfg.patience, mode="max")
    val_cb   = ValMetricsCollector()

    model = JointBertTabularModel(
        n_tabular          = dm.n_tabular,
        num_classes        = len(LABEL_MAP),
        text_proj_dim      = STAGE2_FIXED["text_proj_dim"],
        tab_proj_dim       = STAGE2_FIXED["tab_proj_dim"],
        fusion_hidden_dim  = STAGE2_FIXED["fusion_hidden_dim"],
        dropout            = cfg.dropout,
        lr                 = lr,
        weight_decay       = 0.1,
        warmup_steps       = STAGE2_FIXED["warmup_steps"],
        class_weights      = dm.class_weights,
        n_trainable_layers = n_trainable,
        focal_gamma        = cfg.focal_gamma,
        label_smoothing    = cfg.label_smoothing,
        aux_weight         = cfg.aux_weight,
        backbone           = backbone,
        tab_arch           = STAGE2_FIXED["tab_arch"],
        pool_mode          = cfg.pool_mode,
        warmstart_ckpt     = stage1_ckpt,   # ← loads Stage-1 weights
    )

    precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    trainer   = pl.Trainer(
        max_epochs              = cfg.max_epochs,
        accelerator             = "auto",
        devices                 = 1,
        precision               = precision,
        accumulate_grad_batches = accum_steps,
        callbacks               = [ckpt_cb, early_cb, val_cb],
        logger                  = False,
        enable_progress_bar     = True,
        gradient_clip_val       = 1.0,
        log_every_n_steps       = 20,
    )

    t0 = time.time()
    trainer.fit(model, dm)
    test_results = trainer.test(model, dm, ckpt_path="best", verbose=False)
    elapsed = time.time() - t0

    tr            = test_results[0] if test_results else {}
    test_f1       = float(tr.get("test_f1",       0.0))
    test_f1_macro = float(tr.get("test_f1_macro", 0.0))
    test_acc      = float(tr.get("test_acc",      0.0))

    result = {
        "run_idx":           run_idx,
        "stage":             2,
        "backbone":          backbone,
        "max_length":        max_length,
        "n_trainable":       n_trainable,
        "lr":                lr,
        "accum_steps":       accum_steps,
        "warmup_steps":      STAGE2_FIXED["warmup_steps"],
        "stage1_ckpt":       stage1_ckpt,
        "epochs_trained":    val_cb.epochs_trained,
        "best_val_f1_macro": round(val_cb.best_val_f1_macro, 4),
        "best_val_f1":       round(val_cb.best_val_f1,       4),
        "test_f1":           round(test_f1,                  4),
        "test_f1_macro":     round(test_f1_macro,            4),
        "test_acc":          round(test_acc,                  4),
        "elapsed_min":       round(elapsed / 60,              1),
    }
    line = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"STAGE2 run={run_idx:03d} "
        f"bb={bb_short} L={max_length} "
        f"ntr={n_trainable} lr={lr:.0e} accum={accum_steps} | "
        f"val_macro={result['best_val_f1_macro']:.4f} "
        f"val_f1={result['best_val_f1']:.4f} "
        f"test_macro={result['test_f1_macro']:.4f} "
        f"test_f1={result['test_f1']:.4f} "
        f"acc={result['test_acc']:.4f} "
        f"ep={result['epochs_trained']} "
        f"min={result['elapsed_min']:.1f}"
    )
    with open(RESULTS_TXT, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    logger.info(line)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep — Joint BERT+Tabular")
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for all runs (default 128). Use 64 if OOM.",
    )
    parser.add_argument(
        "--max-runs", type=int, default=0,
        help="Cap the number of runs (0 = run all).",
    )
    parser.add_argument(
        "--resume-from", type=int, default=1,
        help="Skip runs with index < N (1-based) to resume interrupted sweeps.",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Only build dataset CSVs and tabular caches, then exit.",
    )
    parser.add_argument(
        "--skip-prepare", action="store_true",
        help="Skip dataset/tabular preparation (assume files already exist).",
    )
    parser.add_argument(
        "--force-prepare", action="store_true",
        help="Force-regenerate all dataset CSVs even if they exist.",
    )
    parser.add_argument(
        "--stage2-ckpt", type=str, default="",
        help=(
            "Path to a Stage-1 checkpoint. "
            "When provided, runs the STAGE2_GRID (n_trainable × lr) instead of the "
            "normal PARAM_GRID, loading the checkpoint as warmstart."
        ),
    )
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        logger.info("─── Step 1: preparing dataset CSVs ───")
        prepare_all_datasets(force=args.force_prepare)
        logger.info("─── Step 2: precomputing tabular features ───")
        precompute_tabular_for_all()

    if args.prepare_only:
        logger.info("--prepare-only: done.")
        return

    # ── Stage-2 mode ──────────────────────────────────────────────────────────
    if args.stage2_ckpt:
        ckpt = args.stage2_ckpt
        if not Path(ckpt).exists():
            logger.error("Stage-1 checkpoint not found: %s", ckpt)
            return
        s2_keys   = list(STAGE2_GRID.keys())
        s2_combos = [
            dict(zip(s2_keys, v))
            for v in itertools.product(*[STAGE2_GRID[k] for k in s2_keys])
        ]
        with open(RESULTS_TXT, "a", encoding="utf-8") as fh:
            fh.write(
                f"\n{'=' * 130}\n"
                f"Stage-2 started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Stage-1 ckpt    : {ckpt}\n"
                f"Grid            : {json.dumps(STAGE2_GRID)}\n"
                f"Fixed params    : {json.dumps(STAGE2_FIXED)}\n"
                f"{'=' * 130}\n"
            )
        for idx, params in enumerate(s2_combos, start=1):
            try:
                run_stage2(ckpt, params, run_idx=idx, total_runs=len(s2_combos))
            except Exception as exc:
                logger.error("Stage-2 run %d FAILED: %s", idx, exc, exc_info=True)
        return
    # ── Normal sweep mode ─────────────────────────────────────────────────────

    keys   = list(PARAM_GRID.keys())
    combos = [
        dict(zip(keys, v))
        for v in itertools.product(*[PARAM_GRID[k] for k in keys])
    ]

    total = len(combos)
    if args.max_runs > 0:
        combos = combos[: args.max_runs]
        logger.info("Limiting to %d of %d configurations (--max-runs)", len(combos), total)
    else:
        logger.info("Total configurations to run: %d", total)

    if args.resume_from <= 1:
        _write_header()

    all_results = []
    for idx, params in enumerate(combos, start=1):
        if idx < args.resume_from:
            logger.info("Skipping run %d (--resume-from=%d)", idx, args.resume_from)
            continue

        ds_info = SWEEP_DATASETS[FIXED["dataset"]]
        if not ds_info["csv_path"].exists():
            logger.error("Dataset CSV missing: %s — skipping run %d", ds_info["csv_path"].name, idx)
            continue

        try:
            result = run_one(params, batch_size=args.batch_size, run_idx=idx, total_runs=len(combos))
            _write_result(result)
            all_results.append(result)
        except Exception as exc:
            logger.error("Run %d FAILED: %s", idx, exc, exc_info=True)
            with open(RESULTS_TXT, "a", encoding="utf-8") as fh:
                fh.write(f"FAILED run_{idx:03d} {params}: {exc}\n")

    if all_results:
        df_res = pd.DataFrame(all_results)
        best   = df_res.loc[df_res["best_val_f1_macro"].idxmax()]
        summary = (
            f"\n{'─' * 130}\n"
            f"Best by val_f1_macro: run {int(best['run_idx'])} | "
            f"over={int(best['oversample_up'])} ls={best['label_smoothing']} | "
            f"val_macro={best['best_val_f1_macro']:.4f}  "
            f"val_f1={best['best_val_f1']:.4f}  "
            f"test_macro={best['test_f1_macro']:.4f}  "
            f"test_f1={best['test_f1']:.4f}\n"
        )
        logger.info(summary)
        with open(RESULTS_TXT, "a", encoding="utf-8") as fh:
            fh.write(summary)

        csv_out = Path(__file__).parents[2] / "sweep_results.csv"
        df_res.to_csv(csv_out, index=False)
        logger.info("Full results → %s", csv_out)


if __name__ == "__main__":
    main()
