import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from cbr_news.ml.dataset import CBRNewsMultiTaskDataModule
from cbr_news.ml.models.base_model import CBRNewsModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RuoniaOnlyDataModule(CBRNewsMultiTaskDataModule):
    """
    Loads the multitask dataset but exposes only RUONIA labels
    as a single-task dataset, with proper temporal val split.
    """

    def setup(self, stage=None):
        super().setup(stage)

        self.train_dataset = _RuoniaDataset(self.train_dataset)
        self.val_dataset = _RuoniaDataset(self.val_dataset)
        self.test_dataset = _RuoniaDataset(self.test_dataset)

        logger.info("Wrapped to RUONIA-only single-task datasets")
        logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")


class _RuoniaDataset:
    """Thin wrapper: exposes 'labels' key from multitask dataset's 'ruonia_label'."""

    def __init__(self, multitask_dataset):
        self._ds = multitask_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        item = self._ds[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["ruonia_label"],
        }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Baseline training - only main task (RUONIA) with temporal val/test split."""

    logger.info("=" * 60)
    logger.info("BASELINE TRAINING - RUONIA ONLY (TEMPORAL SPLIT)")
    logger.info("=" * 60)

    cfg.model.dropout = 0.3
    cfg.model.learning_rate = 1e-5
    cfg.model.weight_decay = 0.05
    cfg.model.freeze_backbone = True
    cfg.model.num_frozen_layers = 9 

    logger.info(f"Dropout:           {cfg.model.dropout}")
    logger.info(f"Learning rate:     {cfg.model.learning_rate}")
    logger.info(f"Frozen layers:     {cfg.model.num_frozen_layers} / 12")
    logger.info(f"Prediction horizon:{cfg.data.prediction_horizon} days")
    logger.info(f"RUONIA threshold:  {cfg.data.ruonia_threshold}")
    logger.info(f"Train years: {cfg.data.train_years}")
    logger.info(f"Val years:   {cfg.data.val_years}")
    logger.info(f"Test years:  {cfg.data.test_years}")

    pl.seed_everything(cfg.seed, workers=True)

    data_module = RuoniaOnlyDataModule(cfg)
    data_module.setup()

    model = CBRNewsModel(cfg)

    logger.info(f"Model parameters:    {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters:{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    mlflow_logger = MLFlowLogger(
        experiment_name=f"{cfg.training.mlflow.experiment_name}_baseline",
        tracking_uri=cfg.training.mlflow.tracking_uri,
        log_model=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/baseline",
        filename="ruonia-baseline-{epoch:02d}-{val_loss:.4f}",
        monitor=cfg.training.monitor,
        mode=cfg.training.mode,
        save_top_k=1,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.training.monitor,
        patience=cfg.training.patience,
        mode=cfg.training.mode,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=True,
    )

    logger.info("Starting baseline training...")
    trainer.fit(model, data_module)

    logger.info("Testing baseline model...")
    results = trainer.test(model, data_module)

    logger.info("=" * 60)
    logger.info("BASELINE RESULTS (temporal split):")
    logger.info(f"Test Loss:     {results[0]['test_loss']:.4f}")
    logger.info(f"Test Accuracy: {results[0]['test_acc']:.4f}")
    logger.info(f"Test F1:       {results[0]['test_f1']:.4f}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
