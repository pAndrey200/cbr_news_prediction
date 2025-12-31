import logging
from typing import List

import mlflow
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class CBRNewsModel(pl.LightningModule):
    """Модель для классификации новостей Банка России"""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.backbone = AutoModel.from_pretrained(config.model.backbone)

        if config.model.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            if config.model.num_frozen_layers > 0:
                for layer in self.backbone.encoder.layer[
                    -config.model.num_frozen_layers :
                ]:
                    for param in layer.parameters():
                        param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(config.model.dropout)
        self.classifier = nn.Linear(hidden_size, config.model.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

        logger.info(f"Модель инициализирована: {config.model.backbone}")

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def configure_optimizers(self):
        """Настройка оптимизатора и планировщика"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay,
        )

        total_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        warmup_steps = self.config.model.warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _shared_step(self, batch, batch_idx, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)

        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average="weighted")

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)

        output = {"loss": loss, "preds": preds, "labels": labels, "acc": acc, "f1": f1}

        if stage == "train":
            self.train_outputs.append(output)
        elif stage == "val":
            self.val_outputs.append(output)
        else:
            self.test_outputs.append(output)

        return output

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        """В конце тренировочной эпохи"""
        self._log_epoch_metrics("train")
        self.train_outputs.clear()

    def on_validation_epoch_end(self):
        """В конце валидационной эпохи"""
        self._log_epoch_metrics("val")
        self.val_outputs.clear()

    def _log_epoch_metrics(self, stage: str):
        """Логирование метрик эпохи"""
        outputs = self.train_outputs if stage == "train" else self.val_outputs

        if not outputs:
            return

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        all_preds = torch.cat([x["preds"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs])

        epoch_acc = accuracy_score(all_labels.cpu(), all_preds.cpu())
        epoch_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average="weighted")

        self.log(f"{stage}_epoch_loss", avg_loss, prog_bar=True)
        self.log(f"{stage}_epoch_acc", epoch_acc, prog_bar=True)
        self.log(f"{stage}_epoch_f1", epoch_f1, prog_bar=True)

        if self.trainer.logger and hasattr(self.trainer.logger, "experiment"):
            try:
                mlflow.log_metrics(
                    {
                        f"{stage}_epoch_loss": avg_loss.item(),
                        f"{stage}_epoch_acc": epoch_acc,
                        f"{stage}_epoch_f1": epoch_f1,
                    },
                    step=self.current_epoch,
                )
            except Exception as e:
                logger.warning(f"Ошибка логирования в MLflow: {e}")

    def predict(self, texts: List[str], tokenizer, device: str = None):
        """Предсказание для новых текстов"""
        if device is None:
            device = self.device

        self.eval()

        # Токенизация
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.config.data.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        results = []
        for i, text in enumerate(texts):
            result = {
                "text": text,
                "prediction": self.config.data.classes[preds[i].item()],
                "probabilities": {
                    self.config.data.classes[j]: probs[i][j].item()
                    for j in range(len(self.config.data.classes))
                },
            }
            results.append(result)

        return results
