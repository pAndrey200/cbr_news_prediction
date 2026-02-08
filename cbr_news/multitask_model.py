import logging
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class CBRNewsMultiTaskModel(pl.LightningModule):
    """Multi-task модель для предсказания экономических параметров и ключевой ставки"""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Shared encoder (RuBERT)
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

        # Auxiliary task heads (binary classification для каждого экономического параметра)
        self.auxiliary_tasks = config.model.get('auxiliary_tasks', ['usd', 'eur', 'cny', 'inflation', 'ruonia'])
        self.auxiliary_heads = nn.ModuleDict()

        for task in self.auxiliary_tasks:
            self.auxiliary_heads[task] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.model.dropout),
                nn.Linear(hidden_size // 2, 2)  # binary classification
            )

        # Main task head для ключевой ставки
        # Использует признаки от encoder + предсказания auxiliary tasks
        num_auxiliary = len(self.auxiliary_tasks)
        main_input_size = hidden_size + num_auxiliary * 2  # hidden_size + logits от auxiliary tasks

        self.main_head = nn.Sequential(
            nn.Linear(main_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_size // 2, config.model.num_classes)  # 3 classes: up, down, same
        )

        # Loss functions
        self.auxiliary_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.main_loss_fn = nn.CrossEntropyLoss()

        # Task weights (можно настроить для баланса потерь)
        self.auxiliary_weight = config.model.get('auxiliary_weight', 0.3)
        self.main_weight = config.model.get('main_weight', 1.0)

        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

        logger.info(f"Multi-task модель инициализирована: {config.model.backbone}")
        logger.info(f"Auxiliary tasks: {self.auxiliary_tasks}")
        logger.info(f"Auxiliary weight: {self.auxiliary_weight}, Main weight: {self.main_weight}")

    def forward(self, input_ids, attention_mask):
        # Получаем представление от backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)

        # Предсказания для auxiliary tasks
        auxiliary_logits = {}
        auxiliary_features = []

        for task in self.auxiliary_tasks:
            logits = self.auxiliary_heads[task](pooled_output)
            auxiliary_logits[task] = logits
            auxiliary_features.append(logits)

        # Объединяем признаки для главной задачи
        # Concatenate: pooled_output + все auxiliary logits
        if auxiliary_features:
            auxiliary_concat = torch.cat(auxiliary_features, dim=1)
            main_input = torch.cat([pooled_output, auxiliary_concat], dim=1)
        else:
            main_input = pooled_output

        # Предсказание для главной задачи
        main_logits = self.main_head(main_input)

        return {
            'main_logits': main_logits,
            'auxiliary_logits': auxiliary_logits
        }

    def configure_optimizers(self):
        """Настройка оптимизатора и планировщика"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay,
        )

        total_steps = None
        if self.trainer and hasattr(self.trainer, 'datamodule') and self.trainer.datamodule:
            try:
                if self.trainer.datamodule.train_dataset is not None:
                    num_samples = len(self.trainer.datamodule.train_dataset)
                    batch_size = self.config.data.batch_size
                    num_batches = (num_samples + batch_size - 1) // batch_size
                    total_steps = num_batches * self.trainer.max_epochs
            except (AttributeError, TypeError):
                pass

        if total_steps is None:
            total_steps = 10000

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

    def _compute_loss(self, batch, outputs):
        """Вычисление комбинированного loss"""
        # Main task loss
        main_labels = batch["key_rate_label"]
        main_logits = outputs['main_logits']
        main_loss = self.main_loss_fn(main_logits, main_labels)

        # Auxiliary tasks losses
        auxiliary_losses = []
        for task in self.auxiliary_tasks:
            label_key = f"{task}_label"
            if label_key in batch:
                task_labels = batch[label_key]
                task_logits = outputs['auxiliary_logits'][task]
                task_loss = self.auxiliary_loss_fn(task_logits, task_labels)
                auxiliary_losses.append(task_loss)

        # Средний auxiliary loss
        if auxiliary_losses:
            avg_auxiliary_loss = torch.stack(auxiliary_losses).mean()
        else:
            avg_auxiliary_loss = torch.tensor(0.0, device=main_loss.device)

        # Комбинированный loss
        total_loss = self.main_weight * main_loss + self.auxiliary_weight * avg_auxiliary_loss

        return total_loss, main_loss, avg_auxiliary_loss

    def _shared_step(self, batch, batch_idx, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self(input_ids, attention_mask)

        # Вычисляем loss
        total_loss, main_loss, auxiliary_loss = self._compute_loss(batch, outputs)

        # Предсказания для главной задачи
        main_logits = outputs['main_logits']
        main_labels = batch["key_rate_label"]
        main_preds = torch.argmax(main_logits, dim=1)

        # Метрики для главной задачи
        acc = accuracy_score(main_labels.cpu(), main_preds.cpu())
        f1 = f1_score(main_labels.cpu(), main_preds.cpu(), average="weighted", zero_division=0)

        # Логирование
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        self.log(f"{stage}_main_loss", main_loss, prog_bar=True)
        self.log(f"{stage}_aux_loss", auxiliary_loss, prog_bar=False)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)

        # Метрики для auxiliary tasks
        for task in self.auxiliary_tasks:
            label_key = f"{task}_label"
            if label_key in batch:
                task_labels = batch[label_key]
                task_logits = outputs['auxiliary_logits'][task]
                task_preds = torch.argmax(task_logits, dim=1)

                # Фильтруем ignore_index (-100)
                valid_mask = task_labels != -100
                if valid_mask.sum() > 0:
                    valid_labels = task_labels[valid_mask]
                    valid_preds = task_preds[valid_mask]
                    task_acc = accuracy_score(valid_labels.cpu(), valid_preds.cpu())
                    self.log(f"{stage}_{task}_acc", task_acc, prog_bar=False)

        output = {
            "loss": total_loss,
            "main_preds": main_preds,
            "main_labels": main_labels,
            "acc": acc,
            "f1": f1
        }

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
        all_preds = torch.cat([x["main_preds"] for x in outputs])
        all_labels = torch.cat([x["main_labels"] for x in outputs])

        epoch_acc = accuracy_score(all_labels.cpu(), all_preds.cpu())
        epoch_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average="weighted", zero_division=0)

        self.log(f"{stage}_epoch_loss", avg_loss, prog_bar=True)
        self.log(f"{stage}_epoch_acc", epoch_acc, prog_bar=True)
        self.log(f"{stage}_epoch_f1", epoch_f1, prog_bar=True)

    def predict(self, texts: List[str], tokenizer, device: str = None):
        """Предсказание для новых текстов"""
        if device is None:
            device = self.device

        self.eval()

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
            outputs = self(input_ids, attention_mask)
            main_logits = outputs['main_logits']
            probs = F.softmax(main_logits, dim=1)
            preds = torch.argmax(main_logits, dim=1)

            # Также получаем предсказания для auxiliary tasks
            auxiliary_preds = {}
            for task in self.auxiliary_tasks:
                task_logits = outputs['auxiliary_logits'][task]
                task_probs = F.softmax(task_logits, dim=1)
                task_pred = torch.argmax(task_logits, dim=1)
                auxiliary_preds[task] = {
                    'prediction': 'up' if task_pred[0].item() == 1 else 'down',
                    'probabilities': {
                        'down': task_probs[0][0].item(),
                        'up': task_probs[0][1].item()
                    }
                }

        results = []
        for i, text in enumerate(texts):
            result = {
                "text": text,
                "prediction": self.config.data.classes[preds[i].item()],
                "probabilities": {
                    self.config.data.classes[j]: probs[i][j].item()
                    for j in range(len(self.config.data.classes))
                },
                "auxiliary_predictions": {
                    task: auxiliary_preds[task] for task in self.auxiliary_tasks
                }
            }
            results.append(result)

        return results
