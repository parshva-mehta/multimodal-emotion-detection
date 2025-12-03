"""
Training Script for Multimodal Sensor Fusion

Uses PyTorch Lightning for training with Hydra configuration.
Most infrastructure is provided - students need to integrate their fusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import shutil
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


from data import create_dataloaders
from fusion import build_fusion_model
from encoders import build_encoder
from uncertainty import compute_calibration_metrics, CalibrationMetrics


class MultimodalFusionModule(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal fusion training.

    Handles training loop, validation, logging, confusion matrix, and
    train/val loss curve plotting.
    """

    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra configuration object
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Encoders
        self.encoders = nn.ModuleDict()
        modality_output_dims = {}

        # Buffers for test aggregation
        self.test_step_outputs = []

        # Buffers for loss curves
        self._train_losses_epoch = []
        self._val_losses_epoch = []
        self.train_loss_history = []
        self.val_loss_history = []

        # Build encoders for each modality
        for modality in config.dataset.modalities:
            encoder_config = config.model.encoders.get(modality, {})
            input_dim = encoder_config.get("input_dim", 64)
            output_dim = config.model.output_dim

            self.encoders[modality] = build_encoder(
                modality=modality,
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_config=encoder_config,
            )
            modality_output_dims[modality] = output_dim

        # Build fusion model
        self.fusion_model = build_fusion_model(
            fusion_type=config.model.fusion_type,
            modality_dims=modality_output_dims,
            num_classes=config.dataset.get("num_classes", 11),
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.get("num_heads", 4),
            dropout=config.model.dropout,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, features, mask=None):
        """
        Forward pass through encoders and fusion model.

        Args:
            features: Dict of {modality_name: features}
            mask: Optional modality availability mask

        Returns:
            logits: Class predictions
        """
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in features:
                encoded_features[modality] = encoder(features[modality])

        output = self.fusion_model(encoded_features, mask)

        # Handle different fusion output formats
        if isinstance(output, tuple):
            logits = output[0]  # Late fusion returns (fused_logits, per_modality_logits)
        else:
            logits = output

        return logits

    # ------------------------------------------------------------------
    # Training / validation / test steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        features, labels, mask = batch

        logits = self(features, mask)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        # Lightning logging
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/confidence_mean",
            confidences.mean(),
            on_step=False,
            on_epoch=True,
        )

        # For manual loss curve
        self._train_losses_epoch.append(loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        features, labels, mask = batch

        logits = self(features, mask)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.log(
            "val/confidence_mean",
            confidences.mean(),
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val/entropy", entropy, on_epoch=True, prog_bar=False)

        # For manual loss curve
        self._val_losses_epoch.append(loss.detach())

        return {
            "val_loss": loss,
            "val_acc": acc,
            "preds": preds,
            "labels": labels,
            "confidences": confidences,
        }

    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        features, labels, mask = batch

        logits = self(features, mask)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)

        # store for on_test_epoch_end aggregation
        self.test_step_outputs.append(
            {
                "loss": loss.detach(),
                "preds": preds.detach(),
                "labels": labels.detach(),
                "confidences": confidences.detach(),
            }
        )

        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "confidences": confidences,
        }

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        if self.config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate / 100,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif self.config.training.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            return optimizer

    # ------------------------------------------------------------------
    # Epoch-end hooks for loss curves
    # ------------------------------------------------------------------
    def on_train_epoch_end(self):
        if self._train_losses_epoch:
            epoch_mean = torch.stack(self._train_losses_epoch).mean().item()
            self.train_loss_history.append(epoch_mean)
            self._train_losses_epoch.clear()
            print(f"[Epoch {self.current_epoch}] train_loss={epoch_mean:.4f}")

    def on_validation_epoch_end(self):
        if self._val_losses_epoch:
            epoch_mean = torch.stack(self._val_losses_epoch).mean().item()
            self.val_loss_history.append(epoch_mean)
            self._val_losses_epoch.clear()
            print(f"[Epoch {self.current_epoch}] val_loss={epoch_mean:.4f}")

    def on_fit_end(self):
        """Called once after training finishes: save train/val loss curve."""
        if not self.train_loss_history:
            print("on_fit_end: no training loss history collected, skipping loss plot.")
            return

        epochs = range(1, len(self.train_loss_history) + 1)

        fig, ax = plt.subplots()
        ax.plot(epochs, self.train_loss_history, label="Train loss")

        if self.val_loss_history:
            ax.plot(epochs, self.val_loss_history, label="Val loss")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training / Validation Loss")
        ax.legend()
        fig.tight_layout()

        save_root = Path(self.config.experiment.save_dir) / self.config.experiment.name
        save_root.mkdir(parents=True, exist_ok=True)

        out_path = save_root / "loss_curve.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved loss curve to {out_path}")

    # ------------------------------------------------------------------
    # Test epoch end: confusion matrix + test acc
    # ------------------------------------------------------------------
    def on_test_epoch_end(self):
        """
        Aggregate all test batches, compute + save confusion matrix, and
        log a simple test accuracy from the aggregated preds/labels.
        """
        outputs = self.test_step_outputs

        # 0) Safety guard: no test batches
        if not outputs:
            print("on_test_epoch_end: no test outputs collected, skipping aggregation.")
            return

        def _stack_from_outputs(key):
            tensors = [o[key] for o in outputs if key in o]
            if not tensors:
                return None
            return torch.cat(tensors, dim=0)

        preds_t = _stack_from_outputs("preds")
        labels_t = _stack_from_outputs("labels")

        if preds_t is None or labels_t is None:
            print(
                "on_test_epoch_end: missing 'preds' or 'labels' in test_step_outputs; "
                "skipping confusion matrix."
            )
            self.test_step_outputs.clear()
            return

        preds = preds_t.detach().cpu().numpy()
        labels = labels_t.detach().cpu().numpy()

        # 1) num_classes from config
        num_classes = None
        try:
            num_classes = self.config.dataset.get("num_classes", None)
        except Exception:
            num_classes = getattr(
                getattr(self.config, "dataset", {}), "num_classes", None
            )
        if num_classes is None:
            num_classes = 8
        num_classes = int(num_classes)

        # 2) Confusion matrix
        cm = confusion_matrix(labels, preds, labels=np.arange(num_classes))

        # 3) Save directory
        save_root = Path(self.config.experiment.save_dir) / self.config.experiment.name
        save_root.mkdir(parents=True, exist_ok=True)

        cm_npy_path = save_root / "confusion_matrix.npy"
        np.save(cm_npy_path, cm)
        print(f"Saved confusion matrix to {cm_npy_path}")

        # 4) Class names
        dataset_name = getattr(self.config.dataset, "name", "")

        if dataset_name == "ravdess" and num_classes == 8:
            class_names = [
                "neutral",
                "calm",
                "happy",
                "sad",
                "angry",
                "fearful",
                "disgust",
                "surprised",
            ]
        else:
            class_names = [f"C{i}" for i in range(num_classes)]

        # 5) Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        vmax = cm.max()
        thresh = vmax / 2.0 if vmax > 0 else 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

        fig.tight_layout()
        fig_path = save_root / "confusion_matrix.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"Saved confusion matrix figure to {fig_path}")

        # 6) Simple aggregated test accuracy
        try:
            acc = (preds == labels).mean().item()
            self.log("test/acc_agg", acc, prog_bar=True, on_epoch=True, on_step=False)
            print(f"Aggregated test accuracy (from confusion matrix): {acc:.4f}")
        except Exception as e:
            print(f"on_test_epoch_end: failed to compute/log accuracy: {e}")

        # 7) Clear buffer
        self.test_step_outputs.clear()
    main()
