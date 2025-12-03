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


from data import create_dataloaders
from fusion import build_fusion_model
from encoders import build_encoder
from uncertainty import compute_calibration_metrics, CalibrationMetrics


class MultimodalFusionModule(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal fusion training.

    Handles training loop, validation, and logging.
    """

    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra configuration object
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Build encoders for each modality
        self.encoders = nn.ModuleDict()
        modality_output_dims = {}
        self.test_step_outputs = []  # <--- add this

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

        # Metrics storage (if you want to aggregate manually later)
        self.train_metrics = []
        self.val_metrics = []
        
    def test_step(self, batch, batch_idx):
        ...
        out = {
            "loss": loss,
            "y": y,
            "y_hat": y_hat,
            # anything else you used in test_epoch_end
        }
        self.test_step_outputs.append(out)   # <--- accumulate
        return out
    
    def forward(self, features, mask=None):
        """
        Forward pass through encoders and fusion model.

        Args:
            features: Dict of {modality_name: features}
            mask: Optional modality availability mask

        Returns:
            logits: Class predictions
        """
        # Encode each modality
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in features:
                encoded_features[modality] = encoder(features[modality])

        # Fusion
        output = self.fusion_model(encoded_features, mask)

        # Handle different fusion output formats
        if isinstance(output, tuple):
            logits = output[0]  # Late fusion returns (fused_logits, per_modality_logits)
        else:
            logits = output

        return logits

    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        features, labels, mask = batch

        # Forward pass
        logits = self(features, mask)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Optionally log confidences during training
        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/confidence_mean",
            confidences.mean(),
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        features, labels, mask = batch

        # Forward pass
        logits = self(features, mask)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Get confidence & entropy for calibration-ish monitoring
        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()

        # Log metrics
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.log(
            "val/confidence_mean",
            confidences.mean(),
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val/entropy", entropy, on_epoch=True, prog_bar=False)

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

        # Forward pass
        logits = self(features, mask)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        self.log("test/acc", acc, on_epoch=True)

        return {
            "preds": preds,
            "labels": labels,
            "confidences": confidences,
        }

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
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

        # Learning rate scheduler
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
        
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs   # list of dicts from test_step

        # whatever you previously did in test_epoch_end
        # e.g., stack tensors and compute metrics

        """
        Aggregate all test batches and compute + save confusion matrix.
        """
        # Collect preds and labels
        preds = torch.cat([o["preds"] for o in outputs]).cpu().numpy()
        labels = torch.cat([o["labels"] for o in outputs]).cpu().numpy()

        num_classes = int(self.config.dataset.get("num_classes", 8))
        
        all_y = torch.cat([o["y"] for o in outputs], dim=0)
        all_y_hat = torch.cat([o["y_hat"] for o in outputs], dim=0)

        # Compute confusion matrix
        cm = confusion_matrix(labels, preds, labels=np.arange(num_classes))

        # Save raw matrix as .npy
        save_root = Path(self.config.experiment.save_dir) / self.config.experiment.name
        save_root.mkdir(parents=True, exist_ok=True)
        np.save(save_root / "confusion_matrix.npy", cm)
        print(f"Saved confusion matrix to {save_root / 'confusion_matrix.npy'}")

        # Optional: human-readable emotion labels for RAVDESS
        if getattr(self.config.dataset, "name", "") == "ravdess" and num_classes == 8:
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

        # Plot confusion matrix
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

        # Annotate each cell
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
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

        # compute metrics, log them, etc.
        # self.log("test_acc", acc, prog_bar=True)

        # IMPORTANT: clear the buffer so it doesn’t leak across runs
        self.test_step_outputs.clear()


def _collect_logits_labels(model, dataloader, device: str):
    """One full pass to collect logits and labels (works with dict+mask batches)."""
    model.eval().to(device)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                inputs, labels = batch
                mask = None
            elif len(batch) == 3:
                inputs, labels, mask = batch
            else:
                raise ValueError(f"Unexpected batch len={len(batch)}")

            labels = labels.to(device)
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                mask = mask.to(device) if mask is not None else None
                out = model(inputs, mask)
            else:
                out = model(inputs.to(device))

            logits = out[0] if isinstance(out, (tuple, list)) else out
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_logits:
        return torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _per_bin_accuracy(
    confs: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int,
):
    """
    Return bin EDGES (upper edges: 0.1..1.0 for 10 bins) and accuracy per bin (None if empty).
    Matches your screenshot format.
    """
    edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)
    idx = torch.bucketize(confs.clamp(0, 1), edges, right=False) - 1
    idx = idx.clamp(0, num_bins - 1)

    bins_out = [round(float(edges[i + 1].item()), 2) for i in range(num_bins)]  # 0.1..1.0
    acc_out = []
    correct = (preds == labels)

    for b in range(num_bins):
        mask = idx == b
        if mask.any():
            acc_out.append(round(float(correct[mask].float().mean().item()), 4))
        else:
            acc_out.append(None)
    return bins_out, acc_out


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig):
    """
    Main training function.

    Args:
        config: Hydra configuration
    """
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    # Create output directories
    save_dir = Path(config.experiment.save_dir) / config.experiment.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        modalities=config.dataset.modalities,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        modality_dropout=config.training.augmentation.modality_dropout,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = MultimodalFusionModule(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{epoch}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=config.experiment.save_top_k,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=config.training.early_stopping_patience,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Loggers
    tb_logger = TensorBoardLogger(
        save_dir=save_dir,
        name="tb_logs",
    )
    csv_logger = CSVLogger(
        save_dir=save_dir,
        name="csv_logs",  # metrics.csv under csv_logs/version_*/
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu",   # lets PL pick cpu/mps/cuda
        devices=1,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=config.experiment.log_every_n_steps,
        gradient_clip_val=config.training.gradient_clip_norm,
        deterministic=True,
        enable_progress_bar=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    def _is_uncertainty_fusion(cfg) -> bool:
        ft = (cfg.model.fusion_type or "").lower()
        return ft in {
            "uncertainty",
            "uwf",
            "uncertainty_weighted",
            "uncertainty_weighted_late",
        }

    print("\nTesting best model...")
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    trainer.test(model, test_loader, ckpt_path=best_model_path)

    if _is_uncertainty_fusion(config):
        print("\nComputing calibration metrics (uncertainty fusion detected)...")
        best_model = MultimodalFusionModule.load_from_checkpoint(
            best_model_path,
            config=config,
            strict=False,
        )
        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        logits_all, labels_all = _collect_logits_labels(
            best_model,
            test_loader,
            device=device_str,
        )
        if logits_all.numel() == 0:
            print("No logits collected; skipping uncertainty.json.")
        else:
            nll = CalibrationMetrics.negative_log_likelihood(
                logits_all,
                labels_all,
            )
            probs_all = torch.softmax(logits_all, dim=1)
            confs_all, preds_all = torch.max(probs_all, dim=1)
            ece = CalibrationMetrics.expected_calibration_error(
                confs_all,
                preds_all,
                labels_all,
                num_bins=config.evaluation.get("num_calibration_bins", 15),
            )

            bins_list, acc_per_bin = _per_bin_accuracy(
                confs_all,
                preds_all,
                labels_all,
                num_bins=config.evaluation.get("num_calibration_bins", 15),
            )

            CalibrationMetrics.reliability_diagram(
                confs_all.numpy(),
                preds_all.numpy(),
                labels_all.numpy(),
                save_path="./analysis/calibration_diagram.png",
            )
            print("✓ Reliability diagram created")

            experiments_dir = Path(
                config.outputs.get("experiments_dir", "./experiments")
            )
            experiments_dir.mkdir(parents=True, exist_ok=True)
            out_path = experiments_dir / "uncertainty.json"

            out_obj = {
                "dataset": str(config.dataset.name),
                "calibration_metrics": {
                    "ece": round(float(ece), 3),
                    "nll": round(float(nll), 3),
                    "bins": bins_list,
                    "accuracy_per_bin": acc_per_bin,
                },
            }
            with open(out_path, "w") as f:
                json.dump(out_obj, f, indent=2)
            print(f"Saved uncertainty report to: {out_path}")
    else:
        print("\nSkipping calibration metrics: fusion_type is not an uncertainty variant.")
        # Save final results
        results = {
            "best_model_path": str(best_model_path),
            "best_val_loss": float(checkpoint_callback.best_model_score),
            "config": OmegaConf.to_container(config, resolve=True),
        }
        best_ckpt_target = save_dir / "best.ckpt"
        if best_model_path and Path(best_model_path).exists():
            shutil.copy(str(best_model_path), str(best_ckpt_target))
            print(f"Copied best checkpoint to: {best_ckpt_target}")

        results_file = save_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nTraining complete! Results saved to: {results_file}")
        print(f"Best model: {best_model_path}")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
