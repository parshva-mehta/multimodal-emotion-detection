"""
Debug utilities for multimodal RAVDESS training.

Usage (from project root):
    uv run python src/debug.py \
        experiment.name=ravdess_debug \
        dataset.name=ravdess \
        dataset.data_dir="/scratch/pbm52/emotion-detection-mm/multimodal-dataset"
"""

import collections
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from data import create_dataloaders
from train import MultimodalFusionModule


# --------------------------------------------------------------------------
# 1) Label distribution inspection
# --------------------------------------------------------------------------


def inspect_label_distribution(loader, split_name: str):
    print(f"\n=== {split_name} label distribution ===")
    if loader is None:
        print(f"{split_name}: loader is None")
        return

    counts = collections.Counter()
    total_batches = 0
    for batch in loader:
        total_batches += 1
        # expected batch: (features, labels, mask)
        _, labels, _ = batch
        counts.update(labels.tolist())

    if total_batches == 0:
        print(f"{split_name}: no batches")
        return

    print(f"{split_name} batches: {total_batches}")
    if not counts:
        print(f"{split_name}: no labels collected")
        return

    total = sum(counts.values())
    print(f"{split_name} total samples: {total}")
    for cls, cnt in sorted(counts.items()):
        frac = cnt / total
        print(f"  class {cls}: {cnt} ({frac:.3f})")


# --------------------------------------------------------------------------
# 2) Overfit sanity test on a tiny subset
# --------------------------------------------------------------------------


def overfit_one_batch(config: DictConfig, device: str, train_loader):
    print("\n=== Overfit sanity test on 1 batch ===")
    # Grab one batch
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        print("Train loader is empty; cannot run overfit test.")
        return

    features, labels, mask = batch
    features = {
        k: v.to(device) for k, v in features.items()
    }  # dict of modality -> tensor
    labels = labels.to(device)
    mask = mask.to(device) if mask is not None else None

    # Fresh model
    model = MultimodalFusionModule(config).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )

    max_epochs = 50
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(features, mask)
        loss = model.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()

        loss.backward()
        optimizer.step()

        print(
            f"[overfit] epoch {epoch:02d} | loss={loss.item():.4f} | acc={acc:.4f}"
        )

        if acc > 0.98:
            print("✓ Overfit sanity test passed (acc > 0.98).")
            break
    else:
        print(
            "⚠ Overfit sanity test did NOT reach high accuracy. "
            "This suggests a bug in model / labels / loss."
        )


# --------------------------------------------------------------------------
# 3) Encoder output & fused logits stats
# --------------------------------------------------------------------------


def encoder_and_logits_stats(config: DictConfig, device: str, train_loader):
    print("\n=== Encoder & logits stats on one batch ===")
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        print("Train loader is empty; cannot inspect encoder stats.")
        return

    features, labels, mask = batch
    features = {k: v.to(device) for k, v in features.items()}
    labels = labels.to(device)
    mask = mask.to(device) if mask is not None else None

    model = MultimodalFusionModule(config).to(device)
    model.eval()

    with torch.no_grad():
        # Per-modality encoder stats
        for modality, encoder in model.encoders.items():
            if modality not in features:
                print(f"  [{modality}] not present in batch features; skipping.")
                continue
            x = features[modality]
            out = encoder(x)
            print(f"  [{modality}] input shape:  {tuple(x.shape)}")
            print(
                f"  [{modality}] output shape: {tuple(out.shape)}, "
                f"mean={out.mean().item():.4f}, std={out.std().item():.4f}, "
                f"min={out.min().item():.4f}, max={out.max().item():.4f}"
            )

        # Fused logits stats
        logits = model(features, mask)
        probs = F.softmax(logits, dim=1)
        confs, _ = probs.max(dim=1)
        print(
            f"\n  [fusion] logits shape: {tuple(logits.shape)}, "
            f"mean={logits.mean().item():.4f}, std={logits.std().item():.4f}"
        )
        print(
            f"  [fusion] confidences: mean={confs.mean().item():.4f}, "
            f"std={confs.std().item():.4f}, "
            f"min={confs.min().item():.4f}, max={confs.max().item():.4f}"
        )


# --------------------------------------------------------------------------
# 4) Gradient magnitude stats
# --------------------------------------------------------------------------


def gradient_stats(config: DictConfig, device: str, train_loader):
    print("\n=== Gradient magnitude stats (one backward pass) ===")
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        print("Train loader is empty; cannot inspect gradients.")
        return

    features, labels, mask = batch
    features = {k: v.to(device) for k, v in features.items()}
    labels = labels.to(device)
    mask = mask.to(device) if mask is not None else None

    model = MultimodalFusionModule(config).to(device)
    model.train()

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    logits = model(features, mask)
    loss = model.criterion(logits, labels)
    loss.backward()

    grad_means = []
    for p in model.parameters():
        if p.grad is not None:
            grad_means.append(p.grad.detach().abs().mean().item())

    if not grad_means:
        print("No gradients found (all grad=None). Check that parameters require_grad.")
        return

    grad_means_tensor = torch.tensor(grad_means)
    print(
        f"Gradient |mean| across params: "
        f"mean={grad_means_tensor.mean().item():.6f}, "
        f"std={grad_means_tensor.std().item():.6f}, "
        f"min={grad_means_tensor.min().item():.6f}, "
        f"max={grad_means_tensor.max().item():.6f}"
    )


# --------------------------------------------------------------------------
# Hydra entry point
# --------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig):
    print("=" * 80)
    print("DEBUG CONFIGURATION:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    pl.seed_everything(config.seed)

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
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    # Choose device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nUsing device: {device}")

    # 1) Label distributions
    inspect_label_distribution(train_loader, "Train")
    inspect_label_distribution(val_loader, "Val")
    inspect_label_distribution(test_loader, "Test")

    # 2) Overfit sanity test
    overfit_one_batch(config, device, train_loader)

    # 3) Encoder output / logits stats
    encoder_and_logits_stats(config, device, train_loader)

    # 4) Gradient stats
    gradient_stats(config, device, train_loader)

    print("\nDebugging complete.")


if __name__ == "__main__":
    main()
