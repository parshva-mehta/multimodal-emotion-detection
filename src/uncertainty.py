"""
Uncertainty Quantification for Multimodal Fusion

Implements methods for estimating and calibrating confidence scores:
1. MC Dropout for epistemic uncertainty
2. Calibration metrics (ECE, reliability diagrams)
3. Uncertainty-weighted fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty via variance.
    """

    def __init__(self, model: nn.Module, num_samples: int = 10):
        """
        Args:
            model: The model to estimate uncertainty for
            num_samples: Number of MC dropout samples
        """
        super().__init__()
        assert num_samples >= 1, "num_samples must be >= 1"
        self.model = model
        self.num_samples = num_samples

    @staticmethod
    def _set_dropout_train(module: nn.Module) -> None:
        """
        Put only dropout layers into train mode; leave everything else (e.g., BatchNorm) as-is.
        """
        dropout_types = (
            nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout
        )
        for m in module.modules():
            if isinstance(m, dropout_types):
                m.train()

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns:
            mean_logits: (batch_size, num_classes) - mean prediction
            uncertainty: (batch_size,) - prediction uncertainty (variance over class probabilities)
        """
        # Record current mode to restore later
        was_training = self.model.training
        self.model.eval()               # keep non-dropout layers in eval
        self._set_dropout_train(self.model)  # activate only dropout

        logits_samples = []
        with torch.inference_mode():
            for _ in range(self.num_samples):
                logits = self.model(*args, **kwargs)  # (B, C)
                logits_samples.append(logits)

        # Restore original mode
        if was_training:
            self.model.train()
        else:
            self.model.eval()

        # Stack -> (S, B, C)
        logits_stack = torch.stack(logits_samples, dim=0)

        # Mean logits across samples: (B, C)
        mean_logits = logits_stack.mean(dim=0)

        # Convert to probabilities and compute variance across samples
        probs_stack = torch.softmax(logits_stack, dim=-1)     # (S, B, C)
        var_probs = probs_stack.var(dim=0, unbiased=False)    # (B, C)

        # Aggregate class-wise variance into a scalar uncertainty per example
        uncertainty = var_probs.mean(dim=-1)                  # (B,)

        return mean_logits, uncertainty

class CalibrationMetrics:
    """
    Compute calibration metrics for confidence estimates.

    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Negative Log-Likelihood (NLL)
    - Reliability Diagram
    """

    @staticmethod
    def _bin_stats(confidences: torch.Tensor,
                   predictions: torch.Tensor,
                   labels: torch.Tensor,
                   num_bins: int = 15):
        """
        Helper to compute per-bin (size, avg_conf, acc).
        Returns lists (only non-empty bins kept): sizes, avg_confs, accuracies
        """
        # Sanity / shapes
        assert confidences.ndim == predictions.ndim == labels.ndim == 1, "Use 1D tensors"
        assert confidences.shape[0] == predictions.shape[0] == labels.shape[0], "Mismatched shapes"

        # Clamp to [0, 1] and detach to avoid grad shenanigans
        conf = confidences.detach().float().clamp(0, 1)
        preds = predictions.detach().long()
        targs = labels.detach().long()

        # Bin edges: [0, 1], uniform
        # Rightmost included in last bin to cover exactly 1.0
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=conf.device)

        sizes = []
        avg_confs = []
        accs = []

        for b in range(num_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            # Include right edge only for last bin
            if b < num_bins - 1:
                in_bin = (conf >= lo) & (conf < hi)
            else:
                in_bin = (conf >= lo) & (conf <= hi)

            if in_bin.any():
                conf_b = conf[in_bin]
                preds_b = preds[in_bin]
                targs_b = targs[in_bin]

                size = conf_b.numel()
                avg_conf = conf_b.mean()
                acc = (preds_b == targs_b).float().mean()

                sizes.append(size)
                avg_confs.append(avg_conf)
                accs.append(acc)

        if len(sizes) == 0:
            # No valid data; return zeros to avoid div-by-zero
            return [0], [torch.tensor(0., device=conf.device)], [torch.tensor(0., device=conf.device)]

        return sizes, avg_confs, accs

    @staticmethod
    def expected_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        ECE = Σ_b (|acc_b - conf_b|) * (n_b / N)
        """
        sizes, avg_confs, accs = CalibrationMetrics._bin_stats(
            confidences, predictions, labels, num_bins
        )
        N = float(sum(sizes))
        if N == 0:
            return 0.0

        ece = 0.0
        for n_b, c_b, a_b in zip(sizes, avg_confs, accs):
            gap = torch.abs(a_b - c_b).item()
            ece += gap * (n_b / N)
        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        MCE = max_b |acc_b - conf_b|
        """
        _, avg_confs, accs = CalibrationMetrics._bin_stats(
            confidences, predictions, labels, num_bins
        )
        if len(avg_confs) == 0:
            return 0.0
        mce = max(torch.abs(a - c).item() for c, a in zip(avg_confs, accs))
        return float(mce)

    @staticmethod
    def negative_log_likelihood(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Average NLL via cross-entropy.
        """
        assert logits.ndim == 2 and labels.ndim == 1, "logits (N,C), labels (N,)"
        nll = F.cross_entropy(logits, labels.long(), reduction='mean')
        return float(nll.item())

    @staticmethod
    def reliability_diagram(
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
        save_path: str = None
    ) -> None:
        """
        Reliability diagram: bin-wise accuracy vs confidence.
        """
        import matplotlib.pyplot as plt

        # Convert to numpy and clamp
        conf = np.asarray(confidences, dtype=np.float32)
        conf = np.clip(conf, 0.0, 1.0)
        preds = np.asarray(predictions, dtype=np.int64)
        targs = np.asarray(labels, dtype=np.int64)
        assert conf.ndim == preds.ndim == targs.ndim == 1 and len(conf) == len(preds) == len(targs)

        # Binning
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        # Midpoints for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        bin_acc = np.zeros(num_bins, dtype=np.float64)
        bin_conf = np.zeros(num_bins, dtype=np.float64)
        bin_count = np.zeros(num_bins, dtype=np.int64)

        inds = np.digitize(conf, bin_edges[1:-1], right=False)  # 0..num_bins-1
        for b in range(num_bins):
            mask = inds == b
            if np.any(mask):
                c_b = conf[mask]
                p_b = preds[mask]
                y_b = targs[mask]
                bin_count[b] = c_b.size
                bin_conf[b] = float(np.mean(c_b))
                bin_acc[b] = float(np.mean(p_b == y_b))

        # ECE for annotation (ignore empty bins)
        nonempty = bin_count > 0
        if np.any(nonempty):
            weights = bin_count[nonempty] / np.sum(bin_count[nonempty])
            ece = np.sum(np.abs(bin_acc[nonempty] - bin_conf[nonempty]) * weights)
        else:
            ece = 0.0

        # Plot
        plt.figure(figsize=(6, 6))
        # Bars: accuracy per bin at bin centers
        width = 1.0 / num_bins * 0.9
        plt.bar(bin_centers, bin_acc, width=width, align='center', edgecolor='black', linewidth=0.5, alpha=0.8, label='Accuracy')
        # Line: perfect calibration
        plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.0, label='Perfect calibration')

        # Also overlay average confidence as markers where bins non-empty
        plt.scatter(bin_centers[nonempty], bin_conf[nonempty], marker='o', s=20, label='Mean confidence')

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram (ECE = {ece:.3f})')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', linewidth=0.5)

        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
        else:
            plt.show()
        plt.close()


class UncertaintyWeightedFusion(nn.Module):
    """
    Fuse modalities weighted by inverse uncertainty.
    Weight_i ∝ 1 / (uncertainty_i + ε)
    """
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        modality_predictions: Dict[str, torch.Tensor],   # {modality: (B, C)}
        modality_uncertainties: Dict[str, torch.Tensor], # {modality: (B,)}
        modality_mask: torch.Tensor                      # (B, M) in the same modality order
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_logits:   (B, C)
            fusion_weights: (B, M)
        """
        assert isinstance(modality_predictions, dict) and isinstance(modality_uncertainties, dict)
        modalities = list(modality_predictions.keys())

        # Stack tensors
        logits = torch.stack([modality_predictions[m] for m in modalities], dim=1)   # (B, M, C)
        uncert = torch.stack([modality_uncertainties[m] for m in modalities], dim=1) # (B, M)

        mask = modality_mask.to(dtype=logits.dtype, device=logits.device)            # (B, M)

        # Inverse-uncertainty weights
        inv_w = 1.0 / (uncert.to(logits.device) + self.epsilon)                      # (B, M)
        inv_w = inv_w * mask                                                         # zero out missing modalities

        # Normalize
        weight_sum = inv_w.sum(dim=1, keepdim=True) + self.epsilon
        fusion_weights = inv_w / weight_sum                                          # (B, M)

        # Weighted sum of logits over modalities
        fused_logits = (fusion_weights.unsqueeze(-1) * logits).sum(dim=1)            # (B, C)
        return fused_logits, fusion_weights      



class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling.

    Learns a single temperature parameter T that scales logits:
    P_calibrated = softmax(logits / T)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: (batch_size, num_classes)

        Returns:
            scaled_logits: (batch_size, num_classes)
        """
        # Safety: avoid non-positive T at inference time
        T = torch.clamp(self.temperature, min=1e-6)
        return logits / T

    @torch.no_grad()
    def _project_temperature_(self, min_val: float = 1e-6, max_val: float = 1e6) -> None:
        # Ensure T stays positive and finite
        self.temperature.data.clamp_(min=min_val, max=max_val)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> None:
        """
        Learn optimal temperature on a validation set by minimizing NLL.

        Args:
            logits: (N, C) validation logits (detached; model weights frozen)
            labels: (N,) validation labels
            lr: optimizer learning rate
            max_iter: max LBFGS iterations
        """
        assert logits.ndim == 2 and labels.ndim == 1, "logits (N,C), labels (N,)"
        device = self.temperature.device
        logits = logits.detach().to(device)
        labels = labels.detach().long().to(device)

        # Initialize T = 1.0
        with torch.no_grad():
            self.temperature.fill_(1.0)

        # Train the single parameter T
        self.train()

        # Prefer LBFGS per Guo et al.; fall back to Adam if needed
        try:
            optimizer = torch.optim.LBFGS(
                [self.temperature],
                lr=lr,
                max_iter=max_iter,
                line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                # NOTE: no grad needed for logits; only for T
                scaled = logits / torch.clamp(self.temperature, min=1e-6)
                loss = F.cross_entropy(scaled, labels, reduction="mean")
                loss.backward()
                return loss

            loss = optimizer.step(closure)  # returns final loss
            # Small projection to keep T in a sane range
            self._project_temperature_()

        except Exception:
            # Fallback: Adam for robustness
            optimizer = torch.optim.Adam([self.temperature], lr=lr)
            for _ in range(max_iter):
                optimizer.zero_grad(set_to_none=True)
                scaled = logits / torch.clamp(self.temperature, min=1e-6)
                loss = F.cross_entropy(scaled, labels, reduction="mean")
                loss.backward()
                optimizer.step()
                self._project_temperature_()

        # Use eval mode by default after calibration
        self.eval()
        
        raise NotImplementedError("Implement temperature calibration")


class EnsembleUncertainty:
    """
    Estimate uncertainty via ensemble of models.
    
    Train multiple models with different initializations/data splits.
    Uncertainty = variance across ensemble predictions.
    """
    
    def __init__(self, models: list):
        """
        Args:
            models: List of trained models (same architecture)
        """
        self.models = models
        self.num_models = len(models)
    
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and uncertainty from ensemble.
        
        Args:
            inputs: Model inputs
            
        Returns:
            mean_predictions: (batch_size, num_classes) - average prediction
            uncertainty: (batch_size,) - prediction variance
        """
        # TODO: Implement ensemble prediction
        # Steps:
        #   1. Get predictions from all models
        #   2. Compute mean prediction
        #   3. Compute variance as uncertainty measure
        #   4. Return mean and uncertainty
        with torch.inference_mode():
            preds = []
            for model in self.models:
                model.eval()
                logits = model(inputs)                    # (B, C)
                probs = torch.softmax(logits, dim=-1)     # (B, C)
                preds.append(probs)

            # (M, B, C)
            stack = torch.stack(preds, dim=0)
            mean_predictions = stack.mean(dim=0)
            uncertainty = stack.var(dim=0, unbiased=False).mean(dim=-1)

            return mean_predictions, uncertainty


def compute_calibration_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    num_bins: int = 15,
) -> Dict[str, float]:
    """
    Computes ECE, MCE, NLL, Accuracy on the given dataloader.
    Supports batches like:
      - (inputs, labels)
      - (features_dict, labels, mask)
    Also unwraps models that return (logits, aux).
    """
    model.eval().to(device)

    all_confidences, all_predictions, all_labels = [], [], []
    total_nll, total = 0.0, 0

    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                inputs, labels = batch
                mask = None
            elif len(batch) == 3:
                inputs, labels, mask = batch
            else:
                raise ValueError(f"Unexpected batch shape: len={len(batch)}")
        else:
            raise ValueError("Batch must be a tuple/list")

        labels = labels.to(device)

        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            mask = mask.to(device) if mask is not None else None
            logits = model(inputs, mask)
        else:
            inputs = inputs.to(device)
            logits = model(inputs)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        batch_nll = F.cross_entropy(logits, labels, reduction='sum')
        total_nll += float(batch_nll.item())
        total += labels.size(0)

        probs = F.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        all_confidences.append(confs.cpu())
        all_predictions.append(preds.cpu())
        all_labels.append(labels.cpu())

    confidences = torch.cat(all_confidences) if all_confidences else torch.zeros(0)
    predictions = torch.cat(all_predictions) if all_predictions else torch.zeros(0, dtype=torch.long)
    labels = torch.cat(all_labels) if all_labels else torch.zeros(0, dtype=torch.long)

    accuracy = float((predictions == labels).float().mean().item()) if labels.numel() else 0.0
    nll = float(total_nll / max(1, total))

    ece = CalibrationMetrics.expected_calibration_error(
        confidences, predictions, labels, num_bins=num_bins
    )
    mce = CalibrationMetrics.maximum_calibration_error(
        confidences, predictions, labels, num_bins=num_bins
    )

    return {
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "accuracy": accuracy,
    }

if __name__ == '__main__':
    # Test calibration metrics
    print("Testing calibration metrics...")
    
    # Generate fake predictions
    num_samples = 1000
    num_classes = 10
    
    # Well-calibrated predictions
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    
    # Test ECE
    try:
        ece = CalibrationMetrics.expected_calibration_error(
            confidences, predictions, labels
        )
        print(f"✓ ECE computed: {ece:.4f}")
    except NotImplementedError:
        print("✗ ECE not implemented yet")
    
    # Test reliability diagram
    try:
        CalibrationMetrics.reliability_diagram(
            confidences.numpy(),
            predictions.numpy(),
            labels.numpy(),
            save_path='test_reliability.png'
        )
        print("✓ Reliability diagram created")
    except NotImplementedError:
        print("✗ Reliability diagram not implemented yet")
