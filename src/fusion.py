"""
Multimodal Fusion Architectures for Sensor Integration

This module implements three fusion strategies:
1. Early Fusion: Concatenate features before processing
2. Late Fusion: Independent processing, combine predictions
3. Hybrid Fusion: Cross-modal attention + learned weighting
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from attention import CrossModalAttention
from uncertainty import UncertaintyWeightedFusion

import torch.nn.functional as F

class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate encoder outputs and process jointly.
    
    Pros: Joint representation learning across modalities
    Cons: Requires temporal alignment, sensitive to missing modalities
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1,
        num_heads: int = 4,
        **kwargs,   
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
                          Example: {'video': 512, 'imu': 64}
            hidden_dim: Hidden dimension for fusion network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self._num_heads = num_heads
        self.modality_names = list(modality_dims.keys())
        
        # TODO: Implement early fusion architecture
        # Hint: Concatenate all modality features, pass through MLP
        # Architecture suggestion:
        #   concat_dim = sum(modality_dims.values())
        #   Linear(concat_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, num_classes)
        
        self.modality_names = list(modality_dims.keys())
        self.modality_dims = modality_dims
        concat_dim = sum(modality_dims.values())

        layers = []
        layers.append(nn.Linear(concat_dim, hidden_dim))

        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor shape: (batch_size, feature_dim)
            modality_mask: Binary mask (batch_size, num_modalities)
                          1 = available, 0 = missing
                          
        Returns:
            logits: (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Extract features for each modality from dict
        #   2. Handle missing modalities (use zeros or learned embeddings)
        #   3. Concatenate all features
        #   4. Pass through fusion network
        
        ref = None
        for name in self.modality_names:
            if name in modality_features:
                ref = modality_features[name]
                break
        if ref is None:
            raise ValueError("No expected modalities were provided.")
        if ref.dim() != 2:
            raise ValueError(f"Expected rank-2 modality tensors [B, D], got {tuple(ref.shape)}.")

        B, device, dtype = ref.size(0), ref.device, ref.dtype

        if modality_mask is not None:
            if modality_mask.dim() != 2 or modality_mask.size(0) != B or modality_mask.size(1) != len(self.modality_names):
                raise ValueError(f"modality_mask must be [B, {len(self.modality_names)}]")
            modality_mask = modality_mask.to(device=device, dtype=dtype)

        fused = []
        for i, name in enumerate(self.modality_names):
            Dm = self.modality_dims[name]
            if name in modality_features:
                x = modality_features[name]
                if x.dim() != 2 or x.size(0) != B or x.size(1) != Dm:
                    raise ValueError(f"Modality '{name}' must be [B, {Dm}], got {tuple(x.shape)}")
            else:
                x = torch.zeros(B, Dm, device=device, dtype=dtype)

            if modality_mask is not None:
                m = modality_mask[:, i].unsqueeze(-1)  # [B, 1]
                if hasattr(self, "missing_embeddings") and isinstance(self.missing_embeddings, nn.ParameterDict) and name in self.missing_embeddings:
                    token = self.missing_embeddings[name].to(device=device, dtype=dtype).unsqueeze(0).expand(B, Dm)
                    x = m * x + (1.0 - m) * token
                else:
                    x = m * x

            fused.append(x)

        concat = torch.cat(fused, dim=-1)   # [B, sum D_m]
        logits = self.mlp(concat)           # [B, num_classes]
        return logits


class LateFusion(nn.Module):
    """
    Late fusion: Independent classifiers per modality, combine predictions.
    
    Pros: Handles asynchronous sensors, modular per-modality training
    Cons: Limited cross-modal interaction, fusion only at decision level
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1,
        num_heads: int = 4,   # <— added to match build_fusion_model kwargs
        **kwargs,   
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for per-modality classifiers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self._num_heads = num_heads
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        
        # TODO: Create separate classifier for each modality
        # Hint: Use nn.ModuleDict to store per-modality classifiers
        # Each classifier: Linear(modality_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)
        
        # TODO: Learn fusion weights (how to combine predictions)
        # Option 1: Learnable weights (nn.Parameter)
        # Option 2: Attention over predictions
        # Option 3: Simple averaging
        
        self.classifiers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            for name, dim in modality_dims.items()
        })
        self.fusion_logits = nn.Parameter(torch.zeros(self.num_modalities))
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with late fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            
        Returns:
            logits: (batch_size, num_classes) - fused predictions
            per_modality_logits: Dict of individual modality predictions
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Get predictions from each modality classifier
        #   2. Handle missing modalities (mask out or skip)
        #   3. Combine predictions using fusion weights
        #   4. Return both fused and per-modality predictions
        
        per_modality_logits: Dict[str, torch.Tensor] = {}
        ref = None
        for name in self.modality_names:
            if name in modality_features:
                ref = modality_features[name]
                break
        if ref is None:
            raise ValueError("No expected modalities were provided.")
        B, device, dtype = ref.size(0), ref.device, ref.dtype

        logits_list = []
        for name in self.modality_names:
            if name in modality_features:
                x = modality_features[name]
                per_logit = self.classifiers[name](x)  # [B, C]
            else:
                # Missing entirely: treat as zeros
                C = next(self.classifiers.values())[-1].out_features
                per_logit = torch.zeros(B, C, device=device, dtype=dtype)
            per_modality_logits[name] = per_logit
            logits_list.append(per_logit)

        logits_stacked = torch.stack(logits_list, dim=1)
        
        base_w = torch.softmax(self.fusion_logits, dim=0)

        if modality_mask is not None:
            w = base_w.unsqueeze(0).to(device=device, dtype=dtype) * modality_mask.to(device=device, dtype=dtype)
            denom = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
            w = w / denom
        else:
            w = base_w.unsqueeze(0).expand(B, -1)  # [B, M]

        fused_logits = (w.unsqueeze(-1) * logits_stacked).sum(dim=1)  # [B, C]
        
        return fused_logits, per_modality_logits


# class HybridFusion(nn.Module):
#     """
#     Hybrid fusion: Cross-modal attention + learned fusion weights.
    
#     Pros: Rich cross-modal interaction, robust to missing modalities
#     Cons: More complex, higher computation cost
    
#     This is the main focus of the assignment!
#     """
    
#     def __init__(
#         self,
#         modality_dims: Dict[str, int],
#         hidden_dim: int = 256,
#         num_classes: int = 11,
#         num_heads: int = 4,
#         dropout: float = 0.1
#     ):
#         """
#         Args:
#             modality_dims: Dictionary mapping modality name to feature dimension
#             hidden_dim: Hidden dimension for fusion
#             num_classes: Number of output classes
#             num_heads: Number of attention heads
#             dropout: Dropout probability
#         """
#         super().__init__()
#         self.modality_names = list(modality_dims.keys())
#         self.num_modalities = len(self.modality_names)
#         self.hidden_dim = hidden_dim
        
#         # TODO: Project each modality to common hidden dimension
#         # Hint: Use nn.ModuleDict with Linear layers per modality
#         self.proj_layers = nn.ModuleDict({
#             name: nn.Linear(dim, hidden_dim) for name, dim in modality_dims.items()
#         })
        
#         # TODO: Implement cross-modal attention
#         # Use CrossModalAttention from attention.py
#         # Each modality should attend to all other modalities
#         self.attn_layers = nn.ModuleDict({
#             name: CrossModalAttention(
#                 query_dim=self.hidden_dim,
#                 key_dim=self.hidden_dim,
#                 hidden_dim=self.hidden_dim,
#                 num_heads=num_heads,
#                 dropout=dropout,
#             )
#             for name in self.modality_names
#         })

#         # TODO: Learn adaptive fusion weights based on modality availability
#         # Hint: Small MLP that takes modality mask and outputs weights
#         fusion_hidden = max(4, 2 * self.num_modalities)
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(self.num_modalities, fusion_hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(fusion_hidden, self.num_modalities)
#         )
        
#         # TODO: Final classifier
#         # Takes fused representation -> num_classes logits
#         self.classifier = nn.Linear(hidden_dim, num_classes)
    
#     def forward(
#         self,
#         modality_features: Dict[str, torch.Tensor],
#         modality_mask: Optional[torch.Tensor] = None,
#         return_attention: bool = False
#     ) -> Tuple[torch.Tensor, Optional[Dict]]:
#         """
#         Forward pass with hybrid fusion.
        
#         Args:
#             modality_features: Dict of {modality_name: features}
#             modality_mask: Binary mask for available modalities
#             return_attention: If True, return attention weights for visualization
            
#         Returns:
#             logits: (batch_size, num_classes)
#             attention_info: Optional dict with attention weights and fusion weights
#         """
#         # TODO: Implement forward pass
#         # Steps:
#         #   1. Project all modalities to common hidden dimension
#         #   2. Apply cross-modal attention between modality pairs
#         #   3. Compute adaptive fusion weights based on modality_mask
#         #   4. Fuse attended representations with learned weights
#         #   5. Pass through final classifier
#         #   6. Optionally return attention weights for visualization
        
#         ref = modality_features[self.modality_names[0]]
#         B, device, dtype = ref.size(0), ref.device, ref.dtype

#         if modality_mask is None:
#             modality_mask = torch.ones(B, self.num_modalities, device=device, dtype=dtype)
#         else:
#             modality_mask = modality_mask.to(device=device, dtype=dtype)

#         proj_list = []
#         for i, name in enumerate(self.modality_names):
#             if name in modality_features:
#                 x = modality_features[name].to(device=device, dtype=dtype)
#             else:
#                 x = torch.zeros(B, self.proj_layers[name].in_features, device=device, dtype=dtype)
#                 modality_mask[:, i] = 0.0
#             z = self.proj_layers[name](x)
#             proj_list.append(z)


#         Z = torch.stack(proj_list, dim=1)
#         M, D = self.num_modalities, self.hidden_dim
        
        
#         attn_mask = (modality_mask == 0).bool()  # [B, M]

#         attended_list = []
#         attn_info = {} if return_attention else None

#         for i, name in enumerate(self.modality_names):
#             q = Z[:, i:i+1, :]  # [B, 1, D]
#             k = Z               # [B, M, D]
#             v = Z               # [B, M, D]

#             result = self.attn_layers[name](q, k, v, attn_mask)

#             if isinstance(result, tuple):
#                 out, attn_w = result
#             else:
#                 out, attn_w = result, None

#             attended_list.append(out.squeeze(1))  # [B, D]
#             if return_attention:
#                 attn_info[name] = attn_w
                
#         H_att = torch.stack(attended_list, dim=1)
        
#         fusion_logits = self.fusion_mlp(modality_mask)
        
#         fusion_logits = fusion_logits.masked_fill(modality_mask == 0, float('-inf'))
#         fusion_weights = torch.softmax(fusion_logits, dim=-1)
        
#         fusion_weights = torch.where(torch.isfinite(fusion_weights), fusion_weights, torch.zeros_like(fusion_weights))
#         denom = fusion_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
#         fusion_weights = fusion_weights / denom

#         fused = (fusion_weights.unsqueeze(-1) * H_att).sum(dim=1)

#         logits = self.classifier(fused)
        
#         return logits

class HybridFusion(nn.Module):
    """
    Hybrid fusion: Cross-modal attention + learned fusion weights.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim

        # Per-modality projection to a common hidden dim
        self.proj_layers = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim) for name, dim in modality_dims.items()
        })

        # Cross-modal attention per modality
        self.attn_layers = nn.ModuleDict({
            name: CrossModalAttention(
                query_dim=self.hidden_dim,
                key_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for name in self.modality_names
        })

        # Stabilization around attention
        self.pre_ln  = nn.LayerNorm(hidden_dim)
        self.post_ln = nn.LayerNorm(hidden_dim)

        # Content-aware fusion gate (per-modality score from features)
        gate_hidden = max(32, hidden_dim // 2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)  # score per modality
        )

        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        # Reference for batch/device/dtype
        ref = modality_features[self.modality_names[0]]
        B, device, dtype = ref.size(0), ref.device, ref.dtype

        # Build/normalize masks
        # - float mask for gating (1=available, 0=missing)
        # - bool mask for attention (True=invalid key)
        if modality_mask is None:
            mask_f = torch.ones(B, self.num_modalities, device=device, dtype=torch.float32)
        else:
            mask_f = modality_mask.to(device=device)
            if mask_f.dtype not in (torch.float32, torch.float64):
                mask_f = mask_f.float()
        attn_mask = (mask_f <= 0)  # [B, M] bool

        # Project modalities (zero-fill if absent)
        proj_list = []
        for i, name in enumerate(self.modality_names):
            if name in modality_features and modality_features[name] is not None:
                x = modality_features[name].to(device=device, dtype=dtype)
            else:
                Din = self.proj_layers[name].in_features
                x = torch.zeros(B, Din, device=device, dtype=dtype)
                mask_f[:, i] = 0.0
                attn_mask[:, i] = True
            z = self.proj_layers[name](x)  # [B, D]
            proj_list.append(z)

        # Stack and normalize: Z [B, M, D]
        Z = torch.stack(proj_list, dim=1)
        Z = self.pre_ln(Z)

        # Cross-modal attention for each modality, with residual + LN
        attended_list = []
        attn_info = {} if return_attention else None
        for i, name in enumerate(self.modality_names):
            q = Z[:, i:i+1, :]  # [B,1,D]
            k = Z               # [B,M,D]
            v = Z
            out, attn_w = self.attn_layers[name](q, k, v, attn_mask)  # out: [B,1,D]
            out = out.squeeze(1)
            out = self.post_ln(out + Z[:, i, :])  # residual + LN
            attended_list.append(out)
            if return_attention:
                attn_info[name] = attn_w  # [B,H,1,M]

        # H_att: [B, M, D]
        H_att = torch.stack(attended_list, dim=1)

        # Content-aware fusion: score each modality and softmax over available ones
        scores = self.fusion_gate(H_att).squeeze(-1)  # [B, M]
        # Use large negative instead of -inf for AMP/fp16 safety
        scores = scores.masked_fill(attn_mask, -1e4)
        fusion_weights = torch.softmax(scores, dim=-1)  # [B, M]
        # Renormalize (all-missing guard)
        fusion_weights = torch.where(torch.isfinite(fusion_weights), fusion_weights, torch.zeros_like(fusion_weights))
        denom = fusion_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fusion_weights = fusion_weights / denom

        # Fuse and classify
        fused = (fusion_weights.unsqueeze(-1) * H_att).sum(dim=1)  # [B, D]
        logits = self.classifier(fused)

        if return_attention:
            return logits, {
                "fusion_weights": fusion_weights,      # [B, M]
                "per_modality_attention": attn_info,   # dict: name -> [B,H,1,M]
                "H_att": H_att,                        # [B, M, D]
            }
        return logits
        
    
    def compute_adaptive_weights(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights based on modality availability.
        
        Args:
            modality_features: Dict of modality features
            modality_mask: (batch_size, num_modalities) binary mask
            
        Returns:
            weights: (batch_size, num_modalities) normalized fusion weights
        """
        # TODO: Implement adaptive weighting
        # Ideas:
        #   1. Learn weight predictor from modality features + mask
        #   2. Higher weights for more reliable/informative modalities
        #   3. Ensure weights sum to 1 (softmax) and respect mask
        
        scores = []
        ordered_feats = []
        for name in self.modality_names:
            x = modality_features.get(name, None)
            if x is None:
                if ordered_feats:
                    B = ordered_feats[0].size(0)
                    device = ordered_feats[0].device
                    dtype = ordered_feats[0].dtype
                else:
                    B = modality_mask.size(0)
                    device = modality_mask.device
                    dtype = torch.float32
                Dm = getattr(self, "modality_dims", {}).get(name, 1)
                x = torch.zeros(B, Dm, device=device, dtype=dtype)
            ordered_feats.append(x)

            D = x.size(1)
            score = torch.linalg.norm(x, ord=2, dim=1) / (D ** 0.5)
            scores.append(score)  # [B]

        # [B, M]
        scores = torch.stack(scores, dim=1)
        scores = scores.to(modality_mask.device, dtype=modality_mask.dtype)

        # Unavailable modalities to -inf so softmax -> 0
        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = torch.where(modality_mask > 0, scores, torch.full_like(scores, neg_inf))

        # Normalize
        weights = torch.softmax(masked_scores, dim=-1)  # [B, M]
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))

        return weights
        
class LateFusionWithUncertainty(nn.Module):
    """
    Encoders -> per-modality logits + scalar uncertainty -> inverse-uncertainty fusion.
    Accepts (encoded_features: Dict[str, (B,D)], modality_mask: (B,M)).
    """
    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        hidden_dim: int = 0,
        num_heads: int = 0,   # kept for API compatibility
        dropout: float = 0.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.modality_dims = modality_dims
        self.num_classes = num_classes

        # Per-modality classifier & scalar-uncertainty heads
        self.classifiers = nn.ModuleDict()
        self.uncert_heads = nn.ModuleDict()

        for m, d in modality_dims.items():
            if hidden_dim and hidden_dim > 0:
                self.classifiers[m] = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(d, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )
                self.uncert_heads[m] = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(d, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
            else:
                self.classifiers[m] = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))
                self.uncert_heads[m] = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

        self.fuser = UncertaintyWeightedFusion(epsilon=epsilon)

    def forward(
        self,
        encoded_features: Dict[str, torch.Tensor],   # {modality: (B,D)}
        modality_mask: torch.Tensor                  # (B,M) order == self.modalities
    ):
        assert modality_mask is not None, "modality_mask (B,M) is required."
        B = modality_mask.size(0)

        # device/dtype defaults
        sample = next(iter(encoded_features.values())) if encoded_features else None
        feat_device = sample.device if sample is not None else modality_mask.device
        feat_dtype = sample.dtype if sample is not None else torch.float32

        logits_dict, uncert_dict = {}, {}
        for m in self.modalities:
            x = encoded_features.get(m, torch.zeros(B, self.modality_dims[m], device=feat_device, dtype=feat_dtype))
            if x.size(0) != B:
                raise ValueError(f"Batch mismatch for modality '{m}': {x.size(0)} vs {B}")
            logits_m = self.classifiers[m](x)                      # (B,C)
            u_pos    = F.softplus(self.uncert_heads[m](x)).squeeze(-1)  # (B,)
            logits_dict[m] = logits_m
            uncert_dict[m] = u_pos

        fused_logits, fusion_weights = self.fuser(logits_dict, uncert_dict, modality_mask)
        per_modality_logits = torch.stack([logits_dict[m] for m in self.modalities], dim=1)  # (B,M,C)
        return fused_logits, {"per_modality_logits": per_modality_logits, "fusion_weights": fusion_weights}

# Helper functions

def build_fusion_model(
    fusion_type: str,
    modality_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to build fusion models.
    
    Args:
        fusion_type: One of ['early', 'late', 'hybrid']
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments for fusion model
        
    Returns:
        Fusion model instance
    """
    if fusion_type in {"uncertainty"}:
        # Build the LATE fusion wrapper that learns per-modality logits and a positive
        # uncertainty scalar, then fuses via inverse uncertainty with masking.
        return LateFusionWithUncertainty(
            modality_dims=modality_dims,
            num_classes=num_classes,
            hidden_dim=kwargs.get("hidden_dim", 0),
            num_heads=kwargs.get("num_heads", 0),   # kept for API compatibility
            dropout=kwargs.get("dropout", 0.0),
            epsilon=kwargs.get("epsilon", 1e-6),
        )
    
    fusion_classes = {
        'early': EarlyFusion,
        'late': LateFusion,
        'hybrid': HybridFusion
    }
    
    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_classes[fusion_type](
        modality_dims=modality_dims,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Simple test to verify implementation
    print("Testing fusion architectures...")
    
    # Test configuration
    modality_dims = {'video': 512, 'imu': 64}
    num_classes = 11
    batch_size = 4
    
    # Create dummy features
    features = {
        'video': torch.randn(batch_size, 512),
        'imu': torch.randn(batch_size, 64)
    }
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]])  # Different availability patterns
    
    # Test each fusion type
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\nTesting {fusion_type} fusion...")
        try:
            model = build_fusion_model(fusion_type, modality_dims, num_classes)
            
            if fusion_type == 'late':
                logits, per_mod_logits = model(features, mask)
            else:
                logits = model(features, mask)
            
            assert logits.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
            print(f"✓ {fusion_type} fusion working! Output shape: {logits.shape}")
            
        except NotImplementedError:
            print(f"✗ {fusion_type} fusion not implemented yet")
        except Exception as e:
            print(f"✗ {fusion_type} fusion error: {e}")