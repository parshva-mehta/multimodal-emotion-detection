"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. CrossModalAttention: Attention between different modalities
2. TemporalAttention: Attention over time steps in sequences
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

class CrossModalAttention(nn.Module):
    """
    Multi-head cross-modal attention.

    Inputs:
      - query: (B, D) or (B, Tq, D)
      - key:   (B, D) or (B, Tk, D)
      - value: (B, D) or (B, Tk, D)
      - mask:  None, or mask for KEYS with shape (B,), (B,1), or (B,Tk)
               Semantics: True = INVALID key (masked out).
               If a 0/1 "valid" mask is passed (1 = valid), it is auto-inverted.

    Returns:
      - out:  (B, D)  if Tq == 1, else (B, Tq, D)
      - attn: (B, H, Tq, Tk) attention weights (after softmax)
    """

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # --- B: input LayerNorms to stabilize per-modality scales ---
        self.q_in_ln = nn.LayerNorm(query_dim)
        self.k_in_ln = nn.LayerNorm(key_dim)
        self.v_in_ln = nn.LayerNorm(key_dim)

        # Projections
        self.q_proj = nn.Linear(query_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(key_dim,   hidden_dim, bias=True)
        self.v_proj = nn.Linear(key_dim,   hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)

    def _ensure_seq(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D) -> (B, 1, D)
        return x.unsqueeze(1) if x.dim() == 2 else x

    def _normalize_mask(self, mask: torch.Tensor, B: int, Tk: int) -> torch.Tensor:
        """
        Convert various mask forms to boolean INVALID mask of shape (B, Tk).
        Accepts bool or numeric masks. If numeric with 1=valid, it auto-inverts.
        """
        if mask.dtype == torch.bool:
            invalid = mask
        else:
            # Numeric mask: assume 1 = valid, 0 = invalid → convert to invalid=True for <= 0
            invalid = (mask <= 0)

        if invalid.dim() == 1:                 # (B,) -> (B,Tk)
            invalid = invalid.view(B, 1).expand(B, Tk)
        elif invalid.dim() == 2:
            if invalid.size(1) == 1:
                invalid = invalid.expand(B, Tk)
            else:
                assert invalid.size(1) == Tk, f"Mask width {invalid.size(1)} != Tk {Tk}"
        else:
            raise ValueError(f"Mask must be [B] or [B,Tk], got {invalid.shape}")

        return invalid

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Ensure time dimension
        query = self._ensure_seq(query)  # (B, Tq, Dq)
        key   = self._ensure_seq(key)    # (B, Tk, Dk)
        value = self._ensure_seq(value)  # (B, Tk, Dk)

        B, Tq, _ = query.shape
        _, Tk, _ = key.shape

        # --- B: LayerNorm on inputs BEFORE projections ---
        query = self.q_in_ln(query)
        key   = self.k_in_ln(key)
        value = self.v_in_ln(value)

        # Projections
        Q = self.q_proj(query)  # (B, Tq, H*Dh)
        K = self.k_proj(key)    # (B, Tk, H*Dh)
        V = self.v_proj(value)  # (B, Tk, H*Dh)

        # Split heads -> (B, H, T, Dh)
        Q = Q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Tq,Dh)
        K = K.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Tk,Dh)
        V = V.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Tk,Dh)

        # Attention scores: (B, H, Tq, Tk)
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale

        # --- A: numerically safe masking (fp16-friendly) ---
        if mask is not None:
            invalid = self._normalize_mask(mask, B, Tk).to(dtype=torch.bool, device=attn_scores.device)
            neg_large = -1e4  # safer than -inf under mixed precision
            attn_scores = attn_scores.masked_fill(invalid[:, None, None, :], neg_large)

        # Softmax
        attn = torch.softmax(attn_scores, dim=-1)

        # --- A: if an entire row was masked, set distribution to zeros (no NaNs) ---
        if mask is not None:
            invalid = self._normalize_mask(mask, B, Tk).to(dtype=torch.bool, device=attn.device)
            all_masked = invalid.all(dim=-1)  # (B,)
            if all_masked.any():
                attn[all_masked, :, :, :] = 0.0

        attn = self.attn_dropout(attn)

        # Weighted sum -> (B, H, Tq, Dh)
        context = attn @ V

        # Merge heads -> (B, Tq, H*Dh) -> (B, Tq, D)
        context = context.transpose(1, 2).contiguous().view(B, Tq, self.hidden_dim)
        out = self.out_proj(context)  # (B, Tq, D)

        # Squeeze if single query step
        out = out.squeeze(1) if Tq == 1 else out
        return out, attn  # (B, D) or (B, Tq, D), and (B, H, Tq, Tk)       

class TemporalAttention(nn.Module):
    """
    Temporal attention: Attend over sequence of time steps.
    
    Useful for: Variable-length sequences, weighting important timesteps
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # TODO: Implement self-attention over temporal dimension
        # Hint: Similar to CrossModalAttention but Q, K, V from same modality
        
        self.q_proj = nn.Linear(feature_dim, hidden_dim)
        self.k_proj = nn.Linear(feature_dim, hidden_dim)
        self.v_proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
    
    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.
        
        Args:
            sequence: (batch_size, seq_len, feature_dim) - temporal sequence
            mask: Optional (batch_size, seq_len) - binary mask for valid timesteps
            
        Returns:
            attended_sequence: (batch_size, seq_len, hidden_dim) - attended features
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # TODO: Implement temporal self-attention
        # Steps:
        #   1. Project sequence to Q, K, V
        #   2. Compute self-attention over sequence length
        #   3. Apply mask for variable-length sequences
        #   4. Return attended sequence and weights
        
        B, S, _ = sequence.shape
        H = self.num_heads
        Hd = self.head_dim
        scale = Hd ** -0.5

        q = self.q_proj(sequence)  # (B, S, H*Hd)
        k = self.k_proj(sequence)  # (B, S, H*Hd)
        v = self.v_proj(sequence)  # (B, S, H*Hd)

        def to_heads(x):
            return x.view(B, S, H, Hd).permute(0, 2, 1, 3)
        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  

        if mask is not None:
            key_mask = mask.to(dtype=torch.bool)                     
            attn_logits = attn_logits.masked_fill(
                ~key_mask[:, None, None, :],
                float("-inf")
            )

        attention_weights = torch.softmax(attn_logits, dim=-1)        
        if hasattr(self, "attn_drop") and self.attn_drop is not None:
            attention_weights = self.attn_drop(attention_weights)

        context = torch.matmul(attention_weights, v)                 

        if mask is not None:
            query_mask = mask.to(dtype=torch.bool)                   
            context = context * query_mask[:, None, :, None]

        context = context.permute(0, 2, 1, 3).contiguous().view(B, S, H * Hd)  
        attended_sequence = self.out_proj(context)                               
        if hasattr(self, "proj_drop") and self.proj_drop is not None:
            attended_sequence = self.proj_drop(attended_sequence)

        return attended_sequence, attention_weights
        raise NotImplementedError("Implement temporal attention forward pass")
    
    def pool_sequence(
        self,
        sequence: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using attention weights.
        
        Args:
            sequence: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim) - fixed-size representation
        """
        # TODO: Implement attention-based pooling
        # Option 1: Weighted average using mean attention weights
        # Option 2: Learn pooling query vector
        # Option 3: Take output at special [CLS] token position
        
        B, S, D = sequence.shape

        attn_mean_heads = attention_weights.mean(dim=1)
        
        key_importance = attn_mean_heads.mean(dim=1)
        key_importance = key_importance / (key_importance.sum(dim=1, keepdim=True) + 1e-9)
        pooled = torch.bmm(key_importance.unsqueeze(1), sequence).squeeze(1)  # (B, D)
        
        return pooled

class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.
    
    For M modalities, computes M*(M-1)/2 pairwise attention operations.
    Example: {video, audio, IMU} -> {video<->audio, video<->IMU, audio<->IMU}
    """
    
    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dict mapping modality names to feature dimensions
                          Example: {'video': 512, 'audio': 128, 'imu': 64}
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Create CrossModalAttention for each modality pair
        # Hint: Use nn.ModuleDict with keys like "video_to_audio"
        # For each pair (A, B), create attention A->B and B->A
        self.cross_attn = nn.ModuleDict()
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                a = self.modality_names[i]
                b = self.modality_names[j]
                da, db = modality_dims[a], modality_dims[b]
                self.cross_attn[f"{a}_to_{b}"] = CrossModalAttention(
                    da, db, hidden_dim, num_heads, dropout
                )
                self.cross_attn[f"{b}_to_{a}"] = CrossModalAttention(
                    db, da, hidden_dim, num_heads, dropout
                )
    
    def forward(
        self,
        modality_features: dict,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        Apply pairwise attention between all modalities.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor: (batch_size, feature_dim)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            attended_features: Dict of {modality_name: attended_features}
            attention_maps: Dict of {f"{mod_a}_to_{mod_b}": attention_weights}
        """
        # TODO: Implement pairwise attention
        # Steps:
        #   1. For each modality pair (A, B):
        #      - Apply attention A->B (A attends to B)
        #      - Apply attention B->A (B attends to A)
        #   2. Aggregate attended features (options: sum, concat, gating)
        #   3. Handle missing modalities using mask
        #   4. Return attended features and attention maps for visualization
        modality_names = self.modality_names
        B = next(iter(modality_features.values())).size(0)

        if modality_mask is not None:
            assert modality_mask.shape == (B, self.num_modalities), \
                "modality_mask must be (batch, num_modalities)"
            name_to_idx = {m: i for i, m in enumerate(modality_names)}
            avail = {
                m: modality_mask[:, name_to_idx[m]].unsqueeze(1).to(
                    dtype=modality_features[m].dtype
                )
                for m in modality_names
            }
        else:
            avail = {m: None for m in modality_names}

        messages = {m: [] for m in modality_names} 
        attention_maps: dict = {}


        def run_cross(key: str, q, k, v, mq, mk):
            # Try the most complete signature first
            try:
                return self.cross_attn[key](q, k, v, mask_q=mq, mask_k=mk)
            except TypeError:
                try:
                    return self.cross_attn[key](q, k, v)
                except TypeError:
                    out, att = self.cross_attn[key](q, k)
                    if mq is not None:
                        out = out * mq
                    if mk is not None:
                        out = out * mk
                    return out, att

        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                a = modality_names[i]
                b = modality_names[j]
                xa, xb = modality_features[a], modality_features[b]
                ma = avail[a]
                mb = avail[b]

                # A attends to B: (q=xa, k=xb, v=xb) -> message for A
                out_ab, att_ab = run_cross(f"{a}_to_{b}", xa, xb, xb, ma, mb)
                messages[a].append(out_ab)
                attention_maps[f"{a}_to_{b}"] = att_ab

                # B attends to A: (q=xb, k=xa, v=xa) -> message for B
                out_ba, att_ba = run_cross(f"{b}_to_{a}", xb, xa, xa, mb, ma)
                messages[b].append(out_ba)
                attention_maps[f"{b}_to_{a}"] = att_ba

        # Sum messages for each target modality
        attended_features = {}
        first_tensor = next(iter(modality_features.values()))
        for m in modality_names:
            if len(messages[m]) == 0:
                agg = torch.zeros(
                    B, self.hidden_dim,
                    device=first_tensor.device,
                    dtype=first_tensor.dtype
                )
            else:
                agg = torch.stack(messages[m], dim=0).sum(dim=0)  # (B, hidden_dim)

            if avail[m] is not None:
                agg = agg * avail[m]

            attended_features[m] = agg  # (B, hidden_dim)

        # 4) Return
        return attended_features, attention_maps
    
    

class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.
    For M modalities, computes M*(M-1) directional attentions (A<-B and B<-A).
    """

    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim

        # One cross-attn module per direction
        self.cross_attn = nn.ModuleDict()
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                a = self.modality_names[i]
                b = self.modality_names[j]
                da, db = modality_dims[a], modality_dims[b]
                self.cross_attn[f"{a}_to_{b}"] = CrossModalAttention(
                    da, db, hidden_dim, num_heads, dropout
                )
                self.cross_attn[f"{b}_to_{a}"] = CrossModalAttention(
                    db, da, hidden_dim, num_heads, dropout
                )

        # Self projections so we can add a residual self path per modality
        self.self_proj = nn.ModuleDict({
            m: nn.Linear(modality_dims[m], hidden_dim) for m in self.modality_names
        })

        # Light stabilization after summing messages + self
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.msg_dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality_features: dict,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        modality_features: {name: Tensor [B, Din]}
        modality_mask: [B, M] with 1=available, 0=missing (bool/float ok)
        """
        modality_names = self.modality_names
        first = next(iter(modality_features.values()))
        B, device, dtype = first.size(0), first.device, first.dtype

        # Build availability per modality as bool [B]
        if modality_mask is not None:
            assert modality_mask.shape == (B, self.num_modalities), \
                f"modality_mask must be (B,{self.num_modalities})"
            avail_bool = {
                m: (modality_mask[:, i] > 0) if modality_mask.dtype != torch.bool
                   else modality_mask[:, i]
                for i, m in enumerate(modality_names)
            }
        else:
            avail_bool = {m: torch.ones(B, dtype=torch.bool, device=device) for m in modality_names}

        # Containers
        messages = {m: [] for m in modality_names}
        attention_maps: dict = {}

        # Helper: build key mask for CrossModalAttention (True = INVALID key)
        def key_invalid_mask(is_available_bool: torch.Tensor) -> torch.Tensor:
            # is_available_bool: [B] (True if that modality is present)
            return ~is_available_bool  # [B], True means "mask this key out"

        # Iterate pairs (A,B), do A<-B and B<-A if both present; otherwise skip
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                a = modality_names[i]
                b = modality_names[j]
                xa, xb = modality_features[a], modality_features[b]
                a_ok, b_ok = avail_bool[a], avail_bool[b]

                # If either modality is entirely missing for this sample, skip that direction
                # (we’ll still keep self residual)
                if b_ok.any():
                    # A attends to B: q=xa, k=xb, v=xb, mask over keys(B)
                    mask_k_b = key_invalid_mask(b_ok)            # [B]
                    out_ab, att_ab = self.cross_attn[f"{a}_to_{b}"](
                        xa, xb, xb, mask=mask_k_b
                    )
                    messages[a].append(out_ab)  # [B, hidden_dim]
                    attention_maps[f"{a}_to_{b}"] = att_ab
                else:
                    # no message from B to A for those samples → implicit zeros

                    pass

                if a_ok.any():
                    # B attends to A: q=xb, k=xa, v=xa, mask over keys(A)
                    mask_k_a = key_invalid_mask(a_ok)
                    out_ba, att_ba = self.cross_attn[f"{b}_to_{a}"](
                        xb, xa, xa, mask=mask_k_a
                    )
                    messages[b].append(out_ba)
                    attention_maps[f"{b}_to_{a}"] = att_ba
                else:
                    pass

        # Aggregate messages per modality + self residual, then LN
        attended_features = {}
        for m in modality_names:
            if len(messages[m]) == 0:
                msg_sum = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
            else:
                msg_sum = torch.stack(messages[m], dim=0).sum(dim=0)  # (B, hidden_dim)

            self_feat = self.self_proj[m](modality_features[m])       # (B, hidden_dim)
            agg = self.out_ln(self_feat + self.msg_dropout(msg_sum))  # residual + LN

            # If modality is missing, zero it out entirely
            if not avail_bool[m].all():
                maskv = avail_bool[m].to(agg.dtype).unsqueeze(1)      # [B,1]
                agg = agg * maskv

            attended_features[m] = agg  # (B, hidden_dim)

        return attended_features, attention_maps


def visualize_attention(
    attention_weights: torch.Tensor,
    modality_names: list,
    save_path: str = None
) -> None:
    """
    Visualize attention weights between modalities.
    
    Args:
        attention_weights: (num_heads, num_queries, num_keys) or similar
        modality_names: List of modality names for labeling
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # TODO: Implement attention visualization
    # Create heatmap showing which modalities attend to which
    # Useful for understanding fusion behavior
    if attention_weights.dim() == 4:       # (B, H, Q, K)
        mat = attention_weights.mean(dim=(0, 1))
    elif attention_weights.dim() == 3:     # (H, Q, K)
        mat = attention_weights.mean(dim=0)
    elif attention_weights.dim() == 2:     # (Q, K)
        mat = attention_weights
    else:
        raise ValueError("Unsupported attention shape.")

    mat = mat.detach().cpu().numpy()
    M = len(modality_names)
    assert mat.shape[0] == M and mat.shape[1] == M, "Shape must match len(modality_names)."

    fig, ax = plt.subplots(figsize=(1.0 + 0.5*M, 1.0 + 0.5*M))
    im = ax.imshow(mat, aspect="equal")
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    ax.set_xticklabels(modality_names, rotation=45, ha="right")
    ax.set_yticklabels(modality_names)
    ax.set_xlabel("Keys (attended)"); ax.set_ylabel("Queries (attending)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate for small matrices
    if M <= 8:
        for i in range(M):
            for j in range(M):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    # Simple test
    print("Testing attention mechanisms...")
    
    batch_size = 4
    query_dim = 512  # e.g., video features
    key_dim = 64     # e.g., IMU features
    hidden_dim = 256
    num_heads = 4
    
    # Test CrossModalAttention
    print("\nTesting CrossModalAttention...")
    try:
        attn = CrossModalAttention(query_dim, key_dim, hidden_dim, num_heads)
        
        query = torch.randn(batch_size, query_dim)
        key = torch.randn(batch_size, key_dim)
        value = torch.randn(batch_size, key_dim)
        
        attended, weights = attn(query, key, value)
        
        assert attended.shape == (batch_size, hidden_dim)
        print(f"✓ CrossModalAttention working! Output shape: {attended.shape}")
        
    except NotImplementedError:
        print("✗ CrossModalAttention not implemented yet")
    except Exception as e:
        print(f"✗ CrossModalAttention error: {e}")
    
    # Test TemporalAttention
    print("\nTesting TemporalAttention...")
    try:
        seq_len = 10
        feature_dim = 128
        
        temporal_attn = TemporalAttention(feature_dim, hidden_dim, num_heads)
        sequence = torch.randn(batch_size, seq_len, feature_dim)
        
        attended_seq, weights = temporal_attn(sequence)
        
        assert attended_seq.shape == (batch_size, seq_len, hidden_dim)
        print(f"✓ TemporalAttention working! Output shape: {attended_seq.shape}")
        
    except NotImplementedError:
        print("✗ TemporalAttention not implemented yet")
    except Exception as e:
        print(f"✗ TemporalAttention error: {e}")
