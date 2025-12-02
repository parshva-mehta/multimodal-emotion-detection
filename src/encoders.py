"""
Modality-Specific Encoders for Sensor Fusion

Implements lightweight encoders suitable for CPU training:
1. SequenceEncoder: For time-series data (IMU, audio, motion capture)
2. FrameEncoder: For frame-based data (video features)
3. SimpleMLPEncoder: For pre-extracted features
"""

import torch
import torch.nn as nn
from typing import Optional
import torchvision.models as tv_models  # <-- add this

class SequenceEncoder(nn.Module):
    """
    Encoder for sequential/time-series sensor data.
    
    Options: 1D CNN, LSTM, GRU, or Transformer
    Output: Fixed-size embedding per sequence
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = 'lstm',
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for RNN/Transformer
            output_dim: Output embedding dimension
            num_layers: Number of encoder layers
            encoder_type: One of ['lstm', 'gru', 'cnn', 'transformer']
            dropout: Dropout probability
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # TODO: Implement sequence encoder
        # Choose ONE of the following architectures:
        
        if encoder_type == 'lstm':
            # TODO: Implement LSTM encoder
            # self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
            #                    batch_first=True, dropout=dropout)
            # self.projection = nn.Linear(hidden_dim, output_dim)
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )
            self.projection = nn.Linear(hidden_dim, output_dim)
            return
            
        elif encoder_type == 'gru':
            # TODO: Implement GRU encoder
            # Similar to LSTM
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )
            self.projection = nn.Linear(hidden_dim, output_dim)
            return
            
        elif encoder_type == 'cnn':
            # TODO: Implement 1D CNN encoder
            # Stack of Conv1d -> BatchNorm -> ReLU -> Pool
            # Example:
            # self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
            # self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            # self.pool = nn.AdaptiveAvgPool1d(1)
            self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.act = nn.ReLU(inplace=True)
            self.dropout_layer = nn.Dropout(p=dropout)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.projection = nn.Linear(hidden_dim, output_dim)
            return
            
        elif encoder_type == 'transformer':
            # TODO: Implement Transformer encoder
            # encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
            # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=False,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.max_len = 4096
            self.pos_embedding = nn.Embedding(self.max_len, hidden_dim)
            self.projection = nn.Linear(hidden_dim, output_dim)
            return
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
    
    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode variable-length sequences.
        
        Args:
            sequence: (batch_size, seq_len, input_dim) - input sequence
            lengths: Optional (batch_size,) - actual sequence lengths (for padding)
            
        Returns:
            encoding: (batch_size, output_dim) - fixed-size embedding
        """
        # TODO: Implement forward pass based on encoder_type
        # Handle variable-length sequences if lengths provided
        # Return fixed-size embedding via pooling or taking last hidden state
        
        if self.encoder_type in ('lstm', 'gru'):
            B, T, _ = sequence.shape
            if lengths is not None:
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                if self.encoder_type == 'lstm':
                    _, (h_n, _) = self.rnn(packed)
                else:
                    _, h_n = self.rnn(packed)
                last = h_n[-1]
            else:
                outputs, h = self.rnn(sequence)
                if self.encoder_type == 'lstm':
                    h_n = h[0]
                else:
                    h_n = h
                last = h_n[-1]
            return self.projection(last)

        if self.encoder_type == 'cnn':
            x = sequence.transpose(1, 2)  # (B, C_in, T)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.dropout_layer(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.pool(x).squeeze(-1)  # (B, H)
            x = self.dropout_layer(x)
            return self.projection(x)

        if self.encoder_type == 'transformer':
            B, T, _ = sequence.shape
            x = self.input_proj(sequence)  # (B, T, H)
            if lengths is not None:
                device = sequence.device
                arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
                key_padding_mask = arange >= lengths.view(-1, 1)
            else:
                key_padding_mask = None
            positions = torch.arange(T, device=sequence.device).clamp_max(self.max_len - 1)
            pos = self.pos_embedding(positions)  # (T, H)
            x = x + pos.unsqueeze(0)  # (B, T, H)
            x = x.transpose(0, 1)  # (T, B, H)
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (T, B, H)
            x = x.transpose(0, 1)  # (B, T, H)
            if lengths is not None:
                mask = (~key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
                summed = (x * mask).sum(dim=1)                    # (B, H)
                denom = mask.sum(dim=1).clamp_min(1.0)            # (B, 1)
                pooled = summed / denom
            else:
                pooled = x.mean(dim=1)
            return self.projection(pooled)
        


class FrameEncoder(nn.Module):
    """
    Encoder for frame-based data (e.g., video features).
    
    Aggregates frame-level features into video-level embedding.
    """
    
    def __init__(
        self,
        frame_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        temporal_pooling: str = 'attention',
        dropout: float = 0.1
    ):
        """
        Args:
            frame_dim: Dimension of per-frame features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            temporal_pooling: How to pool frames ['average', 'max', 'attention']
            dropout: Dropout probability
        """
        super().__init__()
        self.temporal_pooling = temporal_pooling
        
        # TODO: Implement frame encoder
        # 1. Frame-level processing (optional MLP)
        # 2. Temporal aggregation (pooling or attention)
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self._dropout = nn.Dropout(dropout)
        
        if temporal_pooling == 'attention':
            # TODO: Implement attention-based pooling
            # Learn which frames are important
            # self.attention = nn.Linear(frame_dim, 1)
            self.attention = nn.Linear(hidden_dim, 1)
            pass
        elif temporal_pooling in ['average', 'max']:
            # Simple pooling, no learnable parameters needed
            pass
        else:
            raise ValueError(f"Unknown pooling: {temporal_pooling}")
        
        # TODO: Add projection layer
        # self.projection = nn.Sequential(...)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        
        return
            
    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequence of frames.
        
        Args:
            frames: (batch_size, num_frames, frame_dim) - frame features
            mask: Optional (batch_size, num_frames) - valid frame mask
            
        Returns:
            encoding: (batch_size, output_dim) - video-level embedding
        """
        # TODO: Implement forward pass
        # 1. Process frames (optional)
        # 2. Apply temporal pooling
        # 3. Project to output dimension
        x = self.frame_mlp(frames)  # (B, T, H)
        
        if self.temporal_pooling == 'attention':
            pooled = self.attention_pool(x, mask=mask)  # (B, H)
        elif self.temporal_pooling == 'average':
            if mask is not None:
                mask_f = mask.float().unsqueeze(-1)  # (B, T, 1)
                summed = (x * mask_f).sum(dim=1)     # (B, H)
                denom = mask_f.sum(dim=1).clamp_min(1.0)
                pooled = summed / denom
            else:
                pooled = x.mean(dim=1)
        elif self.temporal_pooling == 'max':
            if mask is not None:
                # set invalid to very negative before max
                very_neg = torch.finfo(x.dtype).min
                mask_bool = mask.bool().unsqueeze(-1)  # (B, T, 1)
                x_masked = x.masked_fill(~mask_bool, very_neg)
                pooled, _ = x_masked.max(dim=1)
            else:
                pooled, _ = x.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.temporal_pooling}")
        
        pooled = self._dropout(pooled)
        return self.projection(pooled)
        
    
    def attention_pool(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool frames using learned attention weights.
        
        Args:
            frames: (batch_size, num_frames, frame_dim)
            mask: Optional (batch_size, num_frames) - valid frames
            
        Returns:
            pooled: (batch_size, frame_dim) - attended frame features
        """
        # TODO: Implement attention pooling
        # 1. Compute attention scores for each frame
        # 2. Apply mask if provided
        # 3. Softmax to get weights
        # 4. Weighted sum of frames
        scores = self.attention(frames).squeeze(-1)  # (B, T)
        if mask is not None:
            very_neg = torch.finfo(scores.dtype).min
            mask_bool = mask.bool()
            scores = scores.masked_fill(~mask_bool, very_neg)
        weights = torch.softmax(scores, dim=1)       # (B, T)
        pooled = torch.einsum('bt,bth->bh', weights, frames)
        return pooled
        

class SimpleMLPEncoder(nn.Module):
    """
    Simple MLP encoder for pre-extracted features.
    
    Use this when working with pre-computed features
    (e.g., ResNet features for images, MFCC for audio).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # TODO: Implement MLP encoder
        # Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x num_layers -> Output
        
        layers = []
        current_dim = input_dim
        
        # TODO: Add hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # TODO: Add output layer
        layers.append(nn.Linear(current_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features through MLP.
        
        Args:
            features: (batch_size, input_dim) - input features
            
        Returns:
            encoding: (batch_size, output_dim) - encoded features
        """
        # TODO: Implement forward pass

        if features.dim() == 3:
            B, T, H = features.shape
            x = features.reshape(B * T, H)          # (B*T, H)
            x = self.encoder(x)                     # (B*T, output_dim)
            x = x.view(B, T, -1).mean(dim=1)        # (B, output_dim)
            return x
        return self.encoder(features)
        
        

class PretrainedCNNEncoder(nn.Module):
    """
    Encoder that uses a pretrained 2D CNN backbone (e.g., ResNet) for frame/image data.

    Supports:
      - Single images: (B, C, H, W)
      - Frame sequences: (B, T, C, H, W) with temporal pooling

    Typical use: video frames, face crops, spectrogram images, etc.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        output_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        temporal_pooling: str = "average",  # 'average' | 'max' | 'attention'
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_pooling = temporal_pooling

        # ---- Build backbone ----
        if backbone == "resnet18":
            self.backbone = tv_models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.backbone = tv_models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.backbone = tv_models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace final classification layer with identity to get feature vector
        if hasattr(self.backbone, "fc"):
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise RuntimeError("Backbone without .fc not supported in this encoder.")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self._dropout = nn.Dropout(dropout)

        # Optional attention over time (when input is (B, T, C, H, W))
        if temporal_pooling == "attention":
            self.attention = nn.Linear(feat_dim, 1)
        elif temporal_pooling in {"average", "max"}:
            self.attention = None
        else:
            raise ValueError(f"Unknown temporal_pooling: {temporal_pooling}")

        # Final projection
        self.projection = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, output_dim),
        )

    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frames:
                - (B, C, H, W)  or
                - (B, T, C, H, W)
            mask: Optional (B, T) indicating valid frames (for sequences)

        Returns:
            (B, output_dim)
        """
        if frames.dim() == 4:
            # (B, C, H, W) -> just run backbone once per image
            feats = self.backbone(frames)  # (B, F)
            feats = self._dropout(feats)
            return self.projection(feats)

        if frames.dim() == 5:
            # (B, T, C, H, W) -> flatten time into batch
            B, T, C, H, W = frames.shape
            x = frames.view(B * T, C, H, W)          # (B*T, C, H, W)
            x = self.backbone(x)                     # (B*T, F)
            x = x.view(B, T, -1)                     # (B, T, F)

            if self.temporal_pooling == "attention":
                feats = self._attention_pool(x, mask=mask)  # (B, F)
            elif self.temporal_pooling == "average":
                if mask is not None:
                    m = mask.float().unsqueeze(-1)  # (B, T, 1)
                    summed = (x * m).sum(dim=1)
                    denom = m.sum(dim=1).clamp_min(1.0)
                    feats = summed / denom
                else:
                    feats = x.mean(dim=1)
            elif self.temporal_pooling == "max":
                if mask is not None:
                    very_neg = torch.finfo(x.dtype).min
                    mask_bool = mask.bool().unsqueeze(-1)
                    x_masked = x.masked_fill(~mask_bool, very_neg)
                    feats, _ = x_masked.max(dim=1)
                else:
                    feats, _ = x.max(dim=1)
            else:
                raise ValueError(f"Unknown temporal_pooling: {self.temporal_pooling}")

            feats = self._dropout(feats)
            return self.projection(feats)

        raise ValueError(f"Expected frames dim 4 or 5, got shape {frames.shape}")

    def _attention_pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attention over time dimension for features x of shape (B, T, F).
        """
        scores = self.attention(x).squeeze(-1)  # (B, T)
        if mask is not None:
            very_neg = torch.finfo(scores.dtype).min
            mask_bool = mask.bool()
            scores = scores.masked_fill(~mask_bool, very_neg)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        pooled = torch.einsum("bt,btf->bf", weights, x)
        return pooled


def build_encoder(
    modality: str,
    input_dim: int,
    output_dim: int,
    encoder_config: dict = None
) -> nn.Module:
    """
    Factory function to build appropriate encoder for each modality.
    
    Args:
        modality: Modality name ('video', 'audio', 'imu', etc.)
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        encoder_config: Optional config dict with encoder hyperparameters
        
    Returns:
        Encoder module appropriate for the modality
    """
    if encoder_config is None:
        encoder_config = {}

    # ---- normalize & sanitize per-modality config ----
    cfg = dict(encoder_config)            # copy so we can mutate safely
    enc_type = cfg.pop("type", None)      # use & remove selector key
    in_dim = cfg.pop("input_dim", input_dim)  # prefer YAML override if present

    # If no explicit type set, choose by heuristic
    if enc_type is None:
        # Common aliases
        mod_lower = modality.lower()
        if mod_lower in {"video", "frames"}:
            enc_type = "frame"
        elif mod_lower in {"imu", "mocap", "audio", "accelerometer", "gyro", "magnetometer",
                           "imu_hand", "imu_chest", "imu_ankle"}:
            enc_type = "sequence"
        elif mod_lower in {"heart_rate", "hr"}:
            enc_type = "mlp"
        else:
            enc_type = "mlp"

    if enc_type == "frame":
        # FrameEncoder kwargs
        temporal_pooling = cfg.pop("temporal_pooling", "attention")
        hidden_dim = cfg.pop("hidden_dim", None)   
        dropout = cfg.pop("dropout", 0.1)

        return FrameEncoder(
            frame_dim=in_dim,
            hidden_dim=hidden_dim if hidden_dim is not None else output_dim * 2 // 1,
            output_dim=output_dim,
            temporal_pooling=temporal_pooling,
            dropout=dropout,
        )

    elif enc_type == "sequence":
        # SequenceEncoder kwargs
        encoder_kind = cfg.pop("encoder_type", "lstm")
        num_layers = cfg.pop("num_layers", 2)
        hidden_dim = cfg.pop("hidden_dim", None)
        dropout = cfg.pop("dropout", 0.1)

        return SequenceEncoder(
            input_dim=in_dim,
            hidden_dim=hidden_dim if hidden_dim is not None else output_dim * 2 // 1,
            output_dim=output_dim,
            num_layers=num_layers,
            encoder_type=encoder_kind,
            dropout=dropout,
        )

    elif enc_type == "mlp":
        # SimpleMLPEncoder kwargs
        num_layers = cfg.pop("num_layers", 2)
        hidden_dim = cfg.pop("hidden_dim", None)
        dropout = cfg.pop("dropout", 0.1)
        batch_norm = cfg.pop("batch_norm", True)

        return SimpleMLPEncoder(
            input_dim=in_dim,
            hidden_dim=hidden_dim if hidden_dim is not None else max(output_dim, 64),
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    elif enc_type == "pretrained_cnn":
        # Pretrained 2D CNN backbone for frames/images
        backbone = cfg.pop("backbone", "resnet18")
        pretrained = cfg.pop("pretrained", True)
        freeze_backbone = cfg.pop("freeze_backbone", False)
        temporal_pooling = cfg.pop("temporal_pooling", "average")
        dropout = cfg.pop("dropout", 0.1)

        return PretrainedCNNEncoder(
            backbone=backbone,
            output_dim=output_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            temporal_pooling=temporal_pooling,
            dropout=dropout,
        )
        
    else:
        raise ValueError(f"Unknown encoder type '{enc_type}' for modality '{modality}'")


if __name__ == '__main__':
    # Test encoders
    print("Testing encoders...")
    
    batch_size = 4
    seq_len = 100
    input_dim = 64
    output_dim = 128
    
    # Test SequenceEncoder
    print("\nTesting SequenceEncoder...")
    for enc_type in ['lstm', 'gru', 'cnn']:
        try:
            encoder = SequenceEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_type=enc_type
            )
            
            sequence = torch.randn(batch_size, seq_len, input_dim)
            output = encoder(sequence)
            
            assert output.shape == (batch_size, output_dim)
            print(f"✓ {enc_type} encoder working! Output shape: {output.shape}")
            
        except NotImplementedError:
            print(f"✗ {enc_type} encoder not implemented yet")
        except Exception as e:
            print(f"✗ {enc_type} encoder error: {e}")
    
    # Test FrameEncoder
    print("\nTesting FrameEncoder...")
    try:
        num_frames = 30
        frame_dim = 512
        
        encoder = FrameEncoder(
            frame_dim=frame_dim,
            output_dim=output_dim,
            temporal_pooling='attention'
        )
        
        frames = torch.randn(batch_size, num_frames, frame_dim)
        output = encoder(frames)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ FrameEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ FrameEncoder not implemented yet")
    except Exception as e:
        print(f"✗ FrameEncoder error: {e}")
    
    # Test SimpleMLPEncoder
    print("\nTesting SimpleMLPEncoder...")
    try:
        encoder = SimpleMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        features = torch.randn(batch_size, input_dim)
        output = encoder(features)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ SimpleMLPEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ SimpleMLPEncoder not implemented yet")
    except Exception as e:
        print(f"✗ SimpleMLPEncoder error: {e}")