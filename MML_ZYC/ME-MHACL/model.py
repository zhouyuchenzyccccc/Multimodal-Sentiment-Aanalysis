# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalEncoder(nn.Module):
    """
    Encodes EEG, eye-tracking, and physiological signals and fuses them via multi-head attention.
    """

    def __init__(self, feat_dim=256, num_heads=8):
        super(MultiModalEncoder, self).__init__()
        # EEG Encoder: input shape (batch, 32 channels, 585 timepoints)
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),  # global average pooling
            nn.Flatten(),
            nn.Linear(128, feat_dim)
        )
        # Eye-Tracking Encoder: input shape (batch, 1, 38)
        self.eye_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(32, feat_dim)
        )
        # Physiological Signals Encoder: input shape (batch, 1, 230)
        self.phy_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(64, feat_dim)
        )
        # Multi-head Attention for feature fusion (8 heads as in related work&#8203;:contentReference[oaicite:4]{index=4})
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=False)

    def forward(self, eeg, eye, phy):
        """
        Forward pass: encode each modality, then fuse.
        - eeg: Tensor of shape (batch, 32, 585)
        - eye: Tensor of shape (batch, 38) or (batch, 1, 38)
        - phy: Tensor of shape (batch, 230) or (batch, 1, 230)
        Returns: fused feature of shape (batch, feat_dim).
        """
        # Encode EEG
        x_eeg = self.eeg_encoder(eeg)  # (batch, feat_dim)
        # Encode eye movement (ensure shape (batch,1,38) for Conv1d)
        if eye.dim() == 2:
            eye = eye.unsqueeze(1)  # (batch,1,38)
        x_eye = self.eye_encoder(eye)  # (batch, feat_dim)
        # Encode physiological (ensure shape (batch,1,230))
        if phy.dim() == 2:
            phy = phy.unsqueeze(1)  # (batch,1,230)
        x_phy = self.phy_encoder(phy)  # (batch, feat_dim)

        # Stack features as sequence of length 3 for attention: shape (3, batch, feat_dim)
        feats = torch.stack([x_eeg, x_eye, x_phy], dim=0)
        # Multi-head attention fusion (queries=keys=values=feats)
        attn_out, _ = self.multihead_attn(feats, feats, feats)
        # Aggregate the outputs by averaging over the modality dimension
        fused = attn_out.mean(dim=0)  # (batch, feat_dim)
        return fused


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning (e.g., SimCLR-style).
    """

    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.5),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    """
    Classifier head for fine-tuning: outputs two logits (arousal, valence).
    """

    def __init__(self, in_dim=256, hidden_dim=128):
        super(Classifier, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc_arousal = nn.Linear(hidden_dim, 2)  # binary arousal (2 classes)
        self.fc_valence = nn.Linear(hidden_dim, 2)  # binary valence (2 classes)

    def forward(self, x):
        h = self.shared(x)
        out_a = self.fc_arousal(h)
        out_v = self.fc_valence(h)
        return out_a, out_v
