import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        length = x.size(1)
        return x + self.pe[:, :length]


class GlobalPredictor(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        num_layers=2,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, feature_dim),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        latent = self.encoder(x)
        last_feat = latent[:, -1, :]
        pred = self.decoder(last_feat)
        return latent, pred

