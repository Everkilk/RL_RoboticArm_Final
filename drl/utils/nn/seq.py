from typing import Union, Sequence, Optional

import torch
from torch import nn

from drl.utils.nn.common import SeqGRU, Linear


class SeqGRUNet(nn.Module):
    def __init__(self, 
        obs_dim: int,
        meta_dim: int, 
        out_dim: Union[int, Sequence[int]], 
        embed_dim: int = 256,
        num_layers: int = 1, 
        hidden_mlp_dims: Sequence[int] = (256, 256),
        use_norm: bool = True, 
        activation: Union[str, nn.Module] = 'SiLU', 
        dropout: float = 0.0
    ):
        super().__init__()
        # Feature Extraction Modules -----------------
        self.seq_fe = SeqGRU(
            input_size=obs_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            use_norm=use_norm,
            dropout=dropout
        )
        
        # Multi-layer Perceptron Head ---------------
        mlp = []
        in_feats = embed_dim + meta_dim
        for hid_feats in hidden_mlp_dims:
            mlp.append(
                Linear(
                    in_features=in_feats, 
                    out_features=hid_feats, 
                    norm_layer='layer' if use_norm else 'none',
                    activation=activation, 
                    dropout=dropout
                )
            )
            in_feats = hid_feats
        mlp.append(nn.Linear(in_features=in_feats, out_features=out_dim))
        self.mlp_head = nn.Sequential(*mlp)

    def forward(self, obs: torch.Tensor, meta: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.cat([self.seq_fe(obs, mask), meta], dim=-1)
        return self.mlp_head(x)