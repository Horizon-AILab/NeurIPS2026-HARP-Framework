from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HARP_S_CNN(nn.Module):
    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        num_classes: int = 2,
        num_layers: int = 1,
        kernel_sizes: Tuple[int, ...] = (3, 4, 5),
        num_filters: int = 256,
        dropout: float = 0.5,
        deep_conv_layers: int = 2,
        fc_hidden_dim: Optional[int] = 512,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if embedding_dim is None:
            embedding_dim = hidden_dim
        if embedding_dim is None:
            raise ValueError("embedding_dim is required")
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.num_filters = int(num_filters)
        self.dropout_value = float(dropout)
        self.deep_conv_layers = int(deep_conv_layers)
        self.fc_hidden_dim = None if fc_hidden_dim is None else int(fc_hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.num_layers,
                    out_channels=self.num_filters,
                    kernel_size=(k, self.embedding_dim),
                )
                for k in self.kernel_sizes
            ]
        )
        self.deep_convs = nn.ModuleList(
            [
                nn.Conv1d(self.num_filters, self.num_filters, kernel_size=3, padding=1)
                for _ in range(self.deep_conv_layers)
            ]
        )
        self.dropout = nn.Dropout(self.dropout_value)
        if self.fc_hidden_dim is None:
            self.fc = nn.Linear(len(self.kernel_sizes) * self.num_filters, self.num_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(len(self.kernel_sizes) * self.num_filters, self.fc_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_value),
                nn.Linear(self.fc_hidden_dim, self.num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outputs: List[torch.Tensor] = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            seq_len = x.size(2)
            if seq_len < k:
                out = F.relu(conv(F.pad(x, (0, 0, 0, k - seq_len)))).squeeze(3)
            else:
                out = F.relu(conv(x)).squeeze(3)
            for deep_conv in self.deep_convs:
                out = F.relu(deep_conv(out))
            conv_outputs.append(F.adaptive_max_pool1d(out, 1).squeeze(2))
        return self.fc(self.dropout(torch.cat(conv_outputs, dim=1)))
