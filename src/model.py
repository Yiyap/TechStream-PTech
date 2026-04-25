"""
Red neuronal para detección binaria de anomalías (logits; sin sigmoid en salida).
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn


class AnomalyDetector(nn.Module):
    """
    MLP con bloques Linear → BatchNorm1d → ReLU → Dropout y cabeza lineal a 1 logit.
    La probabilidad se obtiene en inferencia con ``torch.sigmoid(logits)``.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        blocks: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
            )
            in_dim = out_dim

        self.hidden = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.hidden:
            x = block(x)
        x = self.output_layer(x)
        return x.squeeze(1)

    def get_num_params(self) -> int:
        """Número total de parámetros con gradiente."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    input_dim: int = 7,
    hidden_dims: list[int] | None = None,
    dropout_rate: float = 0.3,
) -> AnomalyDetector:
    """
    Instancia ``AnomalyDetector`` e imprime la arquitectura y el conteo de parámetros.
    """
    model = AnomalyDetector(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    )
    print(model)
    print(f"Parámetros entrenables: {model.get_num_params():,}")
    return model


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    torch.manual_seed(42)
    model = build_model()
    batch = torch.randn(32, 7)
    output = model(batch)
    assert output.shape == torch.Size([32]), output.shape
    print(f"✅ Forward pass OK — Output shape: {output.shape}")
