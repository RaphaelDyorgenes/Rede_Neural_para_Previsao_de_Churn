# Arquiteturas MLP para previsão de churn.
import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)


class ChurnMLP(nn.Module):
    # MLP-v1: 64 → 32 → 1 (com Sigmoid na saída)

    def __init__(self, input_dim: int = 50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ChurnMLPv2(nn.Module):
    # MLP-v2: 128 → 64 → 32 → 1 (logit puro, usa BCEWithLogitsLoss).

    def __init__(self, input_dim: int = 50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_model(
    path: Path | None = None,
    input_dim: int = 50,
    version: str = "v2",
) -> nn.Module:
    # Carrega um modelo treinado do disco.
    # Args:
    #   path: Caminho do checkpoint. Default: models/mlp_best.pt
    #   input_dim: Número de features de entrada.
    #   version: "v1" para ChurnMLP ou "v2" para ChurnMLPv2.
    path = path or MODELS_DIR / "mlp_best.pt"
    model_cls = ChurnMLP if version == "v1" else ChurnMLPv2
    model = model_cls(input_dim=input_dim)
    model.load_state_dict(
        torch.load(path, weights_only=True, map_location="cpu")
    )
    model.eval()
    logger.info(
        "Modelo %s carregado: %s (%d parâmetros)",
        version, path.name, sum(p.numel() for p in model.parameters()),
    )
    return model
