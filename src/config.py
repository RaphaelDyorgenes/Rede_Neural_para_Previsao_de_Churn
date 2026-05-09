# Constantes e configuração centralizada do projeto Telco Churn.
import random
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ================================
# PATHS
# ================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ================================
# REPRODUTIBILIDADE
# ================================
SEED = 42

# ================================
# DATASET
# ================================
TARGET_COL = "Churn Label"

# Colunas a remover: identificadores + vazamento de dados
DROP_COLS = [
    "CustomerID", "Country", "State", "City", "Lat Long",
    "Churn Reason", "Churn Score", "Churn Value", "CLTV",
]

# Features numéricas (passam pelo StandardScaler)
NUMERIC_FEATURES = [
    "Count", "Zip Code", "Latitude", "Longitude",
    "Tenure Months", "Monthly Charges", "Total Charges",
]

# Features categóricas (passam pelo OneHotEncoder)
CATEGORICAL_FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
]


def set_global_seed(seed: int = SEED) -> None:
    # Fixa seed em random, numpy e torch para reprodutibilidade.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info("Seed fixado: %d", seed)
