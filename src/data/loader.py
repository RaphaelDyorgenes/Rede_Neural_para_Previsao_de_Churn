# Carregamento e versionamento do dataset Telco Churn.
import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    # Carrega o dataset CSV limpo.
    path = path or DATA_DIR / "telco_churn_clean.csv"
    df = pd.read_csv(path)
    logger.info("Dataset carregado: %d linhas, %d colunas", *df.shape)
    return df


def compute_dataset_hash(path: str | Path | None = None) -> str:
    # Calcula hash MD5 do dataset para rastreabilidade no MLflow.
    path = path or DATA_DIR / "telco_churn_clean.csv"
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
