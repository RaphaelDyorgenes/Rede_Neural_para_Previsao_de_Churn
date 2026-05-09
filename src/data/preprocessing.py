# Pipeline de pré-processamento reprodutível para Telco Churn.
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DROP_COLS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


# ================================
# TRANSFORMADORES CUSTOM
# ================================
class ColumnDropper(BaseEstimator, TransformerMixin):
    # Remove colunas identificadoras e com vazamento de dados do DataFrame.

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or DROP_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_to_drop = [c for c in self.columns if c in X.columns]
        logger.info("ColumnDropper: removendo %d colunas", len(cols_to_drop))
        return X.drop(columns=cols_to_drop)


class TargetExtractor(BaseEstimator, TransformerMixin):
    # Extrai e binariza a coluna alvo: 'Yes' → 1, 'No' → 0.

    def __init__(self, target_col: str = TARGET_COL):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X[self.target_col] == "Yes").astype(int)


# ================================
# PIPELINE SKLEARN
# ================================
def build_preprocessor() -> ColumnTransformer:
    # Constrói ColumnTransformer com StandardScaler (numérico) + OneHotEncoder (categórico).
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_full_pipeline(model=None) -> Pipeline:
    # Pipeline completa: ColumnDropper → Preprocessor → modelo (opcional).
    steps = [
        ("dropper", ColumnDropper()),
        ("preprocessor", build_preprocessor()),
    ]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)


# ================================
# PERSISTÊNCIA
# ================================
def save_pipeline(pipeline: Pipeline, path: Path | None = None) -> Path:
    # Salva pipeline treinada em disco com joblib.
    path = path or MODELS_DIR / "pipeline.joblib"
    joblib.dump(pipeline, path)
    logger.info("Pipeline salva em %s", path)
    return path


def load_pipeline(path: Path | None = None) -> Pipeline:
    # Carrega pipeline treinada do disco.
    path = path or MODELS_DIR / "pipeline.joblib"
    pipeline = joblib.load(path)
    logger.info("Pipeline carregada de %s", path)
    return pipeline


# ================================
# FUNÇÕES AUXILIARES
# ================================
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Separa features (X) e alvo (y), removendo colunas leaky.
    # Mantida por retrocompatibilidade com notebooks existentes.
    dropper = ColumnDropper()
    extractor = TargetExtractor()
    X = dropper.transform(df).drop(columns=[TARGET_COL], errors="ignore")
    y = extractor.transform(df)
    logger.info(
        "Features: %d colunas | Positivos: %d (%.1f%%)",
        X.shape[1],
        y.sum(),
        y.mean() * 100,
    )
    return X, y
