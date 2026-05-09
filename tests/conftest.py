# Fixtures compartilhadas para testes.
import pytest
import pandas as pd

from src.config import DATA_DIR, DROP_COLS, SEED, TARGET_COL


@pytest.fixture
def full_df():
    # Carrega o dataset completo.
    return pd.read_csv(DATA_DIR / "telco_churn_clean.csv")


@pytest.fixture
def sample_df(full_df):
    # Amostra de 100 linhas do dataset (seed fixo).
    return full_df.sample(n=100, random_state=SEED).reset_index(drop=True)


@pytest.fixture
def sample_features(sample_df):
    # X e y já separados para testes de pipeline.
    X = sample_df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    y = (sample_df[TARGET_COL] == "Yes").astype(int)
    return X, y
