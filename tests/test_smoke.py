# Smoke tests — verifica que os módulos carregam e funcionam juntos.
import numpy as np

from sklearn.linear_model import LogisticRegression

from src.data.loader import load_dataset, compute_dataset_hash
from src.data.preprocessing import build_preprocessor, build_full_pipeline
from src.models.models import ChurnMLP, ChurnMLPv2
from src.config import DROP_COLS, TARGET_COL


def test_load_dataset_returns_dataframe():
    # load_dataset() retorna DataFrame com shape esperado.
    df = load_dataset()
    assert df.shape[0] == 7043
    assert df.shape[1] == 33


def test_compute_dataset_hash():
    # Hash do dataset é string hexadecimal de 32 caracteres.
    h = compute_dataset_hash()
    assert isinstance(h, str)
    assert len(h) == 32


def test_build_preprocessor_fits(sample_df):
    # Preprocessor fita e transforma sem erros no dataset real.
    X = sample_df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(X)
    assert result.shape[0] == len(X)
    assert not np.isnan(result).any()


def test_full_pipeline_with_model(sample_df):
    # Pipeline completa (preprocessor + LogisticRegression) treina e prediz.
    y = (sample_df[TARGET_COL] == "Yes").astype(int)
    pipeline = build_full_pipeline(model=LogisticRegression(max_iter=1000))
    pipeline.fit(sample_df, y)
    preds = pipeline.predict(sample_df)
    assert len(preds) == len(sample_df)
    assert set(preds).issubset({0, 1})


def test_churn_mlp_instantiates():
    # ChurnMLP instancia com input_dim=50.
    model = ChurnMLP(input_dim=50)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_churn_mlpv2_instantiates():
    # ChurnMLPv2 instancia com input_dim=50.
    model = ChurnMLPv2(input_dim=50)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_churn_mlpv2_forward_pass():
    # ChurnMLPv2 produz output com shape correto.
    import torch
    model = ChurnMLPv2(input_dim=50)
    model.eval()
    x = torch.randn(5, 50)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (5, 1)
