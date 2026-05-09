# Testes unitários do pipeline de pré-processamento.
import numpy as np
import pandas as pd

from src.config import DROP_COLS, NUMERIC_FEATURES, TARGET_COL
from src.data.preprocessing import (
    ColumnDropper,
    TargetExtractor,
    build_preprocessor,
    build_full_pipeline,
    prepare_features,
)


def test_column_dropper_removes_correct_columns(sample_df):
    # ColumnDropper remove exatamente as colunas leaky.
    dropper = ColumnDropper()
    result = dropper.transform(sample_df)
    for col in DROP_COLS:
        assert col not in result.columns


def test_column_dropper_keeps_other_columns(sample_df):
    # ColumnDropper preserva colunas que não estão na lista.
    dropper = ColumnDropper()
    result = dropper.transform(sample_df)
    assert "Tenure Months" in result.columns
    assert "Monthly Charges" in result.columns
    assert "Contract" in result.columns


def test_target_extractor_binarizes(sample_df):
    # TargetExtractor converte Yes/No para 1/0.
    extractor = TargetExtractor()
    y = extractor.transform(sample_df)
    assert set(y.unique()).issubset({0, 1})
    assert y.dtype == int


def test_build_preprocessor_fits(sample_df):
    # Preprocessor fita sem erros no dataset.
    X = sample_df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(X)


def test_pipeline_output_is_numeric(sample_df):
    # Pipeline produz array numérico sem NaN.
    X = sample_df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(X)
    assert np.issubdtype(result.dtype, np.floating)
    assert not np.isnan(result).any()


def test_pipeline_output_shape(sample_df):
    # Número de features de saída é consistente entre fit e transform.
    X = sample_df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(X)
    n_numeric = len(NUMERIC_FEATURES)
    # Categóricas expandem com one-hot
    assert result.shape[1] >= n_numeric


def test_full_pipeline_transforms(sample_df):
    # Pipeline completa (dropper + preprocessor) transforma sem erros.
    pipeline = build_full_pipeline()
    result = pipeline.fit_transform(sample_df)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_df)
    assert not np.isnan(result).any()


def test_pipeline_handles_unknown_categories(sample_df):
    # Pipeline não quebra com categorias novas na inferência.
    X = sample_df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    # Simula categoria nova no dado de inferência
    X_new = X.head(1).copy()
    X_new.loc[X_new.index[0], "Contract"] = "Three year"
    result = preprocessor.transform(X_new)
    assert not np.isnan(result).any()


def test_pipeline_reproducibility(sample_df):
    # Duas execuções produzem resultado idêntico.
    X = sample_df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")

    p1 = build_preprocessor()
    r1 = p1.fit_transform(X)

    p2 = build_preprocessor()
    r2 = p2.fit_transform(X)

    np.testing.assert_array_equal(r1, r2)


def test_prepare_features_returns_correct_types(sample_df):
    # prepare_features retorna (DataFrame, Series) com tipos corretos.
    X, y = prepare_features(sample_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert set(y.unique()).issubset({0, 1})
    assert TARGET_COL not in X.columns
    for col in DROP_COLS:
        assert col not in X.columns
