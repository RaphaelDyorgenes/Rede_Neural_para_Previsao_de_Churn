# Testes de validação do schema pandera.
import pytest
import pandera

from src.data.schema import TelcoChurnSchema


def test_schema_validates_real_data(sample_df):
    # Dataset real passa no schema sem erros.
    TelcoChurnSchema.validate(sample_df)


def test_schema_rejects_negative_tenure(sample_df):
    # Tenure Months negativo é rejeitado.
    df_invalid = sample_df.copy()
    df_invalid.loc[0, "Tenure Months"] = -5
    with pytest.raises(pandera.errors.SchemaError):
        TelcoChurnSchema.validate(df_invalid)


def test_schema_rejects_invalid_churn_label(sample_df):
    # Churn Label fora de Yes/No é rejeitado.
    df_invalid = sample_df.copy()
    df_invalid.loc[0, "Churn Label"] = "Maybe"
    with pytest.raises(pandera.errors.SchemaError):
        TelcoChurnSchema.validate(df_invalid)


def test_schema_rejects_invalid_gender(sample_df):
    # Gender fora de Male/Female é rejeitado.
    df_invalid = sample_df.copy()
    df_invalid.loc[0, "Gender"] = "Other"
    with pytest.raises(pandera.errors.SchemaError):
        TelcoChurnSchema.validate(df_invalid)
