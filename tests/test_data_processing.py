import pandas as pd
import pytest

from src.data_processing import RFMTransformer, create_proxy_default


# -------------------------------------------------------------------
# Test RFM Feature Engineering
# -------------------------------------------------------------------
def test_rfm_transformer_creates_expected_columns():
    df = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [100, 200, 50],
        "TransactionStartTime": [
            "2024-01-01",
            "2024-01-05",
            "2024-01-03",
        ],
    })

    transformer = RFMTransformer(reference_date="2024-01-10")
    rfm = transformer.fit_transform(df)

    expected_cols = {
        "CustomerId", "recency", "frequency", "monetary", "monetary_std"
    }

    assert expected_cols.issubset(set(rfm.columns))
    assert len(rfm) == 2


def test_rfm_frequency_correct():
    df = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [100, 200, 50],
        "TransactionStartTime": [
            "2024-01-01",
            "2024-01-05",
            "2024-01-03",
        ],
    })

    transformer = RFMTransformer(reference_date="2024-01-10")
    rfm = transformer.fit_transform(df)

    freq_c1 = rfm.loc[rfm["CustomerId"] == "C1", "frequency"].iloc[0]
    assert freq_c1 == 2


# -------------------------------------------------------------------
# Test Proxy Default Creation
# -------------------------------------------------------------------
def test_proxy_default_creation():
    rfm_df = pd.DataFrame({
        "CustomerId": ["C1", "C2", "C3"],
        "recency": [30, 5, 40],
        "frequency": [1, 10, 1],
        "monetary": [100, 5000, 50],
    })

    result = create_proxy_default(rfm_df)

    assert "default_proxy" in result.columns
    assert result["default_proxy"].isin([0, 1]).all()


def test_proxy_default_missing_columns_raises_error():
    bad_df = pd.DataFrame({
        "recency": [10, 20],
        "frequency": [1, 2],
    })

    with pytest.raises(ValueError):
        create_proxy_default(bad_df)
