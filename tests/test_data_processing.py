import pandas as pd

from src.data_processing import (
    process_data,
    TimeFeatureExtractor,
    AggregateCustomerFeatures,
)


def sample_dataframe():
    return pd.DataFrame(
        {
            "TransactionId": [1, 2, 3, 4],
            "CustomerId": ["C1", "C1", "C2", "C3"],
            "TransactionStartTime": [
                "2024-01-01 10:00:00",
                "2024-01-02 12:00:00",
                "2024-01-03 14:00:00",
                "2024-01-04 16:00:00",
            ],
            "Amount": [100, 200, 50, 300],
            "Value": [100, 200, 50, 300],
            "CurrencyCode": ["UGX", "UGX", "UGX", "UGX"],
            "CountryCode": [256, 256, 256, 256],
            "ProviderId": ["P1", "P1", "P2", "P3"],
            "ProductCategory": ["Cat1", "Cat1", "Cat2", "Cat3"],
            "ChannelId": ["Web", "Android", "Web", "IOS"],
            "PricingStrategy": [1, 1, 2, 2],
        }
    )


def test_time_features_created():
    df = sample_dataframe()
    transformer = TimeFeatureExtractor()
    result = transformer.fit_transform(df)

    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns
    assert "transaction_month" in result.columns
    assert "transaction_year" in result.columns


def test_aggregate_features_created():
    df = sample_dataframe()
    transformer = AggregateCustomerFeatures()
    result = transformer.fit_transform(df)

    assert "total_transaction_amount" in result.columns
    assert "avg_transaction_amount" in result.columns
    assert "transaction_count" in result.columns
    assert "std_transaction_amount" in result.columns


def test_process_data_output_shape():
    df = sample_dataframe()
    processed_df = process_data(df)

    assert processed_df.shape[0] == df.shape[0]
    assert processed_df.shape[1] > 5
