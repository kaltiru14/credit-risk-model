import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

try:
    from woe import WOE
except ImportError:
    WOE = None



# =========================================================
# 1. Custom Transformers
# =========================================================

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour, day, month, year from TransactionStartTime"""

    def __init__(self, time_col="TransactionStartTime"):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])

        X["transaction_hour"] = X[self.time_col].dt.hour
        X["transaction_day"] = X[self.time_col].dt.day
        X["transaction_month"] = X[self.time_col].dt.month
        X["transaction_year"] = X[self.time_col].dt.year

        return X


class AggregateCustomerFeatures(BaseEstimator, TransformerMixin):
    """Create customer-level aggregate transaction features"""

    def __init__(self, customer_id_col="CustomerId", amount_col="Amount"):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        agg = (
            X.groupby(self.customer_id_col)[self.amount_col]
            .agg(
                total_transaction_amount="sum",
                avg_transaction_amount="mean",
                transaction_count="count",
                std_transaction_amount="std",
            )
            .reset_index()
        )

        X = X.merge(agg, on=self.customer_id_col, how="left")
        X["std_transaction_amount"] = X["std_transaction_amount"].fillna(0)

        return X


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence transformer
    NOTE: Must be applied ONLY after target variable exists
    """

    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.encoders = {}

    def fit(self, X, y):
        for col in self.categorical_features:
            woe = WOE()
            woe.fit(X[col], y)
            self.encoders[col] = woe
        return self

    def transform(self, X):
        X = X.copy()
        for col, woe in self.encoders.items():
            X[col] = woe.transform(X[col])
        return X


# =========================================================
# 2. Pipeline Builder
# =========================================================

def build_feature_pipeline(numerical_features, categorical_features):
    """
    Feature engineering pipeline WITHOUT WoE
    (used before target creation)
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("time_features", TimeFeatureExtractor()),
            ("aggregate_features", AggregateCustomerFeatures()),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


# =========================================================
# 3. Public Processing Functions
# =========================================================

def process_data(df):
    """
    Feature engineering BEFORE target creation
    (used for Task 3)
    """

    numerical_features = [
        "Amount",
        "Value",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
    ]

    categorical_features = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]

    pipeline = build_feature_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    processed_array = pipeline.fit_transform(df)

    feature_names = (
        numerical_features
        + list(
            pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(categorical_features)
        )
    )

    processed_df = pd.DataFrame(processed_array, columns=feature_names)

    return processed_df


def apply_woe(df, target_col, categorical_features):
    """
    Apply WoE transformation AFTER target variable exists
    (used in Task 4+)
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    woe_transformer = WoETransformer(categorical_features)
    X_woe = woe_transformer.fit_transform(X, y)

    X_woe[target_col] = y.values

    return X_woe
