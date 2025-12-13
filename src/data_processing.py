import pandas as pd
import numpy as np
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Custom Transformer: RFM Feature Engineering
# -------------------------------------------------------------------
class RFMTransformer(BaseEstimator, TransformerMixin):
    """
    Create customer-level RFM (Recency, Frequency, Monetary) features.
    """

    def __init__(self, reference_date=None):
        self.reference_date = reference_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        required_cols = {"CustomerId", "Amount", "TransactionStartTime"}
        if not required_cols.issubset(X.columns):
            raise ValueError(
                f"Missing required columns: {required_cols - set(X.columns)}"
            )

        logger.info("Starting RFM feature engineering")

        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"], errors="coerce"
        )

        if self.reference_date is None:
            reference_date = df["TransactionStartTime"].max()
        else:
            reference_date = pd.to_datetime(self.reference_date)

        rfm = (
            df.groupby("CustomerId")
            .agg(
                recency=("TransactionStartTime", lambda x: (reference_date - x.max()).days),
                frequency=("TransactionStartTime", "count"),
                monetary=("Amount", "sum"),
                monetary_std=("Amount", "std"),
            )
            .fillna(0)
            .reset_index()
        )

        logger.info("RFM feature engineering completed")
        return rfm


# -------------------------------------------------------------------
# Proxy Default Label Creation
# -------------------------------------------------------------------
def create_proxy_default(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a proxy default label based on RFM behavior.
    High recency, low frequency, and low monetary value are treated as higher risk.
    """

    required_cols = {"recency", "frequency", "monetary"}
    if not required_cols.issubset(df.columns):
        raise ValueError("RFM features must exist before creating proxy default")

    logger.info("Creating proxy default label")

    df = df.copy()

    recency_thresh = df["recency"].quantile(0.75)
    freq_thresh = df["frequency"].quantile(0.25)
    monetary_thresh = df["monetary"].quantile(0.25)

    df["default_proxy"] = np.where(
        (df["recency"] >= recency_thresh)
        & (df["frequency"] <= freq_thresh)
        & (df["monetary"] <= monetary_thresh),
        1,
        0,
    )

    logger.info("Proxy default label created")
    return df


# -------------------------------------------------------------------
# Full Preprocessing Pipeline
# -------------------------------------------------------------------
def build_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Build a preprocessing pipeline using sklearn.
    """

    logger.info("Building preprocessing pipeline")

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor
