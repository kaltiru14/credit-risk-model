# Credit Scoring Business Understanding

## 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
Basel II requires financial institutions to build credit risk models that are:
- Transparent
- Auditable
- Explainable
- Consistent and well-documented

Because regulators must understand exactly why a customer is rated as high- or low-risk, the model cannot behave like a black box.
This means:
- Every feature must be justifiable
- Model logic must be interpretable
- Transformations (e.g., WoE) and assumptions must be documented
- Model monitoring and validation must follow regulatory standards

Thus, even if high-performance machine-learning models exist, interpretability is a regulatory requirement, not just a technical preference.

## 2. Since we lack a direct “default” label, why is creating a proxy variable necessary? What are the potential business risks of predicting using this proxy?
The dataset does not provide historical loan outcomes or a true “default” label.
To train a supervised credit scoring model, we must create a proxy variable that approximates default based on behavioral data (RFM scores, frequency of negative transactions, etc.).
A proxy is necessary because it:
- Enables supervised learning
- Converts customer behavior into a measurable risk signal
- Allows us to categorize customers as good vs bad
- Simulates credit performance in the absence of real loan data

**Business risks of using a proxy**
- The proxy may not perfectly represent real default behavior
- The model may learn patterns that are not aligned with real repayment risk
- Misclassification can occur:
    - **False positives** → good customers incorrectly labeled risky → revenue loss
    - **False negatives** → risky customers incorrectly labeled good → credit losses
- Proxy bias may propagate throughout the model and scoring system
- Model decisions may be challenged by regulators if the proxy is poorly justified

Therefore, the proxy must be constructed carefully, validated statistically, and supported by domain reasoning.
## 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression + WoE) and a complex, high-performance model (e.g., Gradient Boosting) in a regulated financial context?
### Simple Models — Logistic Regression with WoE

**Pros**
- Highly interpretable
- Easy to document and justify
- Regulatory-friendly
- Stable over time
- Straightforward to deploy and monitor

**Cons**
- Limited ability to capture nonlinear patterns
- May have lower predictive performance

### Complex Models — Gradient Boosting, XGBoost, LightGBM
**Pros**
- Strong predictive performance
- Captures complex interactions and nonlinearities
- Often improves accuracy and risk ranking

**Cons**
- Low interpretability (“black box”)
- Requires explainability tools (SHAP, LIME)
- Harder to validate and justify to regulators
- More difficult to deploy and monitor
- More sensitive to data drift
### Summary
In regulated banking environments, interpretability and compliance often outweigh raw performance.

Banks typically use:
- Logistic Regression + WoE for the official scorecard
- Gradient Boosting as an internal challenger model

This balances transparency with performance.

## Task 2 – Exploratory Data Analysis (EDA)

### Objective
The goal of this task was to explore the transaction dataset to understand its structure, identify data quality issues, uncover patterns, and generate insights that would guide feature engineering and modeling.

### Dataset Overview
- Total records: **95,662 transactions**
- Total features: **16 columns**
- Data types include numerical (`Amount`, `Value`, `FraudResult`, etc.) and categorical variables (`ProductCategory`, `ChannelId`, `ProviderId`, etc.)
- No missing values were found in the dataset.

### Key EDA Activities
- Examined dataset structure, data types, and summary statistics
- Analyzed distributions of numerical features to detect skewness and outliers
- Explored categorical feature frequencies to understand dominant behaviors
- Checked for missing values and data completeness
- Identified potential data issues relevant for modeling

### Key Insights
1. **Transaction amounts are highly skewed**, with a few extreme high-value transactions and many low-value transactions, indicating the presence of outliers.
2. **CountryCode shows no variance** (constant value), making it unsuitable as a predictive feature.
3. **FraudResult is highly imbalanced**, with fraudulent transactions being extremely rare (~0.2%).
4. **ProductCategory and ChannelId are dominated by a few categories**, suggesting strong behavioral patterns that can be useful for feature engineering.
5. **No missing values** were detected, simplifying preprocessing and pipeline design.

These findings inform the next steps, particularly feature engineering using customer-level aggregation and behavioral metrics such as RFM.

# Task 3 – Feature Engineering
## Objective

The objective of this task is to build a robust, automated, and reproducible feature engineering pipeline that transforms raw transactional data into a model-ready dataset. This transformation is implemented using scikit-learn Pipelines, ensuring consistency, reusability, and compatibility with downstream modeling and deployment workflows.

## Approach

All feature engineering steps are implemented in src/data_processing.py using sklearn.pipeline.Pipeline and custom transformers. This ensures that the same transformations are applied consistently during training and inference.

## Feature Engineering Steps
**1. Aggregate Customer-Level Features**

Transactional data is aggregated at the customer level (CustomerId) to capture behavioral patterns. The following features are created:
- Total Transaction Amount: Sum of all transaction amounts per customer
- Average Transaction Amount: Mean transaction amount per customer
- Transaction Count: Total number of transactions per customer
- Standard Deviation of Transaction Amounts: Variability of transaction values per customer

These features help represent customer spending behavior and financial stability.

**2. Time-Based Feature Extraction**

From the transaction timestamp (TransactionStartTime), the following features are extracted:

- Transaction Hour – Hour of the day
- Transaction Day – Day of the month
- Transaction Month – Month of the year
- Transaction Year – Year of the transaction

These features capture temporal spending patterns and customer activity cycles.

**3. Categorical Feature Encoding**

Categorical variables are converted into numerical format using One-Hot Encoding, which avoids introducing ordinal relationships between categories. The following fields are encoded:

- CurrencyCode
- CountryCode
- ProviderId
- ProductCategory
- ChannelId
- PricingStrategy

Unseen categories are safely handled using handle_unknown="ignore" to ensure robustness during inference.

**4. Missing Value Handling**

Missing values are handled using imputation to preserve as much data as possible:

- Numerical features: Imputed using the median
- Categorical features: Imputed using the most frequent value

This strategy reduces bias while maintaining data integrity.

**5. Feature Scaling**

All numerical features are standardized using StandardScaler, which transforms features to have:

- Mean = 0
- Standard deviation = 1

Standardization improves model convergence and performance, especially for distance-based and linear models.

**6. Weight of Evidence (WoE) and Information Value (IV)**

Weight of Evidence (WoE) transformation is implemented to support credit-risk-specific modeling, particularly for interpretable models such as Logistic Regression scorecards.
- WoE is implemented using the woe library
- To avoid target leakage, WoE is applied only after the proxy target variable is created (Task 4)
- The WoE dependency is handled as an optional import so that feature engineering and unit tests remain stable even if the library is not installed

This design follows industry best practices for regulated financial modeling.

## Pipeline Design

The full feature engineering process is chained using a scikit-learn Pipeline and ColumnTransformer, combining:

1. Time feature extraction
2. Customer-level aggregation
3. Missing value imputation
4. Categorical encoding
5. Feature scaling

This ensures the pipeline is:
- Reproducible
- Maintainable
- Production-ready
- Testing and Validation

Unit tests are implemented in tests/test_data_processing.py to validate:
- Creation of time-based features
- Creation of aggregate customer features
- Successful execution of the full feature engineering pipeline

All tests pass successfully, confirming the correctness and stability of the implementation.

## Outcome

At the end of Task 3, the raw transaction data is transformed into a clean, structured, and standardized dataset that is fully prepared for:
- Proxy target variable engineering (Task 4)
- Model training and evaluation (Task 5)
- Deployment and inference (Task 6)
## Task 5 – Model Training and Experiment Tracking

### Objective
Develop a structured and reproducible model training pipeline for credit risk prediction, including experiment tracking, hyperparameter tuning, and model evaluation.

### What Was Done
- Split the dataset into training and testing sets with a fixed `random_state` for reproducibility
- Trained multiple machine learning models:
  - Logistic Regression
  - Random Forest
- Performed hyperparameter tuning using `RandomizedSearchCV`
- Evaluated models using the following metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- Logged all experiments to **MLflow**, including:
  - Model parameters
  - Evaluation metrics
  - Trained model artifacts
- Compared model runs using the MLflow UI to identify the best-performing model

### Tools & Libraries
- Python (pandas, numpy)
- scikit-learn
- MLflow
- pytest

### How to Run
```bash
python src/train.py
