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