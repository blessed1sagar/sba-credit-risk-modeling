# How Default Probability is Calculated

## Quick Answer

**Your loan application → 97 features → XGBoost model → Probability (0-100%) → Risk Category**

## Detailed Breakdown

### Step 1: Feature Engineering
Your 15 input fields become 97 numerical features:

```
Input: Gross Approval = $50,000
       Business Age = "Existing or more than 2 years old"
       Approval Date = "2020-03-15"
       NAICS Code = "441110"
       Project State = "CA"

Output: GrossApproval = 50000
        Age_Existing = 1
        IsCovidEra = 1
        Sector_44 = 1
        State_CA = 1
        ... (92 more features)
```

### Step 2: XGBoost Model Prediction

**What is XGBoost?**
- Gradient Boosting Decision Tree algorithm
- 100 decision trees working together
- Each tree learned from 44,667 historical loans

**How it works:**
1. Each tree looks at all 97 features
2. Each tree makes splits like: "IF GrossApproval > $75k AND IsCovidEra = 1 THEN high_risk"
3. Each tree votes for probability of default
4. Final probability = weighted average of all votes

**Example:**
```
Tree #1:  "This loan looks 30% likely to default"
Tree #2:  "This loan looks 10% likely to default"
Tree #3:  "This loan looks 40% likely to default"
...
Tree #100: "This loan looks 25% likely to default"

Final Prediction: 23.4% chance of default
```

**Code:**
```python
# In src/inference_pipeline/predict.py line 217
probabilities = self.model.predict_proba(df_processed)[:, 1]
```

The `[:, 1]` means we take the probability for class 1 (Default/CHGOFF).

### Step 3: Risk Categorization

**Thresholds:**
```
if probability >= 0.28:    # >= 28%
    → 🔴 HIGH RISK → REJECT

elif probability >= 0.15:  # 15-27.9%
    → 🟡 MEDIUM RISK → APPROVE (with monitoring)

else:                       # < 15%
    → 🟢 LOW RISK → APPROVE
```

**Why 28% and not 50%?**
- Missing a default costs more than rejecting a good loan
- 28% threshold catches 83.4% of actual defaults
- Optimized for high recall (sensitivity to defaults)

## Mathematical Formula

```python
# Step 1: Each tree produces a raw score
raw_score = tree_1.score + tree_2.score + ... + tree_100.score

# Step 2: Convert to probability using logistic function
probability = 1 / (1 + exp(-raw_score))

# Step 3: Categorize
if probability >= 0.28:
    risk = "HIGH"
elif probability >= 0.15:
    risk = "MEDIUM"
else:
    risk = "LOW"
```

## Model Training Context

**Training Data:**
- 44,667 historical SBA loans
- 92.55% Paid-in-Full (Good)
- 7.45% Charged-off (Default)

**Class Imbalance Handling:**
```python
scale_pos_weight = 41,337 / 3,330 = 12.41
```
This makes the model pay 12x more attention to defaults during training.

**Performance:**
- ROC-AUC: 0.8317 (83% accuracy in ranking risk)
- Recall: 83.4% (catches 83 out of 100 defaults)
- Precision: 16.1% (only 16% of flagged loans actually default)

## Where is This Calculated?

### Backend (API):
```python
# src/api/main.py:222
probs = predictor.predict(loan_df)
prob = float(probs[0])  # Get probability for single loan

# src/inference_pipeline/predict.py:217
probabilities = self.model.predict_proba(df_processed)[:, 1]
# [:, 1] extracts P(Default) from [P(Good), P(Default)]
```

### Frontend (Streamlit):
```python
# app.py:283
prob = prediction['default_probability']  # Received from API

# app.py:313-327
if prob >= 0.28:
    risk_category = "HIGH"
    recommendation = "REJECT"
```

## Example Walkthrough

**Loan Application:**
- Gross Approval: $100,000
- Interest Rate: 8.5%
- Business Age: Startup
- COVID Era: Yes (2020)
- State: CA
- Has Collateral: No

**Feature Engineering:**
```
GrossApproval = 100000
InitialInterestRate = 8.5
Age_Startup = 1
IsCovidEra = 1
State_CA = 1
HasCollateral = 0
... (91 more features)
```

**XGBoost Prediction:**
- Tree votes average to: 0.345 (34.5%)

**Risk Categorization:**
- 34.5% >= 28% → 🔴 **HIGH RISK**
- Recommendation: **REJECT**

**Interpretation:**
"Based on historical data, loans with similar characteristics have a 34.5% chance of defaulting. This exceeds our 28% threshold, so we recommend rejection or requiring additional collateral."

## FAQ

**Q: Why does the model say 23% when the loan looks good?**
A: The model looks at historical patterns. Even "good-looking" loans can have subtle risk factors. Check the SHAP waterfall plot to see which features increased risk.

**Q: Can I change the 28% threshold?**
A: Yes! Edit `src/api/main.py:226` and `app.py:313`. Lower threshold = more rejections (safer), higher threshold = more approvals (riskier).

**Q: How accurate is the model?**
A: ROC-AUC of 0.8317 means it correctly ranks risk 83% of the time. It catches 83% of actual defaults but also flags many good loans (16% precision).

**Q: What if I disagree with the prediction?**
A: The model is a tool, not a replacement for judgment. Use the SHAP explanation to understand the reasoning, then apply your domain expertise.
