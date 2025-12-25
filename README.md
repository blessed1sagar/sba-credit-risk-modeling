# 📘 Technical Specification: SBA Credit Risk Modelling Pipeline

**Project:** SBA 7(a) Loan Default Prediction
**Model Type:** Binary Classification (XGBoost)
**Target Variable:** `LoanStatus` (0 = Paid in Full, 1 = Charged Off)
**Dataset:** `foia-7a-fy2020-present-asof-250930.csv`
**Version:** Final Approved (December 2025)

---

## I. Executive Summary

This document outlines the definitive data preprocessing, cleaning, and feature engineering standards for the SBA Credit Risk Model. The pipeline transforms raw administrative data into a machine-learning-ready dataset, specifically designed to handle imbalanced classes (~93% Paid / ~7% Default) and temporal economic shifts (COVID-19).

**Final Output:**

- **Features (X):** 79 columns
- **Samples:** 55,831 rows
- **Target (y):** Binary (0 = PIF, 1 = CHGOFF)

---

## II. Data Hygiene & Filtering Strategy

### 1. Target Definition

The model is trained only on loans with a definitive outcome.

| Action             | Description                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------ |
| **Filter**   | `LoanStatus` to include only `'PIF'` (Paid in Full) and `'CHGOFF'` (Charged Off)     |
| **Logic**    | Loans marked as `CANCLD`, `EXEMPT`, or `COMMIT` do not provide valid default signals |
| **Encoding** | PIF → 0, CHGOFF → 1                                                                      |

**Result:** 347,514 rows → 55,954 rows (83.90% removed)

### 2. Missing Data Strategy

We utilize a **"Zero Tolerance for Noise"** strategy. Since the dataset is large and missingness in key columns is negligible, dropping rows ensures data integrity without synthetic noise.

| Feature                   | Missing Count     | Strategy                         | Rationale                                                   |
| ------------------------- | ----------------- | -------------------------------- | ----------------------------------------------------------- |
| `FirstDisbursementDate` | 3 rows            | **Drop Row**               | Negligible count. Dropping is cleaner than imputation.      |
| `LocationID`            | ~120 rows (0.21%) | **Drop Row**               | Random data corruption. Dropping ensures high-quality data. |
| `BankState`             | ~120 rows (0.21%) | **Drop Row**               | Required for `SameStateLending` interaction feature.      |
| `BusinessType`          | ~0 rows           | **Fill with 'INDIVIDUAL'** | Replace whitespace with default category.                   |
| `BusinessAge`           | ~88 rows (0.16%)  | **Map & Fill**             | Map to clean categories, fill NaN with 'Existing'.          |

**Note:** `InitialInterestRate` and `FixedorVariableInterestRate` have **no missing values** after filtering to PIF/CHGOFF, so no handling is required.

---

## III. Feature Selection (Column Dropping)

We remove columns that violate ML principles (Leakage, PII, Noise, Redundancy).

### 1. Target Leakage (Future Information)

Columns that reveal the outcome after the fact.

**Dropped:** `PaidinFullDate`, `ChargeoffDate`, `GrossChargeoffAmount`

**Reason:** These events happen after a default or payoff. Including them would be "cheating" (Data Leakage).

### 2. High Cardinality & PII

Identifiers and Text with too many unique values to generalize.

**Dropped:** `BorrName`, `BorrStreet`, `BorrCity`, `BorrZip`, `BankName`, `BankStreet`, `BankCity`, `BankZip`, `FranchiseName`

**Reason:** Prevents overfitting. The model should learn "Lender Size Risk," not individual branch risk.

### 3. Geographic Redundancy

**Dropped:** `BorrState`, `ProjectCounty`, `SBADistrictOffice`, `CongressionalDistrict`

**Reason:** We strictly use `ProjectState` (where business funds are spent) as the primary geographic signal.

### 4. Administrative Artifacts

**Dropped:** `AsOfDate`, `Program`, `Subprogram`, `ProcessingMethod`

**Reason:** Internal SBA codes that describe how the loan was processed, not the creditworthiness of the borrower.

### 5. Raw Columns After Encoding/Engineering

**Dropped:** `FranchiseCode`, `BankNCUANumber`, `BankFDICNumber`, `NAICSCode`, `NAICSDescription`, `FirstDisbursementDate`, `TerminMonths`, `ApprovalDate`, `BusinessAge`, `FixedorVariableInterestRate`, `CollateralInd`, `SoldSecondMarketInd`, `LocationID`, `BankState`

**Reason:** These have been transformed into engineered features or encoded.

---

## IV. Feature Engineering (The "Business Logic")

Every engineered feature captures a specific dimension of Credit Risk.

> **Naming Convention:** All features use **PascalCase**

### A. Temporal Features (Time & Economics)

| Feature                    | Formula                                           | Signal                                                                                                                  |
| -------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `TimeToDisbursementDays` | `FirstDisbursementDate - ApprovalDate`          | **Process Friction.** Long delays signal administrative issues or borrower struggling to meet closing conditions. |
| `DaysSinceApproval`      | `Snapshot_Date (Max) - ApprovalDate`            | **Cohort Risk.** Captures the economic environment the loan was born into.                                        |
| `IsCovidEra`             | Approved between `2020-03-01` – `2021-12-31` | **Artificial Support.** COVID-era loans received government subsidies (CARES Act), altering default probability.  |
| `ApprovalFY`             | Kept as raw integer (e.g., 2021)                  | **Fiscal Cycle.** SBA policies change on fiscal year boundaries.                                                  |

### B. Financial Features

| Feature         | Formula                                             | Signal                                                                                  |
| --------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `TermInYears` | `TerminMonths / 12`                               | **Loan Duration.** More interpretable than raw months.                            |
| `IsFixedRate` | `FixedorVariableInterestRate == 'F'` → 1, else 0 | **Interest Rate Volatility Risk.** Fixed rate loans are immune to Fed rate hikes. |

### C. Lender & Business Features

| Feature                 | Formula                                     | Signal                                                                                         |
| ----------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `IsCreditUnion`       | `BankNCUANumber.notna()` → 1, else 0     | **Lender Type.** Credit unions may have different underwriting standards.                |
| `IsFranchise`         | `FranchiseCode.notna()` → 1, else 0      | **Business Model.** Franchises have different risk profiles than independent businesses. |
| `HasCollateral`       | `CollateralInd == 'Y'` → 1, else 0       | **Security.** Secured loans have lower loss-given-default.                               |
| `SoldSecondaryMarket` | `SoldSecondMarketInd == 'Y'` → 1, else 0 | **Marketability.** Loans sold to secondary market may have different characteristics.    |

### D. Interaction & Geographic Features

| Feature              | Formula                                      | Signal                                                                                                  |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `SameStateLending` | `(BankState == ProjectState)` → 1, else 0 | **Soft Information.** Local banks often monitor borrowers better than distant lenders.            |
| `LocationIDCount`  | Frequency encoding of `LocationID`         | **Lender Size.** Distinguishes "Big Banks" (High Volume) from "Small Credit Unions" (Low Volume). |

### E. Industry Features

| Feature         | Formula                         | Signal                                                                                                 |
| --------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `NAICSSector` | First 2 digits of `NAICSCode` | **Industry Risk.** Construction (Sector 23) has different failure rates than Retail (Sector 44). |

### F. Business Characteristics

| Original Value                             | Mapped Value          | Category           |
| ------------------------------------------ | --------------------- | ------------------ |
| `Existing or more than 2 years old`      | `Existing`          | Mature             |
| `Startup, Loan Funds will Open Business` | `Startup`           | Pre-Revenue        |
| `New Business or 2 years or less`        | `NewBusiness`       | Early Stage        |
| `Change of Ownership`                    | `ChangeOfOwnership` | Transition         |
| `Unanswered`                             | `Existing`          | Default assumption |

---

## V. Encoding Strategy (Data Transformation)

| Feature               | Method                                 | Rationale                                                                                                                             |
| --------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `ProjectState`      | One-Hot (`State_` prefix)            | States are nominal categories (no order). 54 unique values.                                                                           |
| `NAICSSector`       | **NOT One-Hot (kept as string)** | 24 unique sectors. Consider encoding in modeling phase.                                                                               |
| `BusinessType`      | One-Hot (`Type_` prefix)             | Legal structures (Corp vs. Individual) have distinct liability risks.                                                                 |
| `BusinessAge_Clean` | One-Hot (`Age_` prefix)              | Non-linear risk profile (Startup ≠ Existing).                                                                                        |
| `LocationID`        | Frequency Encoding                     | Cardinality too high for One-Hot. Count captures "Size" signal.                                                                       |
| Binary Flags          | Binary Map (0/1)                       | `IsFixedRate`, `IsCreditUnion`, `IsFranchise`, `HasCollateral`, `SoldSecondaryMarket`, `SameStateLending`, `IsCovidEra` |

---

## VI. Final Execution Pipeline (Strict Order)

```
SECTION 5: HANDLE MISSING VALUES
┌─────────────────────────────────────────────────────────────┐
│ 5.1  Drop FirstDisbursementDate nulls (3 rows)              │
│ 5.2  Drop LocationID/BankState nulls (~120 rows)            │
│ 5.3  Clean BusinessType (replace whitespace with INDIVIDUAL)│
│ 5.4  Clean BusinessAge → BusinessAge_Clean (map categories) │
└─────────────────────────────────────────────────────────────┘
                            ↓
SECTION 6: FEATURE ENGINEERING
┌─────────────────────────────────────────────────────────────┐
│ 6.1  Convert dates to datetime                              │
│ 6.2  TimeToDisbursementDays, DaysSinceApproval              │
│ 6.3  IsCovidEra indicator                                   │
│ 6.4  TermInYears                                            │
│ 6.5  NAICSSector (first 2 digits)                           │
│ 6.6  Binary flags (IsCreditUnion, IsFranchise, IsFixedRate, │
│      HasCollateral, SoldSecondaryMarket)                    │
│ 6.7  SameStateLending (interaction feature)                 │
│ 6.8  LocationIDCount (frequency encoding)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
SECTION 7: CATEGORICAL ENCODING
┌─────────────────────────────────────────────────────────────┐
│ 7.1  One-Hot encode BusinessType, BusinessAge_Clean         │
│      Prefixes: Type_, Age_                                  │
│ 7.2  One-Hot encode ProjectState                            │
│      Prefix: State_                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
SECTION 8: DROP IRRELEVANT FEATURES
┌─────────────────────────────────────────────────────────────┐
│ Drop 34 columns: PII, Leakage, Raw encoded columns          │
└─────────────────────────────────────────────────────────────┘
                            ↓
SECTION 9: FINAL PREPARATION
┌─────────────────────────────────────────────────────────────┐
│ 9.1  Create y: LoanStatus → 0 (PIF) / 1 (CHGOFF)            │
│ 9.2  Create X: All features except LoanStatus               │
│ 9.3  Verify: No missing values in X                         │
└─────────────────────────────────────────────────────────────┘
```

---

## VII. Final Feature Set (79 Features)

### Numeric Features (18)

| #  | Feature                    | Description                           |
| -- | -------------------------- | ------------------------------------- |
| 1  | `GrossApproval`          | Total loan amount approved            |
| 2  | `SBAGuaranteedApproval`  | SBA-guaranteed portion                |
| 3  | `ApprovalFY`             | Fiscal year of approval               |
| 4  | `InitialInterestRate`    | Interest rate at origination          |
| 5  | `RevolverStatus`         | Revolver indicator                    |
| 6  | `JobsSupported`          | Number of jobs supported              |
| 7  | `TimeToDisbursementDays` | Days from approval to disbursement    |
| 8  | `DaysSinceApproval`      | Loan vintage (days since approval)    |
| 9  | `IsCovidEra`             | COVID-era indicator (binary)          |
| 10 | `TermInYears`            | Loan term in years                    |
| 11 | `NAICSSector`            | Industry sector (2-digit code)        |
| 12 | `IsCreditUnion`          | Credit union indicator (binary)       |
| 13 | `IsFranchise`            | Franchise indicator (binary)          |
| 14 | `IsFixedRate`            | Fixed rate indicator (binary)         |
| 15 | `HasCollateral`          | Collateral indicator (binary)         |
| 16 | `SoldSecondaryMarket`    | Secondary market indicator (binary)   |
| 17 | `SameStateLending`       | Same state lending indicator (binary) |
| 18 | `LocationIDCount`        | Lender frequency count                |

### One-Hot Encoded Features (61)

- **Business Type (3):** `Type_CORPORATION`, `Type_INDIVIDUAL`, `Type_PARTNERSHIP`
- **Business Age (4):** `Age_ChangeOfOwnership`, `Age_Existing`, `Age_NewBusiness`, `Age_Startup`
- **State (54):** `State_AK` through `State_WY`

---

## VIII. Target Distribution

| Class                | Label      | Count  | Percentage |
| -------------------- | ---------- | ------ | ---------- |
| **Good Loans** | PIF = 0    | 51,669 | 92.55%     |
| **Bad Loans**  | CHGOFF = 1 | 4,162  | 7.45%      |

> **Note:** Class imbalance requires handling during model training (e.g., SMOTE, class weights).

---

## IX. Data Quality Validation

| Check               | Result             |
| ------------------- | ------------------ |
| Missing values in X | ✓ None            |
| Duplicate rows      | ✓ None (Verified) |
| Data types          | ✓ All numeric     |
| Target encoding     | ✓ Binary (0/1)    |

---

## X. Files Generated

| File                                   | Description                  |
| -------------------------------------- | ---------------------------- |
| `sba_loan_preprocessing_final.ipynb` | Final preprocessing notebook |
| `sba_loan_features.csv`              | Feature matrix (X)           |
| `sba_loan_target.csv`                | Target vector (y)            |

---

*Document Version: Final Approved*
*Last Updated: December 2025*
