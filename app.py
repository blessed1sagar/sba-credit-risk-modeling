"""
Streamlit dashboard for SBA Loan Risk Assessment.

Banker-facing UI that calls FastAPI backend for predictions and explanations.
Provides domain-specific inputs and SHAP waterfall plot visualizations.

Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# API endpoint (read from environment or use default)
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# ============================================================================
# DOMAIN DATA (NAICS Sectors, US States)
# ============================================================================

# NAICS Sector mapping (Name → Code)
NAICS_SECTORS = {
    "Agriculture, Forestry, Fishing and Hunting": "11",
    "Mining, Quarrying, and Oil and Gas Extraction": "21",
    "Utilities": "22",
    "Construction": "23",
    "Manufacturing": "31",
    "Wholesale Trade": "42",
    "Retail Trade": "44",
    "Transportation and Warehousing": "48",
    "Information": "51",
    "Finance and Insurance": "52",
    "Real Estate and Rental and Leasing": "53",
    "Professional, Scientific, and Technical Services": "54",
    "Management of Companies and Enterprises": "55",
    "Administrative and Support Services": "56",
    "Educational Services": "61",
    "Health Care and Social Assistance": "62",
    "Arts, Entertainment, and Recreation": "71",
    "Accommodation and Food Services": "72",
    "Other Services (except Public Administration)": "81",
}

# US States (Name → Abbreviation)
US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"
}

# Business Age categories
BUSINESS_AGE_OPTIONS = [
    "Existing or more than 2 years old",
    "New Business or 2 years or less",
    "Startup, Loan Funds will Open Business",
    "Change of Ownership"
]

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="SBA Loan Risk Assessment",
    page_icon="💼",
    layout="wide"
)

st.title("💼 SBA Loan Risk Assessment Dashboard")
st.markdown("Assess default risk for SBA loan applications using ML-powered predictions and explanations.")

# Sidebar with API status
with st.sidebar:
    st.header("🔗 API Connection")
    st.text(f"Endpoint: {API_URL}")

    # Health check
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Not Available")

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Fill in loan application details
    2. Click **Assess Risk** to get prediction
    3. View default probability and SHAP explanation
    """)

# Main content area
st.header("📝 Loan Application Details")

# Create form with columns for better layout
with st.form("loan_application_form"):
    # ====================
    # FINANCIAL SECTION
    # ====================
    st.subheader("💰 Financial Information")
    col1, col2 = st.columns(2)

    with col1:
        gross_approval = st.number_input(
            "Gross Approval Amount ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000,
            help="Total loan amount requested"
        )

        interest_rate = st.number_input(
            "Initial Interest Rate (%)",
            min_value=0.1,
            max_value=20.0,
            value=6.5,
            step=0.1,
            help="Annual interest rate"
        )

    with col2:
        is_sba_express = st.checkbox(
            "SBA Express Loan",
            value=False,
            help="SBA Express loans have 50% guarantee vs 75% for standard loans"
        )

        is_fixed_rate = st.checkbox(
            "Fixed Interest Rate",
            value=True,
            help="Fixed rate (checked) vs Variable rate (unchecked)"
        )

    col3, col4 = st.columns(2)

    with col3:
        revolver_status = st.checkbox(
            "Revolver Status",
            value=False,
            help="Revolving line of credit"
        )

    with col4:
        jobs_supported = st.number_input(
            "Jobs Supported",
            min_value=0,
            max_value=1000,
            value=5,
            step=1,
            help="Number of jobs created/retained"
        )

    # ====================
    # BUSINESS SECTION
    # ====================
    st.subheader("🏢 Business Information")
    col5, col6 = st.columns(2)

    with col5:
        naics_name = st.selectbox(
            "Industry Sector",
            options=list(NAICS_SECTORS.keys()),
            index=6,  # Default to Retail Trade
            help="NAICS industry classification"
        )
        naics_code = NAICS_SECTORS[naics_name] + "1110"  # Append to make full code

        business_type = st.selectbox(
            "Business Type",
            options=["CORPORATION", "INDIVIDUAL", "PARTNERSHIP"],
            index=0
        )

    with col6:
        business_age = st.selectbox(
            "Business Age",
            options=BUSINESS_AGE_OPTIONS,
            index=0
        )

    # ====================
    # LOCATION SECTION
    # ====================
    st.subheader("📍 Location Information")
    col7, col8 = st.columns(2)

    with col7:
        project_state_name = st.selectbox(
            "Project State",
            options=list(US_STATES.keys()),
            index=4,  # Default to California
            help="State where project is located"
        )
        project_state = US_STATES[project_state_name]

    with col8:
        bank_state_name = st.selectbox(
            "Bank State",
            options=list(US_STATES.keys()),
            index=4,  # Default to California
            help="State where lending bank is located"
        )
        bank_state = US_STATES[bank_state_name]

    # ====================
    # FLAGS SECTION
    # ====================
    st.subheader("🏷️ Additional Flags")
    col9, col10, col11 = st.columns(3)

    with col9:
        is_franchise = st.checkbox("Franchise", value=False)

    with col10:
        is_credit_union = st.checkbox("Credit Union", value=False)

    with col11:
        has_collateral = st.checkbox("Has Collateral", value=True)

    # Submit button
    submit_button = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

# ====================
# PROCESS SUBMISSION
# ====================
if submit_button:
    # Clear any previous results by using a unique container
    st.markdown("---")
    st.header("📊 Risk Assessment Results")
    st.caption(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Client-side calculations
    approval_fy = datetime.now().year
    sba_guaranteed = gross_approval * 0.50 if is_sba_express else gross_approval * 0.75
    location_id = 12345.0  # In production, this would come from a database lookup

    # Build payload
    payload = {
        "GrossApproval": float(gross_approval),
        "SBAGuaranteedApproval": float(sba_guaranteed),
        "InitialInterestRate": float(interest_rate),
        "ApprovalFY": approval_fy,
        "RevolverStatus": 1 if revolver_status else 0,
        "NAICSCode": naics_code,
        "BusinessType": business_type,
        "BusinessAge": business_age,
        "JobsSupported": int(jobs_supported),
        "ProjectState": project_state,
        "BankState": bank_state,
        "LocationID": location_id,
        "ApprovalDate": datetime.now().strftime("%Y-%m-%d"),
        "BankNCUANumber": "12345" if is_credit_union else None,
        "FranchiseCode": "FRNCH" if is_franchise else None,
        "FixedorVariableInterestRate": "F" if is_fixed_rate else "V",
        "CollateralInd": "Y" if has_collateral else "N"
    }

    # Call /predict endpoint
    with st.spinner("Analyzing loan application..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            prediction = response.json()

            # Display prediction
            prob = prediction['default_probability']
            risk = prediction['risk_category']
            recommendation = prediction['recommendation']

            # Results in columns
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric(
                    "Default Probability",
                    f"{prob:.2%}",
                    delta=None,
                    help="Probability that the loan will default"
                )

            with col_b:
                risk_emoji = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
                st.metric(
                    "Risk Category",
                    f"{risk_emoji} {risk}",
                    delta=None
                )

            with col_c:
                if recommendation == "APPROVE":
                    st.success(f"✅ Recommendation: **{recommendation}**")
                else:
                    st.error(f"❌ Recommendation: **{recommendation}**")

            # Decision box
            if prob >= 0.28:
                st.error(
                    f"⚠️ **HIGH RISK**: Default probability ({prob:.2%}) exceeds threshold (28%). "
                    f"Recommend **REJECTION** or require additional collateral/guarantees."
                )
            elif prob >= 0.15:
                st.warning(
                    f"⚠️ **MEDIUM RISK**: Default probability ({prob:.2%}) is moderate. "
                    f"Recommend **APPROVAL** with closer monitoring."
                )
            else:
                st.success(
                    f"✅ **LOW RISK**: Default probability ({prob:.2%}) is low. "
                    f"Recommend **APPROVAL**."
                )

            # Explanation of calculation
            with st.expander("📖 How is this probability calculated?"):
                st.markdown(f"""
                ### Calculation Process:

                **Step 1: Feature Engineering**
                - Your loan application (15 input fields) is transformed into **97 numerical features**
                - Features created: IsCovidEra, NAICSSector, LocationIDCount, one-hot encoded states, etc.

                **Step 2: XGBoost Model Prediction**
                - Model trained on **44,667 historical SBA loans** (92.5% good, 7.5% defaults)
                - XGBoost uses **100 decision trees** to vote on the outcome
                - Each tree analyzes all 97 features and votes for "Paid-in-Full" or "Default"
                - Final probability = weighted average of all tree votes

                **Step 3: Risk Categorization**
                - **Probability = {prob:.2%}** (probability this loan will default)
                - Risk thresholds:
                  - 🔴 **HIGH** (≥28%): Recommend REJECT
                  - 🟡 **MEDIUM** (15-27%): Approve with monitoring
                  - 🟢 **LOW** (<15%): Approve

                **Model Performance:**
                - ROC-AUC: 0.8317 (83% accuracy in ranking risk)
                - Recall: 83.4% (catches 83% of actual defaults)
                - Trained on loans from 2020-2024

                **Why 28% threshold?**
                - Optimized to catch 83% of defaults while minimizing false alarms
                - Lower than 50% because the cost of missing a default is higher than rejecting a good loan
                """)

                st.info("💡 **Tip**: Check the SHAP waterfall plot below to see which features contributed most to this prediction.")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Prediction API Error: {str(e)}")
            st.stop()

    # Call /explain endpoint
    st.markdown("---")
    st.header("🔍 SHAP Waterfall Plot Explanation")
    st.markdown("This plot shows how each feature contributes to moving from the base prediction to the final prediction:")

    with st.spinner("Generating SHAP explanation..."):
        try:
            explain_response = requests.post(f"{API_URL}/explain", json=payload, timeout=15)
            explain_response.raise_for_status()
            explanation = explain_response.json()

            # Prepare SHAP visualization
            import numpy as np
            import shap
            import matplotlib.pyplot as plt

            shap_values = np.array(explanation['shap_values'])  # Convert list to numpy array
            base_value = explanation['base_value']
            feature_names = explanation['feature_names']

            # Create SHAP Explanation object for waterfall plot
            shap_explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                feature_names=feature_names
            )

            # Render SHAP waterfall plot
            st.markdown("**Feature Impact on Prediction:**")

            # Create matplotlib figure with custom size for better readability
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(shap_explanation, max_display=15, show=False)

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Display in Streamlit with container width
            st.pyplot(fig, use_container_width=True)

            # Close figure to free memory
            plt.close(fig)

            st.markdown("""
            **How to read this waterfall plot:**
            - **Base value (E[f(X)])**: The average model prediction across the training dataset ({:.4f})
            - **Red bars** (positive SHAP values): Features pushing the prediction **higher** (towards default)
            - **Blue bars** (negative SHAP values): Features pushing the prediction **lower** (towards paid-in-full)
            - **f(x)**: The final prediction for this loan ({:.4f})
            - Features are ordered by absolute impact, showing the top 15 most influential features

            The waterfall shows the cumulative effect: starting from the base value, each feature either
            increases (red) or decreases (blue) the prediction, ultimately arriving at the final prediction.
            """.format(base_value, base_value + shap_values.sum()))

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Explanation API Error: {str(e)}")
        except Exception as e:
            st.warning(f"⚠️ SHAP visualization error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    "**SBA Loan Risk Assessment Dashboard** | "
    "Powered by XGBoost + SHAP | "
    "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
)
