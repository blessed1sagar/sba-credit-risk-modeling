"""
Feature engineering consistency tests for SBA Loan ML pipeline.

CRITICAL: These tests validate that training and inference feature engineering
produce identical results. This prevents train-serve skew.

Run with: pytest tests/test_feature_consistency.py -v
"""
import pytest
import pandas as pd
import numpy as np
from src.utils.feature_engineering import engineer_features
from src import config


@pytest.fixture
def sample_raw_data():
    """
    Sample raw loan data for testing.

    Includes all required columns for feature engineering.
    """
    return pd.DataFrame([
        {
            # Financials
            "GrossApproval": 50000,
            "SBAGuaranteedApproval": 37500,
            "ApprovalFY": 2020,
            "InitialInterestRate": 6.5,
            "RevolverStatus": 0,
            "JobsSupported": 5,
            # Dates
            "ApprovalDate": "2020-03-15",
            # Business
            "NAICSCode": "441110",
            "BusinessType": "CORPORATION",
            "BusinessAge": "Existing or more than 2 years old",
            "BusinessAge_Clean": "Existing",  # Already cleaned
            # Location
            "ProjectState": "CA",
            "BankState": "CA",
            "LocationID": 12345.0,
            # Flags
            "BankNCUANumber": None,
            "FranchiseCode": None,
            "FixedorVariableInterestRate": "F",
            "CollateralInd": "Y",
        }
    ])


@pytest.fixture
def sample_frequency_map():
    """Sample frequency map for LocationID encoding."""
    return {
        12345.0: 100,
        67890.0: 50,
        11111.0: 10
    }


class TestFeatureEngineering:
    """Test suite for feature engineering consistency."""

    def test_covid_indicator_creation(self, sample_raw_data):
        """Test that IsCovidEra is created correctly and ApprovalDate is dropped."""
        df_result = engineer_features(sample_raw_data.copy())

        # Verify IsCovidEra exists and is correct
        assert 'IsCovidEra' in df_result.columns, "IsCovidEra feature not created"
        assert df_result['IsCovidEra'].iloc[0] == 1, "March 2020 should be COVID period"

        # Verify ApprovalDate is dropped
        assert config.APPROVAL_DATE_COLUMN not in df_result.columns, "ApprovalDate should be dropped"

    def test_naics_sector_extraction(self, sample_raw_data):
        """Test that NAICSSector is extracted correctly from NAICSCode."""
        df_result = engineer_features(sample_raw_data.copy())

        # Verify NAICSSector exists and is correct
        assert 'NAICSSector' in df_result.columns, "NAICSSector feature not created"
        assert df_result['NAICSSector'].iloc[0] == "44", "NAICSSector should be first 2 digits"

    def test_binary_indicators_creation(self, sample_raw_data):
        """Test that all binary indicators are created correctly."""
        df_result = engineer_features(sample_raw_data.copy())

        # Verify all binary indicators exist
        assert 'IsCreditUnion' in df_result.columns
        assert 'IsFranchise' in df_result.columns
        assert 'IsFixedRate' in df_result.columns
        assert 'HasCollateral' in df_result.columns

        # Verify values
        assert df_result['IsCreditUnion'].iloc[0] == 0, "BankNCUANumber is None"
        assert df_result['IsFranchise'].iloc[0] == 0, "FranchiseCode is None"
        assert df_result['IsFixedRate'].iloc[0] == 1, "Interest rate type is 'F'"
        assert df_result['HasCollateral'].iloc[0] == 1, "Collateral indicator is 'Y'"

    def test_same_state_lending(self, sample_raw_data):
        """Test that SameStateLending is created correctly."""
        df_result = engineer_features(sample_raw_data.copy())

        # Verify SameStateLending exists and is correct
        assert 'SameStateLending' in df_result.columns, "SameStateLending feature not created"
        assert df_result['SameStateLending'].iloc[0] == 1, "CA == CA should be same-state"

    def test_frequency_encoding_training_mode(self, sample_raw_data):
        """Test frequency encoding in training mode (no frequency_map provided)."""
        df_result = engineer_features(sample_raw_data.copy())

        # Verify LocationIDCount exists
        assert 'LocationIDCount' in df_result.columns, "LocationIDCount feature not created"
        assert df_result['LocationIDCount'].iloc[0] == 1, "Single row should have frequency 1"

        # Verify LocationID is dropped
        assert config.LOCATION_ID_COLUMN not in df_result.columns, "LocationID should be dropped"

    def test_frequency_encoding_inference_mode(self, sample_raw_data, sample_frequency_map):
        """Test frequency encoding in inference mode (frequency_map provided)."""
        df_result = engineer_features(
            sample_raw_data.copy(),
            frequency_map=sample_frequency_map
        )

        # Verify LocationIDCount uses provided frequency map
        assert 'LocationIDCount' in df_result.columns
        assert df_result['LocationIDCount'].iloc[0] == 100, "Should use frequency from map"

    def test_frequency_encoding_unseen_location(self, sample_frequency_map):
        """Test frequency encoding handles unseen LocationIDs correctly."""
        # Create data with unseen LocationID
        unseen_data = pd.DataFrame([{
            "GrossApproval": 50000,
            "SBAGuaranteedApproval": 37500,
            "ApprovalFY": 2020,
            "InitialInterestRate": 6.5,
            "RevolverStatus": 0,
            "JobsSupported": 5,
            "ApprovalDate": "2020-03-15",
            "NAICSCode": "441110",
            "BusinessType": "CORPORATION",
            "BusinessAge": "Existing or more than 2 years old",
            "BusinessAge_Clean": "Existing",
            "ProjectState": "CA",
            "BankState": "CA",
            "LocationID": 99999.0,  # Unseen LocationID
            "BankNCUANumber": None,
            "FranchiseCode": None,
            "FixedorVariableInterestRate": "F",
            "CollateralInd": "Y",
        }])

        df_result = engineer_features(unseen_data, frequency_map=sample_frequency_map)

        # Should use minimum frequency from training
        assert df_result['LocationIDCount'].iloc[0] == 10, "Unseen LocationID should use min frequency"

    def test_one_hot_encoding_creates_columns(self, sample_raw_data):
        """Test that one-hot encoding creates expected columns."""
        df_result = engineer_features(sample_raw_data.copy())

        # Check BusinessType encoding
        type_cols = [col for col in df_result.columns if col.startswith('Type_')]
        assert len(type_cols) > 0, "BusinessType should be one-hot encoded"
        assert 'Type_CORPORATION' in df_result.columns

        # Check BusinessAge encoding
        age_cols = [col for col in df_result.columns if col.startswith('Age_')]
        assert len(age_cols) > 0, "BusinessAge should be one-hot encoded"

        # Check ProjectState encoding
        state_cols = [col for col in df_result.columns if col.startswith('State_')]
        assert len(state_cols) > 0, "ProjectState should be one-hot encoded"
        assert 'State_CA' in df_result.columns

    def test_raw_columns_dropped(self, sample_raw_data):
        """Test that raw/intermediate columns are dropped."""
        df_result = engineer_features(sample_raw_data.copy())

        # Columns that should be dropped
        dropped_columns = [
            "NAICSCode",
            "BusinessAge",
            "FixedorVariableInterestRate",
            "CollateralInd",
            "LocationID",
            "BankNCUANumber",
            "FranchiseCode",
            "BankState"  # Dropped after SameStateLending creation
        ]

        for col in dropped_columns:
            assert col not in df_result.columns, f"{col} should be dropped"

    def test_no_missing_values(self, sample_raw_data):
        """Test that engineered features have no missing values."""
        df_result = engineer_features(sample_raw_data.copy())

        missing_counts = df_result.isnull().sum()
        assert missing_counts.sum() == 0, "Engineered features should have no missing values"

    def test_inference_column_alignment(self, sample_raw_data):
        """Test that inference mode aligns columns with expected training columns."""
        # Simulate training: get feature columns
        df_training = engineer_features(sample_raw_data.copy())
        training_columns = df_training.columns.tolist()

        # Simulate inference with same data
        df_inference = engineer_features(
            sample_raw_data.copy(),
            frequency_map={12345.0: 100},
            expected_columns=training_columns
        )

        # Verify columns match exactly
        assert df_inference.columns.tolist() == training_columns, \
            "Inference columns should match training columns exactly"

    def test_multi_row_consistency(self, sample_frequency_map):
        """Test that feature engineering is consistent across multiple rows."""
        # Create multi-row dataset
        multi_row_data = pd.DataFrame([
            {
                "GrossApproval": 50000,
                "SBAGuaranteedApproval": 37500,
                "ApprovalFY": 2020,
                "InitialInterestRate": 6.5,
                "RevolverStatus": 0,
                "JobsSupported": 5,
                "ApprovalDate": "2020-03-15",
                "NAICSCode": "441110",
                "BusinessType": "CORPORATION",
                "BusinessAge": "Existing or more than 2 years old",
                "BusinessAge_Clean": "Existing",
                "ProjectState": "CA",
                "BankState": "CA",
                "LocationID": 12345.0,
                "BankNCUANumber": None,
                "FranchiseCode": None,
                "FixedorVariableInterestRate": "F",
                "CollateralInd": "Y",
            },
            {
                "GrossApproval": 75000,
                "SBAGuaranteedApproval": 56250,
                "ApprovalFY": 2019,
                "InitialInterestRate": 7.0,
                "RevolverStatus": 1,
                "JobsSupported": 10,
                "ApprovalDate": "2019-06-20",
                "NAICSCode": "541110",
                "BusinessType": "INDIVIDUAL",
                "BusinessAge": "New Business or 2 years or less",
                "BusinessAge_Clean": "NewBusiness",
                "ProjectState": "NY",
                "BankState": "NY",
                "LocationID": 67890.0,
                "BankNCUANumber": "12345",
                "FranchiseCode": "FRNCH",
                "FixedorVariableInterestRate": "V",
                "CollateralInd": "N",
            }
        ])

        df_result = engineer_features(multi_row_data, frequency_map=sample_frequency_map)

        # Verify both rows processed correctly
        assert len(df_result) == 2, "Should process both rows"
        assert df_result['IsCovidEra'].iloc[0] == 1, "Row 1: March 2020 is COVID"
        assert df_result['IsCovidEra'].iloc[1] == 0, "Row 2: June 2019 is pre-COVID"
        assert df_result['LocationIDCount'].iloc[0] == 100, "Row 1: LocationID 12345 frequency"
        assert df_result['LocationIDCount'].iloc[1] == 50, "Row 2: LocationID 67890 frequency"

    def test_dataframe_dtypes(self, sample_raw_data):
        """Test that engineered features have correct data types."""
        df_result = engineer_features(sample_raw_data.copy())

        # Binary indicators should be int
        binary_features = ['IsCovidEra', 'IsCreditUnion', 'IsFranchise', 'IsFixedRate', 'HasCollateral', 'SameStateLending']
        for feature in binary_features:
            if feature in df_result.columns:
                assert df_result[feature].dtype in [np.int64, np.int32, int], \
                    f"{feature} should be integer type"

        # One-hot encoded features should be int
        one_hot_cols = [col for col in df_result.columns if col.startswith(('Type_', 'Age_', 'State_'))]
        for col in one_hot_cols:
            assert df_result[col].dtype in [np.int64, np.int32, int], \
                f"{col} should be integer type"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
