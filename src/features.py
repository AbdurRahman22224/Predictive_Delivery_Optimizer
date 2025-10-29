"""
Feature engineering and preprocessing pipeline for Predictive Delivery Optimizer
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from utils import set_random_seeds, print_section

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that are available before delivery
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : Dataframe with derived features added
    """
    print_section("Creating Derived Features")
    
    df = df.copy()
    
    # Cost efficiency feature
    df['cost_per_promised_day'] = df['Delivery_Cost_INR'] / df['Promised_Delivery_Days'].replace(0, 1)
    print("Created: cost_per_promised_day")
    
    # Promised days buckets
    df['promised_days_bucket'] = pd.cut(
        df['Promised_Delivery_Days'],
        bins=[0, 1, 3, float('inf')],
        labels=['0-1', '2-3', '4+'],
        include_lowest=True
    ).astype(str)
    print("Created: promised_days_bucket")
    
    # Order value buckets (quartiles)
    df['order_value_bucket'] = pd.qcut(
        df['Order_Value_INR'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    ).astype(str)
    print("Created: order_value_bucket")
    
    # Distance buckets
    df['distance_bucket'] = pd.cut(
        df['Distance_KM'],
        bins=[0, 500, 1500, float('inf')],
        labels=['short', 'medium', 'long'],
        include_lowest=True
    ).astype(str)
    print("Created: distance_bucket")
    
    # Ensure missing weather values are treated as "Unknown"
    df['Weather_Impact'] = df['Weather_Impact'].fillna('Unknown')
    
    return df

def build_preprocessor(df: pd.DataFrame, save_path: str = "processed/preprocessor.joblib") -> ColumnTransformer:
    """
    Build preprocessing pipeline for model training
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data with features
    save_path : str
        Path to save the preprocessor
        
    Returns:
    --------
    ColumnTransformer : Fitted preprocessor
    """
    print_section("Building Preprocessing Pipeline")
    
    # Define feature columns
    numeric_features = [
        'Order_Value_INR',
        'Delivery_Cost_INR',
        'Distance_KM',
        'Fuel_Consumption_L',
        'Toll_Charges_INR',
        'Promised_Delivery_Days',
        'cost_per_promised_day'
    ]
    
    # Columns to ordinal encode
    ordinal_features = ['Priority', 'Weather_Impact']
    
    # Priority ordering: Economy < Standard < Express < Unknown
    priority_categories = [["Economy", "Standard", "Express", "Unknown"]]
    
    # Weather ordering: Unknown (blank) -> fog -> Light_Rain -> Heavy_Rain
    weather_categories = [["Unknown", "Fog", "Light_Rain", "Heavy_Rain"]]
    
    # Nominal categorical features
    nominal_features = ['Carrier', 'Product_Category']
    
    # Bucketed features
    bucket_features = ['promised_days_bucket', 'order_value_bucket', 'distance_bucket']
    
    # Check which features exist in the dataframe
    numeric_features = [f for f in numeric_features if f in df.columns]
    nominal_features = [f for f in nominal_features if f in df.columns]
    bucket_features = [f for f in bucket_features if f in df.columns]
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Ordinal features: {ordinal_features}")
    print(f"Nominal features: {nominal_features}")
    print(f"Bucket features: {bucket_features}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Separate encoders for Priority and Weather_Impact
    priority_encoder = OrdinalEncoder(categories=priority_categories, handle_unknown='use_encoded_value', unknown_value=-1)
    weather_encoder = OrdinalEncoder(categories=weather_categories, handle_unknown='use_encoded_value', unknown_value=-1)
    
    ordinal_transformer_priority = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Economy')),
        ('encoder', priority_encoder)
    ])
    
    ordinal_transformer_weather = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', weather_encoder)
    ])
    
    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Build the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('priority', ordinal_transformer_priority, ['Priority']),
            ('weather', ordinal_transformer_weather, ['Weather_Impact']),
            ('nominal', onehot_transformer, nominal_features + bucket_features)
        ],
        remainder='passthrough'
    )
    
    # Fit the preprocessor
    print("Fitting preprocessor...")
    
    # Prepare data for fitting (drop targets and Order_ID)
    X = df.drop(columns=['Order_ID', 'Actual_Delivery_Days', 'delay_days', 'is_delayed'], errors='ignore')
    
    preprocessor.fit(X)
    
    # Save the preprocessor
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor saved to {save_path}")
    
    return preprocessor

def main():
    """Main function for feature engineering"""
    set_random_seeds(42)
    
    # Load merged data
    print("Loading merged data...")
    df = pd.read_csv("processed/merged.csv")
    
    # Create derived features
    df_features = create_derived_features(df)
    
    # Save features dataframe
    df_features.to_csv("processed/features.csv", index=False)
    
    # Drop leakage features before building preprocessor
    df_features_for_preprocessor = df_features.drop(columns=['Customer_Rating', 'Traffic_Delay_Minutes'], errors='ignore')
    
    # Build preprocessor
    preprocessor = build_preprocessor(df_features_for_preprocessor)
    
    print_section("Feature Engineering Complete")
    print(f"Final feature columns: {len(df_features.columns)}")
    print(f"Target columns: delay_days, is_delayed")
    print(f"Feature columns: {[col for col in df_features.columns if col not in ['Order_ID', 'Actual_Delivery_Days', 'delay_days', 'is_delayed']]}")

if __name__ == "__main__":
    main()
