"""
Data preparation module for Predictive Delivery Optimizer
Merges three datasets and creates target variables
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import from utils module in the same directory
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import set_random_seeds, print_section, save_json

def load_and_merge_data(data_dir: str = "data", random_state: int = 42) -> pd.DataFrame:
    """
    Load and merge three CSV files on Order_ID
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the CSV files
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame : Merged dataset
    """
    set_random_seeds(random_state)
    print_section("Loading and Merging Data")
    
    data_path = Path(data_dir)
    
    # Load the three datasets
    print("Loading delivery_performance.csv...")
    df_perf = pd.read_csv(data_path / "delivery_performance.csv")
    print(f"  Shape: {df_perf.shape}")
    
    print("Loading orders.csv...")
    df_orders = pd.read_csv(data_path / "orders.csv")
    print(f"  Shape: {df_orders.shape}")
    
    print("Loading routes_distance.csv...")
    df_routes = pd.read_csv(data_path / "routes_distance.csv")
    print(f"  Shape: {df_routes.shape}")
    
    # Merge the datasets
    print("\nMerging datasets on Order_ID...")
    df = df_orders.merge(df_perf, on="Order_ID", how="inner")
    df = df.merge(df_routes, on="Order_ID", how="inner")
    
    print(f"Merged dataset shape: {df.shape}")
    print(f"Number of unique orders: {df['Order_ID'].nunique()}")
    
    return df

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that contain post-delivery information (data leakage)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : Dataframe with leakage columns removed
    """
    print_section("Removing Data Leakage Columns")
    
    # Columns to drop (post-delivery information)
    leakage_columns = [
        'Route', 
        'Origin', 
        'Destination', 
        'Customer_Segment', 
        'Order_Date',
        'Special_Handling', 
        'Delivery_Status', 
        'Quality_Issue'
    ]
    
    # Check which columns exist
    columns_to_drop = [col for col in leakage_columns if col in df.columns]
    columns_not_found = [col for col in leakage_columns if col not in df.columns]
    
    print(f"Columns to drop: {columns_to_drop}")
    if columns_not_found:
        print(f"Columns not found (already removed or don't exist): {columns_not_found}")
    
    df_cleaned = df.drop(columns=columns_to_drop)
    
    print(f"\nDropped {len(columns_to_drop)} columns")
    print(f"Shape after dropping: {df_cleaned.shape}")
    
    return df_cleaned

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for classification and regression
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Actual_Delivery_Days and Promised_Delivery_Days
        
    Returns:
    --------
    pd.DataFrame : Dataframe with target variables added
    """
    print_section("Creating Target Variables")
    
    # Create delay_days
    df['delay_days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
    
    # Create binary target: is_delayed
    df['is_delayed'] = (df['delay_days'] > 0).astype(int)
    
    # Print statistics
    print(f"Delay statistics:")
    print(f"  Mean delay: {df['delay_days'].mean():.2f} days")
    print(f"  Median delay: {df['delay_days'].median():.2f} days")
    print(f"  Max delay: {df['delay_days'].max():.2f} days")
    print(f"  Min delay: {df['delay_days'].min():.2f} days")
    
    print(f"\nClass balance (is_delayed):")
    print(df['is_delayed'].value_counts())
    print(f"  Delay rate: {df['is_delayed'].mean()*100:.2f}%")
    
    return df

def print_schema(df: pd.DataFrame) -> None:
    """Print dataset schema and basic info"""
    print_section("Dataset Schema")
    
    print("Column types:")
    print(df.dtypes)
    
    print(f"\nDataframe shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")

def main():
    """Main function to prepare data"""
    # Load and merge data
    df = load_and_merge_data()
    
    # Drop leakage columns
    df_cleaned = drop_leakage_columns(df)
    
    # Create target variables
    df_final = create_target_variables(df_cleaned)
    
    # Save to processed/merged.csv
    output_path = Path("processed/merged.csv")
    df_final.to_csv(output_path, index=False)
    print_section(f"Saved merged data to {output_path}")
    
    # Print schema
    print_schema(df_final)
    
    # Save basic stats
    stats = {
        'n_rows': len(df_final),
        'n_features': len(df_final.columns) - 2,  # excluding targets
        'delay_rate': float(df_final['is_delayed'].mean()),
        'mean_delay_days': float(df_final['delay_days'].mean())
    }
    save_json(stats, "processed/data_stats.json")
    
    return df_final

if __name__ == "__main__":
    df = main()
