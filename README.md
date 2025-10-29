# Predictive Delivery Optimizer

A production-ready machine learning prototype for NexGen Logistics that predicts delivery delays using CatBoost models with hyperparameter tuning via Optuna and MLflow tracking on DagsHub.

## Project Overview

This project builds predictive models to forecast delivery delays by analyzing order characteristics, carrier performance, weather conditions, traffic patterns, and route attributes. The system includes:

- **Classification Model**: Predicts whether a delivery will be delayed (binary: delayed/on-time)
- **Regression Model**: Predicts the number of delay days

## Features

- ✅ Data preprocessing and feature engineering (no data leakage)
- ✅ CatBoost models with Optuna hyperparameter tuning
- ✅ MLflow experiment tracking on DagsHub
- ✅ Streamlit dashboard for business insights
- ✅ Single and batch prediction capabilities
- ✅ Scenario analysis and carrier recommendations

## Project Structure

```
OFI/
├── data/                    # Raw datasets
│   ├── delivery_performance.csv
│   ├── orders.csv
│   └── routes_distance.csv
├── processed/              # Processed data and preprocessors
│   ├── merged.csv
│   ├── features.csv
│   └── preprocessor.joblib
├── models/                 # Trained models
│   ├── best_catboost_classifier.cbm
│   ├── best_catboost_regressor.cbm
│   └── *.json (metadata)
├── src/                    # Source code
│   ├── data_prep.py       # Data merging and cleaning
│   ├── features.py        # Feature engineering
│   ├── modeling.py        # Model training with Optuna
│   └── utils.py           # Helper functions
├── notebooks/              # Jupyter notebooks for EDA
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
# Navigate to project directory
cd OFI

# Install required packages
pip install -r requirements.txt
```

### 2. Configure DagsHub (Optional but Recommended)

The code is pre-configured to use DagsHub for MLflow tracking:

```python
# In src/modeling.py
dagshub.init(
    repo_owner="AbdurRahman22224",
    repo_name="Predictive_Delivery_Optimizer",
    mlflow=True
)
```

You'll need to:
1. Create a repository on DagsHub
2. Update the `repo_owner` and `repo_name` in `src/modeling.py` if different
3. Add your DagsHub token when running the training

Alternatively, you can use local MLflow by commenting out the DagsHub initialization and using:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

## Usage

### 1. Prepare Data and Engineer Features

```bash
# Merge datasets and create target variables
python src/data_prep.py

# Create derived features and build preprocessor
python src/features.py
```

This creates:
- `processed/merged.csv` - Cleaned merged dataset
- `processed/features.csv` - Dataset with derived features
- `processed/preprocessor.joblib` - Preprocessing pipeline

### 2. Train Models

```bash
# Train CatBoost models with Optuna tuning
python src/modeling.py
```

This will:
- Train a CatBoost classifier for delay prediction (is_delayed)
- Train a CatBoost regressor for delay days prediction
- Run 50 trials for classification, 30 trials for regression
- Log all experiments to MLflow (DagsHub)
- Save best models to `models/`

**Expected Performance:**
- Classification F1-score: >0.75
- Classification ROC-AUC: >0.80
- Regression RMSE: <2 days

### 3. Launch Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard includes:
- **Overview**: KPIs, filters, and exploratory visualizations
- **Model Performance**: Metrics, ROC curve, confusion matrix, feature importance
- **Predictions**: Single order form and batch prediction
- **Insights**: Carrier rankings, scenario analysis, actionable recommendations

## Model Details

### Classification Model (CatBoostClassifier)
- **Target**: `is_delayed` (0 = on-time, 1 = delayed)
- **Hyperparameters**: Tuned via Optuna (depth, learning_rate, iterations, etc.)
- **Evaluation**: Accuracy, Precision, Recall, F1-score, ROC-AUC

### Regression Model (CatBoostRegressor)
- **Target**: `delay_days` (actual - promised delivery days)
- **Hyperparameters**: Tuned via Optuna
- **Evaluation**: RMSE, MAE, R²

### Feature Engineering

**Derived Features:**
- `cost_per_promised_day`: Cost efficiency metric
- `promised_days_bucket`: Categorical buckets (0-1, 2-3, 4+ days)
- `order_value_bucket`: Quartile-based buckets (Q1-Q4)
- `distance_bucket`: Route distance categories (short/medium/long)

**Ordinal Encoding:**
- Priority: Economy (0) < Standard (1) < Express (2) < Unknown (3)
- Weather_Impact: Unknown (0) < Fog (1) < Light_Rain (2) < Heavy_Rain (3)

**Preprocessing:**
- Numeric features: Median imputation + StandardScaler
- Categorical features: One-hot encoding
- Ordinal features: Custom ordinal encoding

## Data Leakage Prevention

**Columns Dropped** (contain post-delivery information):
- `Route`, `Origin`, `Destination`, `Customer_Segment`
- `Order_Date`, `Special_Handling`
- `Delivery_Status`, `Quality_Issue`

Only features available **before delivery** are used for training.

## MLflow Tracking

All experiments are logged to MLflow (DagsHub):

- **Parameters**: All hyperparameters from Optuna
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, RMSE, MAE, R²
- **Artifacts**: Models, preprocessors, metadata JSON files
- **Model Registry**: Best models registered and staged (Staging/Production)

View experiments at: https://dagshub.com/AbdurRahman22224/Predictive_Delivery_Optimizer

## Model Explainability

Feature importance is computed from the trained CatBoost models to explain predictions:

- **Feature Importance**: Top features that drive delay predictions
- **Business Insights**: Understanding which factors impact delivery delays most


## Reproducibility

- All random seeds set to 42
- Stratified train/test splits
- Fixed hyperparameter search spaces
- Version-controlled preprocessor pipeline

## Author

Abdur Rahman - ML Engineering Intern at NexGen Logistics

## License

Internal project for NexGen Logistics.

