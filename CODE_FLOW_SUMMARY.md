# Code Flow Summary - Predictive Delivery Optimizer

## Overview
This document explains how the functions are used in the overall codebase and the data flow.

## Data Flow

### 1. Data Preparation (`src/data_prep.py`)
**Functions Used:**
- `load_and_merge_data()` - Merges 3 CSV files on Order_ID
- `drop_leakage_columns()` - Removes post-delivery columns
- `create_target_variables()` - Creates `is_delayed` and `delay_days`

**Output:** `processed/merged.csv`

**Target Variable Formula:**
```python
delay_days = Actual_Delivery_Days - Promised_Delivery_Days
is_delayed = 1 if delay_days > 0 else 0
```

✅ **Correct Formula** - This properly calculates delay as the difference between actual and promised delivery days.

### 2. Feature Engineering (`src/features.py`)
**Functions Used:**
- `create_derived_features()` - Creates all derived features
- `build_preprocessor()` - Builds sklearn pipeline

**Derived Features Created:**
1. `cost_per_promised_day = Delivery_Cost_INR / Promised_Delivery_Days`
   - ✅ **Formula Correct** - Calculates cost efficiency per day

2. `promised_days_bucket` - Bins: "0-1", "2-3", "4+"
   - ✅ **Formula Correct** - Creates meaningful time buckets

3. `order_value_bucket` - Quartiles: Q1, Q2, Q3, Q4
   - ✅ **Formula Correct** - Uses pandas qcut for quartiles

4. `distance_bucket` - Bins: "short" (<500km), "medium" (500-1500km), "long" (>1500km)
   - ✅ **Formula Correct** - Creates distance categories

**Output:** `processed/features.csv`, `processed/preprocessor.joblib`

### 3. Model Training (`src/modeling.py`)
**Functions Used:**
- `load_data()` - Loads features.csv
- `prepare_features_and_targets()` - Separates features and targets
- `load_preprocessor()` - Loads preprocessor
- `train_classifier()` - Trains CatBoostClassifier with Optuna
- `train_regressor()` - Trains CatBoostRegressor with Optuna
- `objective_classifier()` - Optuna objective for classification
- `objective_regressor()` - Optuna objective for regression

**Train/Test Split:**
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

✅ **Split Correct** - 80/20 split with stratification for balanced classes

**Output:** 
- `models/best_catboost_classifier.cbm`
- `models/best_catboost_regressor.cbm`
- `models/best_classifier_info.json`
- `models/best_regressor_info.json`

## 4. Streamlit Dashboard (`app.py`)
**Functions Used:**
- `load_data()` - Loads processed features
- `load_models()` - Loads CatBoost models and preprocessor
- Navigation: Overview, Model Performance, Predictions, Insights

**Prediction Flow:**
1. User inputs features in Streamlit
2. Create input DataFrame with all 14 required features
3. Preprocess using loaded preprocessor
4. Predict using loaded models
5. Display results with probability and expected delay


## Final Feature Set (14 features)
**Numeric (7):**
1. Order_Value_INR
2. Delivery_Cost_INR
3. Distance_KM
4. Fuel_Consumption_L
5. Toll_Charges_INR
6. Promised_Delivery_Days
7. cost_per_promised_day (derived)

**Categorical - OHE (2):**
8. Carrier
9. Product_Category

**Ordinal (2):**
10. Priority
11. Weather_Impact

**Derived Categorical - OHE (3):**
12. promised_days_bucket
13. order_value_bucket
14. distance_bucket

**Excluded Features (to prevent leakage):**
- Customer_Rating (post-delivery information)
- Traffic_Delay_Minutes (not available beforehand)
- carrier_delay_rate_history (removed - not used)

## Model Performance (Actual Results)
- **Classifier:** 
  - F1=0.6667, Accuracy=70%, Precision=0.6923, Recall=0.6429, ROC-AUC=0.7589
  - 50 trials with Optuna ✅
  - Best params: depth=5, learning_rate=0.1249, iterations=425
  
- **Regressor:** 
  - RMSE=1.50 days, MAE=1.14 days, R²=0.1048
  - 50 trials with Optuna ✅
  - Best params: depth=5, learning_rate=0.0692, iterations=705

## Streamlit Dashboard Implementation

### Overview Page Charts:
1. ✅ **Delay Rate by Carrier** (Bar Chart)
2. ✅ **Distance vs Delivery Cost** (Scatter Plot with color by delay)
3. ✅ **Order Priority Distribution** (Pie/Donut Chart)
4. ✅ **Delay Rate by Product Category** (Horizontal Bar Chart)
5. ✅ **Feature Correlation Heatmap**

### Model Performance Page:
- ✅ Displays classifier and regressor metrics
- ✅ Shows hyperparameters in JSON format

### Predictions Page:
- ✅ Single order prediction form
- ✅ Risk level classification (Low/Medium/High)
- ✅ Expected delay days from regressor

### Insights Page:
- ✅ Carrier rankings table
- ✅ Actionable recommendations


