"""
Model training with Optuna hyperparameter tuning and MLflow tracking
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix
)
import warnings
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import dagshub
import mlflow
import mlflow.sklearn
import mlflow.catboost

from utils import set_random_seeds, print_section, save_json

# Initialize DagsHub MLflow tracking
dagshub.init(
    repo_owner="AbdurRahman22224",
    repo_name="Predictive_Delivery_Optimizer",
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/AbdurRahman22224/Predictive_Delivery_Optimizer.mlflow")


def load_data():
    """Load processed features"""
    print_section("Loading Data")
    df = pd.read_csv("processed/features.csv")
    print(f"Data shape: {df.shape}")
    return df

def prepare_features_and_targets(df: pd.DataFrame):
    """Prepare features X and targets y_class, y_reg"""
    # Drop targets and ID
    feature_cols = [col for col in df.columns 
                    if col not in ['Order_ID', 'Actual_Delivery_Days', 'delay_days', 'is_delayed']]
    
    X = df[feature_cols]
    y_class = df['is_delayed']
    y_reg = df['delay_days']
    
    print(f"Features: {X.shape}")
    print(f"Target - Classification: {y_class.value_counts().to_dict()}")
    print(f"Target - Regression: mean={y_reg.mean():.2f}, std={y_reg.std():.2f}")
    
    return X, y_class, y_reg

def load_preprocessor():
    """Load the fitted preprocessor"""
    preprocessor = joblib.load("processed/preprocessor.joblib")
    return preprocessor

def objective_classifier(trial, X_train, y_train, X_val, y_val, preprocessor):
    """Objective function for classification"""
    # Suggest hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 200, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'rsm': trial.suggest_float('rsm', 0.4, 1.0),
        'random_seed': 42,
        'verbose': False
    }
    
    # Transform data
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    # Train model
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(**params)
    model.fit(
        X_train_trans, y_train,
        eval_set=(X_val_trans, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_val_trans)
    y_pred_proba = model.predict_proba(X_val_trans)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    # Report metrics to Optuna
    trial.set_user_attr('accuracy', accuracy)
    trial.set_user_attr('precision', precision)
    trial.set_user_attr('recall', recall)
    trial.set_user_attr('auc', auc)
    
    return f1  # Optimize for F1

def objective_regressor(trial, X_train, y_train, X_val, y_val, preprocessor):
    """Objective function for regression"""
    # Suggest hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 200, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'rsm': trial.suggest_float('rsm', 0.4, 1.0),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }
    
    # Transform data
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    # Train model
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(**params)
    model.fit(
        X_train_trans, y_train,
        eval_set=(X_val_trans, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_val_trans)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Report metrics to Optuna
    trial.set_user_attr('rmse', rmse)
    trial.set_user_attr('mae', mae)
    trial.set_user_attr('r2', r2)
    
    return rmse  # Minimize RMSE

def train_classifier(X, y_class, preprocessor, n_trials=50):
    """Train CatBoost classifier with Optuna"""
    print_section("Training CatBoost Classifier")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Further split training set for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {X_train_split.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_classifier(trial, X_train_split, y_train_split, X_val, y_val, preprocessor),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best trial
    best_trial = study.best_trial
    print(f"\nBest trial:")
    print(f"  F1-score: {best_trial.value:.4f}")
    print(f"  Params: {best_trial.params}")
    
    # Train final model on full training set
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    from catboost import CatBoostClassifier
    best_params = best_trial.params.copy()
    best_params['random_seed'] = 42
    best_params['verbose'] = False
    
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(
        X_train_trans, y_train,
        eval_set=(X_test_trans, y_test),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test_trans)
    y_pred_proba = final_model.predict_proba(X_test_trans)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest set performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")
    
    # Log to MLflow
    try:
        with mlflow.start_run(run_name="CatBoost_Classifier_Best_1"):
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc
            })
            
            # Save model
            model_path = "models/best_catboost_classifier.cbm"
            final_model.save_model(model_path)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact("processed/preprocessor.joblib")
            
            # Log model files as artifacts (avoid deprecated log_model API)
            # The .cbm file is already logged above, so we skip the registry log
    except Exception as e:
        print(f"MLflow logging encountered an issue: {e}")
        print("Continuing with model save to local files...")
    
    # Save metadata
    metadata = {
        'model_type': 'CatBoostClassifier',
        'hyperparameters': best_params,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(auc)
        },
        'n_trials': n_trials
    }
    save_json(metadata, "models/best_classifier_info.json")
    
    return final_model, best_params, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc
    }

def train_regressor(X, y_reg, preprocessor, n_trials=50):
    """Train CatBoost regressor with Optuna"""
    print_section("Training CatBoost Regressor")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    # Further split training set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Train: {X_train_split.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective_regressor(trial, X_train_split, y_train_split, X_val, y_val, preprocessor),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best trial
    best_trial = study.best_trial
    print(f"\nBest trial:")
    print(f"  RMSE: {best_trial.value:.4f}")
    print(f"  Params: {best_trial.params}")
    
    # Train final model
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    from catboost import CatBoostRegressor
    best_params = best_trial.params.copy()
    best_params['random_seed'] = 42
    best_params['verbose'] = False
    
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(
        X_train_trans, y_train,
        eval_set=(X_test_trans, y_test),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluate
    y_pred = final_model.predict(X_test_trans)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest set performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Log to MLflow
    try:
        with mlflow.start_run(run_name="CatBoost_Regressor_Best_1"):
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            
            # Save model
            model_path = "models/best_catboost_regressor.cbm"
            final_model.save_model(model_path)
            mlflow.log_artifact(model_path)
            
            # Log model files as artifacts (avoid deprecated log_model API)
            # The .cbm file is already logged above, so we skip the registry log
    except Exception as e:
        print(f"MLflow logging encountered an issue: {e}")
        print("Continuing with model save to local files...")
    
    # Save metadata
    metadata = {
        'model_type': 'CatBoostRegressor',
        'hyperparameters': best_params,
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'n_trials': n_trials
    }
    save_json(metadata, "models/best_regressor_info.json")
    
    return final_model, best_params, {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def main():
    """Main training function"""
    set_random_seeds(42)
    
    # Load data
    df = load_data()
    
    # Prepare features and targets
    X, y_class, y_reg = prepare_features_and_targets(df)
    
    # Load preprocessor
    preprocessor = load_preprocessor()
    
    # Train classifier
    print("Starting classifier training...")
    model_classifier, params_classifier, metrics_classifier = train_classifier(X, y_class, preprocessor, n_trials=50)
    
    # Train regressor
    print("\nStarting regressor training...")
    model_regressor, params_regressor, metrics_regressor = train_regressor(X, y_reg, preprocessor, n_trials=50)
    
    print_section("Training Complete")
    print("Models saved to models/")
    print("Models logged to MLflow")

if __name__ == "__main__":
    main()
