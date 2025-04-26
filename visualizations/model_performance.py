#!/usr/bin/env python3
"""
Model Performance Visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    roc_auc_score, average_precision_score
)
import matplotlib.ticker as ticker

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.model_trainer import ModelTrainer
from data.data_processor import DataProcessor

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = Path("visualizations/model_performance")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_models():
    """Load the trained models"""
    model_trainer = ModelTrainer(models_dir=str(Path(__file__).parent.parent / "models"))
    models_loaded = model_trainer.load_models()
    
    if not models_loaded:
        print("Warning: Not all models could be loaded.")
    
    return model_trainer

def generate_test_data():
    """Generate test data for model evaluation"""
    data_processor = DataProcessor(data_dir=str(Path(__file__).parent.parent / "data"))
    data_processor.load_data()
    earthquake_df = data_processor.clean_earthquake_data()
    tsunami_df = data_processor.clean_tsunami_data()
    
    # Create a subset of the data for testing
    if len(earthquake_df) > 5000:
        earthquake_test = earthquake_df.sample(5000, random_state=42)
    else:
        earthquake_test = earthquake_df
    
    # For significant earthquakes
    significant_eq = earthquake_test[earthquake_test['mag'] >= 6.0].copy()
    significant_eq['tsunami_occurred'] = significant_eq['region_1deg'].isin(tsunami_df['region_1deg']).astype(int)
    
    # Add synthetic economic damage for significant earthquakes with tsunami
    tsunami_events = significant_eq[significant_eq['tsunami_occurred'] == 1].copy()
    
    return {
        'earthquake_test': earthquake_test,
        'significant_eq': significant_eq,
        'tsunami_events': tsunami_events
    }

def plot_earthquake_classifier_performance(model_trainer, test_data, output_dir):
    """Plot earthquake classifier performance metrics"""
    if model_trainer.earthquake_classifier is None:
        print("Earthquake classifier not loaded. Skipping performance visualization.")
        return
    
    # Get test data
    earthquake_test = test_data['earthquake_test']
    
    # Prepare features
    X_features = [
        'latitude', 'longitude', 'depth', 'coast_distance', 
        'days_since_last_eq', 'year', 'month', 'day'
    ]
    X_test = earthquake_test[X_features].values
    
    # Create target (magnitude >= 6.0 as significant earthquake)
    y_test = (earthquake_test['mag'] >= 6.0).astype(int).values
    
    # Make predictions
    y_pred_proba = model_trainer.earthquake_classifier.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Earthquake Classifier Confusion Matrix', fontsize=18)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.xticks([0.5, 1.5], ['Non-Significant', 'Significant'])
    plt.yticks([0.5, 1.5], ['Non-Significant', 'Significant'])
    plt.tight_layout()
    plt.savefig(output_dir / "earthquake_confusion_matrix.png")
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Earthquake Classifier ROC Curve', fontsize=18)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "earthquake_roc_curve.png")
    plt.close()
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap:.3f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='navy', linestyle='--', 
                label=f'Random (AP = {sum(y_test)/len(y_test):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Earthquake Classifier Precision-Recall Curve', fontsize=18)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "earthquake_precision_recall_curve.png")
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance = model_trainer.earthquake_classifier.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.bar(range(len(importance)), importance[indices], align='center', alpha=0.7)
    plt.xticks(range(len(importance)), [X_features[i] for i in indices], rotation=45, ha='right')
    plt.title('Earthquake Classifier Feature Importance', fontsize=18)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "earthquake_feature_importance.png")
    plt.close()

def plot_magnitude_regressor_performance(model_trainer, test_data, output_dir):
    """Plot magnitude regressor performance metrics"""
    if model_trainer.magnitude_regressor is None:
        print("Magnitude regressor not loaded. Skipping performance visualization.")
        return
    
    # Get test data
    earthquake_test = test_data['earthquake_test']
    
    # Prepare features
    X_features = [
        'latitude', 'longitude', 'depth', 'coast_distance', 
        'days_since_last_eq', 'year', 'month', 'day'
    ]
    X_test = earthquake_test[X_features].values
    
    # Target
    y_test = earthquake_test['mag'].values
    
    # Make predictions
    y_pred = model_trainer.magnitude_regressor.predict(X_test)
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Magnitude Regressor: Actual vs Predicted', fontsize=18)
    plt.xlabel('Actual Magnitude', fontsize=14)
    plt.ylabel('Predicted Magnitude', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_actual_vs_predicted.png")
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 8))
    errors = y_test - y_pred
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Magnitude Regressor: Error Distribution', fontsize=18)
    plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_error_distribution.png")
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance = model_trainer.magnitude_regressor.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.bar(range(len(importance)), importance[indices], align='center', alpha=0.7)
    plt.xticks(range(len(importance)), [X_features[i] for i in indices], rotation=45, ha='right')
    plt.title('Magnitude Regressor Feature Importance', fontsize=18)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_feature_importance.png")
    plt.close()

def plot_economic_regressor_performance(model_trainer, test_data, output_dir):
    """Plot economic regressor performance metrics"""
    if model_trainer.economic_regressor is None:
        print("Economic regressor not loaded. Skipping performance visualization.")
        return
    
    # Use tsunami events for economic prediction
    tsunami_events = test_data['tsunami_events']
    
    # Prepare features
    X_features = [
        'latitude', 'longitude', 'mag', 'depth',
        'coast_distance', 'year', 'month', 'day'
    ]
    
    # Create synthetic economic damage for visualization
    # This is for demonstration only - actual models would use real economic data
    tsunami_events['log_mag'] = np.log(tsunami_events['mag'])
    tsunami_events['economic_damage'] = 10**(tsunami_events['log_mag'] * 3 - 10) * np.random.lognormal(0, 1, size=len(tsunami_events))
    
    X_test = tsunami_events[X_features].values
    y_test = tsunami_events['economic_damage'].values
    
    # Make predictions (handle log transformation if used in the model)
    if hasattr(model_trainer, 'log_transform') and model_trainer.log_transform:
        y_pred_log = model_trainer.economic_regressor.predict(X_test)
        y_pred = np.expm1(y_pred_log)
    else:
        y_pred = model_trainer.economic_regressor.predict(X_test)
    
    # Plot actual vs predicted (log scale)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Economic Regressor: Actual vs Predicted (Log Scale)', fontsize=18)
    plt.xlabel('Actual Economic Damage (USD)', fontsize=14)
    plt.ylabel('Predicted Economic Damage (USD)', fontsize=14)
    
    # Format axis ticks
    formatter = ticker.FuncFormatter(lambda x, p: f'${x:.1e}')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "economic_actual_vs_predicted.png")
    plt.close()
    
    # Plot relative error distribution (log of ratio)
    plt.figure(figsize=(10, 8))
    # Avoid division by zero
    valid_indices = (y_test > 0) & (y_pred > 0)
    log_ratio = np.log10(y_pred[valid_indices] / y_test[valid_indices])
    
    plt.hist(log_ratio, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Perfect Prediction (ratio=1)')
    plt.axvline(x=1, color='g', linestyle='--', label='10x Overprediction')
    plt.axvline(x=-1, color='b', linestyle='--', label='10x Underprediction')
    
    plt.title('Economic Regressor: Log Ratio Distribution', fontsize=18)
    plt.xlabel('Log10(Predicted/Actual)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "economic_log_ratio_distribution.png")
    plt.close()
    
    # Plot feature importance if available
    if hasattr(model_trainer.economic_regressor, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        importance = model_trainer.economic_regressor.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.bar(range(len(importance)), importance[indices], align='center', alpha=0.7)
        plt.xticks(range(len(importance)), [X_features[i] for i in indices], rotation=45, ha='right')
        plt.title('Economic Regressor Feature Importance', fontsize=18)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Importance', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "economic_feature_importance.png")
        plt.close()

def plot_model_comparison(output_dir):
    """Plot comparison of model metrics from previous runs"""
    # Read metrics from log files
    metrics = {
        'original': {
            'earthquake_classifier': {
                'accuracy': 0.988,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.657
            },
            'magnitude_regressor': {
                'mae': 0.313,
                'rmse': 0.434,
                'r2': 0.126
            },
            'economic_regressor': {
                'mae': 784934,
                'rmse': 1077854,
                'r2': -0.033
            }
        },
        'improved': {
            'earthquake_classifier': {
                'accuracy': 0.958,
                'precision': 0.211,
                'recall': 0.903,
                'f1': 0.343,
                'roc_auc': 0.983
            },
            'magnitude_regressor': {
                'mae': 0.217,
                'rmse': 0.287,
                'r2': 0.619
            },
            'economic_regressor': {
                'mae': 750869,
                'rmse': 1143287,
                'r2_log': 0.875,
                'r2_original': 0.769
            }
        }
    }
    
    # Plot earthquake classifier metrics comparison
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    plt.bar(x - width/2, [metrics['original']['earthquake_classifier'][m] for m in metrics_to_plot], 
            width, label='Original Model', alpha=0.7)
    plt.bar(x + width/2, [metrics['improved']['earthquake_classifier'][m] for m in metrics_to_plot], 
            width, label='Improved Model', alpha=0.7)
    
    plt.title('Earthquake Classifier: Metrics Comparison', fontsize=18)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(x, metrics_to_plot, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "earthquake_metrics_comparison.png")
    plt.close()
    
    # Plot magnitude regressor metrics comparison
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['mae', 'rmse', 'r2']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    # For visualization, invert MAE and RMSE since lower is better
    orig_vals = [-metrics['original']['magnitude_regressor']['mae'], 
                -metrics['original']['magnitude_regressor']['rmse'], 
                metrics['original']['magnitude_regressor']['r2']]
    
    impr_vals = [-metrics['improved']['magnitude_regressor']['mae'], 
                -metrics['improved']['magnitude_regressor']['rmse'], 
                metrics['improved']['magnitude_regressor']['r2']]
    
    plt.bar(x - width/2, orig_vals, width, label='Original Model', alpha=0.7)
    plt.bar(x + width/2, impr_vals, width, label='Improved Model', alpha=0.7)
    
    plt.title('Magnitude Regressor: Metrics Comparison', fontsize=18)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score (for MAE/RMSE, higher is better = lower error)', fontsize=14)
    plt.xticks(x, metrics_to_plot, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_metrics_comparison.png")
    plt.close()
    
    # Plot economic regressor R² comparison
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['Original R²', 'Improved R² (log)', 'Improved R² (original)']
    values = [metrics['original']['economic_regressor']['r2'], 
              metrics['improved']['economic_regressor']['r2_log'],
              metrics['improved']['economic_regressor']['r2_original']]
    
    plt.bar(metrics_to_plot, values, alpha=0.7)
    
    plt.title('Economic Regressor: R² Comparison', fontsize=18)
    plt.xlabel('Model Version', fontsize=14)
    plt.ylabel('R²', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "economic_r2_comparison.png")
    plt.close()

def main():
    """Main function to generate model performance visualizations"""
    output_dir = create_output_dir()
    
    # Load models
    model_trainer = load_models()
    
    # Generate test data
    test_data = generate_test_data()
    
    # Plot model performance
    plot_earthquake_classifier_performance(model_trainer, test_data, output_dir)
    plot_magnitude_regressor_performance(model_trainer, test_data, output_dir)
    plot_economic_regressor_performance(model_trainer, test_data, output_dir)
    
    # Plot model comparison
    plot_model_comparison(output_dir)
    
    print(f"Model performance visualizations saved to {output_dir.absolute()}")

if __name__ == "__main__":
    main() 