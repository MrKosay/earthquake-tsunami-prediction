import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            mean_absolute_error, mean_squared_error, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
from datetime import datetime

class ModelTrainer:
    """
    ModelTrainer class for building and training models for earthquake prediction,
    tsunami prediction, and economic impact estimation.
    """
    
    def __init__(self, models_dir='../models'):
        """
        Initialize the ModelTrainer.
        
        Args:
            models_dir (str): Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models for each stage
        self.earthquake_classifier = None
        self.magnitude_regressor = None
        self.depth_regressor = None
        self.tsunami_classifier = None
        self.economic_regressor = None
        
    def train_earthquake_classifier(self, X, y, use_xgboost=True, test_size=0.2, random_state=42):
        """
        Train a binary classifier to predict earthquake occurrence.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector (binary)
            use_xgboost (bool): Whether to use XGBoost instead of RandomForest
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing model and evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Check class imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"Class distribution in training set: {class_counts}")
        
        # Calculate class weight for handling imbalance
        n_samples = len(y_train)
        n_classes = len(unique)
        class_weight = {
            0: n_samples / (n_classes * counts[0]) if len(counts) > 0 else 1.0,
            1: n_samples / (n_classes * counts[1]) if len(counts) > 1 else 1.0
        }
        
        # Choose model with optimized parameters
        if use_xgboost:
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                scale_pos_weight=class_weight[1]/class_weight[0],  # For imbalanced classes
                gamma=1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=random_state
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                class_weight=class_weight,
                random_state=random_state
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Store the model
        self.earthquake_classifier = model
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'earthquake_classifier.joblib')
        joblib.dump(model, model_path)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(range(X.shape[1]), model.feature_importances_))
        }
    
    def train_magnitude_regressor(self, X, y, use_xgboost=True, test_size=0.2, random_state=42):
        """
        Train a regressor to predict earthquake magnitude.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector (continuous magnitude values)
            use_xgboost (bool): Whether to use XGBoost instead of RandomForest
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing model and evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Choose model with optimized parameters
        if use_xgboost:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=300,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=2,
                gamma=0.2,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=1,
                max_features=0.7,
                bootstrap=True,
                random_state=random_state
            )
        
        # Train model
        print("Training magnitude regressor with optimized parameters...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': model.score(X_test, y_test)
        }
        
        # Get feature importances
        feature_importances = model.feature_importances_
        print("Top magnitude regressor features:")
        for i, importance in enumerate(feature_importances):
            if importance > 0.03:  # Only show important features
                print(f"Feature {i}: {importance:.4f}")
        
        # Store the model
        self.magnitude_regressor = model
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'magnitude_regressor.joblib')
        joblib.dump(model, model_path)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(range(X.shape[1]), model.feature_importances_))
        }
    
    def train_depth_regressor(self, X, y, use_xgboost=True, test_size=0.2, random_state=42):
        """
        Train a regressor to predict earthquake depth.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector (continuous depth values in km)
            use_xgboost (bool): Whether to use XGBoost instead of RandomForest
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing model and evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Choose model
        if use_xgboost:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': model.score(X_test, y_test)
        }
        
        # Store the model
        self.depth_regressor = model
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'depth_regressor.joblib')
        joblib.dump(model, model_path)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(range(X.shape[1]), model.feature_importances_))
        }
    
    def train_tsunami_classifier(self, X, y, use_xgboost=True, test_size=0.2, random_state=42):
        """
        Train a binary classifier to predict tsunami occurrence.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector (binary)
            use_xgboost (bool): Whether to use XGBoost instead of RandomForest
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing model and evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Choose model
        if use_xgboost:
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Check if there's more than one class in the predicted values
        # or if the model supports predict_proba for multiple classes
        metrics = {}
        if len(np.unique(y_test)) > 1:
            try:
                # Try to get class 1 probability
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                else:
                    # Only one class in predictions
                    metrics['roc_auc'] = 0.5  # Default for random classifier
            except (IndexError, AttributeError):
                # Handle the case where there's only one class
                metrics['roc_auc'] = 0.5  # Default for random classifier
                
            metrics.update({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            })
        else:
            # Only one class in the test set
            metrics = {
                'accuracy': 1.0 if y_test[0] == y_pred[0] else 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.5  # Default for random classifier
            }
        
        # Store the model
        self.tsunami_classifier = model
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'tsunami_classifier.joblib')
        joblib.dump(model, model_path)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(range(X.shape[1]), model.feature_importances_))
        }
    
    def train_economic_regressor(self, X, y, use_xgboost=True, test_size=0.2, random_state=42):
        """
        Train a regressor to predict economic loss (in USD).
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector (continuous economic damage values)
            use_xgboost (bool): Whether to use XGBoost instead of RandomForest
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing model and evaluation metrics
        """
        # Check for zero or negative values
        valid_indices = y > 0
        if not all(valid_indices):
            print(f"Removing {np.sum(~valid_indices)} zero or negative values from economic data")
            X = X[valid_indices]
            y = y[valid_indices]
            
        if len(y) < 10:
            print("WARNING: Very small economic dataset. Results may be unreliable.")
        
        # Remove extreme outliers using IQR method
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (3.0 * iqr)  # Allow more upside variation
        
        outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        if not all(outlier_mask):
            print(f"Removing {np.sum(~outlier_mask)} outliers from economic data")
            X = X[outlier_mask]
            y = y[outlier_mask]
        
        # Economic data is often log-normally distributed, apply log transform
        # Add small constant to avoid log(0)
        epsilon = 1.0
        y_log = np.log1p(y)
        
        # Print summary statistics
        print(f"Economic damage statistics after preprocessing:")
        print(f"  Count: {len(y)}")
        print(f"  Min: ${np.min(y):,.2f}")
        print(f"  Max: ${np.max(y):,.2f}")
        print(f"  Mean: ${np.mean(y):,.2f}")
        print(f"  Median: ${np.median(y):,.2f}")
        
        # Split data
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X, y_log, test_size=test_size, random_state=random_state
        )
        
        # Keep original y for evaluation
        _, _, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Try several models and select the best
        models = []
        
        # GradientBoostingRegressor with different parameters
        gb_model1 = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features=0.7,
            random_state=random_state
        )
        models.append(('GradientBoost-1', gb_model1))
        
        # GradientBoostingRegressor with different parameters
        gb_model2 = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.6,
            random_state=random_state
        )
        models.append(('GradientBoost-2', gb_model2))
        
        # XGBoost Regressor (if available)
        if use_xgboost:
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.2,
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=random_state
            )
            models.append(('XGBoost', xgb_model))
        
        # Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features=0.6,
            bootstrap=True,
            random_state=random_state
        )
        models.append(('RandomForest', rf_model))
        
        # Linear Regression with regularization from sklearn
        from sklearn.linear_model import Ridge, Lasso
        ridge_model = Ridge(alpha=1.0, random_state=random_state)
        models.append(('Ridge', ridge_model))
        
        lasso_model = Lasso(alpha=0.1, random_state=random_state)
        models.append(('Lasso', lasso_model))
        
        # Train all models and find the best
        best_model = None
        best_name = None
        best_r2 = float('-inf')
        best_predictions = None
        
        print("\nTraining and evaluating multiple economic models:")
        for name, model in models:
            # Train model on log-transformed target
            model.fit(X_train, y_train_log)
            
            # Make predictions (in log space)
            y_pred_log = model.predict(X_test)
            
            # Transform back to original scale
            y_pred = np.expm1(y_pred_log)
            
            # Calculate R² on log-transformed data
            r2_log = model.score(X_test, y_test_log)
            
            # Calculate R² on original scale
            from sklearn.metrics import r2_score
            r2_orig = r2_score(y_test, y_pred)
            
            # Calculate RMSE on original scale
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{name}: R² (log)={r2_log:.4f}, R² (orig)={r2_orig:.4f}, RMSE=${rmse:,.2f}, MAE=${mae:,.2f}")
            
            # Keep track of best model (using R² on log scale)
            if r2_log > best_r2:
                best_r2 = r2_log
                best_model = model
                best_name = name
                best_predictions = y_pred
        
        print(f"\nBest model: {best_name} with R² (log)={best_r2:.4f}")
        
        # Use the best model
        model = best_model
        
        # Make predictions with the best model
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        # Evaluate
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_log': model.score(X_test, y_test_log),  # R² on log-transformed data
            'r2_original': r2_score(y_test, y_pred)     # R² on original scale
        }
        
        # Get feature importances if the model supports it
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            print("\nTop economic regressor features:")
            for i, importance in enumerate(feature_importances):
                if importance > 0.05:  # Only show important features
                    print(f"Feature {i}: {importance:.4f}")
        
        # Store the model
        self.economic_regressor = model
        self.log_transform = True  # Flag to indicate log transformation was used
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'economic_regressor.joblib')
        joblib.dump(model, model_path)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(range(X.shape[1]), model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
        }
    
    def predict_earthquake(self, X):
        """
        Predict earthquake likelihood and characteristics.
        
        Args:
            X (array-like): Feature matrix for prediction
            
        Returns:
            dict: Dictionary containing earthquake predictions
        """
        if self.earthquake_classifier is None:
            raise ValueError("Earthquake classifier not trained.")
        
        # Predict earthquake likelihood
        eq_proba = self.earthquake_classifier.predict_proba(X)[0, 1]
        
        # Only predict magnitude and depth if earthquake likely
        if eq_proba > 0.5 and self.magnitude_regressor is not None and self.depth_regressor is not None:
            # Predict magnitude
            magnitude = self.magnitude_regressor.predict(X)[0]
            
            # Create magnitude range probabilities
            magnitude_ranges = {
                '4-5': 0.0,
                '5-6': 0.0,
                '6-7': 0.0,
                '7+': 0.0
            }
            
            # Simple normal distribution around predicted magnitude
            import scipy.stats as stats
            std_dev = 0.5  # Set standard deviation
            
            magnitude_ranges['4-5'] = float(stats.norm.cdf(5, magnitude, std_dev) - stats.norm.cdf(4, magnitude, std_dev))
            magnitude_ranges['5-6'] = float(stats.norm.cdf(6, magnitude, std_dev) - stats.norm.cdf(5, magnitude, std_dev))
            magnitude_ranges['6-7'] = float(stats.norm.cdf(7, magnitude, std_dev) - stats.norm.cdf(6, magnitude, std_dev))
            magnitude_ranges['7+'] = float(1 - stats.norm.cdf(7, magnitude, std_dev))
            
            # Normalize to sum to 1
            total = sum(magnitude_ranges.values())
            magnitude_ranges = {k: v/total for k, v in magnitude_ranges.items()}
            
            # Predict depth
            depth = self.depth_regressor.predict(X)[0]
            
            # Create confidence intervals
            confidence_interval_mag = [float(max(0, magnitude - 0.75)), float(magnitude + 0.75)]
            confidence_interval_depth = [float(max(0, depth - 7.5)), float(depth + 7.5)]
            
            return {
                'earthquake_likelihood': float(eq_proba),
                'magnitude_prediction': {
                    'estimate': float(magnitude),
                    'range_probs': magnitude_ranges,
                    'confidence_interval': confidence_interval_mag
                },
                'depth_prediction_km': {
                    'estimate': float(depth),
                    'ci': confidence_interval_depth
                }
            }
        else:
            # Return only likelihood if no earthquake expected or models not trained
            return {
                'earthquake_likelihood': float(eq_proba)
            }
    
    def predict_tsunami(self, X, earthquake_prediction):
        """
        Predict tsunami likelihood based on earthquake prediction.
        
        Args:
            X (array-like): Feature matrix for prediction
            earthquake_prediction (dict): Dictionary containing earthquake predictions
            
        Returns:
            dict: Dictionary containing tsunami predictions
        """
        # Only predict tsunami if earthquake is likely
        if earthquake_prediction.get('earthquake_likelihood', 0) > 0.5:
            if self.tsunami_classifier is None:
                raise ValueError("Tsunami classifier not trained.")
                
            # Predict tsunami likelihood
            tsunami_proba = self.tsunami_classifier.predict_proba(X)[0, 1]
            
            return {
                'tsunami_likelihood': float(tsunami_proba)
            }
        else:
            # No tsunami prediction if no earthquake expected
            return {
                'tsunami_likelihood': 0.0
            }
    
    def predict_economic_loss(self, X, earthquake_prediction, tsunami_prediction):
        """
        Predict economic loss based on earthquake and tsunami predictions.
        
        Args:
            X (array-like): Feature matrix for prediction
            earthquake_prediction (dict): Dictionary containing earthquake predictions
            tsunami_prediction (dict): Dictionary containing tsunami predictions
            
        Returns:
            dict: Dictionary containing economic loss predictions
        """
        # Only predict economic loss if earthquake is significant or tsunami is likely
        eq_magnitude = earthquake_prediction.get('magnitude_prediction', {}).get('estimate', 0)
        if tsunami_prediction.get('tsunami_likelihood', 0) > 0.3 or eq_magnitude >= 6.0:
            if self.economic_regressor is None:
                raise ValueError("Economic regressor not trained.")
                
            # Predict economic loss
            if hasattr(self, 'log_transform') and self.log_transform:
                # Prediction in log space
                log_loss = self.economic_regressor.predict(X)[0]
                loss = np.expm1(log_loss)
                loss = max(0, loss)  # Ensure non-negative
            else:
                loss = self.economic_regressor.predict(X)[0]
                loss = max(0, loss)  # Ensure non-negative
            
            # Determine risk category and confidence level
            confidence = "medium"
            if loss < 100000:
                risk_category = 'minimal'
            elif loss < 1000000:
                risk_category = 'low'
            elif loss < 10000000:
                risk_category = 'moderate'
            elif loss < 100000000:
                risk_category = 'high'
                confidence = "medium-low"  # Higher uncertainty for large values
            else:
                risk_category = 'severe'
                confidence = "low"  # Highest uncertainty for extreme values
            
            # Calculate uncertainty range using log-normal distribution properties
            # For log-normal, the confidence interval is asymmetric
            lower_bound = loss * 0.3  # Simplified approach - 70% reduction for lower bound
            upper_bound = loss * 3.0  # Simplified approach - 3x for upper bound
            
            return {
                'economic_loss_usd': {
                    'estimate': float(loss),
                    'risk_category': risk_category,
                    'confidence': confidence,
                    'range': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound)
                    }
                }
            }
        else:
            # No significant economic loss expected
            return {
                'economic_loss_usd': {
                    'estimate': 0.0,
                    'risk_category': 'minimal',
                    'confidence': 'high',
                    'range': {
                        'lower': 0.0,
                        'upper': 10000.0
                    }
                }
            }
    
    def full_prediction(self, X):
        """
        Make a full prediction using all models in the pipeline.
        
        Args:
            X (array-like): Feature matrix for prediction
            
        Returns:
            dict: Dictionary containing all predictions
        """
        # Predict earthquake
        earthquake_prediction = self.predict_earthquake(X)
        
        # Predict tsunami only if earthquake likely
        tsunami_prediction = self.predict_tsunami(X, earthquake_prediction)
        
        # Predict economic loss only if tsunami likely
        economic_prediction = self.predict_economic_loss(X, earthquake_prediction, tsunami_prediction)
        
        # Combine all predictions
        full_prediction = {}
        full_prediction.update(earthquake_prediction)
        full_prediction.update(tsunami_prediction)
        full_prediction.update(economic_prediction)
        
        return full_prediction
    
    def load_models(self):
        """
        Load all trained models from files.
        
        Returns:
            bool: True if all models loaded successfully
        """
        try:
            # Load earthquake classifier
            earthquake_classifier_path = os.path.join(self.models_dir, 'earthquake_classifier.joblib')
            if os.path.exists(earthquake_classifier_path):
                self.earthquake_classifier = joblib.load(earthquake_classifier_path)
            
            # Load magnitude regressor
            magnitude_regressor_path = os.path.join(self.models_dir, 'magnitude_regressor.joblib')
            if os.path.exists(magnitude_regressor_path):
                self.magnitude_regressor = joblib.load(magnitude_regressor_path)
            
            # Load depth regressor
            depth_regressor_path = os.path.join(self.models_dir, 'depth_regressor.joblib')
            if os.path.exists(depth_regressor_path):
                self.depth_regressor = joblib.load(depth_regressor_path)
            
            # Load tsunami classifier
            tsunami_classifier_path = os.path.join(self.models_dir, 'tsunami_classifier.joblib')
            if os.path.exists(tsunami_classifier_path):
                self.tsunami_classifier = joblib.load(tsunami_classifier_path)
            
            # Load economic regressor
            economic_regressor_path = os.path.join(self.models_dir, 'economic_regressor.joblib')
            if os.path.exists(economic_regressor_path):
                self.economic_regressor = joblib.load(economic_regressor_path)
            
            return True
        
        except Exception as e:
            print(f"Error loading models: {e}")
            return False 