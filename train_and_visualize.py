#!/usr/bin/env python3
"""
Train the machine learning models and generate visualizations
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to Python path
sys.path.append(str(Path(__file__).resolve().parent))

# Import project modules
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer

def train_models():
    """Train all models using processed data"""
    print("Starting model training...")
    
    # Initialize data processor and model trainer
    data_processor = DataProcessor(data_dir=str(Path('data')))
    model_trainer = ModelTrainer(models_dir='models/trained')
    
    try:
        # Load and process data
        print("Loading processed data...")
        
        # Load the processed data files
        processed_dir = Path('data/processed')
        earthquake_data = pd.read_csv(processed_dir / 'earthquake_data_processed.csv')
        tsunami_data = pd.read_csv(processed_dir / 'tsunami_data_processed.csv')
        economic_data = pd.read_csv(processed_dir / 'economic_data_processed.csv')
        
        print(f"Loaded processed earthquake data shape: {earthquake_data.shape}")
        print(f"Loaded processed tsunami data shape: {tsunami_data.shape}")
        print(f"Loaded processed economic data shape: {economic_data.shape}")
        
        # Prepare earthquake features for training
        print("Preparing earthquake features for model training...")
        earthquake_features = ['latitude', 'longitude', 'depth', 'magnitude', 'year', 'month', 'day', 'coast_distance', 'days_since_last_eq']
        X_earthquake = earthquake_data[earthquake_features].values
        
        # Train earthquake classifier
        print("Training earthquake classifier...")
        y_earthquake = (earthquake_data['magnitude'] >= 6.0).astype(int).values
        earthquake_results = model_trainer.train_earthquake_classifier(X_earthquake, y_earthquake)
        print(f"Earthquake classifier metrics: {earthquake_results['metrics']}")
        
        # Train magnitude regressor
        print("Training magnitude regressor...")
        y_magnitude = earthquake_data['magnitude'].values
        magnitude_results = model_trainer.train_magnitude_regressor(X_earthquake, y_magnitude)
        print(f"Magnitude regressor metrics: {magnitude_results['metrics']}")
        
        # Train depth regressor
        print("Training depth regressor...")
        y_depth = earthquake_data['depth'].values
        depth_results = model_trainer.train_depth_regressor(X_earthquake, y_depth)
        print(f"Depth regressor metrics: {depth_results['metrics']}")
        
        # Prepare tsunami features
        print("Preparing tsunami features for model training...")
        # Use earthquake features + magnitude for tsunami prediction
        significant_eq = earthquake_data[earthquake_data['magnitude'] >= 6.0].copy()
        
        # Create label for tsunami occurrence (simplified for demo)
        # In production, this would link to actual tsunami data
        tsunami_regions = tsunami_data['region_1deg'].unique() if 'region_1deg' in tsunami_data.columns else []
        significant_eq['tsunami_occurred'] = significant_eq['region_1deg'].isin(tsunami_regions).astype(int)
        
        # Train tsunami classifier
        print("Training tsunami classifier...")
        X_tsunami = significant_eq[earthquake_features].values
        y_tsunami = significant_eq['tsunami_occurred'].values
        tsunami_results = model_trainer.train_tsunami_classifier(X_tsunami, y_tsunami)
        print(f"Tsunami classifier metrics: {tsunami_results['metrics']}")
        
        # Train economic regressor
        print("Training economic regressor...")
        # Simple economic model based on earthquake magnitude and tsunami occurrence
        tsunami_events = significant_eq[significant_eq['tsunami_occurred'] == 1].copy()
        if len(tsunami_events) > 0:
            # Create synthetic economic damage based on magnitude and location
            tsunami_events['damage_usd'] = 10000 * (10 ** tsunami_events['magnitude']) * \
                                        (1 + 0.5 * np.abs(tsunami_events['coast_distance'] - 5) / 5)
            
            X_economic = tsunami_events[earthquake_features].values
            y_economic = tsunami_events['damage_usd'].values
            
            economic_results = model_trainer.train_economic_regressor(X_economic, y_economic)
            print(f"Economic regressor metrics: {economic_results['metrics']}")
        else:
            print("Not enough tsunami events for economic model training")
        
        print("Model training completed successfully!")
        return model_trainer
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_visualizations():
    """Generate all visualizations"""
    print("Generating visualizations...")
    
    # Make sure the visualization output directory exists
    os.makedirs('visualizations/output', exist_ok=True)
    
    # Run the visualizations
    try:
        # Run model performance visualizations
        print("Running model performance visualizations...")
        os.system('python3 visualizations/model_performance.py')
        
        # Run economic analysis visualizations
        print("Running economic analysis visualizations...")
        os.system('python3 visualizations/economic_analysis.py')
        
        # Run earthquake EDA visualizations
        print("Running earthquake EDA visualizations...")
        os.system('python3 visualizations/earthquake_eda.py')
        
        # Run tsunami EDA visualizations
        print("Running tsunami EDA visualizations...")
        os.system('python3 visualizations/tsunami_eda.py')
        
        print("All visualizations generated successfully!")
        return True
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to train models and generate visualizations"""
    print("Starting model training and visualization process...")
    
    # Train the models
    model_trainer = train_models()
    if model_trainer is None:
        print("Model training failed. Aborting visualization generation.")
        return 1
    
    # Generate visualizations
    if not generate_visualizations():
        print("Visualization generation failed.")
        return 1
    
    print("Process completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 