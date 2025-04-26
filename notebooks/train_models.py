#!/usr/bin/env python3
"""
Model Training Script for Earthquake & Tsunami Risk Prediction Platform

This script trains the various models needed for the prediction pipeline:
1. Earthquake occurrence classifier
2. Magnitude regressor
3. Depth regressor
4. Tsunami occurrence classifier
5. Economic loss regressor

Usage:
    python train_models.py [--use-xgboost] [--test-size 0.2] [--random-state 42]
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train models for earthquake and tsunami prediction')
    parser.add_argument('--use-xgboost', action='store_true', default=True, help='Use XGBoost instead of RandomForest')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    logger.info("Starting model training with arguments: %s", args)
    
    try:
        # Initialize data processor
        data_processor = DataProcessor(data_dir=str(Path(__file__).parent.parent / "data"))
        
        # Load and process data
        logger.info("Loading data...")
        data_processor.load_data()
        
        logger.info("Cleaning earthquake data...")
        earthquake_df = data_processor.clean_earthquake_data()
        
        logger.info("Cleaning tsunami data...")
        tsunami_df = data_processor.clean_tsunami_data()
        
        logger.info("Cleaning economic data...")
        economic_df = data_processor.clean_economic_data()
        
        # Initialize model trainer
        model_trainer = ModelTrainer(models_dir=str(Path(__file__).parent.parent / "models"))
        
        # Prepare data for earthquake prediction
        logger.info("Preparing data for earthquake classifier...")
        
        # Debug: Check the columns in the earthquake_df
        logger.info("Earthquake DataFrame columns: %s", earthquake_df.columns.tolist())
        
        # Add more engineered features for better prediction
        # Create distance from key seismic zones (simplistic approach)
        earthquake_df['japan_trench_dist'] = np.sqrt((earthquake_df['latitude'] - 38.0)**2 + (earthquake_df['longitude'] - 142.0)**2)
        earthquake_df['cascadia_dist'] = np.sqrt((earthquake_df['latitude'] - 45.0)**2 + (earthquake_df['longitude'] - (-125.0))**2)
        earthquake_df['month_sin'] = np.sin(2 * np.pi * earthquake_df['month']/12)
        earthquake_df['month_cos'] = np.cos(2 * np.pi * earthquake_df['month']/12)
        
        # Add mag_type as a numerical feature (if there's a pattern in mag types)
        if 'magType' in earthquake_df.columns:
            earthquake_df['magType_code'] = earthquake_df['magType'].astype('category').cat.codes
        
        earthquake_features = [
            'latitude', 'longitude', 'year', 'month', 'day',
            'coast_distance', 'days_since_last_eq', 'japan_trench_dist', 
            'cascadia_dist', 'month_sin', 'month_cos'
        ]
        
        if 'magType' in earthquake_df.columns:
            earthquake_features.append('magType_code')
        
        if 'gap' in earthquake_df.columns and earthquake_df['gap'].isnull().sum() / len(earthquake_df) < 0.3:
            earthquake_df['gap'] = earthquake_df['gap'].fillna(earthquake_df['gap'].median())
            earthquake_features.append('gap')
            
        if 'dmin' in earthquake_df.columns and earthquake_df['dmin'].isnull().sum() / len(earthquake_df) < 0.3:
            earthquake_df['dmin'] = earthquake_df['dmin'].fillna(earthquake_df['dmin'].median())
            earthquake_features.append('dmin')
        
        # Debug: Check if all the features exist in the DataFrame
        for feature in earthquake_features:
            if feature not in earthquake_df.columns:
                logger.error("Feature '%s' not found in earthquake_df", feature)
        
        # Create a binary target for earthquake occurrence
        # For demonstration purposes, consider mag >= 6.0 as significant earthquake
        earthquake_df['significant_earthquake'] = (earthquake_df['mag'] >= 6.0).astype(int)
        
        X_eq, y_eq = data_processor.prepare_model_inputs(
            include_features=earthquake_features,
            target_col='significant_earthquake'
        )
        
        # Debug: Check if X_eq and y_eq are None
        logger.info("X_eq type: %s, y_eq type: %s", type(X_eq), type(y_eq))
        if X_eq is None or y_eq is None:
            logger.error("prepare_model_inputs returned None for X_eq or y_eq")
            # Try simpler approach - prepare features and target directly
            # This is a workaround for debugging
            logger.info("Trying alternative approach...")
            df_subset = earthquake_df[earthquake_features + ['significant_earthquake']]
            logger.info("Subset shape: %s", df_subset.shape)
            y_eq = df_subset['significant_earthquake'].values
            X_eq = df_subset.drop(columns=['significant_earthquake']).values
            logger.info("Manually prepared X_eq shape: %s, y_eq shape: %s", X_eq.shape, y_eq.shape)
        
        # Train earthquake classifier
        logger.info("Training earthquake classifier...")
        eq_results = model_trainer.train_earthquake_classifier(
            X_eq, y_eq,
            use_xgboost=args.use_xgboost,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info("Earthquake classifier metrics: %s", eq_results['metrics'])
        
        # Train magnitude regressor
        logger.info("Training magnitude regressor...")
        
        # Prepare data for magnitude regression manually
        df_subset = earthquake_df[earthquake_features + ['mag']]
        logger.info("Mag subset shape: %s", df_subset.shape)
        y_mag = df_subset['mag'].values
        X_mag = df_subset.drop(columns=['mag']).values
        logger.info("Manually prepared X_mag shape: %s, y_mag shape: %s", X_mag.shape, y_mag.shape)
        
        mag_results = model_trainer.train_magnitude_regressor(
            X_mag, y_mag,
            use_xgboost=args.use_xgboost,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info("Magnitude regressor metrics: %s", mag_results['metrics'])
        
        # Train depth regressor
        logger.info("Training depth regressor...")
        
        # Prepare data for depth regression manually
        df_subset = earthquake_df[earthquake_features + ['depth']]
        logger.info("Depth subset shape: %s", df_subset.shape)
        y_depth = df_subset['depth'].values
        X_depth = df_subset.drop(columns=['depth']).values
        logger.info("Manually prepared X_depth shape: %s, y_depth shape: %s", X_depth.shape, y_depth.shape)
        
        depth_results = model_trainer.train_depth_regressor(
            X_depth, y_depth,
            use_xgboost=args.use_xgboost,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info("Depth regressor metrics: %s", depth_results['metrics'])
        
        # Prepare data for tsunami prediction
        logger.info("Preparing data for tsunami classifier...")
        
        # Merge earthquake and tsunami data for tsunami prediction
        # For demonstration, we're assuming there's a common column or way to join these datasets
        # This would need to be adjusted based on the actual data structure
        combined_df = earthquake_df[earthquake_df['significant_earthquake'] == 1].copy()
        combined_df['tsunami_occurred'] = combined_df['region_1deg'].isin(tsunami_df['region_1deg']).astype(int)
        
        tsunami_features = [
            'latitude', 'longitude', 'mag', 'depth',
            'coast_distance', 'year', 'month', 'day'
        ]
        
        # Prepare tsunami prediction data manually
        df_subset = combined_df[tsunami_features + ['tsunami_occurred']]
        logger.info("Tsunami subset shape: %s", df_subset.shape)
        y_tsunami = df_subset['tsunami_occurred'].values
        X_tsunami = df_subset.drop(columns=['tsunami_occurred']).values
        logger.info("Manually prepared X_tsunami shape: %s, y_tsunami shape: %s", X_tsunami.shape, y_tsunami.shape)
        
        # Train tsunami classifier
        logger.info("Training tsunami classifier...")
        tsunami_results = model_trainer.train_tsunami_classifier(
            X_tsunami, y_tsunami,
            use_xgboost=False,  # Use RandomForest instead of XGBoost
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info("Tsunami classifier metrics: %s", tsunami_results['metrics'])
        
        # Prepare data for economic loss prediction
        logger.info("Preparing data for economic loss regressor...")
        
        # Create a combined dataset for economic impact prediction
        economic_features = [
            'latitude', 'longitude', 'mag', 'depth',
            'coast_distance', 'tsunami_occurred'
        ]
        
        # Create the economic dataset
        combined_econ_df = combined_df.copy()
        
        # Load real economic data instead of using random values
        logger.info("Loading real economic data from dataset1 and dataset2...")
        econ_df1 = pd.read_csv(str(Path(__file__).parent.parent / "data" / "dataset1.csv"))
        econ_df2 = pd.read_csv(str(Path(__file__).parent.parent / "data" / "dataset2.csv"))
        
        # Extract needed columns 
        econ_data = []
        
        # Process dataset1
        for _, row in econ_df1.iterrows():
            if pd.notnull(row.get('Estimated_Economic_Damage_USD', None)):
                econ_data.append({
                    'year': row.get('Year'),
                    'month': row.get('Month'),
                    'day': row.get('Day'),
                    'magnitude': row.get('Magnitude'),
                    'depth': row.get('Depth_km'),
                    'damage': row.get('Estimated_Economic_Damage_USD')
                })
                
        # Process dataset2
        for _, row in econ_df2.iterrows():
            if pd.notnull(row.get('Estimated_Economic_Damage_USD', None)):
                econ_data.append({
                    'year': row.get('Year'),
                    'month': row.get('Month'),
                    'day': row.get('Day'),
                    'magnitude': row.get('Magnitude'),
                    'depth': row.get('Depth_km'),
                    'damage': row.get('Estimated_Economic_Damage_USD')
                })
        
        # Create DataFrame from the collected data
        econ_df = pd.DataFrame(econ_data)
        logger.info(f"Combined economic dataset has {len(econ_df)} events")
        
        # Convert damage values to numeric, handling any string values
        econ_df['damage'] = pd.to_numeric(econ_df['damage'], errors='coerce')
        econ_df = econ_df.dropna(subset=['damage'])
        logger.info(f"Economic dataset after removing non-numeric values: {len(econ_df)} events")
        
        # Instead of random economic damage, find the closest economic damage from real data
        # based on event similarity (magnitude, depth, year)
        economic_damages = []
        
        for _, eq_row in combined_econ_df.iterrows():
            best_match = None
            best_score = float('inf')
            
            for _, econ_row in econ_df.iterrows():
                # Calculate similarity score based on magnitude and depth
                mag_diff = abs(eq_row['mag'] - econ_row['magnitude']) if pd.notnull(econ_row['magnitude']) else 10
                depth_diff = abs(eq_row['depth'] - econ_row['depth']) if pd.notnull(econ_row['depth']) else 100
                year_diff = abs(eq_row['year'] - econ_row['year']) if pd.notnull(econ_row['year']) else 1000
                
                # Weighted similarity score (magnitude is most important)
                similarity = mag_diff * 3 + depth_diff * 0.5 + year_diff * 0.1
                
                if similarity < best_score:
                    best_score = similarity
                    best_match = econ_row['damage']
            
            economic_damages.append(float(best_match) if best_match is not None else 0.0)
        
        combined_econ_df['economic_damage'] = economic_damages
        # Count events with significant economic damage 
        significant_events = sum(1 for d in economic_damages if d > 0)
        logger.info(f"Matched {significant_events} events with economic damage data")
        
        # Add more predictive features for economic impact
        # Add proximity to population centers (simplistic approach)
        # These coordinates represent major coastal cities
        cities = {
            'tokyo': (35.6762, 139.6503),
            'los_angeles': (34.0522, -118.2437),
            'sydney': (-33.8688, 151.2093),
            'shanghai': (31.2304, 121.4737),
            'jakarta': (-6.2088, 106.8456)
        }
        
        for city, (lat, lon) in cities.items():
            combined_econ_df[f'{city}_dist'] = np.sqrt(
                (combined_econ_df['latitude'] - lat)**2 + 
                (combined_econ_df['longitude'] - lon)**2
            )
            economic_features.append(f'{city}_dist')
        
        # Add squared magnitude (non-linear relationship with damage)
        combined_econ_df['mag_squared'] = combined_econ_df['mag']**2
        economic_features.append('mag_squared')
        
        # Add interaction term between magnitude and depth
        combined_econ_df['mag_depth_ratio'] = combined_econ_df['mag'] / (combined_econ_df['depth'] + 1)  # +1 to avoid division by zero
        economic_features.append('mag_depth_ratio')
        
        # Prepare economic prediction data manually
        df_subset = combined_econ_df[economic_features + ['economic_damage']]
        logger.info("Economic subset shape: %s", df_subset.shape)
        y_econ = df_subset['economic_damage'].values
        X_econ = df_subset.drop(columns=['economic_damage']).values
        logger.info("Manually prepared X_econ shape: %s, y_econ shape: %s", X_econ.shape, y_econ.shape)
        
        # Train economic loss regressor
        logger.info("Training economic loss regressor...")
        econ_results = model_trainer.train_economic_regressor(
            X_econ, y_econ,
            use_xgboost=False,  # Use RandomForest instead of XGBoost
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info("Economic loss regressor metrics: %s", econ_results['metrics'])
        
        logger.info("All models trained successfully.")
        
        # Save the preprocessor
        logger.info("Saving preprocessor...")
        preprocessor = data_processor.get_preprocessor()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error("Error during training: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 