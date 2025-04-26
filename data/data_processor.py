import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    """
    DataProcessor class for cleaning and preparing earthquake and tsunami datasets
    for model training and prediction.
    """
    
    def __init__(self, data_dir='../data'):
        """
        Initialize the DataProcessor.
        
        Args:
            data_dir (str): Directory containing the data files
        """
        self.data_dir = data_dir
        self.earthquake_data = None
        self.tsunami_data = None
        self.economic_data = None
        self.earthquake_scaler = StandardScaler()
        self.tsunami_scaler = StandardScaler()
        self.economic_scaler = StandardScaler()
    
    def load_data(self):
        """
        Load all datasets from the data directory.
        
        Returns:
            tuple: Processed datasets (earthquake_df, tsunami_df, economic_df)
        """
        # Load earthquake data
        earthquake_file = os.path.join(self.data_dir, 'earthquake_data.csv')
        self.earthquake_data = pd.read_csv(earthquake_file)
        
        # Load tsunami data
        tsunami_file = os.path.join(self.data_dir, 'tsunami_features.csv')
        self.tsunami_data = pd.read_csv(tsunami_file)
        
        # Load economic data
        economic_file1 = os.path.join(self.data_dir, 'dataset1.csv')
        economic_file2 = os.path.join(self.data_dir, 'dataset2.csv')
        
        economic_data1 = pd.read_csv(economic_file1)
        economic_data2 = pd.read_csv(economic_file2)
        
        # Combine economic datasets
        self.economic_data = self._combine_economic_data(economic_data1, economic_data2)
        
        return self.earthquake_data, self.tsunami_data, self.economic_data
    
    def _combine_economic_data(self, df1, df2):
        """
        Combine economic datasets with appropriate handling of common columns.
        
        Args:
            df1 (DataFrame): First economic dataset
            df2 (DataFrame): Second economic dataset
            
        Returns:
            DataFrame: Combined economic dataset
        """
        # Identify common columns
        common_cols = set(df1.columns) & set(df2.columns)
        
        # Merge datasets
        combined_df = pd.concat([df1, df2], axis=0, ignore_index=True)
        
        return combined_df
    
    def clean_earthquake_data(self):
        """
        Clean and preprocess earthquake data.
        
        Returns:
            DataFrame: Cleaned earthquake data
        """
        if self.earthquake_data is None:
            raise ValueError("Earthquake data not loaded. Call load_data() first.")
        
        df = self.earthquake_data.copy()
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Convert date columns to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['year'] = df['time'].dt.year
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day
        
        # Handle missing values in key columns
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_features:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Calculate additional features
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distance from nearest coast (simplified)
            df['coast_distance'] = np.abs(df['longitude']) % 10
            
            # Calculate seismic region
            df['region_1deg'] = (df['latitude'].round().astype(int).astype(str) + '_' + 
                                df['longitude'].round().astype(int).astype(str))
        
        # Calculate seismic gap if date columns exist
        if 'time' in df.columns:
            df = df.sort_values('time')
            df['prev_eq_time'] = df.groupby('region_1deg')['time'].shift(1)
            df['days_since_last_eq'] = (df['time'] - df['prev_eq_time']).dt.total_seconds() / (24 * 3600)
            df['days_since_last_eq'] = df['days_since_last_eq'].fillna(df['days_since_last_eq'].median())
        
        self.processed_earthquake_data = df
        return df
    
    def clean_tsunami_data(self):
        """
        Clean and preprocess tsunami data.
        
        Returns:
            DataFrame: Cleaned tsunami data
        """
        if self.tsunami_data is None:
            raise ValueError("Tsunami data not loaded. Call load_data() first.")
        
        df = self.tsunami_data.copy()
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Convert event_time to datetime
        if 'event_time' in df.columns:
            df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
        
        # Handle missing values
        for col in ['tsunami_magnitude_iida', 'tsunami_intensity', 'maximum_water_height_m', 'damage_usd']:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Encode categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col not in ['event_time', 'location_name', 'country']:
                df[col] = df[col].astype('category').cat.codes
        
        self.processed_tsunami_data = df
        return df
    
    def clean_economic_data(self):
        """
        Clean and preprocess economic data.
        
        Returns:
            DataFrame: Cleaned economic data
        """
        if self.economic_data is None:
            raise ValueError("Economic data not loaded. Call load_data() first.")
        
        df = self.economic_data.copy()
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Create date column if year, month, day exist
        if all(col in df.columns for col in ['Year', 'Month', 'Day']):
            df['Date'] = pd.to_datetime(
                dict(year=df['Year'], month=df['Month'], day=df['Day']),
                errors='coerce'
            )
        
        # Handle missing values in economic damage
        if 'Estimated_Economic_Damage_USD' in df.columns:
            # Convert to numeric, errors will be converted to NaN
            df['Estimated_Economic_Damage_USD'] = pd.to_numeric(df['Estimated_Economic_Damage_USD'], errors='coerce')
            df['Estimated_Economic_Damage_USD'] = df['Estimated_Economic_Damage_USD'].fillna(0)
        
        # Create risk categories based on damage amount
        if 'Estimated_Economic_Damage_USD' in df.columns:
            damage = df['Estimated_Economic_Damage_USD']
            conditions = [
                (damage < 500000),
                (damage >= 500000) & (damage < 2000000),
                (damage >= 2000000)
            ]
            values = ['low', 'medium', 'high']
            df['Risk_Category'] = np.select(conditions, values, default='unknown')
        
        self.processed_economic_data = df
        return df
    
    def prepare_model_inputs(self, include_features=None, target_col=None):
        """
        Prepare features and target for model training.
        
        Args:
            include_features (list): List of feature column names to include
            target_col (str): Name of the target column
            
        Returns:
            tuple: X (features) and y (target) for model training
        """
        if self.processed_earthquake_data is None:
            raise ValueError("Process data first using clean_* methods")
        
        # Default behavior uses processed earthquake data
        df = self.processed_earthquake_data.copy()
        
        # Filter to include only specified features
        if include_features:
            available_cols = [col for col in include_features if col in df.columns]
            df = df[available_cols]
        
        # Handle target column
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col]
            df = df.drop(columns=[target_col])
        
        # Handle missing values
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Fit and transform the data
        X = preprocessor.fit_transform(df)
        
        # Store the preprocessor for future use
        self.preprocessor = preprocessor
        
        return X, y
    
    def prepare_location_time_features(self, latitude, longitude, date_time):
        """
        Prepare features for a single prediction based on location and time.
        
        Args:
            latitude (float): Latitude of the location
            longitude (float): Longitude of the location
            date_time (datetime): Date and time for prediction
            
        Returns:
            DataFrame: Features ready for model prediction
        """
        # Create a dataframe with the input values - adding all possible features the model might need
        data = {
            # Basic location and time features
            'latitude': [latitude],
            'longitude': [longitude],
            'time': [date_time],
            'year': [date_time.year],
            'month': [date_time.month],
            'day': [date_time.day],
            'hour': [date_time.hour],
            
            # Earthquake features
            'depth': [10.0],  # Default depth in km
            'magnitude': [5.0],  # Default magnitude
            'days_since_last_eq': [365],  # Default days since last earthquake
            
            # Geographical features
            'coast_distance': [np.abs(longitude) % 10],
            'region_1deg': [f"{int(round(latitude))}_{int(round(longitude))}"],
            
            # Additional features that might be needed
            'plate_boundary_distance': [50.0],  # Default distance to plate boundary in km
            'elevation': [0.0],  # Default elevation in meters
            'population_density': [100.0],  # Default population density
            'building_density': [50.0],  # Default building density
            'soil_type': [1],  # Default soil type (categorical)
            'has_historical_tsunamis': [0],  # Default (binary)
            'avg_historical_magnitude': [5.0]  # Default average historical magnitude
        }
        
        df = pd.DataFrame(data)
        return df
    
    def get_preprocessor(self):
        """
        Get the fitted column transformer for preprocessing new data.
        
        Returns:
            ColumnTransformer: Fitted preprocessor
        """
        if not hasattr(self, 'preprocessor'):
            raise ValueError("Preprocessor not created. Call prepare_model_inputs() first.")
        
        return self.preprocessor 