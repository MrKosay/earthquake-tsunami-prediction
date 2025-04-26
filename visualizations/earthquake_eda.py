#!/usr/bin/env python3
"""
Earthquake Data Exploration and Visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data.data_processor import DataProcessor

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Create a custom colormap for earthquake visualizations
colors = ["#F2E6C2", "#F2B999", "#E88973", "#D56355", "#B63A3A", "#802929"]
earthquake_cmap = LinearSegmentedColormap.from_list("earthquake_cmap", colors)

def millions_formatter(x, pos):
    """Format y-axis values in millions"""
    return f'{x/1000000:.1f}M'

def billions_formatter(x, pos):
    """Format y-axis values in billions"""
    return f'{x/1000000000:.1f}B'

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = Path("visualizations/earthquake_eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_and_preprocess_data():
    """Load and preprocess earthquake data"""
    data_processor = DataProcessor(data_dir=str(Path(__file__).parent.parent / "data"))
    data_processor.load_data()
    earthquake_df = data_processor.clean_earthquake_data()
    
    print(f"Loaded earthquake data with {len(earthquake_df)} records")
    return earthquake_df

def plot_magnitude_distribution(df, output_dir):
    """Plot earthquake magnitude distribution"""
    plt.figure(figsize=(12, 8))
    
    # Create histogram of magnitudes
    plt.hist(df['mag'], bins=30, alpha=0.7, color="#B63A3A")
    
    plt.title('Earthquake Magnitude Distribution', fontsize=18)
    plt.xlabel('Magnitude', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for significant magnitude thresholds
    plt.axvline(x=6.0, color='darkred', linestyle='--', 
                label='Major Earthquake (≥6.0)', linewidth=2)
    plt.axvline(x=7.0, color='black', linestyle='--', 
                label='Strong Earthquake (≥7.0)', linewidth=2)
    plt.axvline(x=8.0, color='purple', linestyle='--', 
                label='Great Earthquake (≥8.0)', linewidth=2)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "magnitude_distribution.png")
    plt.close()

def plot_depth_vs_magnitude(df, output_dir):
    """Plot earthquake depth vs magnitude scatter plot"""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(df['mag'], df['depth'], 
                         c=df['mag'], cmap=earthquake_cmap,
                         alpha=0.6, s=50)
    
    plt.title('Earthquake Depth vs. Magnitude', fontsize=18)
    plt.xlabel('Magnitude', fontsize=14)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Magnitude')
    
    # Invert y-axis so deeper earthquakes are lower on the plot
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "depth_vs_magnitude.png")
    plt.close()

def plot_global_earthquake_map(df, output_dir):
    """Plot global earthquake distribution map"""
    plt.figure(figsize=(16, 8))
    
    # Create world map background
    world_map = plt.imread(str(Path(__file__).parent.parent / "visualizations" / "world_map.png"))
    plt.imshow(world_map, extent=[-180, 180, -90, 90])
    
    # Sample the data if it's too large
    if len(df) > 10000:
        plot_df = df.sample(10000, random_state=42)
    else:
        plot_df = df
    
    # Plot earthquakes as scatter points
    scatter = plt.scatter(plot_df['longitude'], plot_df['latitude'],
                         c=plot_df['mag'], cmap=earthquake_cmap,
                         alpha=0.7, s=plot_df['mag']**2, 
                         edgecolors='k', linewidths=0.5)
    
    plt.title('Global Earthquake Distribution', fontsize=18)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Magnitude')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "global_earthquake_map.png")
    plt.close()

def plot_time_series(df, output_dir):
    """Plot earthquake time series"""
    # Create time-based aggregations
    df['year_month'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    monthly_counts = df.groupby('year_month').size()
    
    plt.figure(figsize=(16, 6))
    monthly_counts.plot(color="#B63A3A", linewidth=2)
    
    plt.title('Earthquake Frequency Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Earthquakes', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "earthquake_time_series.png")
    plt.close()

def plot_magnitude_by_region(df, output_dir):
    """Plot earthquake magnitude by region boxplot"""
    # Create regions based on latitude/longitude
    region_bins = {
        'Pacific Ring of Fire': ((df['longitude'] > 120) | (df['longitude'] < -120)) & 
                                ((df['latitude'] > -60) & (df['latitude'] < 60)),
        'East Asia': (df['longitude'] > 90) & (df['longitude'] < 150) & 
                     (df['latitude'] > 0) & (df['latitude'] < 60),
        'South Asia': (df['longitude'] > 60) & (df['longitude'] < 110) & 
                      (df['latitude'] > -10) & (df['latitude'] < 40),
        'Mediterranean': (df['longitude'] > -10) & (df['longitude'] < 40) & 
                         (df['latitude'] > 30) & (df['latitude'] < 45),
        'Americas': (df['longitude'] > -120) & (df['longitude'] < -60) &
                    (df['latitude'] > -60) & (df['latitude'] < 60)
    }
    
    # Assign regions
    df['region'] = 'Other'
    for region, mask in region_bins.items():
        df.loc[mask, 'region'] = region
    
    # Create boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='region', y='mag', data=df, palette=sns.color_palette(colors))
    
    plt.title('Earthquake Magnitude by Region', fontsize=18)
    plt.xlabel('Region', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "magnitude_by_region.png")
    plt.close()

def main():
    """Main function to generate earthquake data visualizations"""
    output_dir = create_output_dir()
    
    # Load the data
    earthquake_df = load_and_preprocess_data()
    
    # Create visualizations
    plot_magnitude_distribution(earthquake_df, output_dir)
    plot_depth_vs_magnitude(earthquake_df, output_dir)
    
    # Check if we have a world map file for background
    world_map_path = Path(__file__).parent / "world_map.png"
    if world_map_path.exists():
        plot_global_earthquake_map(earthquake_df, output_dir)
    else:
        print("World map background file not found. Skipping global earthquake map.")
    
    plot_time_series(earthquake_df, output_dir)
    plot_magnitude_by_region(earthquake_df, output_dir)
    
    print(f"Earthquake data visualizations saved to {output_dir.absolute()}")

if __name__ == "__main__":
    main() 