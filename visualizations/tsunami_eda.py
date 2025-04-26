#!/usr/bin/env python3
"""
Tsunami Data Exploration and Visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data.data_processor import DataProcessor

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Create a custom colormap for tsunami visualizations
colors = ["#CDE7F0", "#90C4DE", "#5AA3CF", "#2A80B9", "#1A5F8C", "#0A3A5C"]
tsunami_cmap = LinearSegmentedColormap.from_list("tsunami_cmap", colors)

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = Path("visualizations/tsunami_eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_and_preprocess_data():
    """Load and preprocess tsunami data"""
    data_processor = DataProcessor(data_dir=str(Path(__file__).parent.parent / "data"))
    data_processor.load_data()
    tsunami_df = data_processor.clean_tsunami_data()
    
    print(f"Loaded tsunami data with {len(tsunami_df)} records")
    return tsunami_df

def plot_tsunami_magnitude_distribution(df, output_dir):
    """Plot tsunami magnitude distribution"""
    plt.figure(figsize=(12, 8))
    
    if 'tsunami_magnitude_iida' in df.columns:
        plt.hist(df['tsunami_magnitude_iida'].dropna(), bins=20, alpha=0.8, color="#1A5F8C")
        
        plt.title('Tsunami Magnitude Distribution (Iida Scale)', fontsize=18)
        plt.xlabel('Tsunami Magnitude', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for significant tsunami thresholds
        plt.axvline(x=3, color='darkblue', linestyle='--', 
                    label='Destructive Tsunami (≥3)', linewidth=2)
        plt.axvline(x=4, color='black', linestyle='--', 
                    label='Very Destructive Tsunami (≥4)', linewidth=2)
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_dir / "tsunami_magnitude_distribution.png")
    else:
        print("Warning: 'tsunami_magnitude_iida' column not found in tsunami data.")
    
    plt.close()

def plot_water_height_distribution(df, output_dir):
    """Plot tsunami water height distribution"""
    plt.figure(figsize=(12, 8))
    
    if 'maximum_water_height_m' in df.columns:
        water_heights = df['maximum_water_height_m'].dropna()
        
        # Remove extreme outliers for better visualization
        q95 = water_heights.quantile(0.95)
        water_heights = water_heights[water_heights <= q95]
        
        plt.hist(water_heights, bins=25, alpha=0.8, color="#2A80B9")
        
        plt.title('Tsunami Maximum Water Height Distribution', fontsize=18)
        plt.xlabel('Maximum Water Height (m)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for significant water height thresholds
        plt.axvline(x=3, color='darkblue', linestyle='--', 
                    label='Potentially Damaging (≥3m)', linewidth=2)
        plt.axvline(x=6, color='black', linestyle='--', 
                    label='Destructive (≥6m)', linewidth=2)
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_dir / "water_height_distribution.png")
    else:
        print("Warning: 'maximum_water_height_m' column not found in tsunami data.")
    
    plt.close()

def plot_tsunami_locations(df, output_dir):
    """Plot tsunami locations on world map"""
    plt.figure(figsize=(16, 8))
    
    # Check if latitude and longitude columns exist
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Create world map background
        world_map = plt.imread(str(Path(__file__).parent / "world_map.png"))
        plt.imshow(world_map, extent=[-180, 180, -90, 90])
        
        # Calculate marker size based on water height if available
        if 'maximum_water_height_m' in df.columns:
            size_var = df['maximum_water_height_m'].fillna(1) * 10
            size_var = size_var.clip(20, 200)  # Limit marker size range
        else:
            size_var = 50
        
        # Plot tsunami locations
        scatter = plt.scatter(df['longitude'], df['latitude'],
                         c=size_var, cmap=tsunami_cmap,
                         alpha=0.7, s=size_var, 
                         edgecolors='k', linewidths=0.5)
        
        plt.title('Global Tsunami Locations', fontsize=18)
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if 'maximum_water_height_m' in df.columns:
            plt.colorbar(scatter, label='Maximum Water Height (m)')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_dir / "tsunami_locations.png")
    else:
        print("Warning: Latitude or longitude columns not found in tsunami data.")
    
    plt.close()

def plot_tsunami_damage(df, output_dir):
    """Plot tsunami economic damage distribution"""
    plt.figure(figsize=(12, 8))
    
    if 'damage_usd' in df.columns:
        damage = df['damage_usd'].dropna()
        
        # Use log scale for better visualization
        damage_log = np.log10(damage[damage > 0])
        
        plt.hist(damage_log, bins=20, alpha=0.8, color="#5AA3CF")
        
        plt.title('Tsunami Economic Damage Distribution (Log Scale)', fontsize=18)
        plt.xlabel('Economic Damage (log10 USD)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels to original values
        tick_locations = np.arange(np.floor(damage_log.min()), np.ceil(damage_log.max()) + 1)
        tick_labels = [f"$10^{int(loc)}$" for loc in tick_locations]
        plt.xticks(tick_locations, tick_labels)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_dir / "tsunami_damage_distribution.png")
    else:
        print("Warning: 'damage_usd' column not found in tsunami data.")
    
    plt.close()

def plot_tsunami_frequency_over_time(df, output_dir):
    """Plot tsunami frequency over time"""
    plt.figure(figsize=(16, 8))
    
    if 'event_time' in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df['event_time']):
            df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
        
        # Group by year
        df['event_year'] = df['event_time'].dt.year
        yearly_counts = df.groupby('event_year').size()
        
        # Plot time series
        yearly_counts.plot(color="#0A3A5C", linewidth=2, marker='o')
        
        plt.title('Tsunami Frequency Over Time', fontsize=18)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Tsunamis', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show years nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_dir / "tsunami_frequency_over_time.png")
    else:
        print("Warning: 'event_time' column not found in tsunami data.")
    
    plt.close()

def main():
    """Main function to generate tsunami data visualizations"""
    output_dir = create_output_dir()
    
    # Load the data
    tsunami_df = load_and_preprocess_data()
    
    # Create visualizations
    plot_tsunami_magnitude_distribution(tsunami_df, output_dir)
    plot_water_height_distribution(tsunami_df, output_dir)
    
    # Check if we have a world map file for background
    world_map_path = Path(__file__).parent / "world_map.png"
    if world_map_path.exists():
        plot_tsunami_locations(tsunami_df, output_dir)
    else:
        print("World map background file not found. Skipping tsunami locations map.")
    
    plot_tsunami_damage(tsunami_df, output_dir)
    plot_tsunami_frequency_over_time(tsunami_df, output_dir)
    
    print(f"Tsunami data visualizations saved to {output_dir.absolute()}")

if __name__ == "__main__":
    main() 