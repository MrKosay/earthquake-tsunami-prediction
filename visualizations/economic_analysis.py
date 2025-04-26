#!/usr/bin/env python3
"""
Economic Damage Analysis and Visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker
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

# Create a custom colormap for economic visualizations
colors = ["#E2F0CB", "#B5D99C", "#86BF7B", "#65A666", "#469D5F", "#2A7F62"]
economic_cmap = LinearSegmentedColormap.from_list("economic_cmap", colors)

def billions_formatter(x, pos):
    """Format y-axis values in billions"""
    return f'${x/1e9:.1f}B'

def millions_formatter(x, pos):
    """Format y-axis values in millions"""
    return f'${x/1e6:.1f}M'

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = Path("visualizations/economic_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_economic_data():
    """Load and preprocess economic data"""
    # Load the raw economic data files
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
                'location': row.get('Location', ''),
                'disaster_type': row.get('Disaster_Type', ''),
                'fatalities': row.get('Fatalities', np.nan),
                'economic_damage': row.get('Estimated_Economic_Damage_USD'),
                'source': row.get('Source', '')
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
                'location': row.get('City_or_Region', ''),
                'country': row.get('Country', ''),
                'disaster_type': row.get('Disaster_Type', ''),
                'fatalities': row.get('Fatalities', np.nan),
                'injuries': row.get('Injuries', np.nan),
                'economic_damage': row.get('Estimated_Economic_Damage_USD'),
                'source': row.get('Source', '')
            })
    
    # Create DataFrame from the collected data
    econ_df = pd.DataFrame(econ_data)
    
    # Convert damage values to numeric, handling any string values
    econ_df['economic_damage'] = pd.to_numeric(econ_df['economic_damage'], errors='coerce')
    econ_df = econ_df.dropna(subset=['economic_damage'])
    
    print(f"Loaded economic data with {len(econ_df)} records")
    return econ_df

def plot_damage_by_magnitude(df, output_dir):
    """Plot economic damage vs magnitude"""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with log scale
    plt.scatter(df['magnitude'], df['economic_damage'], 
                c=df['magnitude'], cmap=economic_cmap,
                alpha=0.7, s=100, edgecolors='k', linewidths=0.5)
    
    plt.title('Economic Damage vs. Earthquake Magnitude', fontsize=18)
    plt.xlabel('Magnitude', fontsize=14)
    plt.ylabel('Economic Damage (USD)', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Format y-axis ticks to show dollar values
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x:.0f}' if x < 1e6 else 
                                                           (f'${x/1e6:.1f}M' if x < 1e9 else 
                                                            f'${x/1e9:.1f}B')))
    
    # Add best fit line
    x = df['magnitude']
    y = np.log10(df['economic_damage'])
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, 10**p(x), "r--", alpha=0.8, 
             label=f"Best fit: Damage ~ 10^({z[0]:.2f}*Magnitude {z[1]:.2f})")
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "damage_vs_magnitude.png")
    plt.close()

def plot_damage_by_year(df, output_dir):
    """Plot economic damage by year with inflation adjustment"""
    plt.figure(figsize=(14, 8))
    
    # Group by year and sum damage
    yearly_damage = df.groupby('year')['economic_damage'].sum().reset_index()
    
    # Add inflation adjustment (rough approximation)
    base_year = 2020
    yearly_damage['adjusted_damage'] = yearly_damage.apply(
        lambda row: row['economic_damage'] * (1.03 ** (base_year - row['year'])), axis=1
    )
    
    # Create bar chart
    x = yearly_damage['year']
    width = 0.35
    
    plt.bar(x - width/2, yearly_damage['economic_damage'], width, 
            label='Nominal Damage', color="#65A666", alpha=0.7)
    plt.bar(x + width/2, yearly_damage['adjusted_damage'], width, 
            label=f'Inflation-Adjusted to {base_year}', color="#2A7F62", alpha=0.7)
    
    plt.title('Economic Damage by Year', fontsize=18)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Economic Damage (USD)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis ticks to show dollar values in billions
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(billions_formatter))
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "damage_by_year.png")
    plt.close()

def plot_damage_distribution(df, output_dir):
    """Plot distribution of economic damages"""
    plt.figure(figsize=(12, 8))
    
    # Use log scale for better visualization
    damage_log = np.log10(df['economic_damage'])
    
    # Create histogram
    plt.hist(damage_log, bins=20, alpha=0.8, color="#469D5F")
    
    plt.title('Economic Damage Distribution (Log Scale)', fontsize=18)
    plt.xlabel('Economic Damage (log10 USD)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis labels to original values
    tick_locations = np.arange(np.floor(damage_log.min()), np.ceil(damage_log.max()) + 1)
    tick_labels = [f"$10^{int(loc)}$" for loc in tick_locations]
    plt.xticks(tick_locations, tick_labels)
    
    # Add vertical lines for damage categories
    plt.axvline(x=6, color='darkgreen', linestyle='--', 
                label='Million Dollar (≥$10^6)', linewidth=2)
    plt.axvline(x=9, color='black', linestyle='--', 
                label='Billion Dollar (≥$10^9)', linewidth=2)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "damage_distribution.png")
    plt.close()

def plot_damage_by_disaster_type(df, output_dir):
    """Plot economic damage by disaster type"""
    plt.figure(figsize=(14, 8))
    
    # Group by disaster type and sum damage
    damage_by_type = df.groupby('disaster_type')['economic_damage'].sum().sort_values(ascending=False)
    
    # Create bar chart
    bars = plt.bar(damage_by_type.index, damage_by_type.values, 
            color=sns.color_palette(colors, len(damage_by_type)), alpha=0.8)
    
    plt.title('Economic Damage by Disaster Type', fontsize=18)
    plt.xlabel('Disaster Type', fontsize=14)
    plt.ylabel('Economic Damage (USD)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Format y-axis ticks to show dollar values in billions
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(billions_formatter))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height/1e9:.1f}B',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "damage_by_disaster_type.png")
    plt.close()

def plot_damage_vs_fatalities(df, output_dir):
    """Plot economic damage vs fatalities"""
    plt.figure(figsize=(12, 8))
    
    # Remove rows with missing fatalities data
    plot_df = df.dropna(subset=['fatalities', 'economic_damage'])
    
    # Create scatter plot with log scales
    scatter = plt.scatter(plot_df['fatalities'], plot_df['economic_damage'], 
                c=plot_df['magnitude'], cmap=economic_cmap,
                alpha=0.7, s=80, edgecolors='k', linewidths=0.5)
    
    plt.title('Economic Damage vs. Fatalities', fontsize=18)
    plt.xlabel('Fatalities (log scale)', fontsize=14)
    plt.ylabel('Economic Damage (USD, log scale)', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Magnitude')
    
    # Format axis ticks
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x:.0f}' if x < 1e6 else 
                                                           (f'${x/1e6:.1f}M' if x < 1e9 else 
                                                            f'${x/1e9:.1f}B')))
    
    # Add annotation for significant events
    for _, row in plot_df.nlargest(5, 'economic_damage').iterrows():
        plt.annotate(
            row.get('location', f"{row['year']}"),
            xy=(row['fatalities'], row['economic_damage']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "damage_vs_fatalities.png")
    plt.close()

def plot_damage_prediction_model(df, output_dir):
    """Plot actual vs predicted damage from a simple model"""
    plt.figure(figsize=(12, 8))
    
    # Create simple prediction model for visualization
    # Log(Damage) ~ a*Magnitude + b*log(Fatalities) + c
    X = df[['magnitude', 'fatalities']].dropna()
    df_model = df.loc[X.index].copy()
    
    X['log_fatalities'] = np.log10(X['fatalities'].clip(1))
    y = np.log10(df_model['economic_damage'])
    
    # Linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X[['magnitude', 'log_fatalities']], y)
    
    # Make predictions
    df_model['predicted_log_damage'] = model.predict(X[['magnitude', 'log_fatalities']])
    df_model['predicted_damage'] = 10 ** df_model['predicted_log_damage']
    
    # Create scatter plot
    plt.scatter(df_model['economic_damage'], df_model['predicted_damage'], 
                c=df_model['magnitude'], cmap=economic_cmap,
                alpha=0.7, s=80, edgecolors='k', linewidths=0.5)
    
    plt.title('Actual vs. Predicted Economic Damage', fontsize=18)
    plt.xlabel('Actual Damage (USD, log scale)', fontsize=14)
    plt.ylabel('Predicted Damage (USD, log scale)', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add perfect prediction line
    min_val = min(df_model['economic_damage'].min(), df_model['predicted_damage'].min())
    max_val = max(df_model['economic_damage'].max(), df_model['predicted_damage'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label="Perfect Prediction")
    
    # Add lines for 10x error margin
    plt.plot([min_val, max_val], [min_val*10, max_val*10], 'k:', alpha=0.5, label="10x Error Margin")
    plt.plot([min_val, max_val], [min_val/10, max_val/10], 'k:', alpha=0.5)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / "damage_prediction_model.png")
    plt.close()
    
    # Return the model metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y, df_model['predicted_log_damage'])
    mae = mean_absolute_error(y, df_model['predicted_log_damage'])
    
    return {
        'r2': r2,
        'mae_log': mae,
        'model_coefficients': {
            'magnitude': model.coef_[0],
            'log_fatalities': model.coef_[1],
            'intercept': model.intercept_
        }
    }

def main():
    """Main function to generate economic data visualizations"""
    output_dir = create_output_dir()
    
    # Load the data
    econ_df = load_economic_data()
    
    # Create visualizations
    plot_damage_by_magnitude(econ_df, output_dir)
    plot_damage_by_year(econ_df, output_dir)
    plot_damage_distribution(econ_df, output_dir)
    plot_damage_by_disaster_type(econ_df, output_dir)
    plot_damage_vs_fatalities(econ_df, output_dir)
    
    # Create and evaluate simple prediction model
    model_metrics = plot_damage_prediction_model(econ_df, output_dir)
    
    print(f"Economic data visualizations saved to {output_dir.absolute()}")
    print(f"Simple prediction model metrics: R²={model_metrics['r2']:.4f}, MAE(log)={model_metrics['mae_log']:.4f}")
    print(f"Model equation: log10(Damage) = {model_metrics['model_coefficients']['magnitude']:.4f}*Magnitude + "
          f"{model_metrics['model_coefficients']['log_fatalities']:.4f}*log10(Fatalities) + "
          f"{model_metrics['model_coefficients']['intercept']:.4f}")

if __name__ == "__main__":
    main() 