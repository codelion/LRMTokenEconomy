#!/usr/bin/env python3
"""
Chain-of-Thought Transcription Analysis Script

This script creates visualizations to analyze the relationship between tokens and characters
in LLM completions, including scatter plots with linear fits and bar charts showing
characters per token ratios.

Usage:
    python cot_transcription.py --input evaluation_stats.csv --output-dir figures --figsize 12,8
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for better performance
plt.switch_backend('Agg')

def validate_input(input_path: Path) -> None:
    """Validate that input file exists and is readable."""
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}")
        sys.exit(1)

def create_output_directory(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

def get_model_groups(df: pd.DataFrame) -> Dict[str, bool]:
    """Get model grouping by open_weights status."""
    model_groups = {}
    for _, row in df[['model_name', 'open_weights']].drop_duplicates().iterrows():
        model_groups[row['model_name']] = row['open_weights']
    return model_groups

def perform_linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Perform linear regression through origin and return slope and R²."""
    # Remove any NaN or infinite values
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return 0.0, 0.0
    
    # Linear regression through origin: y = slope * x
    slope = np.sum(x_clean * y_clean) / np.sum(x_clean * x_clean)
    
    # Calculate R² for fit through origin
    y_pred = slope * x_clean
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, r_squared

def create_scatter_plot(df: pd.DataFrame, output_path: Path, figsize: Tuple[float, float]) -> Dict[str, float]:
    """Create scatter plot of completion tokens vs characters with linear fits."""
    
    # Filter out rows with missing or zero values
    plot_df = df[(df['tokens_completion'] > 0) & (df['char_completion'] > 0)].copy()
    
    if plot_df.empty:
        print("Warning: No valid data points for scatter plot")
        return {}
    
    # Get unique models and assign colors
    models = sorted(plot_df['model_name'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    slopes = {}
    scatter_handles = []
    legend_labels = []
    
    for i, model in enumerate(models):
        model_data = plot_df[plot_df['model_name'] == model]
        
        x = model_data['tokens_completion'].values
        y = model_data['char_completion'].values
        
        # Plot scatter points (store handle for legend)
        scatter_handle = ax.scatter(x, y, c=[colors[i]], alpha=0.6, s=20)
        
        # Perform linear fit through origin
        slope, r_squared = perform_linear_fit(x, y)
        slopes[model] = slope
        
        # Plot fit line (without adding to legend)
        if slope > 0:
            x_max = max(x)
            x_line = np.linspace(0, x_max, 100)
            y_line = slope * x_line
            ax.plot(x_line, y_line, color=colors[i], linestyle='--', alpha=0.8, linewidth=1.5)
            
            # Create legend label with slope
            legend_labels.append(f'{model} (slope: {slope:.1f})')
        else:
            legend_labels.append(f'{model} (slope: N/A)')
        
        # Store scatter handle for legend
        scatter_handles.append(scatter_handle)
    
    # Customize plot
    ax.set_xlabel('Completion Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Completion Characters', fontsize=12, fontweight='bold')
    ax.set_title('Completion Tokens vs Characters by LLM', 
                fontsize=14, fontweight='bold')
    
    # Add legend using only scatter plot handles
    ax.legend(scatter_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created scatter plot: {output_path.name}")
    return slopes

def create_slope_bar_chart(slopes: Dict[str, float], model_groups: Dict[str, bool], 
                          output_path: Path, figsize: Tuple[float, float]) -> None:
    """Create bar chart showing characters per token slope by LLM."""
    
    if not slopes:
        print("Warning: No slope data available for bar chart")
        return
    
    # Prepare data sorted by slope values in ascending order
    sorted_items = sorted(slopes.items(), key=lambda x: x[1])
    models = [item[0] for item in sorted_items]
    slope_values = [item[1] for item in sorted_items]
    
    # Color code by open_weights (maintaining correspondence with sorted models)
    colors = []
    for model in models:
        is_open = model_groups.get(model, False)
        colors.append('#2F85A7' if is_open else '#8095A3')  # Green for open, red for closed
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(range(len(models)), slope_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    # ax.set_xlabel('LLM Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Characters per Token', fontsize=12, fontweight='bold')
    ax.set_title('Characters per Token Ratio by LLM', 
                fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, slope_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2F85A7', alpha=0.8, label='Open Weights'),
        Patch(facecolor='#8095A3', alpha=0.8, label='Closed Weights')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created slope bar chart: {output_path.name}")

def create_filtered_scatter_plot(df: pd.DataFrame, output_path: Path, figsize: Tuple[float, float]) -> None:
    """Create scatter plot with only selected LLMs for illustration."""
    
    # Define the prefixes to filter by
    filter_prefixes = ['o4', 'gemini', 'claude', 'deephermes', 'magistral-medium']
    
    # Filter data to only include LLMs that start with specified prefixes
    filtered_df = df[df['model_name'].str.lower().str.startswith(tuple(filter_prefixes), na=False)]
    
    # Filter out rows with missing or zero values
    plot_df = filtered_df[(filtered_df['tokens_completion'] > 0) & (filtered_df['char_completion'] > 0)].copy()
    
    if plot_df.empty:
        print("Warning: No data found for filtered models")
        return
    
    # Get unique models and assign colors
    models = sorted(plot_df['model_name'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Create the plot with larger figure size
    larger_figsize = (figsize[0] * 1.3, figsize[1] * 1.2)  # Make it 30% wider and 20% taller
    fig, ax = plt.subplots(figsize=larger_figsize)
    
    slopes = {}
    scatter_handles = []
    legend_labels = []
    
    for i, model in enumerate(models):
        model_data = plot_df[plot_df['model_name'] == model]
        
        x = model_data['tokens_completion'].values
        y = model_data['char_completion'].values
        
        # Plot scatter points (store handle for legend)
        scatter_handle = ax.scatter(x, y, c=[colors[i]], alpha=0.7, s=40)
        
        # Perform linear fit through origin
        slope, r_squared = perform_linear_fit(x, y)
        slopes[model] = slope
        
        # Plot fit line (without adding to legend)
        if slope > 0:
            x_max = max(x)
            x_line = np.linspace(0, x_max, 100)
            y_line = slope * x_line
            ax.plot(x_line, y_line, color=colors[i], linestyle='--', alpha=0.8, linewidth=2)
            
            # Create legend label with slope
            legend_labels.append(f'{model} (slope: {slope:.1f})')
        else:
            legend_labels.append(f'{model} (slope: N/A)')
        
        # Store scatter handle for legend
        scatter_handles.append(scatter_handle)
    
    # Customize plot
    ax.set_xlabel('Completion Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Completion Characters', fontsize=12, fontweight='bold')
    ax.set_title('Completion Tokens vs Characters for Selected LLMs', 
                fontsize=14, fontweight='bold')
    
    # Add legend using only scatter plot handles in the top left of plot area
    ax.legend(scatter_handles, legend_labels, loc='upper left', fontsize=10, 
             framealpha=0.9, edgecolor='black')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created filtered scatter plot: {output_path.name}")
    print(f"  Included models: {', '.join(models)}")

def main():
    """Main function to create transcription analysis plots."""
    parser = argparse.ArgumentParser(
        description="Create Chain-of-Thought transcription analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cot_transcription.py                    # Use default parameters
    python cot_transcription.py --figsize 14,10    # Custom figure size
    python cot_transcription.py --input data.csv --output-dir plots --figsize 14,10
        """
    )
    
    parser.add_argument('--input', default='evaluation_stats.csv',
                       help='Path to input CSV file with evaluation data (default: evaluation_stats.csv)')
    parser.add_argument('--output-dir', default='figures/cot_transcription',
                       help='Output directory for plots (default: figures/cot_transcription)')
    parser.add_argument('--figsize', default='8,6',
                       help='Figure size as width,height in inches (default: 12,8)')
    
    args = parser.parse_args()
    
    # Parse figure size
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print("Error: Invalid figsize format. Use 'width,height' (e.g., '12,8')")
        sys.exit(1)
    
    # Validate inputs
    input_path = Path(args.input)
    validate_input(input_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    create_output_directory(output_dir)
    
    print(f"Loading data from: {input_path}")
    
    # Load data
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Check required columns
    required_columns = ['model_name', 'tokens_completion', 'char_completion', 'open_weights']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        sys.exit(1)
    
    print(f"Found {df['model_name'].nunique()} unique models")
    
    # Get model groupings
    model_groups = get_model_groups(df)
    
    print("Creating transcription analysis plots...")
    
    # Create scatter plot and get slopes
    scatter_path = output_dir / "tokens_vs_characters_scatter.png"
    slopes = create_scatter_plot(df, scatter_path, figsize)
    
    # Create filtered scatter plot for illustration
    filtered_scatter_path = output_dir / "tokens_vs_characters_selected_models.png"
    create_filtered_scatter_plot(df, filtered_scatter_path, figsize)
    
    # Create slope bar chart
    if slopes:
        bar_path = output_dir / "characters_per_token_by_model.png"
        create_slope_bar_chart(slopes, model_groups, bar_path, figsize)
        
        # Print summary statistics
        print(f"\nCharacters per Token Summary:")
        print(f"Mean slope: {np.mean(list(slopes.values())):.1f}")
        print(f"Std deviation: {np.std(list(slopes.values())):.1f}")
        print(f"Range: {min(slopes.values()):.1f} - {max(slopes.values()):.1f}")
        
        # Group by open/closed weights
        open_slopes = [slopes[model] for model, is_open in model_groups.items() 
                      if is_open and model in slopes]
        closed_slopes = [slopes[model] for model, is_open in model_groups.items() 
                        if not is_open and model in slopes]
        
        if open_slopes:
            print(f"Open weights models - Mean: {np.mean(open_slopes):.1f}, Count: {len(open_slopes)}")
        if closed_slopes:
            print(f"Closed weights models - Mean: {np.mean(closed_slopes):.1f}, Count: {len(closed_slopes)}")
    
    print(f"\nCompleted! Created transcription analysis plots in {output_dir}")
    print(f"  - 1 scatter plot (all models)")  
    print(f"  - 1 filtered scatter plot (selected models)")
    print(f"  - 1 slope bar chart")
    print(f"Figure size: {figsize[0]}x{figsize[1]} inches")

if __name__ == "__main__":
    main()