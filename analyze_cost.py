#!/usr/bin/env python3
"""
Cost Analysis Script

This script creates visualizations to analyze LLM model pricing data,
showing minimum, maximum, and average costs for prompt and completion tokens.

Usage:
    python analyze_cost.py --input model_prices.csv --output-dir figures/cost
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
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

def get_model_open_weights(df_pricing: pd.DataFrame) -> pd.DataFrame:
    """Add open_weights information to pricing data by matching with evaluation data."""
    try:
        # Try to load evaluation data to get open_weights info
        eval_df = pd.read_csv('evaluation_stats.csv')
        
        # Create mapping from model names to open_weights status
        # Handle model name variations between datasets
        model_mapping = {}
        for _, row in eval_df[['model_name', 'open_weights']].drop_duplicates().iterrows():
            eval_model = row['model_name']
            is_open = row['open_weights']
            
            # Try to match with pricing data model names
            for _, price_row in df_pricing.iterrows():
                price_model = price_row['model_name']
                price_short = price_model.split('/')[-1]
                
                # Various matching strategies
                if (eval_model == price_model or 
                    eval_model == price_short or
                    eval_model.replace('-', '') == price_short.replace('-', '') or
                    price_short.startswith(eval_model) or
                    eval_model.startswith(price_short)):
                    model_mapping[price_model] = is_open
                    break
        
        # Add open_weights column to pricing data
        df_pricing['open_weights'] = df_pricing['model_name'].map(model_mapping)
        
        # Fill missing values with False (assume closed if not found)
        df_pricing['open_weights'] = df_pricing['open_weights'].fillna(False)
        
        print(f"Mapped open_weights for {sum(~df_pricing['open_weights'].isna())} models")
        return df_pricing
        
    except FileNotFoundError:
        print("Warning: evaluation_stats.csv not found, assuming all models are closed weights")
        df_pricing['open_weights'] = False
        return df_pricing

def create_pricing_bar_chart(df: pd.DataFrame, output_path: Path, figsize: Tuple[float, float]) -> None:
    """Create bar chart showing completion token pricing with min/max shades and mean lines."""
    
    # Prepare data - extract model names for better display
    df['short_name'] = df['model_name'].str.split('/').str[-1]
    
    # Sort by average completion cost for better visualization
    df_sorted = df.sort_values('completion_avg').reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(df_sorted))
    bar_width = 0.8
    
    # Define colors based on open_weights status
    open_colors = {'max': 'lightgreen', 'min': 'darkgreen'}
    closed_colors = {'max': 'lightcoral', 'min': 'darkred'}
    
    # Create bars with different colors for open/closed weights
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        is_open = row['open_weights']
        colors = open_colors if is_open else closed_colors
        
        # Max value bar
        ax.bar(i, row['completion_max'], width=bar_width, 
              color=colors['max'], alpha=0.6, 
              edgecolor='black', linewidth=0.5)
        
        # Min value bar (overlay)
        ax.bar(i, row['completion_min'], width=bar_width, 
              color=colors['min'], alpha=0.8, 
              edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('$/1M tokens', fontsize=12, fontweight='bold')
    # ax.set_xlabel('LLM Model', fontsize=12, fontweight='bold')
    ax.set_title('Completion Token Pricing', fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.6, label='Open Weights (Max)'),
        Patch(facecolor='darkgreen', alpha=0.8, label='Open Weights (Min)'),
        Patch(facecolor='lightcoral', alpha=0.6, label='Closed Weights (Max)'),
        Patch(facecolor='darkred', alpha=0.8, label='Closed Weights (Min)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted['short_name'], rotation=45, ha='right', fontsize=10)
    
    # Add value labels for both min and max values
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        if row['completion_min'] == row['completion_max']:
            # If min=max, only add one label
            ax.text(i, row['completion_max'] + 0.2, f'${row["completion_max"]:.2f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            # Add max value label
            ax.text(i, row['completion_max'] + 0.2, f'${row["completion_max"]:.2f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Add min value label
            ax.text(i, row['completion_min'] + 0.1, f'${row["completion_min"]:.2f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created pricing bar chart: {output_path.name}")

def create_cost_heatmap(df_pricing: pd.DataFrame, output_path: Path, figsize: Tuple[float, float]) -> None:
    """Create heatmap showing completion cost by prompt type and LLM model."""
    
    try:
        # Load evaluation data
        eval_df = pd.read_csv('evaluation_stats.csv')
        print(f"Loaded evaluation data with {len(eval_df)} records")
        
        # Calculate mean completion tokens per model and prompt
        completion_stats = eval_df.groupby(['model_name', 'prompt_id', 'type', 'open_weights'])['tokens_completion'].mean().reset_index()
        
        # Merge with pricing data
        # Create mapping from pricing model names to evaluation model names
        model_mapping = {}
        eval_models = set(eval_df['model_name'].unique())
        pricing_models = set(df_pricing['model_name'].unique())
        
        print(f"Debug: Found {len(eval_models)} models in evaluation data:")
        for model in sorted(eval_models):
            print(f"  - {model}")
        print(f"Debug: Found {len(pricing_models)} models in pricing data:")
        for model in sorted(pricing_models):
            print(f"  - {model}")
        
        for _, price_row in df_pricing.iterrows():
            price_model = price_row['model_name']
            price_short = price_model.split('/')[-1]
            
            # Find matching evaluation model
            matched = False
            for eval_model in eval_df['model_name'].unique():
                if (eval_model == price_model or 
                    eval_model == price_short):
                    model_mapping[eval_model] = price_model
                    print(f"Debug: Matched '{eval_model}' -> '{price_model}'")
                    matched = True
                    break
            
            if not matched:
                print(f"Debug: Could not match pricing model '{price_model}' to any evaluation model")
        
        # Add pricing model names to completion stats
        completion_stats['pricing_model'] = completion_stats['model_name'].map(model_mapping)
        
        # Debug: Show which models were excluded
        excluded_models = completion_stats[completion_stats['pricing_model'].isna()]['model_name'].unique()
        if len(excluded_models) > 0:
            print(f"Debug: Excluding {len(excluded_models)} models without pricing data:")
            for model in excluded_models:
                print(f"  - {model}")
        
        # Filter out rows where we couldn't find pricing data
        initial_count = len(completion_stats)
        completion_stats = completion_stats.dropna(subset=['pricing_model'])
        final_count = len(completion_stats)
        print(f"Debug: Filtered from {initial_count} to {final_count} records after pricing match")
        
        # Merge with pricing data (using minimum completion cost)
        pricing_minimal = df_pricing[['model_name', 'completion_min']].copy()
        completion_stats = completion_stats.merge(
            pricing_minimal, 
            left_on='pricing_model', 
            right_on='model_name', 
            suffixes=('', '_pricing')
        )
        
        # Calculate completion cost: mean_tokens * (min_cost_per_million / 1M)
        completion_stats['completion_cost'] = completion_stats['tokens_completion'] * completion_stats['completion_min'] / 1_000_000
        
        # Create pivot table for heatmap
        heatmap_data = completion_stats.pivot_table(
            index='model_name', 
            columns='prompt_id', 
            values='completion_cost', 
            aggfunc='mean'
        )
        
        # Get prompt type mapping for sorting
        prompt_type_map = completion_stats.groupby('prompt_id')['type'].first().to_dict()
        
        # Sort columns (prompts) by type, then by prompt_id
        sorted_prompts = sorted(heatmap_data.columns, key=lambda x: (prompt_type_map.get(x, ''), x))
        heatmap_data = heatmap_data.reindex(columns=sorted_prompts)
        
        # Get model groups for sorting
        model_groups = completion_stats.groupby('model_name')['open_weights'].first().to_dict()
        
        # Sort models by open_weights status
        sorted_models = sorted(heatmap_data.index, key=lambda x: (not model_groups.get(x, False), x))
        heatmap_data = heatmap_data.reindex(sorted_models)
        
        
        # Create figure with larger width for many prompts
        fig_width = max(figsize[0], len(heatmap_data.columns) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.4f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Completion Cost ($)'},
                   ax=ax,
                   linewidths=0.5)
        
        ax.set_title('Completion Cost by Model and Prompt', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prompt ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('LLM Model', fontsize=12, fontweight='bold')
        
        # Add group labels
        # Find the boundary between open and closed weights models
        open_models = [m for m in sorted_models if model_groups.get(m, False)]
        closed_models = [m for m in sorted_models if not model_groups.get(m, False)]
        
        if open_models and closed_models:
            # Add group labels
            open_center = (len(open_models) - 1) / 2
            closed_center = len(open_models) + (len(closed_models) - 1) / 2
            
            ax.text(-2.5, open_center, 'Open\nWeights', rotation=90, 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(-2.5, closed_center, 'Closed\nWeights', rotation=90, 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Created cost heatmap: {output_path.name}")
        print(f"  Cost range: ${heatmap_data.min().min():.4f} - ${heatmap_data.max().max():.4f}")
        
    except FileNotFoundError:
        print("Warning: evaluation_stats.csv not found, skipping heatmap creation")
    except Exception as e:
        print(f"Error creating heatmap: {e}")

def create_mean_cost_bar_chart(df_pricing: pd.DataFrame, output_path: Path, figsize: Tuple[float, float]) -> None:
    """Create stacked bar chart showing min and max completion costs by LLM and type."""
    
    try:
        # Load evaluation data
        eval_df = pd.read_csv('evaluation_stats.csv')
        print(f"Loaded evaluation data with {len(eval_df)} records")
        
        # Check for truncated completions (exactly 30000 tokens)
        truncated_data = eval_df[eval_df['tokens_completion'] == 30000]
        if len(truncated_data) > 0:
            print(f"Debug: Found {len(truncated_data)} truncated completions (30000 tokens exactly)")
            
            # Group by prompt_id and model to see which combinations have truncations
            truncated_groups = truncated_data.groupby(['prompt_id', 'model_name']).size().reset_index(name='count')
            
            # For each truncated group, exclude the entire prompt for that model
            excluded_combinations = set()
            for _, row in truncated_groups.iterrows():
                prompt_id = row['prompt_id']
                model_name = row['model_name']
                excluded_combinations.add((prompt_id, model_name))
                print(f"Debug: Excluding prompt '{prompt_id}' for model '{model_name}' due to truncation")
            
            # Filter out all records for these prompt-model combinations
            initial_count = len(eval_df)
            eval_df = eval_df[~eval_df.apply(lambda x: (x['prompt_id'], x['model_name']) in excluded_combinations, axis=1)]
            final_count = len(eval_df)
            print(f"Debug: Filtered from {initial_count} to {final_count} records after removing truncated prompts")
        
        # Create model mapping (same logic as heatmap)
        model_mapping = {}
        for _, price_row in df_pricing.iterrows():
            price_model = price_row['model_name']
            price_short = price_model.split('/')[-1]
            
            for eval_model in eval_df['model_name'].unique():
                if (eval_model == price_model or eval_model == price_short):
                    model_mapping[eval_model] = price_model
                    break
        
        # Calculate completion costs per record
        eval_df['pricing_model'] = eval_df['model_name'].map(model_mapping)
        eval_df = eval_df.dropna(subset=['pricing_model'])
        
        # Merge with pricing data - use both min and max pricing
        pricing_data = df_pricing[['model_name', 'completion_min', 'completion_max']].copy()
        eval_df = eval_df.merge(
            pricing_data, 
            left_on='pricing_model', 
            right_on='model_name', 
            suffixes=('', '_pricing')
        )
        
        # Calculate mean tokens per model and type
        token_stats = eval_df.groupby(['model_name', 'type'])['tokens_completion'].mean().reset_index()
        token_stats.columns = ['model_name', 'type', 'mean_tokens']
        
        # Merge with pricing data to get min/max pricing rates
        cost_stats = token_stats.merge(
            pricing_data, 
            left_on='model_name', 
            right_on='model_name', 
            how='left'
        )
        
        # Calculate min and max costs using mean tokens but min/max pricing
        cost_stats['min_cost'] = cost_stats['mean_tokens'] * cost_stats['completion_min'] / 1_000_000
        cost_stats['max_cost'] = cost_stats['mean_tokens'] * cost_stats['completion_max'] / 1_000_000
        
        # Get open_weights information for models
        model_open_weights = {}
        for _, row in eval_df[['model_name', 'open_weights']].drop_duplicates().iterrows():
            model_open_weights[row['model_name']] = row['open_weights']
        
        # Calculate overall min cost per model for sorting
        model_overall_costs = cost_stats.groupby('model_name')['min_cost'].mean().sort_values()
        models = list(model_overall_costs.index)
        
        # Get unique types and assign more saturated colors
        types = sorted(cost_stats['type'].unique())
        # Use more saturated color schemes
        type_colors = {
            'Math': '#FF6B6B',           # Bright red
            'knowledge': '#4ECDC4',       # Bright teal
            'logic puzzle': '#45B7D1'    # Bright blue
        }
        # Fallback colors if we have different types
        if len(types) > len(type_colors):
            colors = plt.cm.Dark2(np.linspace(0, 1, len(types)))
            type_colors = dict(zip(types, colors))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        x_pos = np.arange(len(models))
        bar_width = 0.8 / len(types)
        
        # Create stacked bars for each type
        for i, ptype in enumerate(types):
            type_data = cost_stats[cost_stats['type'] == ptype].set_index('model_name')
            
            min_costs = [type_data.loc[model, 'min_cost'] if model in type_data.index else 0 for model in models]
            max_costs = [type_data.loc[model, 'max_cost'] if model in type_data.index else 0 for model in models]
            
            # Calculate the additional height for max (max - min)
            additional_heights = [max_cost - min_cost for min_cost, max_cost in zip(min_costs, max_costs)]
            
            # Only show bars where we have data
            x_positions = [x + i * bar_width for x in x_pos]
            valid_indices = [j for j, min_cost in enumerate(min_costs) if min_cost > 0]
            
            if valid_indices:
                x_valid = [x_positions[j] for j in valid_indices]
                min_costs_valid = [min_costs[j] for j in valid_indices]
                additional_heights_valid = [additional_heights[j] for j in valid_indices]
                
                # Determine edge colors based on open/closed weights
                edge_colors = []
                for j in valid_indices:
                    model = models[j]
                    is_open = model_open_weights.get(model, False)
                    edge_colors.append('darkgreen' if is_open else 'darkred')
                
                # Create base bars (min values) with base color
                bars_min = ax.bar(x_valid, min_costs_valid, bar_width, 
                                 color=type_colors[ptype], 
                                 label=f'{ptype} (Min)' if i == 0 else "",
                                 linewidth=1.5)
                
                # Create lighter shade for max bars
                import matplotlib.colors as mcolors
                base_color = mcolors.to_rgb(type_colors[ptype])
                light_color = tuple(min(1.0, c + 0.3) for c in base_color)  # Lighter shade
                
                # Create additional bars (max - min) with lighter color
                bars_max = ax.bar(x_valid, additional_heights_valid, bar_width, 
                                 bottom=min_costs_valid,
                                 color=light_color, 
                                 label=f'{ptype} (Max)' if i == 0 else "",
                                 linewidth=1.5)
                
                # Set individual edge colors for each bar
                for bar, edge_color in zip(bars_min, edge_colors):
                    bar.set_edgecolor(edge_color)
                for bar, edge_color in zip(bars_max, edge_colors):
                    bar.set_edgecolor(edge_color)
        
        # Customize plot
        ax.set_xlabel('LLM Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Completion Cost ($)', fontsize=12, fontweight='bold')
        ax.set_title('Min/Max Completion Cost by Model and Prompt Type', fontsize=14, fontweight='bold')
        
        # Set x-axis
        ax.set_xticks([x + bar_width * (len(types) - 1) / 2 for x in x_pos])
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Create custom legend for both prompt type and open/closed weights
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # Prompt type legend elements (show both min and max)
        type_legend_elements = []
        for ptype in types:
            # Dark shade for min
            type_legend_elements.append(Patch(facecolor=type_colors[ptype], alpha=0.9, label=f'{ptype} (Min)'))
            # Light shade for max
            type_legend_elements.append(Patch(facecolor=type_colors[ptype], alpha=0.5, label=f'{ptype} (Max)'))
        
        # Open/closed weights legend elements
        weight_legend_elements = [
            Line2D([0], [0], color='darkgreen', linewidth=3, label='Open Weights'),
            Line2D([0], [0], color='darkred', linewidth=3, label='Closed Weights')
        ]
        
        # Combine legends
        all_legend_elements = type_legend_elements + weight_legend_elements
        ax.legend(handles=all_legend_elements, fontsize=9, title='Prompt Type & Model Type', title_fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Created min/max cost bar chart: {output_path.name}")
        
        # Print summary statistics
        overall_stats = cost_stats.groupby('type')[['min_cost', 'max_cost']].agg(['mean', 'min', 'max'])
        print(f"\nCost Summary by Type:")
        for ptype in types:
            if ptype in overall_stats.index:
                min_stats = overall_stats.loc[ptype, 'min_cost']
                max_stats = overall_stats.loc[ptype, 'max_cost']
                print(f"  {ptype}: Min=${min_stats['min']:.4f}-${min_stats['max']:.4f}, Max=${max_stats['min']:.4f}-${max_stats['max']:.4f}")
        
    except FileNotFoundError:
        print("Warning: evaluation_stats.csv not found, skipping min/max cost bar chart creation")
    except Exception as e:
        print(f"Error creating min/max cost bar chart: {e}")

def create_individual_type_bar_charts(df_pricing: pd.DataFrame, output_dir: Path, figsize: Tuple[float, float]) -> None:
    """Create separate bar charts for each prompt type."""
    
    try:
        # Load evaluation data
        eval_df = pd.read_csv('evaluation_stats.csv')
        print(f"Loaded evaluation data with {len(eval_df)} records for individual type charts")
        
        # Check for truncated completions (exactly 30000 tokens)
        truncated_data = eval_df[eval_df['tokens_completion'] == 30000]
        if len(truncated_data) > 0:
            # Group by prompt_id and model to see which combinations have truncations
            truncated_groups = truncated_data.groupby(['prompt_id', 'model_name']).size().reset_index(name='count')
            
            # For each truncated group, exclude the entire prompt for that model
            excluded_combinations = set()
            for _, row in truncated_groups.iterrows():
                prompt_id = row['prompt_id']
                model_name = row['model_name']
                excluded_combinations.add((prompt_id, model_name))
            
            # Filter out all records for these prompt-model combinations
            eval_df = eval_df[~eval_df.apply(lambda x: (x['prompt_id'], x['model_name']) in excluded_combinations, axis=1)]
        
        # Create model mapping (same logic as heatmap)
        model_mapping = {}
        for _, price_row in df_pricing.iterrows():
            price_model = price_row['model_name']
            price_short = price_model.split('/')[-1]
            
            for eval_model in eval_df['model_name'].unique():
                if (eval_model == price_model or eval_model == price_short):
                    model_mapping[eval_model] = price_model
                    break
        
        # Calculate completion costs per record
        eval_df['pricing_model'] = eval_df['model_name'].map(model_mapping)
        eval_df = eval_df.dropna(subset=['pricing_model'])
        
        # Merge with pricing data
        pricing_minimal = df_pricing[['model_name', 'completion_min']].copy()
        eval_df = eval_df.merge(
            pricing_minimal, 
            left_on='pricing_model', 
            right_on='model_name', 
            suffixes=('', '_pricing')
        )
        
        # Calculate completion cost per record
        eval_df['completion_cost'] = eval_df['tokens_completion'] * eval_df['completion_min'] / 1_000_000
        
        # Get open_weights information for models
        model_open_weights = {}
        for _, row in eval_df[['model_name', 'open_weights']].drop_duplicates().iterrows():
            model_open_weights[row['model_name']] = row['open_weights']
        
        # Get unique types
        types = sorted(eval_df['type'].unique())
        
        # Define colors for each type
        type_colors = {
            'Math': '#FF6B6B',           # Bright red
            'knowledge': '#4ECDC4',       # Bright teal
            'logic puzzle': '#45B7D1'    # Bright blue
        }
        
        # Create individual charts for each type
        for ptype in types:
            # Filter data for this type
            type_data = eval_df[eval_df['type'] == ptype]
            
            # Calculate mean tokens per model for this type
            token_stats = type_data.groupby('model_name')['tokens_completion'].mean().reset_index()
            token_stats.columns = ['model_name', 'mean_tokens']
            
            # Merge with pricing data to get min/max pricing rates
            pricing_data = df_pricing[['model_name', 'completion_min', 'completion_max']].copy()
            cost_stats = token_stats.merge(
                pricing_data, 
                left_on='model_name', 
                right_on='model_name', 
                how='left'
            )
            
            # Calculate min and max costs using mean tokens but min/max pricing
            cost_stats['min_cost'] = cost_stats['mean_tokens'] * cost_stats['completion_min'] / 1_000_000
            cost_stats['max_cost'] = cost_stats['mean_tokens'] * cost_stats['completion_max'] / 1_000_000
            cost_stats['count'] = type_data.groupby('model_name').size().values
            
            # Sort by min cost
            cost_stats = cost_stats.sort_values('min_cost')
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            models = cost_stats['model_name'].tolist()
            min_costs = cost_stats['min_cost'].tolist()
            max_costs = cost_stats['max_cost'].tolist()
            
            # Calculate the additional height for max (max - min)
            additional_heights = [max_cost - min_cost for min_cost, max_cost in zip(min_costs, max_costs)]
            
            # Determine bar colors based on open/closed weights
            min_colors = []
            max_colors = []
            for model in models:
                is_open = model_open_weights.get(model, False)
                if is_open:
                    min_colors.append('darkgreen')
                    max_colors.append('lightgreen')
                else:
                    min_colors.append('darkred')
                    max_colors.append('lightcoral')
            
            # Create stacked bars - min values
            bars_min = ax.bar(range(len(models)), min_costs, 
                             color=min_colors, 
                             label='Min Cost',
                             edgecolor='black', linewidth=0.5)
            
            # Create additional bars for max - min
            bars_max = ax.bar(range(len(models)), additional_heights, 
                             bottom=min_costs,
                             color=max_colors, 
                             label='Max Cost',
                             edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_xlabel('LLM Model', fontsize=12, fontweight='bold')
            ax.set_ylabel('Completion Cost ($)', fontsize=12, fontweight='bold')
            ax.set_title(f'Min/Max Completion Cost - {ptype.title()} Prompts', fontsize=14, fontweight='bold')
            
            # Set x-axis
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Create legend for both cost type and open/closed weights
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gray', label='Min Cost'),
                Patch(facecolor='lightgray', label='Max Cost'),
                Patch(facecolor='#2E8B57', label='Open Weights'),
                Patch(facecolor='#B22222', label='Closed Weights')
            ]
            ax.legend(handles=legend_elements, fontsize=10, title='Cost Type & Model Type', title_fontsize=11)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars (show both min and max)
            for i, (min_cost, max_cost) in enumerate(zip(min_costs, max_costs)):
                if min_cost == max_cost:
                    # If min=max, only show one label
                    ax.text(i, max_cost + max(max_costs) * 0.01, 
                           f'${max_cost:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                else:
                    # Show both min and max labels
                    ax.text(i, max_cost + max(max_costs) * 0.01, 
                           f'${max_cost:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    ax.text(i, min_cost + (max_cost - min_cost) * 0.1, 
                           f'${min_cost:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            safe_filename = ptype.replace(' ', '_').replace('/', '_').lower()
            output_path = output_dir / f"mean_cost_{safe_filename}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Created individual cost chart for {ptype}: {output_path.name}")
            print(f"  Models: {len(models)}, Cost range: ${min(min_costs):.4f} - ${max(max_costs):.4f}")
        
        print(f"Created {len(types)} individual prompt type charts")
        
    except FileNotFoundError:
        print("Warning: evaluation_stats.csv not found, skipping individual type charts creation")
    except Exception as e:
        print(f"Error creating individual type charts: {e}")

def main():
    """Main function to create cost analysis plots."""
    parser = argparse.ArgumentParser(
        description="Create LLM cost analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_cost.py                              # Use default parameters
    python analyze_cost.py --input model_prices.csv    # Custom input file
    python analyze_cost.py --output-dir figures/cost   # Custom output directory
        """
    )
    
    parser.add_argument('--input', default='model_prices.csv',
                       help='Path to input CSV file with model pricing data (default: model_prices.csv)')
    parser.add_argument('--output-dir', default='figures/cost',
                       help='Output directory for plots (default: figures/cost)')
    parser.add_argument('--figsize', default='8,6',
                       help='Figure size as width,height in inches (default: 8,6)')
    
    args = parser.parse_args()
    
    # Parse figure size
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print("Error: Invalid figsize format. Use 'width,height' (e.g., '8,6')")
        sys.exit(1)
    
    # Validate inputs
    input_path = Path(args.input)
    validate_input(input_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    create_output_directory(output_dir)
    
    print(f"Loading pricing data from: {input_path}")
    
    # Load data
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} models with {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Check required columns
    required_columns = ['model_name', 'prompt_min', 'prompt_max', 'prompt_avg', 
                       'completion_min', 'completion_max', 'completion_avg']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        sys.exit(1)
    
    print("Creating cost analysis plots...")
    
    # Add open_weights information
    df = get_model_open_weights(df)
    
    # Create pricing bar chart
    pricing_path = output_dir / "model_pricing_comparison.png"
    create_pricing_bar_chart(df, pricing_path, figsize)
    
    # Create cost heatmap with larger size
    heatmap_path = output_dir / "completion_cost_heatmap.png"
    heatmap_figsize = (16, 12)
    create_cost_heatmap(df, heatmap_path, heatmap_figsize)
    
    # Create mean cost bar chart
    bar_chart_path = output_dir / "mean_completion_cost_by_type.png"
    create_mean_cost_bar_chart(df, bar_chart_path, figsize)
    
    # Create individual charts for each prompt type
    create_individual_type_bar_charts(df, output_dir, figsize)
    
    # Print summary statistics
    print(f"\nPricing Summary:")
    print(f"Prompt tokens - Mean: ${df['prompt_avg'].mean():.2f}, Range: ${df['prompt_min'].min():.2f} - ${df['prompt_max'].max():.2f}")
    print(f"Completion tokens - Mean: ${df['completion_avg'].mean():.2f}, Range: ${df['completion_min'].min():.2f} - ${df['completion_max'].max():.2f}")
    
    # Show most/least expensive models
    most_expensive_prompt = df.loc[df['prompt_avg'].idxmax()]
    least_expensive_prompt = df.loc[df['prompt_avg'].idxmin()]
    most_expensive_completion = df.loc[df['completion_avg'].idxmax()]
    least_expensive_completion = df.loc[df['completion_avg'].idxmin()]
    
    print(f"\nMost expensive prompt: {most_expensive_prompt['model_name']} (${most_expensive_prompt['prompt_avg']:.2f})")
    print(f"Least expensive prompt: {least_expensive_prompt['model_name']} (${least_expensive_prompt['prompt_avg']:.2f})")
    print(f"Most expensive completion: {most_expensive_completion['model_name']} (${most_expensive_completion['completion_avg']:.2f})")
    print(f"Least expensive completion: {least_expensive_completion['model_name']} (${least_expensive_completion['completion_avg']:.2f})")
    
    print(f"\nCompleted! Created cost analysis plots in {output_dir}")
    print(f"  - 1 pricing bar chart")
    print(f"  - 1 completion cost heatmap")
    print(f"  - 1 combined mean cost bar chart by type")
    print(f"  - Individual cost charts for each prompt type")
    print(f"Figure size: {figsize[0]}x{figsize[1]} inches")

if __name__ == "__main__":
    main()