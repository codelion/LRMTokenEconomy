#!/usr/bin/env python3
"""
Unified Prompt Analysis Script

This script analyzes token usage patterns for knowledge, logic puzzle, and math type prompts,
focusing on completion tokens and reasoning character ratios.

Usage:
    python analyze_prompts.py --types knowledge,logic_puzzle,math --output-dir figures/eco_all
    python analyze_prompts.py --types knowledge --output-dir figures/eco_knowledge
    python analyze_prompts.py --types math --output-dir figures/eco_math
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for better performance
plt.switch_backend('Agg')

# Type mappings for filtering and display
TYPE_MAPPINGS = {
    'knowledge': 'knowledge',
    'logic_puzzle': 'logic puzzle', 
    'math': 'Math'
}

TYPE_DISPLAY_NAMES = {
    'knowledge': 'Knowledge',
    'logic_puzzle': 'Logic Puzzle',
    'math': 'Math'
}

# Preset configurations for common analysis patterns
PRESETS = {
    'all': {
        'types': ['knowledge', 'logic_puzzle', 'math'],
        'output_dir': 'figures/eco_all',
        'description': 'Analyze all prompt types'
    },
    'knowledge': {
        'types': ['knowledge'],
        'output_dir': 'figures/eco_knowledge', 
        'description': 'Analyze knowledge prompts only'
    },
    'logic_puzzle': {
        'types': ['logic_puzzle'],
        'output_dir': 'figures/eco_logic_puzzles',
        'description': 'Analyze logic puzzle prompts only'
    },
    'math': {
        'types': ['math'],
        'output_dir': 'figures/eco_math',
        'description': 'Analyze math prompts only'
    },
    'reasoning': {
        'types': ['logic_puzzle', 'math'],
        'output_dir': 'figures/eco_reasoning',
        'description': 'Analyze reasoning-heavy prompts (logic puzzles + math)'
    },
    'knowledge_math': {
        'types': ['knowledge', 'math'],
        'output_dir': 'figures/eco_knowledge_math',
        'description': 'Analyze knowledge and math prompts'
    }
}


def create_output_directory(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")


def load_evaluation_stats(csv_path: Path) -> pd.DataFrame:
    """Load evaluation statistics from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded evaluation stats with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading evaluation stats: {e}")
        sys.exit(1)


def determine_calculation_method(df: pd.DataFrame, selected_types: Set[str]) -> Dict[str, str]:
    """Determine which calculation method to use for each LLM based on data validity."""
    # Filter for selected types
    type_conditions = [df['type'] == TYPE_MAPPINGS[t] for t in selected_types]
    filtered_df = df[pd.concat(type_conditions, axis=1).any(axis=1)].copy()
    
    method_per_llm = {}
    
    for model_name in filtered_df['model_name'].unique():
        model_data = filtered_df[filtered_df['model_name'] == model_name]
        if model_data.empty:
            continue
            
        # Check if token method is valid
        token_method_valid = True
        has_nonzero_reasoning = False
        
        for _, row in model_data.iterrows():
            tokens_reasoning = row.get('tokens_reasoning', 0)
            tokens_completion = row.get('tokens_completion', 0)
            
            # Check if we have valid token data
            if (isinstance(tokens_reasoning, (int, float)) and 
                isinstance(tokens_completion, (int, float)) and tokens_completion > 0):
                
                # Ensure tokens_reasoning is not negative
                if tokens_reasoning > 0:
                    has_nonzero_reasoning = True
                    # If tokens_reasoning > tokens_completion, invalid
                    if tokens_reasoning > tokens_completion:
                        token_method_valid = False
                        break
        
        # Decide method
        if token_method_valid and has_nonzero_reasoning:
            method_per_llm[model_name] = "tokens"
            print(f"Using token method for {model_name}")
        elif not has_nonzero_reasoning:
            method_per_llm[model_name] = "chars"
            print(f"Using character method for {model_name} (tokens_reasoning is zero)")
        else:
            method_per_llm[model_name] = "chars"
            print(f"Using character method for {model_name} (tokens_reasoning > tokens_completion detected)")
    
    return method_per_llm

def calculate_reasoning_ratios(metrics: Dict[str, Any], method: str) -> List[float]:
    """Calculate reasoning ratios for each individual run using the specified method."""
    ratios = []
    
    for i in range(len(metrics['completion_tokens'])):
        if method == "tokens":
            # Method 1: Use tokens_reasoning / tokens_completion
            if (i < len(metrics['tokens_reasoning']) and 
                metrics['tokens_reasoning'] and 
                isinstance(metrics['tokens_reasoning'][i], (int, float)) and
                metrics['tokens_reasoning'][i] > 0):
                
                tokens_reasoning = float(metrics['tokens_reasoning'][i])
                tokens_completion = float(metrics['completion_tokens'][i])
                if tokens_completion > 0:
                    ratio = tokens_reasoning / tokens_completion
                    ratios.append(ratio)
                else:
                    ratios.append(0.0)
            else:
                ratios.append(0.0)
                
        else:  # method == "chars"
            # Method 2: Use char_reasoning / char_completion
            if (i < len(metrics['char_reasoning']) and 
                i < len(metrics['char_completion']) and
                metrics['char_reasoning'] and 
                metrics['char_completion'] and
                isinstance(metrics['char_reasoning'][i], (int, float)) and
                isinstance(metrics['char_completion'][i], (int, float))):
                
                char_reasoning = float(metrics['char_reasoning'][i])
                char_completion = float(metrics['char_completion'][i])
                if char_completion > 0:
                    ratio = char_reasoning / char_completion
                    ratios.append(ratio)
                else:
                    ratios.append(0.0)
            else:
                ratios.append(0.0)
    
    return ratios

def load_and_process_data(df: pd.DataFrame, selected_types: Set[str]) -> List[Dict[str, Any]]:
    """Load and process CSV data, extracting metrics for selected prompt types only."""
    all_metrics = []
    
    # Filter for selected types
    type_conditions = [df['type'] == TYPE_MAPPINGS[t] for t in selected_types]
    filtered_df = df[pd.concat(type_conditions, axis=1).any(axis=1)].copy()
    
    print(f"Filtered to {len(filtered_df)} results for types: {', '.join(selected_types)}")
    
    # Determine calculation method for each LLM
    print(f"\nDetermining calculation methods for each LLM...")
    method_per_llm = determine_calculation_method(df, selected_types)
    
    # Group by model and prompt_id to aggregate multiple runs
    grouped = filtered_df.groupby(['model_name', 'prompt_id'])
    
    for (model_name, prompt_id), group in grouped:
        # Extract metrics for this model/prompt combination
        completion_tokens = group['tokens_completion'].tolist()
        tokens_reasoning = group['tokens_reasoning'].tolist()
        tokens_output = group['tokens_output'].tolist()
        char_output = group['char_output'].tolist()
        char_reasoning = group['char_reasoning'].tolist()
        char_completion = group['char_completion'].tolist()
        success_rates = group['success_rate'].tolist()
        
        # Skip if no valid completion token data
        if not completion_tokens or not all(isinstance(t, (int, float)) and t > 0 for t in completion_tokens):
            continue
        
        # Get model configuration from the CSV data
        first_row = group.iloc[0]
        calculation_method = method_per_llm.get(model_name, "chars")
        
        metrics = {
            'llm': model_name,
            'prompt_id': prompt_id,
            'type': first_row['type'],
            'completion_tokens': completion_tokens,
            'tokens_reasoning': tokens_reasoning,
            'tokens_output': tokens_output,
            'char_output': char_output,
            'char_reasoning': char_reasoning,
            'char_completion': char_completion,
            'success_rates': success_rates,
            'full_cot': first_row['full_cot'],
            'open_weights': first_row['open_weights'],
            'total_responses': len(completion_tokens),
            'calculation_method': calculation_method
        }
        
        # Calculate reasoning ratios using the determined method
        metrics['reasoning_ratios'] = calculate_reasoning_ratios(metrics, calculation_method)
        
        all_metrics.append(metrics)
    
    print(f"\nSummary:")
    print(f"Total processed results: {len(all_metrics)}")
    print(f"Unique models: {len(set(m['llm'] for m in all_metrics))}")
    print(f"Unique prompts: {len(set(m['prompt_id'] for m in all_metrics))}")
    
    return all_metrics

def get_analysis_title_suffix(selected_types: Set[str]) -> str:
    """Get appropriate title suffix based on selected types."""
    if len(selected_types) == 1:
        return TYPE_DISPLAY_NAMES[list(selected_types)[0]]
    elif len(selected_types) == len(TYPE_MAPPINGS):
        return "All Prompt Types"
    else:
        type_names = [TYPE_DISPLAY_NAMES[t] for t in sorted(selected_types)]
        return " + ".join(type_names)

def main():
    """Main function to analyze token economy for selected prompt types."""
    # Build preset descriptions for help text
    preset_help = "\n".join([f"  {name}: {config['description']}" for name, config in PRESETS.items()])
    
    parser = argparse.ArgumentParser(
        description="Analyze token economy for prompt types with specific visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This script creates ten specific visualizations:
1. Heatmap: Success rates for each prompt and LLM
2. Heatmap: Relative completion tokens (normalized by reference models)
3. Bar chart: Average relative completion tokens across all prompts
4. Heatmap: Correct/incorrect completion token length ratio
5. Heatmap: Reasoning ratios for each prompt and LLM
6. Stacked bar chart: Token composition (reasoning + output)
7. Stacked bar chart: Output-first composition (output + reasoning)
8. Heatmap: Reasoning tokens by prompt and LLM
9. Heatmap: Output tokens by prompt and LLM
10. Stacked bar chart: Token composition by prompt (averaged across LLMs)

Available presets:
{preset_help}

Examples:
    python analyze_prompts.py --preset all
    python analyze_prompts.py --preset knowledge
    python analyze_prompts.py --preset reasoning
    python analyze_prompts.py --types knowledge,math --output-dir figures/eco_knowledge_math
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preset', 
                       choices=list(PRESETS.keys()),
                       help='Use a predefined configuration preset')
    group.add_argument('--types', 
                       help='Comma-separated list of prompt types to analyze. Options: knowledge, logic_puzzle, math')
    
    parser.add_argument('--csv-file', default='evaluation_stats.csv',
                       help='Path to evaluation statistics CSV file (default: evaluation_stats.csv)')
    parser.add_argument('--output-dir',
                       help='Output directory for generated files (overrides preset default)')
    parser.add_argument('--figsize', default='10,6',
                       help='Figure size as width,height in inches (default: 10,6)')
    parser.add_argument('--reference-models', default='claude-4-sonnet-0522-thinking,o4-mini-high-long,gemini-2.5-flash-high',
                       help='Comma-separated list of reference models for token normalization (default: claude-4-sonnet-0522-thinking,o4-mini-high-long,gemini-2.5-flash-high)')
    
    args = parser.parse_args()
    
    # Handle preset vs types logic
    if args.preset:
        preset_config = PRESETS[args.preset]
        selected_types = set(preset_config['types'])
        output_dir_default = preset_config['output_dir']
        print(f"Using preset '{args.preset}': {preset_config['description']}")
    elif args.types:
        # Parse and validate types
        try:
            selected_types = set(t.strip() for t in args.types.split(','))
            invalid_types = selected_types - set(TYPE_MAPPINGS.keys())
            if invalid_types:
                print(f"Error: Invalid types: {invalid_types}")
                print(f"Valid types: {list(TYPE_MAPPINGS.keys())}")
                sys.exit(1)
        except Exception as e:
            print(f"Error parsing types: {e}")
            sys.exit(1)
        output_dir_default = 'figures/eco_custom'
    else:
        # Default to 'all' preset if nothing specified
        preset_config = PRESETS['all']
        selected_types = set(preset_config['types'])
        output_dir_default = preset_config['output_dir']
        print(f"No preset or types specified, using default preset 'all': {preset_config['description']}")
    
    # Parse figure size
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print("Error: Invalid figsize format. Use 'width,height' (e.g., '10,6')")
        sys.exit(1)
    
    # Parse reference models
    reference_models = [model.strip() for model in args.reference_models.split(',')]
    print(f"Reference models for token normalization: {reference_models}")
    
    # Validate inputs
    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir if args.output_dir else output_dir_default)
    
    # Validate paths exist
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        sys.exit(1)
    
    create_output_directory(output_dir)
    
    print(f"Starting prompt analysis for types: {', '.join(sorted(selected_types))}")
    print(f"CSV file: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Figure size: {figsize[0]}x{figsize[1]} inches")
    print()
    
    # Load evaluation statistics
    df = load_evaluation_stats(csv_path)
    
    # Process data and extract metrics
    print(f"\nExtracting metrics for selected prompt types...")
    metrics_data = load_and_process_data(df, selected_types)
    
    if not metrics_data:
        print("No metrics data extracted. Exiting.")
        sys.exit(1)
    
    # Get title suffix for analysis
    title_suffix = get_analysis_title_suffix(selected_types)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Heatmap: Success rates for each prompt and LLM
    success_heatmap_path = output_dir / "success_rate_heatmap.png"
    create_success_rate_heatmap(metrics_data, success_heatmap_path, figsize, title_suffix)
    
    # 2. Heatmap: Relative completion tokens (normalized by reference models)
    relative_tokens_heatmap_path = output_dir / "relative_completion_tokens_heatmap.png"
    create_relative_completion_tokens_heatmap(metrics_data, relative_tokens_heatmap_path, figsize, title_suffix, reference_models)
    
    # 3. Bar chart: Average relative completion tokens across all prompts
    avg_relative_tokens_chart_path = output_dir / "average_relative_completion_tokens_chart.png"
    create_average_relative_tokens_chart(metrics_data, avg_relative_tokens_chart_path, figsize, title_suffix, reference_models)
    
    # 4. Heatmap: Correct/incorrect completion token length ratio
    correct_incorrect_ratio_heatmap_path = output_dir / "correct_incorrect_ratio_heatmap.png"
    create_correct_incorrect_ratio_heatmap(metrics_data, correct_incorrect_ratio_heatmap_path, figsize, title_suffix)
    
    # 5. Heatmap: Reasoning ratios for each prompt and LLM
    heatmap_path = output_dir / "reasoning_ratio_heatmap.png"
    create_reasoning_heatmap(metrics_data, heatmap_path, figsize, title_suffix)
    
    # 6. Stacked bar chart: Token composition (reasoning + output)
    stacked_chart_path = output_dir / "token_composition_stacked_chart.png"
    create_stacked_token_chart(metrics_data, stacked_chart_path, figsize)
    
    # 7. Stacked bar chart: Output-first composition (output + reasoning)
    output_first_chart_path = output_dir / "token_composition_output_first_chart.png"
    create_output_first_stacked_chart(metrics_data, output_first_chart_path, figsize)
    
    # 8. Heatmap: Reasoning tokens by prompt and LLM
    reasoning_tokens_heatmap_path = output_dir / "reasoning_tokens_heatmap.png"
    create_reasoning_tokens_heatmap(metrics_data, reasoning_tokens_heatmap_path, figsize, title_suffix)
    
    # 9. Heatmap: Output tokens by prompt and LLM
    output_tokens_heatmap_path = output_dir / "output_tokens_heatmap.png"
    create_output_tokens_heatmap(metrics_data, output_tokens_heatmap_path, figsize, title_suffix)
    
    # 10. Stacked bar chart: Token composition by prompt (averaged across LLMs)
    prompt_composition_chart_path = output_dir / "token_composition_by_prompt_chart.png"
    create_prompt_composition_chart(metrics_data, prompt_composition_chart_path, figsize, title_suffix)
    
    # Save summary data
    types_suffix = "_".join(sorted(selected_types))
    summary_path = output_dir / f"{types_suffix}_analysis_summary.csv"
    save_summary_data(metrics_data, summary_path)
    
    print(f"\nCompleted! Generated {title_suffix} prompt analysis in {output_dir}")
    print(f"  - 10 visualization files")
    print(f"  - 1 summary CSV file")
    print(f"  - Total data points analyzed: {len(metrics_data)}")

def create_success_rate_heatmap(metrics_data: List[Dict[str, Any]], output_path: Path, 
                               figsize: Tuple[float, float], title_suffix: str) -> None:
    """Create heatmap showing success rates for each prompt and LLM."""
    
    # Organize data by prompt and model
    prompt_model_success = defaultdict(dict)
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        llm = entry['llm']
        avg_success_rate = np.mean(entry['success_rates']) if entry['success_rates'] else 0
        prompt_model_success[prompt_id][llm] = avg_success_rate
    
    # Get all unique prompts and models
    all_prompts = sorted(prompt_model_success.keys())
    all_models = sorted(set(llm for prompt_data in prompt_model_success.values() for llm in prompt_data.keys()))
    
    # Create matrix for heatmap
    matrix = []
    for prompt in all_prompts:
        row = []
        for model in all_models:
            success_rate = prompt_model_success[prompt].get(model, 0)
            row.append(success_rate)
        matrix.append(row)
    
    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(max(12, len(all_models) * 0.8), max(6, len(all_prompts) * 0.8)))
    
    # Create heatmap with green color scheme (higher success = darker green)
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_models)))
    ax.set_yticks(range(len(all_prompts)))
    ax.set_xticklabels([model[:20] for model in all_models], rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(all_prompts, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i, prompt in enumerate(all_prompts):
        for j, model in enumerate(all_models):
            success_rate = matrix[i][j]
            text_color = 'white' if success_rate < 0.5 else 'black'
            ax.text(j, i, f'{success_rate:.2f}', ha='center', va='center', 
                   color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_title(f'Success Rates by {title_suffix} Prompt and LLM\n(Green=High Success, Red=Low Success)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title_suffix} Prompts', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created success rate heatmap: {output_path.name}")

def create_relative_completion_tokens_heatmap(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                            figsize: Tuple[float, float], title_suffix: str, 
                                            reference_models: List[str]) -> None:
    """Create heatmap showing relative completion tokens normalized by reference models."""
    
    # First, organize data by prompt and model to get average completion tokens
    prompt_model_tokens = defaultdict(dict)
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        llm = entry['llm']
        avg_completion_tokens = np.mean(entry['completion_tokens']) if entry['completion_tokens'] else 0
        prompt_model_tokens[prompt_id][llm] = avg_completion_tokens
    
    # Get all unique prompts and models
    all_prompts = sorted(prompt_model_tokens.keys())
    all_models = sorted(set(llm for prompt_data in prompt_model_tokens.values() for llm in prompt_data.keys()))
    
    # Calculate normalization factors per prompt based on reference models
    prompt_normalization_factors = {}
    
    for prompt in all_prompts:
        reference_tokens = []
        matched_models = []
        
        for ref_model in reference_models:
            # Try exact match first, then partial match (case insensitive)
            ref_token_value = None
            matched_model = None
            
            # Try exact match first
            for model in prompt_model_tokens[prompt]:
                if model.lower() == ref_model.lower():
                    ref_token_value = prompt_model_tokens[prompt][model]
                    matched_model = model
                    break
            
            # If no exact match, try partial match
            if ref_token_value is None:
                for model in prompt_model_tokens[prompt]:
                    if ref_model.lower() in model.lower() or model.lower() in ref_model.lower():
                        ref_token_value = prompt_model_tokens[prompt][model]
                        matched_model = model
                        break
            
            if ref_token_value is not None and ref_token_value > 0:
                reference_tokens.append(ref_token_value)
                matched_models.append(matched_model)
        
        if reference_tokens:
            prompt_normalization_factors[prompt] = np.mean(reference_tokens)
            print(f"Prompt '{prompt}': matched {len(reference_tokens)} reference models, normalization factor = {prompt_normalization_factors[prompt]:.1f}")
        else:
            # Fallback: use mean of all models for this prompt
            all_tokens_for_prompt = [tokens for tokens in prompt_model_tokens[prompt].values() if tokens > 0]
            if all_tokens_for_prompt:
                prompt_normalization_factors[prompt] = np.mean(all_tokens_for_prompt)
                print(f"Prompt '{prompt}': no reference models found, using mean of all models = {prompt_normalization_factors[prompt]:.1f}")
            else:
                prompt_normalization_factors[prompt] = 1.0
                print(f"Prompt '{prompt}': no valid token data, using normalization factor = 1.0")
    
    # Create normalized matrix for heatmap
    matrix = []
    for prompt in all_prompts:
        row = []
        normalization_factor = prompt_normalization_factors[prompt]
        
        for model in all_models:
            absolute_tokens = prompt_model_tokens[prompt].get(model, 0)
            relative_tokens = absolute_tokens / normalization_factor if normalization_factor > 0 else 0
            row.append(relative_tokens)
        matrix.append(row)
    
    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(max(12, len(all_models) * 0.8), max(6, len(all_prompts) * 0.8)))
    
    # Create heatmap with diverging colormap centered at 1.0 (reference models)
    vmax = max(max(row) for row in matrix) if matrix and any(matrix) else 2.0
    vmin = 0
    im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=vmin, vmax=min(vmax, 3.0))
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_models)))
    ax.set_yticks(range(len(all_prompts)))
    ax.set_xticklabels([model[:20] for model in all_models], rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(all_prompts, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Completion Tokens (1.0 = Reference Models Mean)', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i, prompt in enumerate(all_prompts):
        for j, model in enumerate(all_models):
            relative_tokens = matrix[i][j]
            # Color text based on background intensity
            text_color = 'white' if relative_tokens > 1.5 else 'black'
            ax.text(j, i, f'{relative_tokens:.2f}', ha='center', va='center', 
                   color=text_color, fontsize=8, fontweight='bold')
    
    # Format reference models for title
    ref_models_str = ', '.join(reference_models[:3])  # Show first 3 reference models
    if len(reference_models) > 3:
        ref_models_str += f' (+{len(reference_models)-3} more)'
    
    ax.set_title(f'Relative Completion Tokens by {title_suffix} Prompt and LLM\n(Normalized by: {ref_models_str})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title_suffix} Prompts', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created relative completion tokens heatmap: {output_path.name}")

def create_average_relative_tokens_chart(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                        figsize: Tuple[float, float], title_suffix: str, 
                                        reference_models: List[str]) -> None:
    """Create bar chart showing average relative completion tokens across all prompts."""
    
    # First, organize data by prompt and model to get average completion tokens
    prompt_model_tokens = defaultdict(dict)
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        llm = entry['llm']
        avg_completion_tokens = np.mean(entry['completion_tokens']) if entry['completion_tokens'] else 0
        prompt_model_tokens[prompt_id][llm] = avg_completion_tokens
    
    # Get all unique prompts and models
    all_prompts = sorted(prompt_model_tokens.keys())
    all_models = sorted(set(llm for prompt_data in prompt_model_tokens.values() for llm in prompt_data.keys()))
    
    # Calculate normalization factors per prompt based on reference models
    prompt_normalization_factors = {}
    
    for prompt in all_prompts:
        reference_tokens = []
        
        for ref_model in reference_models:
            # Try exact match first, then partial match (case insensitive)
            ref_token_value = None
            
            # Try exact match first
            for model in prompt_model_tokens[prompt]:
                if model.lower() == ref_model.lower():
                    ref_token_value = prompt_model_tokens[prompt][model]
                    break
            
            # If no exact match, try partial match
            if ref_token_value is None:
                for model in prompt_model_tokens[prompt]:
                    if ref_model.lower() in model.lower() or model.lower() in ref_model.lower():
                        ref_token_value = prompt_model_tokens[prompt][model]
                        break
            
            if ref_token_value is not None and ref_token_value > 0:
                reference_tokens.append(ref_token_value)
        
        if reference_tokens:
            prompt_normalization_factors[prompt] = np.mean(reference_tokens)
        else:
            # Fallback: use mean of all models for this prompt
            all_tokens_for_prompt = [tokens for tokens in prompt_model_tokens[prompt].values() if tokens > 0]
            if all_tokens_for_prompt:
                prompt_normalization_factors[prompt] = np.mean(all_tokens_for_prompt)
            else:
                prompt_normalization_factors[prompt] = 1.0
    
    # Calculate average relative tokens for each model across all prompts
    model_avg_relative_tokens = {}
    
    for model in all_models:
        relative_tokens_per_prompt = []
        
        for prompt in all_prompts:
            absolute_tokens = prompt_model_tokens[prompt].get(model, 0)
            normalization_factor = prompt_normalization_factors[prompt]
            relative_tokens = absolute_tokens / normalization_factor if normalization_factor > 0 else 0
            if relative_tokens > 0:  # Only include prompts where this model has data
                relative_tokens_per_prompt.append(relative_tokens)
        
        if relative_tokens_per_prompt:
            model_avg_relative_tokens[model] = np.mean(relative_tokens_per_prompt)
        else:
            model_avg_relative_tokens[model] = 0
    
    # Sort models by average relative tokens (ascending - most efficient first)
    sorted_models = sorted(model_avg_relative_tokens.keys(), key=lambda x: model_avg_relative_tokens[x])
    
    # Prepare data for plotting
    relative_values = [model_avg_relative_tokens[model] for model in sorted_models]
    
    # Get model weight information from metrics_data
    model_open_weights = {}
    for entry in metrics_data:
        model_open_weights[entry['llm']] = entry['open_weights']
    
    # Determine colors based on open/closed weights and reference status
    colors = []
    for model in sorted_models:
        is_reference = any(ref_model.lower() in model.lower() or model.lower() in ref_model.lower() 
                          for ref_model in reference_models)
        is_open_weights = model_open_weights.get(model, False)
        
        if is_open_weights:
            # Open weights = blue family
            if is_reference:
                colors.append('darkblue')  # Reference open weight models
            else:
                colors.append('lightblue')  # Non-reference open weight models
        else:
            # Closed weights = red family
            if is_reference:
                colors.append('darkred')  # Reference closed weight models
            else:
                colors.append('lightcoral')  # Non-reference closed weight models
    
    # Create figure
    fig_width = max(figsize[0], len(sorted_models) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))
    
    # Create bar chart
    x_pos = np.arange(len(sorted_models))
    bars = ax.bar(x_pos, relative_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add horizontal line at y=1.0 (reference level)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Reference Level (1.0)')
    
    # Add value labels on bars
    for i, (model, value) in enumerate(zip(sorted_models, relative_values)):
        ax.text(i, value + max(relative_values) * 0.01, f'{value:.2f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Customize chart
    ax.set_title(f'Average Relative Completion Tokens Across All {title_suffix} Prompts\n(Lower is More Token-Efficient)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Relative Completion Tokens', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model[:15] for model in sorted_models], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(relative_values) * 1.1)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', alpha=0.8, label='Closed Weight (Reference)'),
        Patch(facecolor='lightcoral', alpha=0.8, label='Closed Weight (Other)'),
        Patch(facecolor='darkblue', alpha=0.8, label='Open Weight (Reference)'),
        Patch(facecolor='lightblue', alpha=0.8, label='Open Weight (Other)'),
        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.7, label='Reference Level (1.0)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created average relative tokens chart: {output_path.name}")

def create_correct_incorrect_ratio_heatmap(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                          figsize: Tuple[float, float], title_suffix: str) -> None:
    """Create heatmap showing ratio of correct to incorrect completion token lengths."""
    
    # Organize data by prompt and model
    prompt_model_ratios = defaultdict(dict)
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        llm = entry['llm']
        
        # Collect completion tokens for correct and incorrect responses
        correct_tokens = []
        incorrect_tokens = []
        
        for i in range(len(entry['completion_tokens'])):
            completion_tokens = entry['completion_tokens'][i]
            
            # Get success rate for this run
            if (i < len(entry['success_rates']) and 
                isinstance(entry['success_rates'][i], (int, float))):
                success_rate = entry['success_rates'][i]
                is_correct = success_rate >= 1.0
            else:
                is_correct = False  # Default to incorrect if no success rate data
            
            # Add to appropriate list
            if is_correct:
                correct_tokens.append(completion_tokens)
            else:
                incorrect_tokens.append(completion_tokens)
        
        # Calculate ratio only if we have both correct and incorrect responses
        if correct_tokens and incorrect_tokens:
            avg_correct = np.mean(correct_tokens)
            avg_incorrect = np.mean(incorrect_tokens)
            
            # Calculate ratio: correct / incorrect
            # Values > 1 mean correct responses are longer
            # Values < 1 mean correct responses are shorter
            ratio = avg_correct / avg_incorrect
            prompt_model_ratios[prompt_id][llm] = ratio
        # Leave empty (NaN) if we don't have both types of responses
    
    # Get all unique prompts and models
    all_prompts = sorted(prompt_model_ratios.keys())
    all_models = sorted(set(llm for prompt_data in prompt_model_ratios.values() for llm in prompt_data.keys()))
    
    if not all_prompts or not all_models:
        print("No valid data for correct/incorrect ratio heatmap")
        return
    
    # Create matrix for heatmap
    matrix = []
    for prompt in all_prompts:
        row = []
        for model in all_models:
            ratio = prompt_model_ratios[prompt].get(model, np.nan)
            row.append(ratio)
        matrix.append(row)
    
    # Convert to numpy array for easier handling
    matrix = np.array(matrix)
    
    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(max(12, len(all_models) * 0.8), max(6, len(all_prompts) * 0.8)))
    
    # Create heatmap with diverging colormap centered at 1.0
    # Use a mask for NaN values to leave them white/empty
    masked_matrix = np.ma.masked_where(np.isnan(matrix), matrix)
    
    # Set color scale centered at 1.0 (equal lengths)
    vmin = max(0.1, np.nanmin(matrix)) if not np.all(np.isnan(matrix)) else 0.5
    vmax = min(3.0, np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 2.0
    
    im = ax.imshow(masked_matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_models)))
    ax.set_yticks(range(len(all_prompts)))
    ax.set_xticklabels([model[:20] for model in all_models], rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(all_prompts, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correct/Incorrect Completion Token Ratio\n(>1: Correct longer, <1: Incorrect longer)', 
                   rotation=270, labelpad=25, fontweight='bold')
    
    # Add horizontal line at 1.0 on colorbar to show equal point
    cbar.ax.axhline(y=1.0, color='black', linewidth=2, alpha=0.8)
    
    # Add text annotations only for non-NaN values
    for i, prompt in enumerate(all_prompts):
        for j, model in enumerate(all_models):
            ratio = matrix[i][j]
            if not np.isnan(ratio):
                # Color text based on background intensity
                text_color = 'white' if ratio > 1.5 or ratio < 0.67 else 'black'
                ax.text(j, i, f'{ratio:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_title(f'Correct vs Incorrect Completion Token Length Ratios by {title_suffix} Prompt and LLM\n(Empty cells: only correct OR only incorrect responses)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title_suffix} Prompts', fontsize=12, fontweight='bold')
    
    # Count how many prompt/model combinations have both correct and incorrect responses
    valid_cells = np.sum(~np.isnan(matrix))
    total_cells = len(all_prompts) * len(all_models)
    
    # Add summary text
    ax.text(0.02, 0.98, f'Valid ratios: {valid_cells}/{total_cells} cells\n(Both correct & incorrect responses)', 
           transform=ax.transAxes, fontsize=9, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created correct/incorrect ratio heatmap: {output_path.name}")
    print(f"  - {valid_cells} out of {total_cells} cells have both correct and incorrect responses")

def create_reasoning_heatmap(metrics_data: List[Dict[str, Any]], output_path: Path, 
                            figsize: Tuple[float, float], title_suffix: str) -> None:
    """Create heatmap showing reasoning ratios for each prompt and LLM."""
    
    # Organize data by prompt and model
    prompt_model_ratios = defaultdict(dict)
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        llm = entry['llm']
        avg_ratio = np.mean(entry['reasoning_ratios']) if entry['reasoning_ratios'] else 0
        prompt_model_ratios[prompt_id][llm] = avg_ratio
    
    # Get all unique prompts and models
    all_prompts = sorted(prompt_model_ratios.keys())
    all_models = sorted(set(llm for prompt_data in prompt_model_ratios.values() for llm in prompt_data.keys()))
    
    # Create matrix for heatmap
    matrix = []
    for prompt in all_prompts:
        row = []
        for model in all_models:
            ratio = prompt_model_ratios[prompt].get(model, 0)
            row.append(ratio)
        matrix.append(row)
    
    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(max(12, len(all_models) * 0.8), max(6, len(all_prompts) * 0.8)))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_models)))
    ax.set_yticks(range(len(all_prompts)))
    ax.set_xticklabels([model[:20] for model in all_models], rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(all_prompts, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Reasoning Character Ratio', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i, prompt in enumerate(all_prompts):
        for j, model in enumerate(all_models):
            ratio = matrix[i][j]
            text_color = 'white' if ratio > 0.5 else 'black'
            ax.text(j, i, f'{ratio:.2f}', ha='center', va='center', 
                   color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_title(f'Reasoning Character Ratios by {title_suffix} Prompt and LLM\n(Red=High Reasoning, Blue=Low Reasoning)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title_suffix} Prompts', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created reasoning ratio heatmap: {output_path.name}")

def create_reasoning_tokens_heatmap(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                   figsize: Tuple[float, float], title_suffix: str) -> None:
    """Create heatmap showing reasoning token counts for each prompt and LLM."""
    
    # Organize data by prompt and model
    prompt_model_tokens = defaultdict(dict)
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        llm = entry['llm']
        
        # Calculate average reasoning tokens based on method
        reasoning_tokens = []
        for i in range(len(entry['completion_tokens'])):
            completion_tokens = entry['completion_tokens'][i]
            
            if entry['calculation_method'] == 'tokens':
                # Use direct token values
                if (i < len(entry['tokens_reasoning']) and 
                    entry['tokens_reasoning'] and 
                    isinstance(entry['tokens_reasoning'][i], (int, float))):
                    reasoning_tokens.append(entry['tokens_reasoning'][i])
                else:
                    reasoning_tokens.append(0)
            else:
                # Calculate from character ratios
                if (i < len(entry['char_reasoning']) and 
                    i < len(entry['char_completion']) and
                    entry['char_reasoning'] and 
                    entry['char_completion'] and
                    isinstance(entry['char_reasoning'][i], (int, float)) and
                    isinstance(entry['char_completion'][i], (int, float))):
                    char_reasoning = entry['char_reasoning'][i]
                    char_completion = entry['char_completion'][i]
                    if char_completion > 0:
                        reasoning_ratio = char_reasoning / char_completion
                        reasoning_tokens.append(completion_tokens * reasoning_ratio)
                    else:
                        reasoning_tokens.append(0)
                else:
                    reasoning_tokens.append(0)
        
        avg_reasoning_tokens = np.mean(reasoning_tokens) if reasoning_tokens else 0
        prompt_model_tokens[prompt_id][llm] = avg_reasoning_tokens
    
    # Get all unique prompts and models
    all_prompts = sorted(prompt_model_tokens.keys())
    all_models = sorted(set(llm for prompt_data in prompt_model_tokens.values() for llm in prompt_data.keys()))
    
    # Create matrix for heatmap
    matrix = []
    for prompt in all_prompts:
        row = []
        for model in all_models:
            tokens = prompt_model_tokens[prompt].get(model, 0)
            row.append(tokens)
        matrix.append(row)
    
    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(max(12, len(all_models) * 0.8), max(6, len(all_prompts) * 0.8)))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0)
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_models)))
    ax.set_yticks(range(len(all_prompts)))
    ax.set_xticklabels([model[:20] for model in all_models], rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(all_prompts, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Reasoning Tokens', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i, prompt in enumerate(all_prompts):
        for j, model in enumerate(all_models):
            tokens = matrix[i][j]
            text_color = 'white' if tokens > np.max(matrix) * 0.7 else 'black'
            ax.text(j, i, f'{tokens:.0f}', ha='center', va='center', 
                   color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_title(f'Average Reasoning Tokens by {title_suffix} Prompt and LLM', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title_suffix} Prompts', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created reasoning tokens heatmap: {output_path.name}")

def create_output_tokens_heatmap(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                figsize: Tuple[float, float], title_suffix: str) -> None:
    """Create heatmap showing output token counts for each prompt and LLM."""
    
    # Organize data by prompt and model
    prompt_model_tokens = defaultdict(dict)
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        llm = entry['llm']
        
        # Calculate average output tokens based on method
        output_tokens = []
        for i in range(len(entry['completion_tokens'])):
            completion_tokens = entry['completion_tokens'][i]
            
            if entry['calculation_method'] == 'tokens':
                # Use direct token values
                if (i < len(entry['tokens_output']) and 
                    entry['tokens_output'] and 
                    isinstance(entry['tokens_output'][i], (int, float))):
                    output_tokens.append(entry['tokens_output'][i])
                else:
                    output_tokens.append(completion_tokens)
            else:
                # Calculate from character ratios (output = completion - reasoning)
                if (i < len(entry['char_reasoning']) and 
                    i < len(entry['char_completion']) and
                    entry['char_reasoning'] and 
                    entry['char_completion'] and
                    isinstance(entry['char_reasoning'][i], (int, float)) and
                    isinstance(entry['char_completion'][i], (int, float))):
                    char_reasoning = entry['char_reasoning'][i]
                    char_completion = entry['char_completion'][i]
                    if char_completion > 0:
                        reasoning_ratio = char_reasoning / char_completion
                        reasoning_tokens = completion_tokens * reasoning_ratio
                        output_tokens.append(completion_tokens - reasoning_tokens)
                    else:
                        output_tokens.append(completion_tokens)
                else:
                    output_tokens.append(completion_tokens)
        
        avg_output_tokens = np.mean(output_tokens) if output_tokens else 0
        prompt_model_tokens[prompt_id][llm] = avg_output_tokens
    
    # Get all unique prompts and models
    all_prompts = sorted(prompt_model_tokens.keys())
    all_models = sorted(set(llm for prompt_data in prompt_model_tokens.values() for llm in prompt_data.keys()))
    
    # Create matrix for heatmap
    matrix = []
    for prompt in all_prompts:
        row = []
        for model in all_models:
            tokens = prompt_model_tokens[prompt].get(model, 0)
            row.append(tokens)
        matrix.append(row)
    
    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(max(12, len(all_models) * 0.8), max(6, len(all_prompts) * 0.8)))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='Greens', aspect='auto', vmin=0)
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_models)))
    ax.set_yticks(range(len(all_prompts)))
    ax.set_xticklabels([model[:20] for model in all_models], rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(all_prompts, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Output Tokens', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i, prompt in enumerate(all_prompts):
        for j, model in enumerate(all_models):
            tokens = matrix[i][j]
            text_color = 'white' if tokens > np.max(matrix) * 0.7 else 'black'
            ax.text(j, i, f'{tokens:.0f}', ha='center', va='center', 
                   color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_title(f'Average Output Tokens by {title_suffix} Prompt and LLM', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title_suffix} Prompts', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created output tokens heatmap: {output_path.name}")

def create_stacked_token_chart(metrics_data: List[Dict[str, Any]], output_path: Path, 
                              figsize: Tuple[float, float]) -> None:
    """Create stacked bar chart showing reasoning and output token composition."""
    
    # Aggregate data by model
    model_data = defaultdict(lambda: {
        'completion_tokens': [],
        'reasoning_tokens': [],
        'output_tokens': [],
        'open_weights': False,
        'calculation_method': 'chars'
    })
    
    for entry in metrics_data:
        llm = entry['llm']
        model_data[llm]['open_weights'] = entry['open_weights']
        model_data[llm]['calculation_method'] = entry['calculation_method']
        
        # Calculate reasoning and output tokens based on method
        for i in range(len(entry['completion_tokens'])):
            completion_tokens = entry['completion_tokens'][i]
            
            if entry['calculation_method'] == 'tokens':
                # Use direct token values
                if (i < len(entry['tokens_reasoning']) and 
                    i < len(entry['tokens_output']) and
                    entry['tokens_reasoning'] and 
                    entry['tokens_output']):
                    reasoning_tokens = entry['tokens_reasoning'][i]
                    output_tokens = entry['tokens_output'][i]
                else:
                    # Fallback if token data missing
                    reasoning_tokens = 0
                    output_tokens = completion_tokens
            else:
                # Calculate from character ratios
                if (i < len(entry['char_reasoning']) and 
                    i < len(entry['char_completion']) and
                    entry['char_reasoning'] and 
                    entry['char_completion']):
                    char_reasoning = entry['char_reasoning'][i]
                    char_completion = entry['char_completion'][i]
                    if char_completion > 0:
                        reasoning_ratio = char_reasoning / char_completion
                        reasoning_tokens = completion_tokens * reasoning_ratio
                        output_tokens = completion_tokens - reasoning_tokens
                    else:
                        reasoning_tokens = 0
                        output_tokens = completion_tokens
                else:
                    # Fallback if character data missing
                    reasoning_tokens = 0
                    output_tokens = completion_tokens
            
            model_data[llm]['completion_tokens'].append(completion_tokens)
            model_data[llm]['reasoning_tokens'].append(reasoning_tokens)
            model_data[llm]['output_tokens'].append(output_tokens)
    
    # Calculate averages and sort by total completion tokens
    model_averages = {}
    for llm, data in model_data.items():
        if data['completion_tokens']:
            avg_completion = np.mean(data['completion_tokens'])
            avg_reasoning = np.mean(data['reasoning_tokens'])
            avg_output = np.mean(data['output_tokens'])
            model_averages[llm] = {
                'completion': avg_completion,
                'reasoning': avg_reasoning,
                'output': avg_output,
                'open_weights': data['open_weights']
            }
    
    # Sort models by completion tokens
    sorted_models = sorted(model_averages.keys(), key=lambda x: model_averages[x]['completion'])
    
    # Prepare data for plotting
    reasoning_values = [model_averages[model]['reasoning'] for model in sorted_models]
    output_values = [model_averages[model]['output'] for model in sorted_models]
    completion_values = [model_averages[model]['completion'] for model in sorted_models]
    
    # Colors based on open/closed weights
    colors_reasoning = []
    colors_output = []
    for model in sorted_models:
        if model_averages[model]['open_weights']:
            colors_reasoning.append('darkblue')
            colors_output.append('lightblue')
        else:
            colors_reasoning.append('darkred')
            colors_output.append('lightcoral')
    
    # Create figure
    fig_width = max(figsize[0], len(sorted_models) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))
    
    # Create stacked bars
    x_pos = np.arange(len(sorted_models))
    bars_reasoning = ax.bar(x_pos, reasoning_values, color=colors_reasoning, alpha=0.8, label='Reasoning Tokens')
    bars_output = ax.bar(x_pos, output_values, bottom=reasoning_values, color=colors_output, alpha=0.8, label='Output Tokens')
    
    # Add completion token labels on top
    for i, (model, completion) in enumerate(zip(sorted_models, completion_values)):
        ax.text(i, completion + max(completion_values) * 0.01, f'{completion:.0f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Customize chart
    ax.set_title('Token Composition: Reasoning vs Output Tokens by Model', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Tokens', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model[:15] for model in sorted_models], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create custom legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', alpha=0.8, label='Closed Weight (Reasoning)'),
        Patch(facecolor='lightcoral', alpha=0.8, label='Closed Weight (Output)'),
        Patch(facecolor='darkblue', alpha=0.8, label='Open Weight (Reasoning)'),
        Patch(facecolor='lightblue', alpha=0.8, label='Open Weight (Output)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created stacked token composition chart: {output_path.name}")

def create_output_first_stacked_chart(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                     figsize: Tuple[float, float]) -> None:
    """Create stacked bar chart with output tokens at bottom, reasoning at top, sorted by output tokens."""
    
    # Aggregate data by model (reuse logic from original stacked chart)
    model_data = defaultdict(lambda: {
        'completion_tokens': [],
        'reasoning_tokens': [],
        'output_tokens': [],
        'open_weights': False,
        'calculation_method': 'chars'
    })
    
    for entry in metrics_data:
        llm = entry['llm']
        model_data[llm]['open_weights'] = entry['open_weights']
        model_data[llm]['calculation_method'] = entry['calculation_method']
        
        # Calculate reasoning and output tokens based on method
        for i in range(len(entry['completion_tokens'])):
            completion_tokens = entry['completion_tokens'][i]
            
            if entry['calculation_method'] == 'tokens':
                # Use direct token values
                if (i < len(entry['tokens_reasoning']) and 
                    i < len(entry['tokens_output']) and
                    entry['tokens_reasoning'] and 
                    entry['tokens_output']):
                    reasoning_tokens = entry['tokens_reasoning'][i]
                    output_tokens = entry['tokens_output'][i]
                else:
                    # Fallback if token data missing
                    reasoning_tokens = 0
                    output_tokens = completion_tokens
            else:
                # Calculate from character ratios
                if (i < len(entry['char_reasoning']) and 
                    i < len(entry['char_completion']) and
                    entry['char_reasoning'] and 
                    entry['char_completion']):
                    char_reasoning = entry['char_reasoning'][i]
                    char_completion = entry['char_completion'][i]
                    if char_completion > 0:
                        reasoning_ratio = char_reasoning / char_completion
                        reasoning_tokens = completion_tokens * reasoning_ratio
                        output_tokens = completion_tokens - reasoning_tokens
                    else:
                        reasoning_tokens = 0
                        output_tokens = completion_tokens
                else:
                    # Fallback if character data missing
                    reasoning_tokens = 0
                    output_tokens = completion_tokens
            
            model_data[llm]['completion_tokens'].append(completion_tokens)
            model_data[llm]['reasoning_tokens'].append(reasoning_tokens)
            model_data[llm]['output_tokens'].append(output_tokens)
    
    # Calculate averages and sort by output tokens
    model_averages = {}
    for llm, data in model_data.items():
        if data['completion_tokens']:
            avg_completion = np.mean(data['completion_tokens'])
            avg_reasoning = np.mean(data['reasoning_tokens'])
            avg_output = np.mean(data['output_tokens'])
            model_averages[llm] = {
                'completion': avg_completion,
                'reasoning': avg_reasoning,
                'output': avg_output,
                'open_weights': data['open_weights']
            }
    
    # Sort models by output tokens (ascending)
    sorted_models = sorted(model_averages.keys(), key=lambda x: model_averages[x]['output'])
    
    # Prepare data for plotting
    output_values = [model_averages[model]['output'] for model in sorted_models]
    reasoning_values = [model_averages[model]['reasoning'] for model in sorted_models]
    completion_values = [model_averages[model]['completion'] for model in sorted_models]
    
    # Colors based on open/closed weights
    colors_output = []
    colors_reasoning = []
    for model in sorted_models:
        if model_averages[model]['open_weights']:
            colors_output.append('lightblue')
            colors_reasoning.append('darkblue')
        else:
            colors_output.append('lightcoral')
            colors_reasoning.append('darkred')
    
    # Create figure
    fig_width = max(figsize[0], len(sorted_models) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))
    
    # Create stacked bars (output at bottom, reasoning on top)
    x_pos = np.arange(len(sorted_models))
    bars_output = ax.bar(x_pos, output_values, color=colors_output, alpha=0.8, label='Output Tokens')
    bars_reasoning = ax.bar(x_pos, reasoning_values, bottom=output_values, color=colors_reasoning, alpha=0.8, label='Reasoning Tokens')
    
    # Add output token labels at baseline (y=0)
    for i, (model, output_tokens) in enumerate(zip(sorted_models, output_values)):
        ax.text(i, 0, f'{output_tokens:.0f}', ha='center', va='bottom', 
               fontweight='bold', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    # Add completion token labels on top
    for i, (model, completion) in enumerate(zip(sorted_models, completion_values)):
        ax.text(i, completion + max(completion_values) * 0.01, f'{completion:.0f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Customize chart
    ax.set_title('Token Composition: Output (Bottom) + Reasoning (Top) Tokens by Model\n(Sorted by Output Tokens)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Tokens', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model[:15] for model in sorted_models], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create custom legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', alpha=0.8, label='Closed Weight (Output)'),
        Patch(facecolor='darkred', alpha=0.8, label='Closed Weight (Reasoning)'),
        Patch(facecolor='lightblue', alpha=0.8, label='Open Weight (Output)'),
        Patch(facecolor='darkblue', alpha=0.8, label='Open Weight (Reasoning)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created output-first stacked token composition chart: {output_path.name}")

def create_prompt_composition_chart(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                   figsize: Tuple[float, float], title_suffix: str) -> None:
    """Create stacked bar chart showing reasoning and output token composition by prompt (averaged across LLMs)."""
    
    # Aggregate data by prompt across all models
    prompt_data = defaultdict(lambda: {
        'completion_tokens': [],
        'reasoning_tokens': [],
        'output_tokens': []
    })
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        
        # Calculate reasoning and output tokens based on method
        for i in range(len(entry['completion_tokens'])):
            completion_tokens = entry['completion_tokens'][i]
            
            if entry['calculation_method'] == 'tokens':
                # Use direct token values
                if (i < len(entry['tokens_reasoning']) and 
                    i < len(entry['tokens_output']) and
                    entry['tokens_reasoning'] and 
                    entry['tokens_output']):
                    reasoning_tokens = entry['tokens_reasoning'][i]
                    output_tokens = entry['tokens_output'][i]
                else:
                    # Fallback if token data missing
                    reasoning_tokens = 0
                    output_tokens = completion_tokens
            else:
                # Calculate from character ratios
                if (i < len(entry['char_reasoning']) and 
                    i < len(entry['char_completion']) and
                    entry['char_reasoning'] and 
                    entry['char_completion']):
                    char_reasoning = entry['char_reasoning'][i]
                    char_completion = entry['char_completion'][i]
                    if char_completion > 0:
                        reasoning_ratio = char_reasoning / char_completion
                        reasoning_tokens = completion_tokens * reasoning_ratio
                        output_tokens = completion_tokens - reasoning_tokens
                    else:
                        reasoning_tokens = 0
                        output_tokens = completion_tokens
                else:
                    # Fallback if character data missing
                    reasoning_tokens = 0
                    output_tokens = completion_tokens
            
            prompt_data[prompt_id]['completion_tokens'].append(completion_tokens)
            prompt_data[prompt_id]['reasoning_tokens'].append(reasoning_tokens)
            prompt_data[prompt_id]['output_tokens'].append(output_tokens)
    
    # Calculate averages and sort by total completion tokens
    prompt_averages = {}
    for prompt_id, data in prompt_data.items():
        if data['completion_tokens']:
            avg_completion = np.mean(data['completion_tokens'])
            avg_reasoning = np.mean(data['reasoning_tokens'])
            avg_output = np.mean(data['output_tokens'])
            prompt_averages[prompt_id] = {
                'completion': avg_completion,
                'reasoning': avg_reasoning,
                'output': avg_output
            }
    
    # Sort prompts by completion tokens
    sorted_prompts = sorted(prompt_averages.keys(), key=lambda x: prompt_averages[x]['completion'])
    
    # Prepare data for plotting
    reasoning_values = [prompt_averages[prompt]['reasoning'] for prompt in sorted_prompts]
    output_values = [prompt_averages[prompt]['output'] for prompt in sorted_prompts]
    completion_values = [prompt_averages[prompt]['completion'] for prompt in sorted_prompts]
    
    # Create figure
    fig_width = max(figsize[0], len(sorted_prompts) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))
    
    # Create stacked bars
    x_pos = np.arange(len(sorted_prompts))
    bars_reasoning = ax.bar(x_pos, reasoning_values, color='steelblue', alpha=0.8, label='Reasoning Tokens')
    bars_output = ax.bar(x_pos, output_values, bottom=reasoning_values, color='lightsteelblue', alpha=0.8, label='Output Tokens')
    
    # Add completion token labels on top
    for i, (prompt, completion) in enumerate(zip(sorted_prompts, completion_values)):
        ax.text(i, completion + max(completion_values) * 0.01, f'{completion:.0f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Customize chart
    ax.set_title(f'Token Composition by {title_suffix} Prompt (Averaged Across All LLMs)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{title_suffix} Prompts', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Tokens', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([prompt.replace('_', ' ').title() for prompt in sorted_prompts], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created prompt composition chart: {output_path.name}")

def save_summary_data(metrics_data: List[Dict[str, Any]], output_path: Path) -> None:
    """Save summary data to CSV file."""
    
    summary_data = []
    for entry in metrics_data:
        avg_completion_tokens = np.mean(entry['completion_tokens'])
        avg_reasoning_ratio = np.mean(entry['reasoning_ratios']) if entry['reasoning_ratios'] else 0
        avg_success_rate = np.mean(entry['success_rates']) if entry['success_rates'] else 0
        
        summary_data.append({
            'llm': entry['llm'],
            'prompt_id': entry['prompt_id'],
            'type': entry['type'],
            'open_weights': entry['open_weights'],
            'full_cot': entry['full_cot'],
            'avg_completion_tokens': avg_completion_tokens,
            'avg_reasoning_ratio': avg_reasoning_ratio,
            'avg_success_rate': avg_success_rate,
            'total_responses': entry['total_responses']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    print(f"Saved summary data: {output_path.name}")

if __name__ == "__main__":
    main()