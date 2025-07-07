#!/usr/bin/env python3
"""
Word Statistics Analysis Script for Chain-of-Thought Traces

This script analyzes the first words of each line in thinking traces from LLMs with full_cot=true,
generating word frequency statistics and a comparative heatmap visualization.

Usage:
    python analyze_wordstats.py --data-dir data --config query_config_full.json --output-dir figures/wordstatistic
"""

import argparse
import sys
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for better performance
plt.switch_backend('Agg')

def validate_inputs(data_dir: Path, config_path: Path) -> None:
    """Validate that input paths exist and are accessible."""
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    if not data_dir.is_dir():
        print(f"Error: Data path is not a directory: {data_dir}")
        sys.exit(1)
        
    if not config_path.exists():
        print(f"Error: Config file does not exist: {config_path}")
        sys.exit(1)
    
    if not config_path.is_file():
        print(f"Error: Config path is not a file: {config_path}")
        sys.exit(1)

def create_output_directory(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

def load_config(config_path: Path) -> Dict[str, bool]:
    """Load configuration and return mapping of model names to full_cot status."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        full_cot_models = {}
        for llm in config.get('llms', []):
            model_name = llm.get('name')
            full_cot = llm.get('full_cot', False)
            if model_name:
                full_cot_models[model_name] = full_cot
        
        print(f"Loaded configuration with {len(full_cot_models)} models")
        full_cot_count = sum(1 for is_cot in full_cot_models.values() if is_cot)
        print(f"Found {full_cot_count} models with full_cot=true")
        
        return full_cot_models
        
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def discover_output_files(data_dir: Path) -> List[Path]:
    """Discover all output_queries_*.json files in the data directory."""
    pattern = "output_queries_*.json"
    files = list(data_dir.glob(pattern))
    files.sort()  # Sort for consistent processing order
    
    if not files:
        print(f"Warning: No {pattern} files found in {data_dir}")
    else:
        print(f"Found {len(files)} output query files")
    
    return files

def normalize_word(word: str) -> str:
    """Normalize a word by converting to lowercase and removing punctuation."""
    # Remove punctuation and convert to lowercase
    normalized = re.sub(r'[^\w]', '', word.lower())
    return normalized

def extract_first_words_from_trace(thinking_trace: str) -> List[str]:
    """Extract the first word from each non-empty line in a thinking trace."""
    if not thinking_trace or not isinstance(thinking_trace, str):
        return []
    
    first_words = []
    lines = thinking_trace.split('\n')
    
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            words = line.split()
            if words:  # Make sure there's at least one word
                first_word = normalize_word(words[0])
                if first_word:  # Only add non-empty normalized words
                    first_words.append(first_word)
    
    return first_words

def count_total_words_in_trace(thinking_trace: str) -> int:
    """Count total words in a thinking trace."""
    if not thinking_trace or not isinstance(thinking_trace, str):
        return 0
    
    # Split into words and count
    words = thinking_trace.split()
    return len(words)

def load_and_process_output_files(files: List[Path], full_cot_models: Dict[str, bool]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Load output files and extract first words from thinking traces for full_cot models."""
    model_first_words = defaultdict(list)
    model_total_words = defaultdict(int)
    
    # Get list of models with full_cot=true
    target_models = {name for name, is_cot in full_cot_models.items() if is_cot}
    print(f"Target models with full_cot=true: {sorted(target_models)}")
    
    total_processed = 0
    total_thinking_traces = 0
    
    for file_path in files:
        print(f"Processing: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            file_processed = 0
            
            for result in results:
                llm_name = result.get('llm', '')
                thinking_traces = result.get('thinking', [])
                
                # Only process models with full_cot=true
                if llm_name in target_models and thinking_traces:
                    for trace in thinking_traces:
                        first_words = extract_first_words_from_trace(trace)
                        total_words = count_total_words_in_trace(trace)
                        
                        model_first_words[llm_name].extend(first_words)
                        model_total_words[llm_name] += total_words
                        total_thinking_traces += 1
                    
                    file_processed += 1
            
            print(f"  Processed {file_processed} results from {file_path.name}")
            total_processed += file_processed
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"\nSummary:")
    print(f"Total processed results: {total_processed}")
    print(f"Total thinking traces analyzed: {total_thinking_traces}")
    print(f"Models with data: {sorted(model_first_words.keys())}")
    
    # Print word counts per model
    for model in sorted(model_first_words.keys()):
        first_word_count = len(model_first_words[model])
        total_word_count = model_total_words[model]
        print(f"  {model}: {first_word_count} first words, {total_word_count} total words")
    
    return dict(model_first_words), dict(model_total_words)

def calculate_word_statistics(model_first_words: Dict[str, List[str]]) -> Tuple[Dict[str, Counter], List[str]]:
    """Calculate word frequency statistics and determine top words globally."""
    
    # Calculate word frequencies for each model
    model_word_counts = {}
    all_words = Counter()
    
    for model, words in model_first_words.items():
        word_counter = Counter(words)
        model_word_counts[model] = word_counter
        all_words.update(word_counter)
    
    # Get top 30 words globally
    top_30_global = [word for word, _ in all_words.most_common(30)]
    
    print(f"\nWord Statistics:")
    print(f"Total unique words across all models: {len(all_words)}")
    print(f"Top 30 most common words globally: {top_30_global[:10]}...")  # Show first 10
    
    # Print top words for each model
    for model in sorted(model_word_counts.keys()):
        counter = model_word_counts[model]
        total_words = sum(counter.values())
        top_5 = counter.most_common(5)
        print(f"\n{model} (total: {total_words} words):")
        for word, count in top_5:
            percentage = (count / total_words) * 100
            print(f"  {word}: {count} ({percentage:.1f}%)")
    
    return model_word_counts, top_30_global

def create_heatmap(model_word_counts: Dict[str, Counter], top_words: List[str], 
                  output_path: Path, figsize: Tuple[float, float]) -> None:
    """Create heatmap showing word usage percentages by model."""
    
    models = sorted(model_word_counts.keys())
    
    # Calculate percentage matrix
    percentage_matrix = []
    
    for word in top_words:
        row = []
        for model in models:
            counter = model_word_counts[model]
            total_words = sum(counter.values())
            word_count = counter.get(word, 0)
            percentage = (word_count / total_words * 100) if total_words > 0 else 0
            row.append(percentage)
        percentage_matrix.append(row)
    
    # Convert to numpy array for plotting
    data = np.array(percentage_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(data, 
               xticklabels=models, 
               yticklabels=top_words,
               annot=True, 
               fmt='.1f',
               cmap='YlOrRd',
               cbar_kws={'label': 'Usage Percentage (%)'},
               ax=ax,
               linewidths=0.5)
    
    # Customize plot
    ax.set_title('First Words Usage in Chain-of-Thought Traces\n(% of first words per model)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Model (full_cot=true)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Most Common First Words', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created word statistics heatmap: {output_path.name}")
    print(f"  Data shape: {data.shape} ({len(top_words)} words × {len(models)} models)")
    print(f"  Percentage range: {data.min():.1f}% - {data.max():.1f}%")

def calculate_cosine_similarity_matrix(model_word_counts: Dict[str, Counter]) -> Tuple[np.ndarray, List[str]]:
    """Calculate cosine similarity between models based on their word usage patterns."""
    
    models = sorted(model_word_counts.keys())
    
    # Get all unique words across all models
    all_words = set()
    for counter in model_word_counts.values():
        all_words.update(counter.keys())
    all_words = sorted(all_words)
    
    # Create feature vectors for each model (normalized word frequencies)
    feature_vectors = []
    for model in models:
        counter = model_word_counts[model]
        total_words = sum(counter.values())
        
        # Create normalized frequency vector
        vector = []
        for word in all_words:
            frequency = counter.get(word, 0) / total_words if total_words > 0 else 0
            vector.append(frequency)
        
        feature_vectors.append(vector)
    
    # Calculate cosine similarity matrix
    feature_matrix = np.array(feature_vectors)
    similarity_matrix = cosine_similarity(feature_matrix)
    
    print(f"Calculated cosine similarity matrix for {len(models)} models")
    print(f"Feature space dimensions: {len(all_words)} unique words")
    
    return similarity_matrix, models

def create_similarity_heatmap(similarity_matrix: np.ndarray, models: List[str], 
                            output_path: Path, figsize: Tuple[float, float]) -> None:
    """Create heatmap showing cosine similarity between models with hierarchical clustering."""
    
    # Convert similarity to distance matrix for clustering
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is exactly zero for squareform
    np.fill_diagonal(distance_matrix, 0)
    
    # Perform hierarchical clustering
    condensed_distances = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Get the order from dendrogram
    dendro = dendrogram(linkage_matrix, labels=models, no_plot=True)
    cluster_order = dendro['leaves']
    
    # Reorder similarity matrix and model names according to clustering
    ordered_similarity = similarity_matrix[np.ix_(cluster_order, cluster_order)]
    ordered_models = [models[i] for i in cluster_order]
    
    print(f"Clustered model order: {ordered_models}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with clustered order
    sns.heatmap(ordered_similarity, 
               xticklabels=ordered_models, 
               yticklabels=ordered_models,
               annot=True, 
               fmt='.3f',
               cmap='RdGy_r',
               vmin=0, 
               vmax=1,
               square=True,
               cbar_kws={'label': 'Cosine Similarity'},
               ax=ax,
               linewidths=0.5)
    
    # Customize plot
    ax.set_title('Cosine Similarity of Word Usage Patterns\nbetween LLM Models (clustered by similarity)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('LLM Model', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created cosine similarity heatmap: {output_path.name}")
    print(f"  Similarity range: {ordered_similarity.min():.3f} - {ordered_similarity.max():.3f}")
    
    # Print most and least similar model pairs using original indices
    n = len(models)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle, excluding diagonal
            similarities.append((similarity_matrix[i,j], models[i], models[j]))
    
    similarities.sort(reverse=True)
    
    print(f"  Most similar pairs:")
    for sim, model1, model2 in similarities[:3]:
        print(f"    {model1} ↔ {model2}: {sim:.3f}")
    
    print(f"  Least similar pairs:")
    for sim, model1, model2 in similarities[-3:]:
        print(f"    {model1} ↔ {model2}: {sim:.3f}")

def create_word_statistics_chart(model_first_words: Dict[str, List[str]], model_total_words: Dict[str, int], 
                               output_path: Path, figsize: Tuple[float, float]) -> None:
    """Create bar chart showing word statistics (total first words, unique words, ratios)."""
    
    models = sorted(model_first_words.keys())
    
    # Calculate statistics
    stats_data = []
    for model in models:
        first_words = model_first_words[model]
        total_words = model_total_words[model]
        
        total_first_words = len(first_words)
        unique_first_words = len(set(first_words))
        unique_ratio = unique_first_words / total_first_words if total_first_words > 0 else 0
        first_to_total_ratio = total_first_words / total_words if total_words > 0 else 0
        
        stats_data.append({
            'model': model,
            'total_first_words': total_first_words,
            'unique_first_words': unique_first_words,
            'unique_ratio': unique_ratio,
            'total_words': total_words,
            'first_to_total_ratio': first_to_total_ratio
        })
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Chain-of-Thought Word Statistics by Model', fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(models))
    bar_width = 0.6
    
    # 1. Total First Words
    total_first = [data['total_first_words'] for data in stats_data]
    bars1 = ax1.bar(x_pos, total_first, bar_width, color='skyblue', alpha=0.8, edgecolor='black')
    ax1.set_title('Total First Words', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, total_first):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_first)*0.01, 
                f'{value:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. Unique First Words
    unique_first = [data['unique_first_words'] for data in stats_data]
    bars2 = ax2.bar(x_pos, unique_first, bar_width, color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.set_title('Unique First Words', fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars2, unique_first):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(unique_first)*0.01, 
                f'{value:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 3. Unique Word Ratio (unique/total first words)
    unique_ratios = [data['unique_ratio'] for data in stats_data]
    bars3 = ax3.bar(x_pos, unique_ratios, bar_width, color='lightgreen', alpha=0.8, edgecolor='black')
    ax3.set_title('Unique Words Ratio\n(Unique / Total First Words)', fontweight='bold')
    ax3.set_ylabel('Ratio')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars3, unique_ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. First Words to Total Words Ratio
    first_to_total_ratios = [data['first_to_total_ratio'] for data in stats_data]
    bars4 = ax4.bar(x_pos, first_to_total_ratios, bar_width, color='gold', alpha=0.8, edgecolor='black')
    ax4.set_title('First Words Ratio\n(First Words / Total Words)', fontweight='bold')
    ax4.set_ylabel('Ratio')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars4, first_to_total_ratios):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(first_to_total_ratios)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created word statistics chart: {output_path.name}")
    
    # Print summary statistics
    print(f"\nWord Statistics Summary:")
    for data in stats_data:
        model = data['model']
        print(f"  {model}:")
        print(f"    Total first words: {data['total_first_words']:,}")
        print(f"    Unique first words: {data['unique_first_words']:,}")
        print(f"    Unique ratio: {data['unique_ratio']:.3f}")
        print(f"    Total words in CoT: {data['total_words']:,}")
        print(f"    First/Total ratio: {data['first_to_total_ratio']:.3f}")
        print()

def save_detailed_statistics(model_word_counts: Dict[str, Counter], top_words: List[str], 
                           output_dir: Path) -> None:
    """Save detailed word statistics to CSV files."""
    
    # Create detailed statistics DataFrame
    rows = []
    models = sorted(model_word_counts.keys())
    
    for word in top_words:
        row = {'word': word}
        for model in models:
            counter = model_word_counts[model]
            total_words = sum(counter.values())
            word_count = counter.get(word, 0)
            percentage = (word_count / total_words * 100) if total_words > 0 else 0
            row[f'{model}_count'] = word_count
            row[f'{model}_percentage'] = percentage
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = output_dir / "word_statistics_detailed.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed statistics: {csv_path.name}")
    
    # Create summary statistics
    summary_lines = []
    summary_lines.append("Word Statistics Summary")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    
    for model in models:
        counter = model_word_counts[model]
        total_words = sum(counter.values())
        unique_words = len(counter)
        top_10 = counter.most_common(10)
        
        summary_lines.append(f"Model: {model}")
        summary_lines.append(f"  Total first words: {total_words}")
        summary_lines.append(f"  Unique words: {unique_words}")
        summary_lines.append(f"  Top 10 words:")
        
        for word, count in top_10:
            percentage = (count / total_words) * 100
            summary_lines.append(f"    {word}: {count} ({percentage:.1f}%)")
        summary_lines.append("")
    
    summary_path = output_dir / "word_statistics_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Saved summary statistics: {summary_path.name}")

def main():
    """Main function to analyze word statistics in chain-of-thought traces."""
    parser = argparse.ArgumentParser(
        description="Analyze first words in chain-of-thought thinking traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_wordstats.py                                    # Use defaults
    python analyze_wordstats.py --data-dir data --config query_config_full.json
    python analyze_wordstats.py --output-dir figures/wordstatistic --figsize 12,10
        """
    )
    
    parser.add_argument('--data-dir', default='data',
                       help='Directory containing output_queries_*.json files (default: data)')
    parser.add_argument('--config', default='query_config_full.json',
                       help='Path to query configuration file (default: query_config_full.json)')
    parser.add_argument('--output-dir', default='figures/wordstatistic',
                       help='Output directory for generated files (default: figures/wordstatistic)')
    parser.add_argument('--figsize', default='14,10',
                       help='Figure size as width,height in inches (default: 14,10)')
    
    args = parser.parse_args()
    
    # Parse figure size
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print("Error: Invalid figsize format. Use 'width,height' (e.g., '14,10')")
        sys.exit(1)
    
    # Validate inputs
    data_dir = Path(args.data_dir)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    
    validate_inputs(data_dir, config_path)
    create_output_directory(output_dir)
    
    print("Starting word statistics analysis for chain-of-thought traces...")
    print(f"Data directory: {data_dir}")
    print(f"Config file: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Figure size: {figsize[0]}x{figsize[1]} inches")
    print()
    
    # Load configuration
    full_cot_models = load_config(config_path)
    
    # Discover and load output files
    output_files = discover_output_files(data_dir)
    if not output_files:
        print("No output files found. Exiting.")
        sys.exit(1)
    
    # Process files and extract first words
    model_first_words, model_total_words = load_and_process_output_files(output_files, full_cot_models)
    
    if not model_first_words:
        print("No word data extracted. Exiting.")
        sys.exit(1)
    
    # Calculate word statistics
    model_word_counts, top_30_words = calculate_word_statistics(model_first_words)
    
    # Create word usage heatmap
    heatmap_path = output_dir / "first_words_heatmap.png"
    create_heatmap(model_word_counts, top_30_words, heatmap_path, figsize)
    
    # Calculate and create cosine similarity heatmap
    similarity_matrix, similarity_models = calculate_cosine_similarity_matrix(model_word_counts)
    similarity_path = output_dir / "cosine_similarity_heatmap.png"
    create_similarity_heatmap(similarity_matrix, similarity_models, similarity_path, figsize)
    
    # Create word statistics chart
    stats_chart_path = output_dir / "word_statistics_chart.png"
    create_word_statistics_chart(model_first_words, model_total_words, stats_chart_path, figsize)
    
    # Save detailed statistics
    save_detailed_statistics(model_word_counts, top_30_words, output_dir)
    
    print(f"\nCompleted! Generated word statistics analysis in {output_dir}")
    print(f"  - 1 word usage heatmap")
    print(f"  - 1 cosine similarity heatmap") 
    print(f"  - 1 word statistics chart")
    print(f"  - 1 detailed statistics CSV")
    print(f"  - 1 summary statistics file")

if __name__ == "__main__":
    main()