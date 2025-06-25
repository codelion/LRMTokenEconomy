# This script is given two parameters in the commandline:
# 1. A path to a folder with json datasets
# 2. A path to a configurations file in json
#
# The script will read the configuration file, which contains descriptions of reasoning LLMs.
# It will then traverse the dataset folder and open each json file with a filename that starts with "output_queries".
# Only datafiles that match a llm from the configuration file are processed. (Optional: add command line options to process all files)
# 
# In the datafiles you will find samples prompts for a combination of different prompts ("prompt_id") and reasoning LLMs ("llm").
# For each entry, evaluate the following:
# - the length of the output ("output" field)
# - the length of the thinking ("thinking" field)
# - the number of tokens in the output ("tokens_completion" field)
# For each of those acquire the following statitics: Number of samples, mean, standard deviation, minimum, maximum.
#
# Note that there may be missing fields or string field may have the value "null". Implement measures to handle this gracefully.

# Output:
# Output a compact representation as a json file.
# Create plots that are saved as png files and pdf to a figure/ folder (create of it does not exist).
# 
# Create the following heat maps: (prompt_id, llm) -> (output_length (mean), thinking_length (mean), output_tokens (mean), number of samples)

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze LLM output statistics from JSON datasets.")
    parser.add_argument("dataset_folder", type=str, help="Path to the folder with json datasets.")
    parser.add_argument("config_file", type=str, help="Path to a configurations file in json.")
    parser.add_argument("-o", "--output", type=str, default="statistics.json", help="Path for the output JSON statistics file.")
    parser.add_argument("--all", action="store_true", help="Process all LLMs, ignoring the config file.")
    return parser.parse_args()

def load_config(config_file):
    """Loads LLM configurations from a JSON file."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return {llm['name'] for llm in config.get('llms', [])}
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_file}")
        return None

def load_and_process_data(dataset_folder, configured_llms, process_all):
    """Loads and processes data from JSON files in the dataset folder."""
    all_samples = []
    found_llms_in_data = set()
    processed_files_count = 0
    data_files = [f for f in os.listdir(dataset_folder) if f.startswith("output_queries") and f.endswith(".json")]

    for filename in data_files:
        filepath = os.path.join(dataset_folder, filename)
        file_had_valid_records = False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Could not process {filename}: {e}. Skipping.")
            continue

        eval_filename = filename.replace("output_queries", "detailed_evaluations")
        eval_filepath = os.path.join(dataset_folder, eval_filename)
        eval_data = {}
        if os.path.exists(eval_filepath):
            try:
                with open(eval_filepath, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"Warning: Could not process evaluation file {eval_filename}. Skipping scores for this file.")

        results = data.get('results', [])
        if not isinstance(results, list):
            results = []

        for record in results:
            llm_name = record.get('llm')
            if not llm_name:
                continue
            
            found_llms_in_data.add(llm_name)

            if not process_all and llm_name not in configured_llms:
                continue
            
            if not file_had_valid_records:
                file_had_valid_records = True

            prompt_id = record.get('prompt_id')
            outputs = record.get('output', [])
            thinkings = record.get('thinking', [])
            tokens = record.get('tokens_completion', [])

            outputs = outputs if isinstance(outputs, list) else [outputs]
            thinkings = thinkings if isinstance(thinkings, list) else [thinkings]
            tokens = tokens if isinstance(tokens, list) else [tokens]

            eval_list = eval_data.get(prompt_id, {}).get(llm_name, [])

            for i in range(len(outputs)):
                output_val = outputs[i]
                thinking_val = thinkings[i] if i < len(thinkings) else None
                token_val = tokens[i] if i < len(tokens) else None
                
                score = np.nan
                if i < len(eval_list) and isinstance(eval_list[i], dict) and 'total_score' in eval_list[i]:
                    score = eval_list[i]['total_score']

                all_samples.append({
                    'prompt_id': prompt_id,
                    'llm': llm_name,
                    'output_length': len(output_val) if isinstance(output_val, str) else 0,
                    'thinking_length': len(thinking_val) if isinstance(thinking_val, str) else 0,
                    'tokens_completion': pd.to_numeric(token_val, errors='coerce'),
                    'score': score
                })
        
        if file_had_valid_records:
            processed_files_count += 1
            
    df = pd.DataFrame(all_samples)
    return df, found_llms_in_data, processed_files_count, len(data_files)

def report_llm_stats(found_llms_in_data, configured_llms, process_all, processed_files_count, total_files_count):
    """Prints statistics about processed and skipped LLMs and files."""
    if not process_all:
        skipped_llms = found_llms_in_data - configured_llms
        if skipped_llms:
            print("\nWarning: The following LLMs were found in data files but are not in the configuration file and were skipped:")
            for llm in sorted(list(skipped_llms)):
                print(f"- {llm}")
            print("To process them, add them to your config file or use the --all flag.")

        unseen_configured_llms = configured_llms - found_llms_in_data
        if unseen_configured_llms:
            print("\nInfo: The following LLMs are in the configuration file but were not found in any data files:")
            for llm in sorted(list(unseen_configured_llms)):
                print(f"- {llm}")
    
    print(f"\nProcessed {processed_files_count} out of {total_files_count} matching data file(s).")

def calculate_statistics(df):
    """Calculates summary statistics from the dataframe."""
    stats = df.groupby(['prompt_id', 'llm']).agg(
        output_length_count=('output_length', 'size'),
        output_length_mean=('output_length', 'mean'),
        output_length_median=('output_length', 'median'),
        output_length_std=('output_length', 'std'),
        output_length_min=('output_length', 'min'),
        output_length_max=('output_length', 'max'),
        thinking_length_mean=('thinking_length', 'mean'),
        thinking_length_median=('thinking_length', 'median'),
        thinking_length_std=('thinking_length', 'std'),
        thinking_length_min=('thinking_length', 'min'),
        thinking_length_max=('thinking_length', 'max'),
        tokens_completion_mean=('tokens_completion', 'mean'),
        tokens_completion_median=('tokens_completion', 'median'),
        tokens_completion_std=('tokens_completion', 'std'),
        tokens_completion_min=('tokens_completion', 'min'),
        tokens_completion_max=('tokens_completion', 'max'),
        score_mean=('score', 'mean'),
        score_median=('score', 'median'),
        score_std=('score', 'std'),
        score_min=('score', 'min'),
        score_max=('score', 'max'),
    ).reset_index()
    stats.rename(columns={'output_length_count': 'num_samples'}, inplace=True)
    return stats

def save_statistics(stats, output_file):
    """Saves statistics to a JSON file."""
    stats.to_json(output_file, orient='records', indent=4)
    print(f"Statistics saved to {output_file}")

def generate_heatmaps(stats, figure_dir):
    """Generates and saves heatmap plots."""
    print("\nGenerating heatmaps...")
    metrics_to_plot = {
        'output_length_mean': 'Mean Output Length',
        'thinking_length_mean': 'Mean Thinking Length',
        'tokens_completion_mean': 'Mean Output Tokens',
        'num_samples': 'Number of Samples'
    }

    for metric, title in metrics_to_plot.items():
        if metric not in stats.columns:
            continue
        try:
            pivot_df = stats.pivot(index='prompt_id', columns='llm', values=metric)
            
            height = max(6, len(pivot_df.index) * 0.6)
            width = max(10, len(pivot_df.columns) * 1.0)

            plt.figure(figsize=(width, height))

            pivot_for_plot = pivot_df.replace(0, np.nan)
            
            plot_fmt = "d" if metric == 'num_samples' else ".1f"

            sns.heatmap(pivot_for_plot, annot=pivot_df, fmt=plot_fmt, cmap="viridis", linewidths=.5, norm=LogNorm())
            plt.title(title, fontsize=16)
            plt.xlabel("LLM", fontsize=12)
            plt.ylabel("Prompt ID", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            sanitized_title = title.replace(' ', '_').lower()
            plt.savefig(os.path.join(figure_dir, f"{sanitized_title}.png"), dpi=300)
            plt.savefig(os.path.join(figure_dir, f"{sanitized_title}.pdf"))
            plt.close()
        except Exception as e:
            print(f"Could not generate plot for {metric}: {e}")

def _plot_correctness_bar_chart(data, y_col, title, filename_base, figure_dir):
    """Generates a compact bar chart for correctness analysis."""
    labels = data[y_col]
    correct_means = data['thinking_length_mean_Correct']
    correct_medians = data['thinking_length_median_Correct']
    incorrect_means = data['thinking_length_mean_Incorrect']
    incorrect_medians = data['thinking_length_median_Incorrect']

    y = np.arange(len(labels))
    height = 0.35

    fig_width = 12
    fig_height = max(6, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Mean bars
    ax.barh(y - height/2, correct_means, height, label='Mean Thinking (Correct)', color='tab:green', alpha=0.7)
    ax.barh(y + height/2, incorrect_means, height, label='Mean Thinking (Incorrect)', color='tab:red', alpha=0.7)

    # Median markers
    ax.plot(correct_medians, y - height/2, 'k|', markersize=12, mew=2, label='Median')
    ax.plot(incorrect_medians, y + height/2, 'k|', markersize=12, mew=2)

    ax.set_xlabel('Thinking Length (characters)')
    ax.set_title(title, fontsize=16)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    # Manually create a single legend entry for Median
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='lower right')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"{filename_base}.png"), dpi=300)
    plt.savefig(os.path.join(figure_dir, f"{filename_base}.pdf"))
    plt.close(fig)

def _plot_bar_chart(data, y_col, title, filename_base, figure_dir):
    labels = data[y_col]
    output_means = data['output_length_mean']
    output_medians = data['output_length_median']
    thinking_means = data['thinking_length_mean']
    thinking_medians = data['thinking_length_median']

    y = np.arange(len(labels))
    height = 0.35

    fig_width = 12
    fig_height = max(6, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Mean bars
    ax.barh(y - height/2, output_means, height, label='Mean Output Length', alpha=0.7)
    ax.barh(y + height/2, thinking_means, height, label='Mean Thinking Length', alpha=0.7)

    # Median markers
    ax.plot(output_medians, y - height/2, 'k|', markersize=12, mew=2, label='Median')
    ax.plot(thinking_medians, y + height/2, 'k|', markersize=12, mew=2)


    ax.set_xlabel('Average Length (characters)')
    ax.set_title(title, fontsize=16)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # To display the highest value at the top
    ax.legend(loc='lower right')
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"{filename_base}.png"), dpi=300)
    plt.savefig(os.path.join(figure_dir, f"{filename_base}.pdf"))
    plt.close(fig)

def _plot_single_metric_bar_chart(data, y_col, metric_mean_col, metric_median_col, x_label, title, filename_base, figure_dir):
    """Generates a compact bar chart for a single metric."""
    labels = data[y_col]
    means = data[metric_mean_col]
    medians = data[metric_median_col]

    y = np.arange(len(labels))
    height = 0.6

    fig_width = 12
    fig_height = max(6, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Mean bars
    ax.barh(y, means, height, label=f'Mean {x_label}', alpha=0.7)

    # Median markers
    ax.plot(medians, y, 'k|', markersize=12, mew=2, label='Median')

    ax.set_xlabel(x_label)
    ax.set_title(title, fontsize=16)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # To display the highest value at the top
    ax.legend(loc='lower right')
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"{filename_base}.png"), dpi=300)
    plt.savefig(os.path.join(figure_dir, f"{filename_base}.pdf"))
    plt.close(fig)

def generate_bar_charts(df, figure_dir):
    """Generates and saves aggregated bar charts."""
    print("\nGenerating aggregated bar charts...")
    
    # Aggregate and sort by mean thinking length in descending order
    prompt_stats = df.groupby('prompt_id').agg(
        output_length_mean=('output_length', 'mean'),
        output_length_median=('output_length', 'median'),
        thinking_length_mean=('thinking_length', 'mean'),
        thinking_length_median=('thinking_length', 'median')
    ).reset_index().sort_values('thinking_length_mean', ascending=False)

    llm_stats = df.groupby('llm').agg(
        output_length_mean=('output_length', 'mean'),
        output_length_median=('output_length', 'median'),
        thinking_length_mean=('thinking_length', 'mean'),
        thinking_length_median=('thinking_length', 'median')
    ).reset_index().sort_values('thinking_length_mean', ascending=False)

    try:
        _plot_bar_chart(prompt_stats, 'prompt_id', 'Aggregated Output and Thinking Lengths by Prompt', 'lengths_by_prompt', figure_dir)
    except Exception as e:
        print(f"Could not generate plot for lengths by prompt: {e}")

    try:
        _plot_bar_chart(llm_stats, 'llm', 'Aggregated Output and Thinking Lengths by LLM', 'lengths_by_llm', figure_dir)
    except Exception as e:
        print(f"Could not generate plot for lengths by llm: {e}")

def generate_token_count_bar_charts(df, figure_dir):
    """Generates and saves aggregated bar charts for token counts."""
    print("\nGenerating aggregated token count bar charts...")
    
    token_df = df.dropna(subset=['tokens_completion']).copy()
    if token_df.empty:
        print("Skipping token count bar charts: no data with token information.")
        return

    # Aggregate and sort by mean token count in descending order
    prompt_stats = token_df.groupby('prompt_id').agg(
        tokens_completion_mean=('tokens_completion', 'mean'),
        tokens_completion_median=('tokens_completion', 'median')
    ).reset_index().sort_values('tokens_completion_mean', ascending=False)

    llm_stats = token_df.groupby('llm').agg(
        tokens_completion_mean=('tokens_completion', 'mean'),
        tokens_completion_median=('tokens_completion', 'median')
    ).reset_index().sort_values('tokens_completion_mean', ascending=False)

    try:
        _plot_single_metric_bar_chart(
            prompt_stats, 
            y_col='prompt_id', 
            metric_mean_col='tokens_completion_mean',
            metric_median_col='tokens_completion_median',
            x_label='Token Count',
            title='Aggregated Token Count by Prompt', 
            filename_base='token_count_by_prompt', 
            figure_dir=figure_dir
        )
    except Exception as e:
        print(f"Could not generate plot for token count by prompt: {e}")

    try:
        _plot_single_metric_bar_chart(
            llm_stats, 
            y_col='llm', 
            metric_mean_col='tokens_completion_mean',
            metric_median_col='tokens_completion_median',
            x_label='Token Count',
            title='Aggregated Token Count by LLM', 
            filename_base='token_count_by_llm', 
            figure_dir=figure_dir
        )
    except Exception as e:
        print(f"Could not generate plot for token count by llm: {e}")

def generate_correctness_plots(df, figure_dir):
    """Generates bar charts showing correlation between correctness and response length."""
    print("\nGenerating correctness vs. length plots...")
    if 'score' not in df.columns or df['score'].isnull().all():
        print("Skipping correctness plots: no score data found.")
        return

    plot_df = df.dropna(subset=['score']).copy()
    if plot_df.empty:
        print("Skipping correctness plots: no valid score data found.")
        return
        
    # Define correctness: score > 0 is Correct, otherwise Incorrect.
    plot_df['Correctness'] = np.where(plot_df['score'] > 0, 'Correct', 'Incorrect')

    agg_stats = plot_df.groupby(['prompt_id', 'Correctness']).agg(
        thinking_length_mean=('thinking_length', 'mean'),
        thinking_length_median=('thinking_length', 'median')
    ).unstack(fill_value=0)

    # Flatten the multi-index columns
    agg_stats.columns = [f'{val}_{col}' for val, col in agg_stats.columns]
    agg_stats.reset_index(inplace=True)

    # Ensure all required columns exist, filling with 0 if not present
    for col in ['thinking_length_mean_Correct', 'thinking_length_median_Correct', 'thinking_length_mean_Incorrect', 'thinking_length_median_Incorrect']:
        if col not in agg_stats.columns:
            agg_stats[col] = 0

    # Sort by mean thinking length of correct answers
    agg_stats.sort_values(by='thinking_length_mean_Correct', ascending=False, inplace=True)

    try:
        _plot_correctness_bar_chart(agg_stats, 'prompt_id', 'Thinking Length by Prompt and Correctness', 'correctness_vs_thinking_length', figure_dir)
    except Exception as e:
        print(f"Could not generate plot for correctness vs length: {e}")

def generate_token_length_scatter_plot(df, figure_dir):
    """Generates a scatter plot of tokens vs. total length with linear regression."""
    print("\nGenerating tokens vs. length scatter plot...")
    
    plot_df = df.dropna(subset=['tokens_completion', 'output_length', 'thinking_length']).copy()
    if plot_df.empty:
        print("Skipping tokens vs. length scatter plot: no data with token information.")
        return

    plot_df['total_length'] = plot_df['output_length'] + plot_df['thinking_length']

    # Filter out models that have no token information at all
    models_with_tokens = plot_df.groupby('llm')['tokens_completion'].count() > 0
    models_to_plot = models_with_tokens[models_with_tokens].index
    plot_df = plot_df[plot_df['llm'].isin(models_to_plot)]

    if plot_df.empty:
        print("Skipping tokens vs. length scatter plot: no models with token information found.")
        return

    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(data=plot_df, x='total_length', y='tokens_completion', hue='llm', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    llm_colors = {labels[i]: handles[i].get_color() for i in range(len(labels))}

    for i, llm in enumerate(labels):
        model_df = plot_df[plot_df['llm'] == llm]
        if model_df.empty:
            new_labels.append(f'{llm} (no data)')
            continue

        x = model_df['total_length']
        y = model_df['tokens_completion']

        # Linear regression: y = m*x (no offset)
        x_reshaped = x.values.reshape(-1, 1)
        
        slope = 0.0
        if not np.all(x_reshaped == 0):
            try:
                slope = np.linalg.lstsq(x_reshaped, y.values, rcond=None)[0][0]
            except np.linalg.LinAlgError:
                slope = np.nan
        
        if not np.isnan(slope):
            # Use unique points for a cleaner line fit
            x_fit = np.array(sorted(x.unique()))
            y_fit = slope * x_fit
            
            # Get color from the scatter plot handle
            color = handles[i].get_color()
            
            plt.plot(x_fit, y_fit, linestyle='--', color=color)
            new_labels.append(f'{llm} (slope={slope:.2f})')
        else:
            new_labels.append(f'{llm} (fit failed)')

    ax.legend(handles, new_labels, title='LLM')
    
    plt.title('Tokens vs. Response + Thinking Length', fontsize=16)
    plt.xlabel('Sum of Response and Thinking Length (characters)', fontsize=12)
    plt.ylabel('Number of Tokens', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    filename_base = 'tokens_vs_length_scatter'
    plt.savefig(os.path.join(figure_dir, f"{filename_base}.png"), dpi=300)
    plt.savefig(os.path.join(figure_dir, f"{filename_base}.pdf"))
    plt.close()

    # Individual plots PDF
    pdf_filename = os.path.join(figure_dir, 'tokens_vs_length_scatter_individual.pdf')
    with PdfPages(pdf_filename) as pdf:
        for llm in models_to_plot:
            model_df = plot_df[plot_df['llm'] == llm]
            if model_df.empty:
                continue

            plt.figure(figsize=(10, 6))
            
            x = model_df['total_length']
            y = model_df['tokens_completion']
            
            color = llm_colors.get(llm, 'blue')
            sns.scatterplot(x=x, y=y, alpha=0.6, color=color)

            # Linear regression
            x_reshaped = x.values.reshape(-1, 1)
            slope = 0.0
            if not np.all(x_reshaped == 0):
                try:
                    slope = np.linalg.lstsq(x_reshaped, y.values, rcond=None)[0][0]
                except np.linalg.LinAlgError:
                    slope = np.nan
            
            if not np.isnan(slope):
                x_fit = np.array(sorted(x.unique()))
                y_fit = slope * x_fit
                plt.plot(x_fit, y_fit, linestyle='--', color=color, label=f'Fit (slope={slope:.2f})')
                plt.legend()

            plt.title(f'Tokens vs. Length for {llm}', fontsize=16)
            plt.xlabel('Sum of Response and Thinking Length (characters)', fontsize=12)
            plt.ylabel('Number of Tokens', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            pdf.savefig()
            plt.close()
    print(f"Individual LLM scatter plots saved to {pdf_filename}")

def main():
    """Main function to run the analysis."""
    args = parse_args()

    configured_llms = load_config(args.config_file)
    if configured_llms is None:
        return

    df, found_llms, processed_count, total_count = load_and_process_data(
        args.dataset_folder, configured_llms, args.all
    )

    report_llm_stats(found_llms, configured_llms, args.all, processed_count, total_count)

    if df.empty:
        print("No samples found to process.")
        return

    stats = calculate_statistics(df)
    
    save_statistics(stats, args.output)

    figure_dir = 'figure'
    os.makedirs(figure_dir, exist_ok=True)

    generate_heatmaps(stats, figure_dir)
    generate_bar_charts(df, figure_dir)
    generate_token_count_bar_charts(df, figure_dir)
    generate_correctness_plots(df, figure_dir)
    generate_token_length_scatter_plot(df, figure_dir)

    print(f"\nPlots saved to '{figure_dir}' directory.")

if __name__ == "__main__":
    main()