"""
Cross-Validation Summary Script for Duke Dataset

Aggregates results from multiple folds and generates comprehensive performance summary.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from collections import defaultdict

LABEL_NAME_MAPS = {
    "Arterial T1w": ["C", "O", "Q"],
    "Portven T1w": ["K"],
    "Late T1w": ["E", "N", "P"],
    "AX T2w": ["A"],
    "COR T2w": ["J"],
    "AX FatSat T1w": ["B"],
    "AX Dixon In": ["G"],
    "AX Dixon Opp": ["H"],
    "AX DWI": ["I"],
    "AX ADC": ["M"],
    "Localizer": ["L"],
    "MRCP": ["D"],
    "Other": ["F"]
}

# Create reverse mapping: letter code -> label name
LETTER_TO_LABEL_NAME = {}
for label_name, letter_codes in LABEL_NAME_MAPS.items():
    for letter in letter_codes:
        LETTER_TO_LABEL_NAME[letter] = label_name

# Create sort order mapping: letter code -> order index
LETTER_SORT_ORDER = {}
order_idx = 0
for label_name, letter_codes in LABEL_NAME_MAPS.items():
    for letter in letter_codes:
        LETTER_SORT_ORDER[letter] = order_idx
        order_idx += 1


def get_label_display_name(letter_code):
    """
    Get display name for a letter code.
    
    Args:
        letter_code: Single letter code (A, B, C, etc.)
        
    Returns:
        String in format "A - AX T2w" or just "A" if not found
    """
    if letter_code in LETTER_TO_LABEL_NAME:
        return f"{letter_code} - {LETTER_TO_LABEL_NAME[letter_code]}"
    return letter_code


def get_label_sort_order(letter_code):
    """
    Get sort order for a letter code based on LABEL_NAME_MAPS.
    
    Args:
        letter_code: Single letter code (A, B, C, etc.)
        
    Returns:
        Integer representing sort order (lower = earlier)
    """
    return LETTER_SORT_ORDER.get(letter_code, 999)

def load_fold_results(cv_dir, num_folds=5, eval_fold='evaluation'):
    """
    Load evaluation results from all folds.
    
    Args:
        cv_dir: Path to cross-validation directory containing fold_* subdirectories
        num_folds: Number of folds to load
        
    Returns:
        Dictionary with fold results and metadata
    """
    all_results = {
        'summary': [],
        'detailed': [],
        'metadata': []
    }
    
    missing_folds = []
    
    for fold_idx in range(num_folds):
        fold_dir = Path(cv_dir) / f'fold_{fold_idx}'
        eval_dir = fold_dir / eval_fold
        
        # Check if fold directory exists
        if not fold_dir.exists():
            missing_folds.append(fold_idx)
            print(f"⚠ Warning: fold_{fold_idx} directory not found")
            continue
        
        if not eval_dir.exists():
            missing_folds.append(fold_idx)
            print(f"⚠ Warning: fold_{fold_idx}/evaluation directory not found")
            continue
        
        # Load summary metrics
        summary_path = eval_dir / 'summary_metrics.csv'
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df['fold'] = fold_idx
            all_results['summary'].append(df)
        else:
            print(f"⚠ Warning: {summary_path} not found")
        
        # Load detailed metrics
        detailed_path = eval_dir / 'detailed_metrics.csv'
        if detailed_path.exists():
            df = pd.read_csv(detailed_path)
            df['fold'] = fold_idx
            all_results['detailed'].append(df)
        
        # Load metadata if available
        metadata_path = eval_dir / 'inference_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                metadata['fold'] = fold_idx
                all_results['metadata'].append(metadata)
    
    # Combine all results
    if all_results['summary']:
        all_results['summary'] = pd.concat(all_results['summary'], ignore_index=True)
    else:
        all_results['summary'] = pd.DataFrame()
    
    if all_results['detailed']:
        all_results['detailed'] = pd.concat(all_results['detailed'], ignore_index=True)
    else:
        all_results['detailed'] = pd.DataFrame()
    
    return all_results, missing_folds


def compute_cv_statistics(summary_df):
    """
    Compute cross-validation statistics (mean, std, min, max) for each task.
    
    Args:
        summary_df: DataFrame with summary metrics from all folds
        
    Returns:
        DataFrame with aggregated statistics
    """
    if summary_df.empty:
        return pd.DataFrame()
    
    # Group by task and compute statistics
    metrics = ['Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_F1', 
               'Weighted_Precision', 'Weighted_Recall', 'Weighted_F1']
    
    stats_list = []
    
    for task in summary_df['Task'].unique():
        task_df = summary_df[summary_df['Task'] == task]
        
        stats = {'Task': task}
        
        for metric in metrics:
            if metric in task_df.columns:
                values = task_df[metric].values
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values)
                stats[f'{metric}_min'] = np.min(values)
                stats[f'{metric}_max'] = np.max(values)
                stats[f'{metric}_median'] = np.median(values)
        
        # Sample count (should be consistent across folds)
        if 'Samples' in task_df.columns:
            stats['Total_Samples'] = task_df['Samples'].sum()
            stats['Avg_Samples_Per_Fold'] = task_df['Samples'].mean()
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def compute_per_class_statistics(detailed_df):
    """
    Compute per-class statistics across folds.
    
    Args:
        detailed_df: DataFrame with detailed per-class metrics from all folds
        
    Returns:
        DataFrame with per-class aggregated statistics
    """
    if detailed_df.empty:
        return pd.DataFrame()
    
    # Filter out aggregate rows (macro avg, weighted avg)
    class_df = detailed_df[~detailed_df['Label'].str.contains('avg', case=False, na=False)].copy()
    
    if class_df.empty:
        return pd.DataFrame()
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    stats_list = []
    
    for task in class_df['Task'].unique():
        task_data = class_df[class_df['Task'] == task]
        
        for label in task_data['Label'].unique():
            label_data = task_data[task_data['Label'] == label]
            
            stats = {
                'Task': task,
                'Label': label
            }
            
            for metric in metrics:
                if metric in label_data.columns:
                    values = label_data[metric].values
                    stats[f'{metric}_mean'] = np.mean(values)
                    stats[f'{metric}_std'] = np.std(values)
                    stats[f'{metric}_min'] = np.min(values)
                    stats[f'{metric}_max'] = np.max(values)
                    stats[f'{metric}_median'] = np.median(values)
            
            # Support statistics
            if 'Support' in label_data.columns:
                stats['Total_Support'] = label_data['Support'].sum()
                stats['Avg_Support_Per_Fold'] = label_data['Support'].mean()
            
            stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def generate_per_class_plots(class_stats_df, output_dir):
    """
    Generate per-class performance visualizations.
    
    Args:
        class_stats_df: DataFrame with per-class statistics
        output_dir: Directory to save plots
    """
    if class_stats_df.empty:
        print("⚠ No per-class data to plot")
        return {}
    
    output_files = {}
    tasks = class_stats_df['Task'].unique()
    
    for task in tasks:
        task_data = class_stats_df[class_stats_df['Task'] == task].copy()
        # Sort by LABEL_NAME_MAPS order instead of F1-Score
        task_data['_sort_order'] = task_data['Label'].apply(get_label_sort_order)
        task_data = task_data.sort_values('_sort_order')
        
        labels = task_data['Label'].values
        # Create display labels with full names
        display_labels = [get_label_display_name(label) for label in labels]
        f1_mean = task_data['F1-Score_mean'].values
        f1_std = task_data['F1-Score_std'].values
        precision_mean = task_data['Precision_mean'].values
        recall_mean = task_data['Recall_mean'].values
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(labels) * 0.5)))
        
        # Plot 1: F1-Score with error bars
        y_pos = np.arange(len(labels))
        axes[0].barh(y_pos, f1_mean, xerr=f1_std, alpha=0.7, color='steelblue')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(display_labels, fontsize=9)
        axes[0].set_xlabel('F1-Score (Mean ± Std)', fontsize=11)
        axes[0].set_title(f'{task.replace("label_", "")} - F1-Score by Class', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlim([0, 1.05])
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].invert_yaxis()
        
        # Plot 2: Precision vs Recall
        x_pos = np.arange(len(labels))
        width = 0.35
        axes[1].barh(x_pos - width/2, precision_mean, width, label='Precision', alpha=0.7)
        axes[1].barh(x_pos + width/2, recall_mean, width, label='Recall', alpha=0.7)
        axes[1].set_yticks(x_pos)
        axes[1].set_yticklabels(display_labels, fontsize=9)
        axes[1].set_xlabel('Score (Mean)', fontsize=11)
        axes[1].set_title(f'{task.replace("label_", "")} - Precision vs Recall by Class',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 1.05])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'per_class_{task}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        output_files[f'per_class_{task}'] = plot_path
    
    # Create summary heatmap across all tasks and classes
    fig, axes = plt.subplots(1, 3, figsize=(20, max(8, len(class_stats_df) * 0.3)))
    
    for idx, metric in enumerate(['Precision_mean', 'Recall_mean', 'F1-Score_mean']):
        pivot_data = class_stats_df.pivot(index='Label', columns='Task', values=metric)
        pivot_data.columns = [col.replace('label_', '') for col in pivot_data.columns]
        
        # Create display names for y-axis labels
        display_index = [get_label_display_name(label) for label in pivot_data.index]
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=axes[idx], 
                   cbar_kws={'label': metric.replace('_mean', '')},
                   yticklabels=display_index)
        axes[idx].set_title(f'{metric.replace("_mean", "")} Heatmap', 
                          fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Task', fontsize=11)
        axes[idx].set_ylabel('Class', fontsize=11)
        axes[idx].tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'per_class_heatmap_all_tasks.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    output_files['per_class_heatmap_all'] = plot_path
    
    return output_files


def generate_fold_comparison_plots(summary_df, output_dir):
    """
    Generate plots comparing performance across folds.
    
    Args:
        summary_df: DataFrame with summary metrics from all folds
        output_dir: Directory to save plots
    """
    if summary_df.empty:
        print("⚠ No data to plot")
        return {}
    
    output_files = {}
    tasks = summary_df['Task'].unique()
    num_tasks = len(tasks)
    
    # Plot 1: Accuracy across folds for all tasks
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for task in tasks:
        task_df = summary_df[summary_df['Task'] == task].sort_values('fold')
        ax.plot(task_df['fold'], task_df['Accuracy'], marker='o', 
                label=task.replace('label_', ''), linewidth=2, markersize=8)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Across Folds for All Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(summary_df['fold'].unique())))
    ax.set_ylim([0, 1.05])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'fold_comparison_accuracy.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    output_files['fold_comparison_accuracy'] = plot_path
    
    # Plot 2: Macro F1 across folds for all tasks
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for task in tasks:
        task_df = summary_df[summary_df['Task'] == task].sort_values('fold')
        ax.plot(task_df['fold'], task_df['Macro_F1'], marker='s', 
                label=task.replace('label_', ''), linewidth=2, markersize=8)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_title('Macro F1 Score Across Folds for All Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(summary_df['fold'].unique())))
    ax.set_ylim([0, 1.05])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'fold_comparison_macro_f1.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    output_files['fold_comparison_f1'] = plot_path
    
    # Plot 3: Box plots showing distribution across folds
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy box plot
    data_acc = []
    labels_acc = []
    for task in tasks:
        task_df = summary_df[summary_df['Task'] == task]
        data_acc.append(task_df['Accuracy'].values)
        labels_acc.append(task.replace('label_', ''))
    
    bp1 = axes[0].boxplot(data_acc, labels=labels_acc, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Distribution Across Folds', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.05])
    
    # Macro F1 box plot
    data_f1 = []
    for task in tasks:
        task_df = summary_df[summary_df['Task'] == task]
        data_f1.append(task_df['Macro_F1'].values)
    
    bp2 = axes[1].boxplot(data_f1, labels=labels_acc, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    axes[1].set_ylabel('Macro F1 Score', fontsize=12)
    axes[1].set_title('Macro F1 Distribution Across Folds', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fold_distribution_boxplots.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    output_files['fold_distribution'] = plot_path
    
    # Plot 4: Heatmap of performance across tasks and folds
    pivot_acc = summary_df.pivot(index='Task', columns='fold', values='Accuracy')
    pivot_acc.index = [idx.replace('label_', '') for idx in pivot_acc.index]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, num_tasks * 0.6)))
    
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', 
                vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Accuracy'})
    axes[0].set_title('Accuracy Heatmap: Tasks × Folds', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('Task', fontsize=12)
    
    pivot_f1 = summary_df.pivot(index='Task', columns='fold', values='Macro_F1')
    pivot_f1.index = [idx.replace('label_', '') for idx in pivot_f1.index]
    
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Macro F1'})
    axes[1].set_title('Macro F1 Heatmap: Tasks × Folds', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Fold', fontsize=12)
    axes[1].set_ylabel('Task', fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_heatmap.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    output_files['performance_heatmap'] = plot_path
    
    return output_files


def generate_summary_report(cv_stats_df, summary_df, class_stats_df, output_dir, cv_dir, missing_folds):
    """
    Generate comprehensive markdown report.
    
    Args:
        cv_stats_df: DataFrame with aggregated CV statistics
        summary_df: DataFrame with per-fold summary metrics
        class_stats_df: DataFrame with per-class statistics
        output_dir: Directory to save report
        cv_dir: Path to CV directory
        missing_folds: List of missing fold indices
    """
    report_path = os.path.join(output_dir, 'cv_summary_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Cross-Validation Performance Summary\n\n")
        f.write(f"**Experiment Directory**: `{cv_dir}`\n\n")
        f.write(f"**Number of Folds**: {len(summary_df['fold'].unique())}\n")
        if missing_folds:
            f.write(f"**Missing Folds**: {missing_folds}\n")
        f.write("\n---\n\n")
        
        # Overall statistics
        f.write("## Overall Performance Statistics\n\n")
        f.write("Mean ± Standard Deviation across all folds:\n\n")
        
        # Format the CV statistics table for better readability
        display_df = cv_stats_df[['Task', 'Accuracy_mean', 'Accuracy_std', 
                                   'Macro_F1_mean', 'Macro_F1_std',
                                   'Weighted_F1_mean', 'Weighted_F1_std']].copy()
        display_df.columns = ['Task', 'Accuracy (mean)', 'Accuracy (std)', 
                              'Macro F1 (mean)', 'Macro F1 (std)',
                              'Weighted F1 (mean)', 'Weighted F1 (std)']
        display_df['Task'] = display_df['Task'].str.replace('label_', '')
        
        f.write(display_df.round(4).to_markdown(index=False))
        f.write("\n\n")
        
        # Best and worst performing tasks
        f.write("## Performance Rankings\n\n")
        
        f.write("### By Accuracy\n\n")
        sorted_by_acc = cv_stats_df.sort_values('Accuracy_mean', ascending=False)
        f.write("**Top 3 Tasks:**\n")
        for i, row in sorted_by_acc.head(3).iterrows():
            task = row['Task'].replace('label_', '')
            f.write(f"{i+1}. **{task}**: {row['Accuracy_mean']:.4f} ± {row['Accuracy_std']:.4f}\n")
        
        f.write("\n**Bottom 3 Tasks:**\n")
        for i, row in sorted_by_acc.tail(3).iterrows():
            task = row['Task'].replace('label_', '')
            f.write(f"- **{task}**: {row['Accuracy_mean']:.4f} ± {row['Accuracy_std']:.4f}\n")
        
        f.write("\n### By Macro F1 Score\n\n")
        sorted_by_f1 = cv_stats_df.sort_values('Macro_F1_mean', ascending=False)
        f.write("**Top 3 Tasks:**\n")
        for i, row in sorted_by_f1.head(3).iterrows():
            task = row['Task'].replace('label_', '')
            f.write(f"{i+1}. **{task}**: {row['Macro_F1_mean']:.4f} ± {row['Macro_F1_std']:.4f}\n")
        
        f.write("\n**Bottom 3 Tasks:**\n")
        for i, row in sorted_by_f1.tail(3).iterrows():
            task = row['Task'].replace('label_', '')
            f.write(f"- **{task}**: {row['Macro_F1_mean']:.4f} ± {row['Macro_F1_std']:.4f}\n")
        
        f.write("\n---\n\n")
        
        # Per-task detailed statistics
        f.write("## Detailed Task-by-Task Analysis\n\n")
        
        for _, row in cv_stats_df.iterrows():
            task = row['Task'].replace('label_', '')
            f.write(f"### {task}\n\n")
            
            f.write("| Metric | Mean | Std | Min | Max | Median |\n")
            f.write("|--------|------|-----|-----|-----|--------|\n")
            
            metrics_to_show = [
                ('Accuracy', 'Accuracy'),
                ('Macro_Precision', 'Macro Precision'),
                ('Macro_Recall', 'Macro Recall'),
                ('Macro_F1', 'Macro F1'),
                ('Weighted_F1', 'Weighted F1')
            ]
            
            for metric_key, metric_name in metrics_to_show:
                if f'{metric_key}_mean' in row:
                    f.write(f"| {metric_name} | {row[f'{metric_key}_mean']:.4f} | "
                           f"{row[f'{metric_key}_std']:.4f} | {row[f'{metric_key}_min']:.4f} | "
                           f"{row[f'{metric_key}_max']:.4f} | {row[f'{metric_key}_median']:.4f} |\n")
            
            f.write("\n")
        
        f.write("\n---\n\n")
        
        # Fold-by-fold comparison
        f.write("## Fold-by-Fold Performance\n\n")
        
        for fold_idx in sorted(summary_df['fold'].unique()):
            fold_df = summary_df[summary_df['fold'] == fold_idx]
            avg_acc = fold_df['Accuracy'].mean()
            avg_f1 = fold_df['Macro_F1'].mean()
            
            f.write(f"### Fold {fold_idx}\n\n")
            f.write(f"- **Average Accuracy**: {avg_acc:.4f}\n")
            f.write(f"- **Average Macro F1**: {avg_f1:.4f}\n")
            f.write(f"- **Test Samples**: {fold_df['Samples'].iloc[0] if len(fold_df) > 0 else 'N/A'}\n")
            f.write("\n")
        
        f.write("\n---\n\n")
        
        # Overall summary
        f.write("## Summary\n\n")
        overall_acc = cv_stats_df['Accuracy_mean'].mean()
        overall_acc_std = cv_stats_df['Accuracy_std'].mean()
        overall_f1 = cv_stats_df['Macro_F1_mean'].mean()
        overall_f1_std = cv_stats_df['Macro_F1_std'].mean()
        
        f.write(f"- **Overall Average Accuracy**: {overall_acc:.4f} (avg std: {overall_acc_std:.4f})\n")
        f.write(f"- **Overall Average Macro F1**: {overall_f1:.4f} (avg std: {overall_f1_std:.4f})\n")
        f.write(f"- **Number of Tasks**: {len(cv_stats_df)}\n")
        f.write(f"- **Number of Folds Evaluated**: {len(summary_df['fold'].unique())}\n")
        
        if 'Total_Samples' in cv_stats_df.columns:
            total_samples = cv_stats_df['Total_Samples'].iloc[0]
            f.write(f"- **Total Test Samples Across All Folds**: {int(total_samples)}\n")
        
        f.write("\n---\n\n")
        
        # Per-class statistics section
        if not class_stats_df.empty:
            f.write("## Per-Class Performance Statistics\n\n")
            f.write("Mean ± Standard Deviation for each class across all folds:\n\n")
            
            for task in class_stats_df['Task'].unique():
                task_data = class_stats_df[class_stats_df['Task'] == task].copy()
                # Sort by LABEL_NAME_MAPS order instead of F1-Score
                task_data['_sort_order'] = task_data['Label'].apply(get_label_sort_order)
                task_data = task_data.sort_values('_sort_order')
                
                f.write(f"### {task.replace('label_', '')}\n\n")
                
                # Add display names to the data
                task_data_display = task_data.copy()
                task_data_display['Class'] = task_data_display['Label'].apply(
                    lambda x: get_label_display_name(x)
                )
                
                display_cols = ['Class', 'Precision_mean', 'Precision_std', 
                               'Recall_mean', 'Recall_std', 'F1-Score_mean', 'F1-Score_std',
                               'Total_Support']
                display_data = task_data_display[display_cols].copy()
                display_data.columns = ['Class', 'Precision (mean)', 'Precision (std)',
                                       'Recall (mean)', 'Recall (std)', 
                                       'F1-Score (mean)', 'F1-Score (std)', 'Total Support']
                
                f.write(display_data.round(4).to_markdown(index=False))
                f.write("\n\n")
                
                # Identify best and worst performing classes by F1-Score
                best_idx = task_data['F1-Score_mean'].idxmax()
                worst_idx = task_data['F1-Score_mean'].idxmin()
                best_class = task_data.loc[best_idx]
                worst_class = task_data.loc[worst_idx]
                
                best_display = get_label_display_name(best_class['Label'])
                worst_display = get_label_display_name(worst_class['Label'])
                
                f.write(f"**Best performing class**: {best_display} "
                       f"(F1: {best_class['F1-Score_mean']:.4f} ± {best_class['F1-Score_std']:.4f})\n\n")
                f.write(f"**Worst performing class**: {worst_display} "
                       f"(F1: {worst_class['F1-Score_mean']:.4f} ± {worst_class['F1-Score_std']:.4f})\n\n")
            
            f.write("\n---\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("### Task-Level Performance\n\n")
        f.write("- [Fold Comparison - Accuracy](fold_comparison_accuracy.png)\n")
        f.write("- [Fold Comparison - Macro F1](fold_comparison_macro_f1.png)\n")
        f.write("- [Performance Distribution](fold_distribution_boxplots.png)\n")
        f.write("- [Performance Heatmap](performance_heatmap.png)\n\n")
        
        if not class_stats_df.empty:
            f.write("### Per-Class Performance\n\n")
            for task in class_stats_df['Task'].unique():
                task_name = task.replace('label_', '')
                f.write(f"- [{task_name} - Per-Class Metrics](per_class_{task}.png)\n")
            f.write("- [All Tasks - Per-Class Heatmap](per_class_heatmap_all_tasks.png)\n")
    
    return report_path


def main():
    """
    Main entry point for CV summary script.
    """
    parser = argparse.ArgumentParser(
        description='Summarize cross-validation performance across folds.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python summarize_cv.py logs/20260209_160303_5fold_cv
  python summarize_cv.py logs/20260209_160303_5fold_cv --num_folds 5 --output_dir cv_summary
        """
    )
    parser.add_argument('cv_dir', type=str,
                       help='Path to cross-validation directory containing fold_* subdirectories')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of folds (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save summary outputs (default: cv_dir/cv_summary)')
    parser.add_argument('--eval_fold', type=str, default='evaluation',
                       help='Name of the evaluation subdirectory within each fold (default: evaluation)')
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.cv_dir, 'cv_summary')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Cross-Validation Performance Summary")
    print("="*80)
    print(f"CV Directory: {args.cv_dir}")
    print(f"Number of Folds: {args.num_folds}")
    print(f"Output Directory: {output_dir}")
    print(f"Evaluation Subdirectory: {args.eval_fold}")
    print("="*80)
    
    # Load results from all folds
    print("\n📂 Loading fold results...")
    all_results, missing_folds = load_fold_results(args.cv_dir, args.num_folds, args.eval_fold)
    
    if all_results['summary'].empty:
        print("❌ Error: No fold results found. Please check the CV directory path.")
        return
    
    print(f"✓ Loaded results from {len(all_results['summary']['fold'].unique())} folds")
    
    # Compute CV statistics
    print("\n📊 Computing cross-validation statistics...")
    cv_stats_df = compute_cv_statistics(all_results['summary'])
    
    # Save aggregated statistics
    cv_stats_path = os.path.join(output_dir, 'cv_statistics.csv')
    cv_stats_df.to_csv(cv_stats_path, index=False)
    print(f"✓ CV statistics saved to {cv_stats_path}")
    
    # Compute per-class statistics
    print("\n📊 Computing per-class statistics...")
    class_stats_df = compute_per_class_statistics(all_results['detailed'])
    
    if not class_stats_df.empty:
        class_stats_path = os.path.join(output_dir, 'per_class_statistics.csv')
        class_stats_df.to_csv(class_stats_path, index=False)
        print(f"✓ Per-class statistics saved to {class_stats_path}")
    else:
        print("⚠ No per-class data available")
    
    # Save per-fold summary
    per_fold_path = os.path.join(output_dir, 'per_fold_summary.csv')
    all_results['summary'].to_csv(per_fold_path, index=False)
    print(f"✓ Per-fold summary saved to {per_fold_path}")
    
    # Save per-fold detailed metrics
    if not all_results['detailed'].empty:
        per_fold_detailed_path = os.path.join(output_dir, 'per_fold_detailed_metrics.csv')
        all_results['detailed'].to_csv(per_fold_detailed_path, index=False)
        print(f"✓ Per-fold detailed metrics saved to {per_fold_detailed_path}")
    
    # Generate task-level plots
    print("\n📈 Generating task-level comparison plots...")
    plot_files = generate_fold_comparison_plots(all_results['summary'], output_dir)
    for name, path in plot_files.items():
        print(f"✓ {name}: {path}")
    
    # Generate per-class plots
    if not class_stats_df.empty:
        print("\n📈 Generating per-class performance plots...")
        class_plot_files = generate_per_class_plots(class_stats_df, output_dir)
        for name, path in class_plot_files.items():
            print(f"✓ {name}: {path}")
        plot_files.update(class_plot_files)
    
    # Generate report
    print("\n📝 Generating summary report...")
    report_path = generate_summary_report(
        cv_stats_df, all_results['summary'], class_stats_df, output_dir, args.cv_dir, missing_folds
    )
    print(f"✓ Summary report saved to {report_path}")
    
    # Display key results
    print("\n" + "="*80)
    print("KEY RESULTS")
    print("="*80)
    
    overall_acc = cv_stats_df['Accuracy_mean'].mean()
    overall_acc_std = cv_stats_df['Accuracy_std'].mean()
    overall_f1 = cv_stats_df['Macro_F1_mean'].mean()
    overall_f1_std = cv_stats_df['Macro_F1_std'].mean()
    
    print(f"\n📊 Overall Performance:")
    print(f"  • Average Accuracy: {overall_acc:.4f} (avg std: {overall_acc_std:.4f})")
    print(f"  • Average Macro F1: {overall_f1:.4f} (avg std: {overall_f1_std:.4f})")
    
    print(f"\n🏆 Best Performing Task (by Accuracy):")
    best_task = cv_stats_df.loc[cv_stats_df['Accuracy_mean'].idxmax()]
    print(f"  • {best_task['Task'].replace('label_', '')}: "
          f"{best_task['Accuracy_mean']:.4f} ± {best_task['Accuracy_std']:.4f}")
    
    print(f"\n⚠️  Worst Performing Task (by Accuracy):")
    worst_task = cv_stats_df.loc[cv_stats_df['Accuracy_mean'].idxmin()]
    print(f"  • {worst_task['Task'].replace('label_', '')}: "
          f"{worst_task['Accuracy_mean']:.4f} ± {worst_task['Accuracy_std']:.4f}")
    
    print(f"\n✓ All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
