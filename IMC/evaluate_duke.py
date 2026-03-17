""" Evaluation script for Duke dataset prediction results.
Generates classification reports, confusion matrices, and performance plots.
Duke labels do not contain 'na' values, so evaluation is simplified.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import os
from IMC.data.constants import DUKE_LABEL_NAMES, DUKE_ORIGINAL_LABEL_NAMES


def evaluate_performance(pred_values, true_values, label_names):
    """
    Evaluate performance for Duke labels (no 'na' handling needed).
    
    Args:
        pred_values: Array of predicted values
        true_values: Array of true values
        label_names: List of label names for this task
        
    Returns:
        Dictionary containing evaluation metrics
    """
    pred_values = np.array(pred_values)
    true_values = np.array(true_values)
    
    # Get unique labels present in predictions/ground truth
    unique_labels = list(set(true_values) | set(pred_values))
    filtered_names = [name for name in label_names if name in unique_labels]
    
    results = {}
    
    try:
        report = classification_report(
            true_values, pred_values,
            labels=filtered_names,
            target_names=filtered_names,
            zero_division=0,
            output_dict=True
        )
        results = {
            'report': report,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall'],
            'samples': len(true_values)
        }
    except Exception as e:
        results = {'error': str(e)}
    
    return results


def run_evaluation(pred_df, labels_df, output_dir):
    """
    Run comprehensive evaluation on Duke dataset predictions.
    
    Args:
        pred_df: DataFrame with predictions
        labels_df: DataFrame with ground truth labels
        output_dir: Directory to save output files
        
    Returns:
        Tuple of (summary_df, detailed_metrics_df, output_files)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use Duke label mappings
    label_map = DUKE_ORIGINAL_LABEL_NAMES
    
    # Align dataframes by Filepath
    labels_df = labels_df.set_index('Filepath')
    pred_df = pred_df.set_index('Filepath')
    
    common_filepaths = pred_df.index.intersection(labels_df.index)
    pred_common = pred_df.loc[common_filepaths]
    labels_common = labels_df.loc[common_filepaths]
    
    print(f"Evaluating on {len(common_filepaths)} common samples")
    
    all_results = {}
    detailed_metrics = []
    output_files = {}
    
    # Evaluate each task
    for task_name in label_map.keys():
        if task_name in pred_common.columns and task_name in labels_common.columns:
            pred_values = pred_common[task_name].values
            true_values = labels_common[task_name].values
            label_names = label_map[task_name]
            
            print(f"\nEvaluating {task_name}...")
            results = evaluate_performance(pred_values, true_values, label_names)
            all_results[task_name] = results
            
            # Extract detailed metrics per label
            if 'report' in results:
                report = results['report']
                for label, scores in report.items():
                    if isinstance(scores, dict) and 'precision' in scores:
                        detailed_metrics.append({
                            'Task': task_name,
                            'Label': label,
                            'Precision': scores.get('precision'),
                            'Recall': scores.get('recall'),
                            'F1-Score': scores.get('f1-score'),
                            'Support': scores.get('support')
                        })
            
            # Generate confusion matrix
            cm = confusion_matrix(true_values, pred_values, labels=label_names)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, 
                       yticklabels=label_names, cmap='Blues')
            plt.title(f'Confusion Matrix for {task_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            cm_path = os.path.join(output_dir, f'confusion_matrix_{task_name}.png')
            plt.savefig(cm_path, bbox_inches='tight', dpi=150)
            plt.close()
            output_files[f'confusion_matrix_{task_name}'] = cm_path
            print(f"  ✓ Confusion matrix saved to {cm_path}")
    
    # Save detailed metrics
    detailed_metrics_df = pd.DataFrame(detailed_metrics)
    detailed_metrics_path = os.path.join(output_dir, 'detailed_metrics.csv')
    detailed_metrics_df.to_csv(detailed_metrics_path, index=False)
    output_files['detailed_metrics'] = detailed_metrics_path
    print(f"\n✓ Detailed metrics saved to {detailed_metrics_path}")
    
    # Create summary dataframe
    summary_data = []
    for task_name, results in all_results.items():
        if 'error' not in results:
            row = {
                'Task': task_name,
                'Samples': results['samples'],
                'Accuracy': results['accuracy'],
                'Macro_Precision': results['macro_precision'],
                'Macro_Recall': results['macro_recall'],
                'Macro_F1': results['macro_f1'],
                'Weighted_Precision': results['weighted_precision'],
                'Weighted_Recall': results['weighted_recall'],
                'Weighted_F1': results['weighted_f1'],
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_metrics_path = os.path.join(output_dir, 'summary_metrics.csv')
    summary_df.to_csv(summary_metrics_path, index=False)
    output_files['summary_metrics'] = summary_metrics_path
    print(f"✓ Summary metrics saved to {summary_metrics_path}")
    
    # Generate performance comparison plots
    generate_performance_plots(summary_df, output_dir, output_files)
    
    # Generate evaluation report
    generate_report(summary_df, detailed_metrics_df, output_files, output_dir)
    
    return summary_df, detailed_metrics_df, output_files


def generate_performance_plots(summary_df, output_dir, output_files):
    """
    Generate performance comparison plots.
    
    Args:
        summary_df: Summary metrics dataframe
        output_dir: Directory to save plots
        output_files: Dictionary to store output file paths
    """
    tasks = summary_df['Task'].tolist()
    task_labels = [t.replace('label_', '') for t in tasks]
    x = np.arange(len(tasks))
    
    # Create comprehensive performance plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Accuracy
    accuracy = summary_df['Accuracy'].tolist()
    axes[0, 0].bar(x, accuracy, alpha=0.7, color='steelblue')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Classification Accuracy by Task')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(task_labels, rotation=45, ha='right')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accuracy):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 2: Macro F1 Score
    macro_f1 = summary_df['Macro_F1'].tolist()
    axes[0, 1].bar(x, macro_f1, alpha=0.7, color='coral')
    axes[0, 1].set_ylabel('Macro F1 Score')
    axes[0, 1].set_title('Macro F1 Score by Task')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(task_labels, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(macro_f1):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 3: Weighted F1 Score
    weighted_f1 = summary_df['Weighted_F1'].tolist()
    axes[1, 0].bar(x, weighted_f1, alpha=0.7, color='mediumseagreen')
    axes[1, 0].set_ylabel('Weighted F1 Score')
    axes[1, 0].set_title('Weighted F1 Score by Task')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(task_labels, rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(weighted_f1):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 4: Precision vs Recall
    macro_precision = summary_df['Macro_Precision'].tolist()
    macro_recall = summary_df['Macro_Recall'].tolist()
    width = 0.35
    axes[1, 1].bar(x - width/2, macro_precision, width, label='Precision', alpha=0.7)
    axes[1, 1].bar(x + width/2, macro_recall, width, label='Recall', alpha=0.7)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Macro Precision vs Recall by Task')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(task_labels, rotation=45, ha='right')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_overview.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    output_files['performance_overview_plot'] = plot_path
    print(f"✓ Performance overview plot saved to {plot_path}")
    
    # Sample distribution plot
    plt.figure(figsize=(12, 6))
    samples = summary_df['Samples'].tolist()
    bars = plt.bar(task_labels, samples, alpha=0.7, color='mediumpurple')
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution by Classification Task')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, samples):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(samples)*0.01, 
                f'{int(count)}', ha='center', va='bottom')
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, 'sample_distribution.png')
    plt.savefig(plot_path2, bbox_inches='tight', dpi=150)
    plt.close()
    output_files['sample_distribution_plot'] = plot_path2
    print(f"✓ Sample distribution plot saved to {plot_path2}")


def generate_report(summary_df, detailed_metrics_df, output_files, output_dir):
    """
    Generate evaluation report in markdown format.
    
    Args:
        summary_df: Summary metrics dataframe
        detailed_metrics_df: Detailed metrics dataframe
        output_files: Dictionary of output file paths
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Duke Dataset Evaluation Report\n\n")
        
        f.write("## Overall Performance Summary\n\n")
        f.write(summary_df.round(4).to_markdown(index=False))
        f.write("\n\n")
        
        # Key statistics
        f.write("## Key Statistics\n\n")
        f.write(f"- **Number of Tasks**: {len(summary_df)}\n")
        f.write(f"- **Average Accuracy**: {summary_df['Accuracy'].mean():.4f}\n")
        f.write(f"- **Average Macro F1**: {summary_df['Macro_F1'].mean():.4f}\n")
        f.write(f"- **Average Weighted F1**: {summary_df['Weighted_F1'].mean():.4f}\n")
        f.write(f"- **Best Performing Task (Accuracy)**: {summary_df.loc[summary_df['Accuracy'].idxmax(), 'Task']} ({summary_df['Accuracy'].max():.4f})\n")
        f.write(f"- **Worst Performing Task (Accuracy)**: {summary_df.loc[summary_df['Accuracy'].idxmin(), 'Task']} ({summary_df['Accuracy'].min():.4f})\n")
        f.write("\n")
        
        f.write("## Performance by Task\n\n")
        for _, row in summary_df.iterrows():
            task = row['Task'].replace('label_', '')
            f.write(f"### {task}\n\n")
            f.write(f"- **Samples**: {int(row['Samples'])}\n")
            f.write(f"- **Accuracy**: {row['Accuracy']:.4f}\n")
            f.write(f"- **Macro F1**: {row['Macro_F1']:.4f}\n")
            f.write(f"- **Weighted F1**: {row['Weighted_F1']:.4f}\n")
            f.write(f"- **Precision (Macro)**: {row['Macro_Precision']:.4f}\n")
            f.write(f"- **Recall (Macro)**: {row['Macro_Recall']:.4f}\n")
            f.write("\n")
        
        f.write("## Detailed Per-Class Metrics\n\n")
        f.write(detailed_metrics_df.round(4).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Confusion Matrices\n\n")
        for task_name in DUKE_LABEL_NAMES.keys():
            cm_key = f'confusion_matrix_{task_name}'
            if cm_key in output_files:
                f.write(f"### {task_name.replace('label_', '')}\n\n")
                f.write(f"![Confusion Matrix for {task_name}](confusion_matrix_{task_name}.png)\n\n")
        
        f.write("## Performance Visualizations\n\n")
        if 'performance_overview_plot' in output_files:
            f.write("### Performance Overview\n\n")
            f.write("![Performance Overview](performance_overview.png)\n\n")
        
        if 'sample_distribution_plot' in output_files:
            f.write("### Sample Distribution\n\n")
            f.write("![Sample Distribution](sample_distribution.png)\n\n")
    
    output_files['report'] = report_path
    print(f"✓ Evaluation report saved to {report_path}")


def main():
    """
    Main entry point for command-line usage.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate Duke dataset prediction results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate_duke.py predictions.csv labels.csv --output_dir results
        """
    )
    parser.add_argument('prediction_csv', type=str, 
                       help='Path to the prediction CSV file.')
    parser.add_argument('label_csv', type=str, 
                       help='Path to the ground truth label CSV file.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                       help='Directory to save the output files (default: evaluation_results)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Duke Dataset Evaluation")
    print("="*80)
    print(f"Predictions: {args.prediction_csv}")
    print(f"Labels: {args.label_csv}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Load data
    pred_df = pd.read_csv(args.prediction_csv)
    labels_df = pd.read_csv(args.label_csv)
    
    print(f"\nLoaded {len(pred_df)} predictions")
    print(f"Loaded {len(labels_df)} ground truth labels")
    
    # Run evaluation
    summary_df, detailed_metrics_df, output_files = run_evaluation(
        pred_df, labels_df, args.output_dir
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    for name, path in output_files.items():
        print(f"  - {name}: {path}")
    
    print(f"\n✓ All results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
