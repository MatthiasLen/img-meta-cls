"""
Cross-Validation Summary Script for ADNI Dataset.

Aggregates evaluation results from all fold directories and generates a
comprehensive performance summary including:
- Per-fold and aggregate statistics (mean ± std, min, max, median)
- Per-class metrics across folds
- Visualisation plots (fold comparison, heatmaps, per-class bar charts)
- Markdown report

Usage
-----
    python -m IMC.net4_adni.summarize_cv ./logs/net04_adni_cv
    python -m IMC.net4_adni.summarize_cv ./logs/net04_adni_cv \\
        --num_folds 5 --output_dir ./logs/net04_adni_cv/cv_summary

Authors: Tuan Truong
Date: 2026
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# ADNI task / label definitions (mirror adni_dataloader_local.py)
# ---------------------------------------------------------------------------

ADNI_LABEL_NAMES = {
    "label_AcquisitionPlane": ["AX", "COR", "SAG", "na"],
    "label_SequenceContrast": ["ASL", "CAL", "DWI", "OTHER", "PD", "T1", "T2", "T2FLAIR", "na"],
    "label_Localizer":        ["yes", "no", "na"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fold_results(cv_dir: str, num_folds: int = 5, eval_subdir: str = "evaluation"):
    """
    Load evaluation results (summary_metrics.csv, detailed_metrics.csv) from all folds.

    Returns
    -------
    all_results : dict with keys 'summary', 'detailed'
    missing_folds : list of fold indices that could not be loaded
    """
    all_results = {"summary": [], "detailed": []}
    missing_folds = []

    for fold_idx in range(num_folds):
        fold_dir = Path(cv_dir) / f"fold_{fold_idx}"
        eval_dir = fold_dir / eval_subdir

        if not fold_dir.exists():
            missing_folds.append(fold_idx)
            print(f"⚠  fold_{fold_idx}: directory not found")
            continue

        if not eval_dir.exists():
            missing_folds.append(fold_idx)
            print(f"⚠  fold_{fold_idx}/{eval_subdir}: not found")
            continue

        summary_path = eval_dir / "summary_metrics.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["fold"] = fold_idx
            all_results["summary"].append(df)
        else:
            print(f"⚠  {summary_path} not found")

        detailed_path = eval_dir / "detailed_metrics.csv"
        if detailed_path.exists():
            df = pd.read_csv(detailed_path)
            df["fold"] = fold_idx
            all_results["detailed"].append(df)

    for key in ("summary", "detailed"):
        if all_results[key]:
            all_results[key] = pd.concat(all_results[key], ignore_index=True)
        else:
            all_results[key] = pd.DataFrame()

    return all_results, missing_folds


def compute_cv_statistics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean / std / min / max / median per task across folds."""
    if summary_df.empty:
        return pd.DataFrame()

    metrics = [
        "Accuracy", "Macro_F1", "Weighted_F1",
    ]
    # Add macro precision / recall if present
    for col in ("Macro_Precision", "Macro_Recall"):
        if col in summary_df.columns:
            metrics.append(col)

    rows = []
    for task in summary_df["Task"].unique():
        task_df = summary_df[summary_df["Task"] == task]
        stats: dict = {"Task": task}
        for m in metrics:
            if m in task_df.columns:
                vals = task_df[m].dropna().values
                stats[f"{m}_mean"]   = np.mean(vals)
                stats[f"{m}_std"]    = np.std(vals)
                stats[f"{m}_min"]    = np.min(vals)
                stats[f"{m}_max"]    = np.max(vals)
                stats[f"{m}_median"] = np.median(vals)
        if "Samples" in task_df.columns:
            stats["Total_Samples"]         = task_df["Samples"].sum()
            stats["Avg_Samples_Per_Fold"]  = task_df["Samples"].mean()
        rows.append(stats)

    return pd.DataFrame(rows)


def compute_per_class_statistics(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-class mean / std across folds (excludes avg rows)."""
    if detailed_df.empty:
        return pd.DataFrame()

    class_df = detailed_df[~detailed_df["Label"].str.contains("avg", case=False, na=False)].copy()
    if class_df.empty:
        return pd.DataFrame()

    metrics = ["Precision", "Recall", "F1-Score"]
    rows = []
    for task in class_df["Task"].unique():
        for label in class_df[class_df["Task"] == task]["Label"].unique():
            label_df = class_df[(class_df["Task"] == task) & (class_df["Label"] == label)]
            stats: dict = {"Task": task, "Label": label}
            for m in metrics:
                if m in label_df.columns:
                    vals = label_df[m].dropna().values
                    stats[f"{m}_mean"]   = np.mean(vals)
                    stats[f"{m}_std"]    = np.std(vals)
                    stats[f"{m}_min"]    = np.min(vals)
                    stats[f"{m}_max"]    = np.max(vals)
                    stats[f"{m}_median"] = np.median(vals)
            if "Support" in label_df.columns:
                stats["Total_Support"] = label_df["Support"].sum()
                stats["Avg_Support"]   = label_df["Support"].mean()
            rows.append(stats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_fold_comparison_plots(summary_df: pd.DataFrame, output_dir: str) -> dict:
    """Line plots + box plots comparing task performance across folds."""
    if summary_df.empty:
        print("⚠  No summary data — skipping fold comparison plots.")
        return {}

    output_files: dict = {}
    tasks = summary_df["Task"].unique()

    # --- Accuracy across folds ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for task in tasks:
        t_df = summary_df[summary_df["Task"] == task].sort_values("fold")
        ax.plot(t_df["fold"], t_df["Accuracy"], marker="o",
                label=task.replace("label_", ""), linewidth=2, markersize=7)
    ax.set(xlabel="Fold", ylabel="Accuracy",
           title="Accuracy Across Folds", ylim=[0, 1.05])
    ax.set_xticks(sorted(summary_df["fold"].unique()))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "fold_comparison_accuracy.png")
    plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    output_files["fold_comparison_accuracy"] = p

    # --- Macro F1 across folds ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for task in tasks:
        t_df = summary_df[summary_df["Task"] == task].sort_values("fold")
        ax.plot(t_df["fold"], t_df["Macro_F1"], marker="s",
                label=task.replace("label_", ""), linewidth=2, markersize=7)
    ax.set(xlabel="Fold", ylabel="Macro F1",
           title="Macro F1 Across Folds", ylim=[0, 1.05])
    ax.set_xticks(sorted(summary_df["fold"].unique()))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "fold_comparison_macro_f1.png")
    plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    output_files["fold_comparison_f1"] = p

    # --- Box plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    task_labels = [t.replace("label_", "") for t in tasks]
    data_acc = [summary_df[summary_df["Task"] == t]["Accuracy"].values for t in tasks]
    data_f1  = [summary_df[summary_df["Task"] == t]["Macro_F1"].values for t in tasks]

    for ax_idx, (ax, data, title, color) in enumerate(zip(
        axes,
        [data_acc, data_f1],
        ["Accuracy Distribution", "Macro F1 Distribution"],
        ["lightblue", "lightcoral"],
    )):
        bp = ax.boxplot(data, labels=task_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
        ax.set(title=title, ylim=[0, 1.05])
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(output_dir, "fold_distribution_boxplots.png")
    plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    output_files["fold_distribution"] = p

    # --- Heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(tasks) * 0.8)))
    for ax, metric, cmap, label in zip(
        axes,
        ["Accuracy", "Macro_F1"],
        ["YlGnBu", "YlOrRd"],
        ["Accuracy", "Macro F1"],
    ):
        pivot = summary_df.pivot(index="Task", columns="fold", values=metric)
        pivot.index = [i.replace("label_", "") for i in pivot.index]
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                    vmin=0, vmax=1, ax=ax, cbar_kws={"label": label})
        ax.set_title(f"{label} Heatmap: Tasks × Folds", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, "performance_heatmap.png")
    plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    output_files["performance_heatmap"] = p

    return output_files


def generate_per_class_plots(class_stats_df: pd.DataFrame, output_dir: str) -> dict:
    """Horizontal bar charts and heatmap for per-class F1 / P / R across folds."""
    if class_stats_df.empty:
        print("⚠  No per-class data — skipping per-class plots.")
        return {}

    output_files: dict = {}

    for task in class_stats_df["Task"].unique():
        task_df = class_stats_df[class_stats_df["Task"] == task].dropna(
            subset=["F1-Score_mean"]
        ).sort_values("F1-Score_mean", ascending=True)

        labels         = task_df["Label"].values
        f1_mean        = task_df["F1-Score_mean"].values
        f1_std         = task_df["F1-Score_std"].values
        precision_mean = task_df["Precision_mean"].values
        recall_mean    = task_df["Recall_mean"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(labels) * 0.5)))

        y_pos = np.arange(len(labels))
        axes[0].barh(y_pos, f1_mean, xerr=f1_std, alpha=0.75, color="steelblue")
        axes[0].set_yticks(y_pos); axes[0].set_yticklabels(labels, fontsize=9)
        axes[0].set(xlabel="F1-Score (Mean ± Std)",
                    title=f"{task.replace('label_', '')} — F1 by Class",
                    xlim=[0, 1.05])
        axes[0].grid(True, alpha=0.3, axis="x")

        width = 0.35
        axes[1].barh(y_pos - width / 2, precision_mean, width, label="Precision", alpha=0.75)
        axes[1].barh(y_pos + width / 2, recall_mean, width, label="Recall", alpha=0.75)
        axes[1].set_yticks(y_pos); axes[1].set_yticklabels(labels, fontsize=9)
        axes[1].set(xlabel="Score (Mean)",
                    title=f"{task.replace('label_', '')} — Precision vs Recall",
                    xlim=[0, 1.05])
        axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        p = os.path.join(output_dir, f"per_class_{task}.png")
        plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
        output_files[f"per_class_{task}"] = p

    # Overall heatmap across tasks
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(class_stats_df) // 3)))
    for idx, metric in enumerate(["Precision_mean", "Recall_mean", "F1-Score_mean"]):
        if metric not in class_stats_df.columns:
            continue
        pivot = class_stats_df.pivot(index="Label", columns="Task", values=metric)
        pivot.columns = [c.replace("label_", "") for c in pivot.columns]
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=axes[idx],
                    cbar_kws={"label": metric.replace("_mean", "")})
        axes[idx].set_title(metric.replace("_mean", ""), fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, "per_class_heatmap_all_tasks.png")
    plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    output_files["per_class_heatmap_all"] = p

    return output_files


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_summary_report(
    cv_stats_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    class_stats_df: pd.DataFrame,
    output_dir: str,
    cv_dir: str,
    missing_folds: list,
) -> str:
    report_path = os.path.join(output_dir, "cv_summary_report.md")

    with open(report_path, "w") as f:
        f.write("# Cross-Validation Performance Summary — ADNI\n\n")
        f.write(f"**Experiment Directory**: `{cv_dir}`\n\n")
        f.write(f"**Number of Folds**: {len(summary_df['fold'].unique())}\n")
        if missing_folds:
            f.write(f"**Missing Folds**: {missing_folds}\n")
        f.write("\n---\n\n")

        f.write("## Overall Performance Statistics\n\n")
        f.write("Mean ± Std across all folds:\n\n")

        display_cols = ["Task", "Accuracy_mean", "Accuracy_std",
                        "Macro_F1_mean", "Macro_F1_std",
                        "Weighted_F1_mean", "Weighted_F1_std"]
        display_cols = [c for c in display_cols if c in cv_stats_df.columns]
        display_df   = cv_stats_df[display_cols].copy()
        display_df["Task"] = display_df["Task"].str.replace("label_", "")
        f.write(display_df.round(4).to_markdown(index=False))
        f.write("\n\n")

        # Per-fold comparison
        f.write("## Fold-by-Fold Performance\n\n")
        for fold_idx in sorted(summary_df["fold"].unique()):
            fold_df  = summary_df[summary_df["fold"] == fold_idx]
            avg_acc  = fold_df["Accuracy"].mean()
            avg_f1   = fold_df["Macro_F1"].mean()
            n_samp   = fold_df["Samples"].iloc[0] if "Samples" in fold_df.columns and len(fold_df) > 0 else "N/A"
            f.write(f"### Fold {fold_idx}\n\n")
            f.write(f"- **Average Accuracy** : {avg_acc:.4f}\n")
            f.write(f"- **Average Macro F1** : {avg_f1:.4f}\n")
            f.write(f"- **Test Samples**     : {n_samp}\n\n")

        f.write("---\n\n")

        # Per-class section
        if not class_stats_df.empty:
            f.write("## Per-Class Performance Statistics\n\n")
            for task in class_stats_df["Task"].unique():
                task_data = class_stats_df[class_stats_df["Task"] == task].copy()
                task_data = task_data.sort_values("F1-Score_mean", ascending=False)
                f.write(f"### {task.replace('label_', '')}\n\n")
                display_cols = ["Label", "Precision_mean", "Precision_std",
                                "Recall_mean", "Recall_std",
                                "F1-Score_mean", "F1-Score_std", "Total_Support"]
                display_cols = [c for c in display_cols if c in task_data.columns]
                f.write(task_data[display_cols].round(4).to_markdown(index=False))
                f.write("\n\n")
            f.write("---\n\n")

        # Visualisations section
        f.write("## Visualisations\n\n")
        f.write("- [Accuracy — Fold Comparison](fold_comparison_accuracy.png)\n")
        f.write("- [Macro F1 — Fold Comparison](fold_comparison_macro_f1.png)\n")
        f.write("- [Distribution Box Plots](fold_distribution_boxplots.png)\n")
        f.write("- [Performance Heatmap](performance_heatmap.png)\n\n")
        if not class_stats_df.empty:
            for task in class_stats_df["Task"].unique():
                f.write(f"- [Per-Class: {task.replace('label_', '')}](per_class_{task}.png)\n")
            f.write("- [Per-Class Heatmap (all tasks)](per_class_heatmap_all_tasks.png)\n")

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize 5-fold CV performance for the ADNI net04 experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m IMC.net4_adni.summarize_cv ./logs/net04_adni_cv
  python -m IMC.net4_adni.summarize_cv ./logs/net04_adni_cv --output_dir ./logs/net04_adni_cv/cv_summary
        """,
    )
    parser.add_argument("cv_dir", type=str,
                        help="Path to CV directory containing fold_* subdirectories.")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <cv_dir>/cv_summary).")
    parser.add_argument("--eval_subdir", type=str, default="evaluation",
                        help="Name of evaluation subdirectory inside each fold dir.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.cv_dir, "cv_summary")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Cross-Validation Performance Summary — ADNI")
    print("=" * 80)
    print(f"CV Directory     : {args.cv_dir}")
    print(f"Folds            : {args.num_folds}")
    print(f"Output Directory : {output_dir}")
    print("=" * 80)

    # ---- Load ----
    print("\n📂 Loading fold results …")
    all_results, missing_folds = load_fold_results(args.cv_dir, args.num_folds, args.eval_subdir)

    if all_results["summary"].empty:
        print("❌ No fold results found — check --cv_dir and --eval_subdir.")
        return

    n_loaded = len(all_results["summary"]["fold"].unique())
    print(f"✓ Loaded results from {n_loaded}/{args.num_folds} folds")

    # ---- Statistics ----
    print("\n📊 Computing CV statistics …")
    cv_stats_df = compute_cv_statistics(all_results["summary"])
    cv_stats_df.to_csv(os.path.join(output_dir, "cv_statistics.csv"), index=False)
    print(f"✓ cv_statistics.csv saved")

    print("\n📊 Computing per-class statistics …")
    class_stats_df = compute_per_class_statistics(all_results["detailed"])
    if not class_stats_df.empty:
        class_stats_df.to_csv(os.path.join(output_dir, "per_class_statistics.csv"), index=False)
        print(f"✓ per_class_statistics.csv saved")

    all_results["summary"].to_csv(os.path.join(output_dir, "per_fold_summary.csv"), index=False)
    if not all_results["detailed"].empty:
        all_results["detailed"].to_csv(
            os.path.join(output_dir, "per_fold_detailed_metrics.csv"), index=False
        )

    # ---- Plots ----
    print("\n📈 Generating fold comparison plots …")
    plot_files = generate_fold_comparison_plots(all_results["summary"], output_dir)
    for name, path in plot_files.items():
        print(f"  ✓ {name}: {path}")

    if not class_stats_df.empty:
        print("\n📈 Generating per-class plots …")
        class_plot_files = generate_per_class_plots(class_stats_df, output_dir)
        for name, path in class_plot_files.items():
            print(f"  ✓ {name}: {path}")
        plot_files.update(class_plot_files)

    # ---- Report ----
    print("\n📝 Generating Markdown report …")
    report_path = generate_summary_report(
        cv_stats_df, all_results["summary"], class_stats_df,
        output_dir, args.cv_dir, missing_folds,
    )
    print(f"✓ Report: {report_path}")

    # ---- Key results ----
    print("\n" + "=" * 80)
    print("KEY RESULTS")
    print("=" * 80)
    if not cv_stats_df.empty and "Accuracy_mean" in cv_stats_df.columns:
        overall_acc = cv_stats_df["Accuracy_mean"].mean()
        overall_f1  = cv_stats_df["Macro_F1_mean"].mean()
        print(f"\n📊 Overall (avg across tasks):")
        print(f"   Accuracy  : {overall_acc:.4f}")
        print(f"   Macro F1  : {overall_f1:.4f}")

        best  = cv_stats_df.loc[cv_stats_df["Accuracy_mean"].idxmax()]
        worst = cv_stats_df.loc[cv_stats_df["Accuracy_mean"].idxmin()]
        print(f"\n🏆 Best task  : {best['Task'].replace('label_', '')}  "
              f"acc={best['Accuracy_mean']:.4f} ± {best['Accuracy_std']:.4f}")
        print(f"⚠️  Worst task : {worst['Task'].replace('label_', '')}  "
              f"acc={worst['Accuracy_mean']:.4f} ± {worst['Accuracy_std']:.4f}")

    print(f"\n✓ All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
