import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

# --- Label Name Mappings ---
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
    "Other": ["F"],
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
    """Get display name for a letter code."""
    if letter_code in LETTER_TO_LABEL_NAME:
        return f"{letter_code} - {LETTER_TO_LABEL_NAME[letter_code]}"
    return letter_code


def get_label_full_name(letter_code):
    """Get full label name for a letter code."""
    return LETTER_TO_LABEL_NAME.get(letter_code, letter_code)


def get_label_sort_order(letter_code):
    """Get sort order for a letter code."""
    return LETTER_SORT_ORDER.get(letter_code, 999)


# --- Configuration ---
LABEL_CSV_PATH = os.environ.get("LABEL_CSV_PATH", "")
METADATA_PATH = os.environ.get("METADATA_PATH", "")
LOCAL_DATASET_PATH_PREFIX = os.environ.get("LOCAL_DATASET_PATH", "")
TARGET_LABEL = "SequenceType_Code_norm"
OUTPUT_DIR = os.environ.get(
    "XGBOOST_OUTPUT_DIR",
    os.path.join("logs", f"xgboost_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
)

missing = [
    name
    for name, value in {
        "LABEL_CSV_PATH": LABEL_CSV_PATH,
        "METADATA_PATH": METADATA_PATH,
        "LOCAL_DATASET_PATH": LOCAL_DATASET_PATH_PREFIX,
    }.items()
    if not value
]
if missing:
    raise RuntimeError(
        "Missing Duke dataset configuration for the XGBoost baseline. Set the "
        f"environment variables {', '.join(missing)} before running this script."
    )

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# --- Load Data ---
print("Loading data...")
labels_df = pd.read_csv(LABEL_CSV_PATH)
metadata_df = pd.read_parquet(METADATA_PATH)

labels_df["Filepath"] = labels_df["Filepath"].apply(lambda x: os.path.join(LOCAL_DATASET_PATH_PREFIX, x))
labels_df.set_index("Filepath", inplace=True)
metadata_df.set_index("Filepath", inplace=True)

# --- Ensure Data Alignment ---
common_indices = labels_df.index.intersection(metadata_df.index)
print(f"Common samples: {len(common_indices)}/{len(labels_df)} labels, {len(metadata_df)} metadata")
labels_df = labels_df.loc[common_indices]
metadata_df = metadata_df.loc[common_indices]

# --- Prepare Data for CV ---
le = LabelEncoder()
X = metadata_df
y = labels_df[TARGET_LABEL]

# Fit label encoder once on all labels
le.fit(y)
print(f"Label classes: {le.classes_}")


# --- Cross-Validation Using Pre-defined Folds ---
print(f"Starting cross-validation on {len(X)} samples...")

folds = list(range(5))
all_predictions = []
all_true_labels = []
all_filepaths = []
fold_metrics = []
fold_class_metrics = []  # Store per-class metrics for each fold

for test_fold in folds:
    val_fold = (test_fold + 1) % len(folds)
    train_fold = list(set(folds) - {test_fold, val_fold})
    print(f"Training on folds: {train_fold}, Validating on fold: {val_fold}, Testing on fold: {test_fold}")

    # Get train indices and filter both X and y consistently
    train_mask = labels_df["split"].isin([f"fold_{f}" for f in train_fold])
    train_indices = labels_df[train_mask].index
    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    y_train = le.transform(y_train)  # Encode labels for training
    print(f"Training samples: {len(X_train)}, Class distribution: {np.bincount(y_train)}")

    # Get test indices and filter both X and y consistently
    test_mask = labels_df["split"] == f"fold_{test_fold}"
    test_indices = labels_df[test_mask].index
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    y_test = le.transform(y_test)  # Encode labels for testing
    print(f"Testing samples: {len(X_test)}, Class distribution: {np.bincount(y_test)}")

    # --- Train XGBoost ---
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        device="cuda",
    )

    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)

    # Store predictions and filepaths for this fold
    all_predictions.extend(y_pred)
    all_true_labels.extend(y_test)
    all_filepaths.extend(test_indices.tolist())

    # Generate classification report for this fold
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    print(
        f"Classification Report for fold_{test_fold}:\n{classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)}"
    )

    # Save fold metrics
    fold_metric = {
        "fold": test_fold,
        "samples": len(y_test),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    fold_metrics.append(fold_metric)

    # Collect per-class metrics for this fold
    for label in le.classes_:
        if label in report and isinstance(report[label], dict):
            fold_class_metrics.append(
                {
                    "fold": test_fold,
                    "class": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1-score": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
            )

    # Save fold predictions
    fold_pred_df = pd.DataFrame(
        {
            "Filepath": test_indices.tolist(),
            "True_Label": le.inverse_transform(y_test),
            "Predicted_Label": le.inverse_transform(y_pred),
        }
    )
    fold_pred_path = os.path.join(OUTPUT_DIR, f"predictions_fold_{test_fold}.csv")
    fold_pred_df.to_csv(fold_pred_path, index=False)
    print(f"✓ Fold {test_fold} predictions saved to {fold_pred_path}")

    # Generate confusion matrix for this fold
    cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))

    # Create display labels for confusion matrix
    cm_labels = [get_label_display_name(label) for label in le.classes_]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=cm_labels, yticklabels=cm_labels, cmap="Blues")
    plt.title(f"Confusion Matrix - Fold {test_fold}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_fold_{test_fold}.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}\n")

# --- Generate Overall CV Report ---
print("\n" + "=" * 80)
print("GENERATING CROSS-VALIDATION REPORT")
print("=" * 80)

# Save all predictions
all_pred_df = pd.DataFrame(
    {
        "Filepath": all_filepaths,
        "True_Label": le.inverse_transform(all_true_labels),
        "Predicted_Label": le.inverse_transform(all_predictions),
    }
)
all_pred_path = os.path.join(OUTPUT_DIR, "predictions_all_folds.csv")
all_pred_df.to_csv(all_pred_path, index=False)
print(f"✓ All predictions saved to {all_pred_path}")

# Generate overall metrics
overall_report = classification_report(
    all_true_labels, all_predictions, target_names=le.classes_, output_dict=True, zero_division=0
)

overall_metrics = {
    "total_samples": len(all_true_labels),
    "accuracy": overall_report["accuracy"],
    "macro_precision": overall_report["macro avg"]["precision"],
    "macro_recall": overall_report["macro avg"]["recall"],
    "macro_f1": overall_report["macro avg"]["f1-score"],
    "weighted_precision": overall_report["weighted avg"]["precision"],
    "weighted_recall": overall_report["weighted avg"]["recall"],
    "weighted_f1": overall_report["weighted avg"]["f1-score"],
}

print("\nOverall CV Performance:")
print(f"  Total samples: {overall_metrics['total_samples']}")
print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
print(f"  Macro F1: {overall_metrics['macro_f1']:.4f}")
print(f"  Weighted F1: {overall_metrics['weighted_f1']:.4f}")

# Save fold metrics summary
fold_metrics_df = pd.DataFrame(fold_metrics)
fold_metrics_path = os.path.join(OUTPUT_DIR, "fold_metrics_summary.csv")
fold_metrics_df.to_csv(fold_metrics_path, index=False)
print(f"\n✓ Fold metrics summary saved to {fold_metrics_path}")

# Save overall metrics
overall_metrics_path = os.path.join(OUTPUT_DIR, "overall_metrics.json")
with open(overall_metrics_path, "w") as f:
    json.dump(overall_metrics, f, indent=2)
print(f"✓ Overall metrics saved to {overall_metrics_path}")

# Generate overall confusion matrix
cm_overall = confusion_matrix(all_true_labels, all_predictions, labels=range(len(le.classes_)))

# Create display labels for confusion matrix
cm_labels = [get_label_display_name(label) for label in le.classes_]

plt.figure(figsize=(14, 12))
sns.heatmap(cm_overall, annot=True, fmt="d", xticklabels=cm_labels, yticklabels=cm_labels, cmap="Blues")
plt.title("Overall Confusion Matrix - All Folds", fontsize=14, fontweight="bold")
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
cm_overall_path = os.path.join(OUTPUT_DIR, "confusion_matrix_overall.png")
plt.savefig(cm_overall_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"✓ Overall confusion matrix saved to {cm_overall_path}")

# Compute per-class statistics across folds
fold_class_metrics_df = pd.DataFrame(fold_class_metrics)
fold_class_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "per_fold_class_metrics.csv"), index=False)
print("✓ Per-fold per-class metrics saved")

class_stats_list = []
for class_label in le.classes_:
    class_data = fold_class_metrics_df[fold_class_metrics_df["class"] == class_label]
    if not class_data.empty:
        class_stats = {
            "class_code": class_label,
            "class_name": get_label_full_name(class_label),
            "class_display": get_label_display_name(class_label),
            "sort_order": get_label_sort_order(class_label),
            "precision_mean": class_data["precision"].mean(),
            "precision_std": class_data["precision"].std(),
            "precision_min": class_data["precision"].min(),
            "precision_max": class_data["precision"].max(),
            "precision_median": class_data["precision"].median(),
            "recall_mean": class_data["recall"].mean(),
            "recall_std": class_data["recall"].std(),
            "recall_min": class_data["recall"].min(),
            "recall_max": class_data["recall"].max(),
            "recall_median": class_data["recall"].median(),
            "f1_mean": class_data["f1-score"].mean(),
            "f1_std": class_data["f1-score"].std(),
            "f1_min": class_data["f1-score"].min(),
            "f1_max": class_data["f1-score"].max(),
            "f1_median": class_data["f1-score"].median(),
            "total_support": class_data["support"].sum(),
        }
        class_stats_list.append(class_stats)

class_stats_df = pd.DataFrame(class_stats_list)
# Sort by label order
class_stats_df = class_stats_df.sort_values("sort_order").reset_index(drop=True)
class_stats_path = os.path.join(OUTPUT_DIR, "class_statistics.csv")
class_stats_df.to_csv(class_stats_path, index=False)
print(f"✓ Class statistics (mean±std across folds) saved to {class_stats_path}")

# Generate additional visualizations

# Plot: Line plot showing class performance across folds
print("\n📈 Generating additional visualizations...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for _, class_row in class_stats_df.iterrows():
    class_label = class_row["class_code"]
    display_name = class_row["class_display"]
    class_data = fold_class_metrics_df[fold_class_metrics_df["class"] == class_label].sort_values("fold")
    axes[0].plot(class_data["fold"], class_data["precision"], marker="o", label=display_name, linewidth=2)
    axes[1].plot(class_data["fold"], class_data["recall"], marker="s", label=display_name, linewidth=2)
    axes[2].plot(class_data["fold"], class_data["f1-score"], marker="^", label=display_name, linewidth=2)

for ax, metric in zip(axes, ["Precision", "Recall", "F1-Score"]):
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{metric} Across Folds by Class", fontsize=12, fontweight="bold")
    ax.set_xticks(folds)
    ax.set_ylim([0, 1.05])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
line_plot_path = os.path.join(OUTPUT_DIR, "class_performance_across_folds.png")
plt.savefig(line_plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"✓ Class performance line plot saved to {line_plot_path}")

# Plot: Box plots for class performance distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

data_precision = []
data_recall = []
data_f1 = []
labels_list = []

for _, class_row in class_stats_df.iterrows():
    class_label = class_row["class_code"]
    class_data = fold_class_metrics_df[fold_class_metrics_df["class"] == class_label]
    data_precision.append(class_data["precision"].values)
    data_recall.append(class_data["recall"].values)
    data_f1.append(class_data["f1-score"].values)
    labels_list.append(class_row["class_display"])

bp1 = axes[0].boxplot(data_precision, labels=labels_list, patch_artist=True)
for patch in bp1["boxes"]:
    patch.set_facecolor("lightblue")
axes[0].set_ylabel("Precision", fontsize=11)
axes[0].set_title("Precision Distribution Across Folds", fontsize=12, fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].set_ylim([0, 1.05])

bp2 = axes[1].boxplot(data_recall, labels=labels_list, patch_artist=True)
for patch in bp2["boxes"]:
    patch.set_facecolor("lightcoral")
axes[1].set_ylabel("Recall", fontsize=11)
axes[1].set_title("Recall Distribution Across Folds", fontsize=12, fontweight="bold")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(True, alpha=0.3, axis="y")
axes[1].set_ylim([0, 1.05])

bp3 = axes[2].boxplot(data_f1, labels=labels_list, patch_artist=True)
for patch in bp3["boxes"]:
    patch.set_facecolor("lightgreen")
axes[2].set_ylabel("F1-Score", fontsize=11)
axes[2].set_title("F1-Score Distribution Across Folds", fontsize=12, fontweight="bold")
axes[2].tick_params(axis="x", rotation=45)
axes[2].grid(True, alpha=0.3, axis="y")
axes[2].set_ylim([0, 1.05])

plt.tight_layout()
box_plot_path = os.path.join(OUTPUT_DIR, "class_distribution_boxplots.png")
plt.savefig(box_plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"✓ Class distribution box plots saved to {box_plot_path}")

# Plot: Per-class bar chart with error bars
fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(le.classes_) * 0.5)))

classes_display = class_stats_df["class_display"].values
f1_mean = class_stats_df["f1_mean"].values
f1_std = class_stats_df["f1_std"].values
precision_mean = class_stats_df["precision_mean"].values
recall_mean = class_stats_df["recall_mean"].values

y_pos = np.arange(len(classes_display))

axes[0].barh(y_pos, f1_mean, xerr=f1_std, alpha=0.7, color="steelblue")
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(classes_display, fontsize=10)
axes[0].set_xlabel("F1-Score (Mean ± Std)", fontsize=11)
axes[0].set_title("F1-Score by Class", fontsize=12, fontweight="bold")
axes[0].set_xlim([0, 1.05])
axes[0].grid(True, alpha=0.3, axis="x")
axes[0].invert_yaxis()

width = 0.35
axes[1].barh(y_pos - width / 2, precision_mean, width, label="Precision", alpha=0.7)
axes[1].barh(y_pos + width / 2, recall_mean, width, label="Recall", alpha=0.7)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(classes_display, fontsize=10)
axes[1].set_xlabel("Score (Mean)", fontsize=11)
axes[1].set_title("Precision vs Recall by Class", fontsize=12, fontweight="bold")
axes[1].set_xlim([0, 1.05])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="x")
axes[1].invert_yaxis()

plt.tight_layout()
per_class_plot_path = os.path.join(OUTPUT_DIR, "per_class_performance.png")
plt.savefig(per_class_plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"✓ Per-class performance plot saved to {per_class_plot_path}")

# Plot: Heatmap of class performance across folds
# Add display names to fold_class_metrics_df for plotting
fold_class_metrics_df["class_display"] = fold_class_metrics_df["class"].apply(get_label_display_name)

pivot_precision = fold_class_metrics_df.pivot(index="class_display", columns="fold", values="precision")
pivot_recall = fold_class_metrics_df.pivot(index="class_display", columns="fold", values="recall")
pivot_f1 = fold_class_metrics_df.pivot(index="class_display", columns="fold", values="f1-score")

# Reorder rows based on sort order
display_to_sort = {get_label_display_name(code): get_label_sort_order(code) for code in le.classes_}
sorted_indices = sorted(pivot_precision.index, key=lambda x: display_to_sort.get(x, 999))
pivot_precision = pivot_precision.loc[sorted_indices]
pivot_recall = pivot_recall.loc[sorted_indices]
pivot_f1 = pivot_f1.loc[sorted_indices]

fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(le.classes_) * 0.6)))

sns.heatmap(
    pivot_precision, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=axes[0], cbar_kws={"label": "Precision"}
)
axes[0].set_title("Precision Heatmap: Classes × Folds", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Fold", fontsize=11)
axes[0].set_ylabel("Class", fontsize=11)

sns.heatmap(
    pivot_recall, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, ax=axes[1], cbar_kws={"label": "Recall"}
)
axes[1].set_title("Recall Heatmap: Classes × Folds", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Fold", fontsize=11)
axes[1].set_ylabel("Class", fontsize=11)

sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=axes[2], cbar_kws={"label": "F1-Score"})
axes[2].set_title("F1-Score Heatmap: Classes × Folds", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Fold", fontsize=11)
axes[2].set_ylabel("Class", fontsize=11)

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "performance_heatmap.png")
plt.savefig(heatmap_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"✓ Performance heatmap saved to {heatmap_path}")

# Generate performance comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Accuracy per fold
folds_list = fold_metrics_df["fold"].tolist()
accuracy_list = fold_metrics_df["accuracy"].tolist()
axes[0, 0].bar(folds_list, accuracy_list, alpha=0.7, color="steelblue")
axes[0, 0].axhline(y=overall_metrics["accuracy"], color="red", linestyle="--", label="Overall")
axes[0, 0].set_xlabel("Fold")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_title("Accuracy by Fold")
axes[0, 0].set_ylim([0, 1])
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(accuracy_list):
    axes[0, 0].text(folds_list[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom")

# Plot 2: Macro F1 per fold
macro_f1_list = fold_metrics_df["macro_f1"].tolist()
axes[0, 1].bar(folds_list, macro_f1_list, alpha=0.7, color="coral")
axes[0, 1].axhline(y=overall_metrics["macro_f1"], color="red", linestyle="--", label="Overall")
axes[0, 1].set_xlabel("Fold")
axes[0, 1].set_ylabel("Macro F1 Score")
axes[0, 1].set_title("Macro F1 Score by Fold")
axes[0, 1].set_ylim([0, 1])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(macro_f1_list):
    axes[0, 1].text(folds_list[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom")

# Plot 3: Weighted F1 per fold
weighted_f1_list = fold_metrics_df["weighted_f1"].tolist()
axes[1, 0].bar(folds_list, weighted_f1_list, alpha=0.7, color="mediumseagreen")
axes[1, 0].axhline(y=overall_metrics["weighted_f1"], color="red", linestyle="--", label="Overall")
axes[1, 0].set_xlabel("Fold")
axes[1, 0].set_ylabel("Weighted F1 Score")
axes[1, 0].set_title("Weighted F1 Score by Fold")
axes[1, 0].set_ylim([0, 1])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(weighted_f1_list):
    axes[1, 0].text(folds_list[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom")

# Plot 4: Sample distribution per fold
samples_list = fold_metrics_df["samples"].tolist()
axes[1, 1].bar(folds_list, samples_list, alpha=0.7, color="mediumpurple")
axes[1, 1].set_xlabel("Fold")
axes[1, 1].set_ylabel("Number of Samples")
axes[1, 1].set_title("Sample Distribution by Fold")
axes[1, 1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(samples_list):
    axes[1, 1].text(folds_list[i], v + max(samples_list) * 0.01, f"{int(v)}", ha="center", va="bottom")

plt.tight_layout()
performance_plot_path = os.path.join(OUTPUT_DIR, "performance_overview.png")
plt.savefig(performance_plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"✓ Performance overview plot saved to {performance_plot_path}")

# Generate per-class metrics
detailed_metrics = []
for label, scores in overall_report.items():
    if isinstance(scores, dict) and "precision" in scores:
        detailed_metrics.append(
            {
                "Label": label,
                "Precision": scores["precision"],
                "Recall": scores["recall"],
                "F1-Score": scores["f1-score"],
                "Support": scores["support"],
            }
        )

detailed_metrics_df = pd.DataFrame(detailed_metrics)
detailed_metrics_path = os.path.join(OUTPUT_DIR, "detailed_metrics.csv")
detailed_metrics_df.to_csv(detailed_metrics_path, index=False)
print(f"✓ Detailed per-class metrics saved to {detailed_metrics_path}")

# Generate markdown report
report_path = os.path.join(OUTPUT_DIR, "cv_report.md")
with open(report_path, "w") as f:
    f.write("# XGBoost Cross-Validation Report\n\n")
    f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Target Label**: {TARGET_LABEL}\n\n")
    f.write(f"**Number of Folds**: {len(folds)}\n\n")
    f.write(f"**Number of Classes**: {len(le.classes_)}\n\n")

    f.write("---\n\n")

    f.write("## Class Label Mapping\n\n")
    f.write("| Code | Full Name |\n")
    f.write("|------|----------|\n")
    for _, row in class_stats_df.iterrows():
        f.write(f"| {row['class_code']} | {row['class_name']} |\n")
    f.write("\n")

    f.write("---\n\n")

    f.write("## Overall Performance\n\n")
    f.write(f"- **Total Samples**: {overall_metrics['total_samples']}\n")
    f.write(f"- **Accuracy**: {overall_metrics['accuracy']:.4f}\n")
    f.write(f"- **Macro Precision**: {overall_metrics['macro_precision']:.4f}\n")
    f.write(f"- **Macro Recall**: {overall_metrics['macro_recall']:.4f}\n")
    f.write(f"- **Macro F1**: {overall_metrics['macro_f1']:.4f}\n")
    f.write(f"- **Weighted Precision**: {overall_metrics['weighted_precision']:.4f}\n")
    f.write(f"- **Weighted Recall**: {overall_metrics['weighted_recall']:.4f}\n")
    f.write(f"- **Weighted F1**: {overall_metrics['weighted_f1']:.4f}\n\n")

    f.write("---\n\n")

    f.write("## Fold-Level Performance\n\n")
    f.write("Mean ± Standard Deviation across all folds:\n\n")

    display_fold_df = fold_metrics_df[["fold", "samples", "accuracy", "macro_f1", "weighted_f1"]].copy()
    display_fold_df.columns = ["Fold", "Samples", "Accuracy", "Macro F1", "Weighted F1"]
    f.write(display_fold_df.round(4).to_markdown(index=False))
    f.write("\n\n")

    f.write("### Fold Statistics\n\n")
    f.write(
        f"- **Mean Accuracy**: {fold_metrics_df['accuracy'].mean():.4f} ± {fold_metrics_df['accuracy'].std():.4f}\n"
    )
    f.write(
        f"- **Mean Macro F1**: {fold_metrics_df['macro_f1'].mean():.4f} ± {fold_metrics_df['macro_f1'].std():.4f}\n"
    )
    f.write(
        f"- **Mean Weighted F1**: {fold_metrics_df['weighted_f1'].mean():.4f} ± {fold_metrics_df['weighted_f1'].std():.4f}\n"
    )
    f.write(
        f"- **Best Fold (Accuracy)**: Fold {fold_metrics_df.loc[fold_metrics_df['accuracy'].idxmax(), 'fold']} ({fold_metrics_df['accuracy'].max():.4f})\n"
    )
    f.write(
        f"- **Worst Fold (Accuracy)**: Fold {fold_metrics_df.loc[fold_metrics_df['accuracy'].idxmin(), 'fold']} ({fold_metrics_df['accuracy'].min():.4f})\n\n"
    )

    f.write("---\n\n")

    f.write("## Per-Class Performance Statistics\n\n")
    f.write("Mean ± Standard Deviation for each class across all folds:\n\n")

    display_class_df = class_stats_df[
        [
            "class_display",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "total_support",
        ]
    ].copy()
    display_class_df.columns = [
        "Class",
        "Precision (mean)",
        "Precision (std)",
        "Recall (mean)",
        "Recall (std)",
        "F1-Score (mean)",
        "F1-Score (std)",
        "Total Support",
    ]
    f.write(display_class_df.round(4).to_markdown(index=False))
    f.write("\n\n")

    f.write("### Class Performance Rankings\n\n")

    # Sort by F1-score
    sorted_by_f1 = class_stats_df.sort_values("f1_mean", ascending=False)

    f.write("#### By F1-Score\n\n")
    f.write("**Top 3 Classes:**\n\n")
    for i, (idx, row) in enumerate(sorted_by_f1.head(3).iterrows(), 1):
        f.write(f"{i}. **{row['class_display']}**: {row['f1_mean']:.4f} ± {row['f1_std']:.4f}\n")

    f.write("\n**Bottom 3 Classes:**\n\n")
    for idx, row in sorted_by_f1.tail(3).iterrows():
        f.write(f"- **{row['class_display']}**: {row['f1_mean']:.4f} ± {row['f1_std']:.4f}\n")

    f.write("\n#### By Precision\n\n")
    sorted_by_precision = class_stats_df.sort_values("precision_mean", ascending=False)
    f.write("**Top 3 Classes:**\n\n")
    for i, (idx, row) in enumerate(sorted_by_precision.head(3).iterrows(), 1):
        f.write(f"{i}. **{row['class_display']}**: {row['precision_mean']:.4f} ± {row['precision_std']:.4f}\n")

    f.write("\n#### By Recall\n\n")
    sorted_by_recall = class_stats_df.sort_values("recall_mean", ascending=False)
    f.write("**Top 3 Classes:**\n\n")
    for i, (idx, row) in enumerate(sorted_by_recall.head(3).iterrows(), 1):
        f.write(f"{i}. **{row['class_display']}**: {row['recall_mean']:.4f} ± {row['recall_std']:.4f}\n")

    f.write("\n---\n\n")

    f.write("## Detailed Class Analysis\n\n")

    for _, row in class_stats_df.iterrows():
        class_display = row["class_display"]
        class_name = row["class_name"]
        f.write(f"### {class_display}\n\n")
        f.write(f"**Full Name**: {class_name}\n\n")

        f.write("| Metric | Mean | Std | Min | Max | Median |\n")
        f.write("|--------|------|-----|-----|-----|--------|\n")

        f.write(
            f"| Precision | {row['precision_mean']:.4f} | {row['precision_std']:.4f} | "
            f"{row['precision_min']:.4f} | {row['precision_max']:.4f} | {row['precision_median']:.4f} |\n"
        )
        f.write(
            f"| Recall | {row['recall_mean']:.4f} | {row['recall_std']:.4f} | "
            f"{row['recall_min']:.4f} | {row['recall_max']:.4f} | {row['recall_median']:.4f} |\n"
        )
        f.write(
            f"| F1-Score | {row['f1_mean']:.4f} | {row['f1_std']:.4f} | "
            f"{row['f1_min']:.4f} | {row['f1_max']:.4f} | {row['f1_median']:.4f} |\n"
        )

        f.write(f"\n**Total Support**: {int(row['total_support'])}\n\n")

    f.write("---\n\n")

    f.write("## Performance Across All Folds (by Class)\n\n")
    f.write("This section shows the performance of each class in every fold.\n\n")

    for _, row in class_stats_df.iterrows():
        class_code = row["class_code"]
        class_display = row["class_display"]
        class_name = row["class_name"]

        f.write(f"### {class_display}\n\n")

        # Get data for this class across all folds
        class_fold_data = fold_class_metrics_df[fold_class_metrics_df["class"] == class_code].sort_values("fold")

        if not class_fold_data.empty:
            fold_table = class_fold_data[["fold", "precision", "recall", "f1-score", "support"]].copy()
            fold_table.columns = ["Fold", "Precision", "Recall", "F1-Score", "Support"]
            f.write(fold_table.round(4).to_markdown(index=False))
            f.write("\n\n")

            # Add statistics
            f.write(f"**Statistics (across {len(class_fold_data)} folds):**\n")
            f.write(
                f"- Precision: {class_fold_data['precision'].mean():.4f} ± {class_fold_data['precision'].std():.4f} "
                f"(min: {class_fold_data['precision'].min():.4f}, max: {class_fold_data['precision'].max():.4f})\n"
            )
            f.write(
                f"- Recall: {class_fold_data['recall'].mean():.4f} ± {class_fold_data['recall'].std():.4f} "
                f"(min: {class_fold_data['recall'].min():.4f}, max: {class_fold_data['recall'].max():.4f})\n"
            )
            f.write(
                f"- F1-Score: {class_fold_data['f1-score'].mean():.4f} ± {class_fold_data['f1-score'].std():.4f} "
                f"(min: {class_fold_data['f1-score'].min():.4f}, max: {class_fold_data['f1-score'].max():.4f})\n"
            )
            f.write(f"- Total Support: {int(class_fold_data['support'].sum())}\n")
            f.write("\n")

    f.write("---\n\n")

    f.write("## Confusion Matrices\n\n")
    f.write("### Overall\n\n")
    f.write("![Overall Confusion Matrix](confusion_matrix_overall.png)\n\n")

    f.write("### By Fold\n\n")
    for fold in folds:
        f.write(f"#### Fold {fold}\n\n")
        f.write(f"![Confusion Matrix Fold {fold}](confusion_matrix_fold_{fold}.png)\n\n")

    f.write("---\n\n")

    f.write("## Performance Visualizations\n\n")

    f.write("### Fold-Level Performance\n\n")
    f.write("![Performance Overview](performance_overview.png)\n\n")

    f.write("### Per-Class Performance\n\n")
    f.write("![Class Performance Across Folds](class_performance_across_folds.png)\n\n")
    f.write("![Class Distribution Boxplots](class_distribution_boxplots.png)\n\n")
    f.write("![Per-Class Performance](per_class_performance.png)\n\n")
    f.write("![Performance Heatmap](performance_heatmap.png)\n\n")

    f.write("---\n\n")

    f.write("## Model Configuration\n\n")
    f.write("```python\n")
    f.write("XGBClassifier(\n")
    f.write("    objective='multi:softmax',\n")
    f.write(f"    num_class={len(le.classes_)},\n")
    f.write("    eval_metric='mlogloss',\n")
    f.write("    use_label_encoder=False,\n")
    f.write("    n_estimators=100,\n")
    f.write("    max_depth=6,\n")
    f.write("    learning_rate=0.1,\n")
    f.write("    subsample=0.8,\n")
    f.write("    colsample_bytree=0.8,\n")
    f.write("    device='cuda'\n")
    f.write(")\n")
    f.write("```\n\n")

    f.write("---\n\n")

    f.write("## Files Generated\n\n")
    f.write("### Predictions\n")
    f.write("- `predictions_all_folds.csv` - Combined predictions from all folds\n")
    f.write("- `predictions_fold_[0-4].csv` - Per-fold predictions\n\n")
    f.write("### Metrics\n")
    f.write("- `fold_metrics_summary.csv` - Fold-level metrics summary\n")
    f.write("- `overall_metrics.json` - Overall aggregated metrics\n")
    f.write("- `detailed_metrics.csv` - Detailed per-class metrics (overall)\n")
    f.write("- `per_fold_class_metrics.csv` - Per-class metrics for each fold\n")
    f.write("- `class_statistics.csv` - Per-class statistics (mean, std, min, max, median)\n\n")
    f.write("### Visualizations\n")
    f.write("- `confusion_matrix_overall.png` - Overall confusion matrix\n")
    f.write("- `confusion_matrix_fold_[0-4].png` - Per-fold confusion matrices\n")
    f.write("- `performance_overview.png` - Fold-level performance comparison\n")
    f.write("- `class_performance_across_folds.png` - Class metrics across folds (line plots)\n")
    f.write("- `class_distribution_boxplots.png` - Distribution of class performance\n")
    f.write("- `per_class_performance.png` - Per-class bar charts with error bars\n")
    f.write("- `performance_heatmap.png` - Heatmap of class performance across folds\n")

print(f"✓ Comprehensive CV report saved to {report_path}")

print("\n" + "=" * 80)
print("CROSS-VALIDATION COMPLETE")
print("=" * 80)
print(f"\nAll results saved to: {OUTPUT_DIR}")

print("\n📊 Overall Summary:")
print(f"  - Total samples: {overall_metrics['total_samples']}")
print(f"  - Number of classes: {len(le.classes_)}")
print(f"  - Overall accuracy: {overall_metrics['accuracy']:.4f}")
print(f"  - Mean fold accuracy: {fold_metrics_df['accuracy'].mean():.4f} ± {fold_metrics_df['accuracy'].std():.4f}")
print(f"  - Overall macro F1: {overall_metrics['macro_f1']:.4f}")
print(f"  - Mean fold macro F1: {fold_metrics_df['macro_f1'].mean():.4f} ± {fold_metrics_df['macro_f1'].std():.4f}")

print("\n🏆 Best Performing Class (by F1):")
best_class = class_stats_df.loc[class_stats_df["f1_mean"].idxmax()]
print(f"  - {best_class['class_display']}: {best_class['f1_mean']:.4f} ± {best_class['f1_std']:.4f}")

print("\n⚠️  Worst Performing Class (by F1):")
worst_class = class_stats_df.loc[class_stats_df["f1_mean"].idxmin()]
print(f"  - {worst_class['class_display']}: {worst_class['f1_mean']:.4f} ± {worst_class['f1_std']:.4f}")

print("\n📈 Generated Visualizations:")
print("  - Fold-level performance comparison")
print("  - Class performance across folds (line plots)")
print("  - Class distribution boxplots")
print("  - Per-class bar charts with error bars")
print("  - Performance heatmaps (classes × folds)")
print("  - Confusion matrices (overall + per fold)")

print("\n📄 View the full report at:")
print(f"  {report_path}")
print("=" * 80)
