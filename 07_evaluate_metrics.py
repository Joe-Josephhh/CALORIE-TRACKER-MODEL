"""
07_evaluate_metrics.py — Comprehensive Evaluation for best_model_finetuned.keras

What this file does:
1. Loads the fine-tuned model (best_model_finetuned.keras) and 80 class labels.
2. Runs batch inference over all 600 test split images.
3. Computes the Confusion Matrix and plots a high-resolution heatmap (outputs/results/confusion_matrix.png).
4. Generates a per-class Classification Report (Precision, Recall, F1-Score) and saves to CSV.
5. Identifies top-performing classes and the most common misclassifications for report analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---- Configuration ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CONFIG_DIR = Path("configs")
RESULTS_DIR = Path("outputs/results")
CHECKPOINT_PATH = Path("outputs/checkpoints/best_model_finetuned.keras")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Step 1: Load Model & Data ----
print(f"Loading fine-tuned model: {CHECKPOINT_PATH} ...")
model = tf.keras.models.load_model(CHECKPOINT_PATH)

test_df = pd.read_csv(CONFIG_DIR / "test_split.csv")
class_names = pd.read_csv(CONFIG_DIR / "class_names.csv")["class_name"].tolist()
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for idx, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

print(f"Total test images: {len(test_df)} across {NUM_CLASSES} classes.")


def preprocess_image(filepath, label):
    image_bytes = tf.io.read_file(filepath)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image.set_shape([*IMG_SIZE, 3])
    image = preprocess_input(image)
    return image, label


test_ds = tf.data.Dataset.from_tensor_slices((
    test_df["filepath"].values,
    test_df["label"].map(class_to_idx).values
)).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---- Step 2: Compute Predictions ----
print("Running test predictions ...")
y_true = test_df["label"].map(class_to_idx).values
raw_preds = model.predict(test_ds, verbose=1)
y_pred = np.argmax(raw_preds, axis=1)

# Top-1 and Top-5 accuracy calculation
top_1_correct = np.sum(y_pred == y_true)
top_5_correct = np.sum([y_true[i] in np.argsort(raw_preds[i])[::-1][:5] for i in range(len(y_true))])

top_1_acc = (top_1_correct / len(y_true)) * 100
top_5_acc = (top_5_correct / len(y_true)) * 100

print(f"\nFinal Test Results:")
print(f"  Top-1 Accuracy: {top_1_acc:.2f}% ({top_1_correct}/{len(y_true)})")
print(f"  Top-5 Accuracy: {top_5_acc:.2f}% ({top_5_correct}/{len(y_true)})")

# ---- Step 3: Classification Report ----
report_dict = classification_report(
    y_true, y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = RESULTS_DIR / "classification_report.csv"
report_df.to_csv(report_csv_path)
print(f"Classification report saved to: {report_csv_path}")

# Print summary averages
macro_avg = report_dict["macro avg"]
weighted_avg = report_dict["weighted avg"]
print(f"\nMacro Average    -> Precision: {macro_avg['precision']:.4f}, Recall: {macro_avg['recall']:.4f}, F1: {macro_avg['f1-score']:.4f}")
print(f"Weighted Average -> Precision: {weighted_avg['precision']:.4f}, Recall: {weighted_avg['recall']:.4f}, F1: {weighted_avg['f1-score']:.4f}")

# ---- Step 4: Confusion Matrix Plot ----
print("Plotting Confusion Matrix heatmap ...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(26, 22))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=True
)
plt.title(f"Confusion Matrix — Fine-Tuned EfficientNetB0 (Top-1 Acc: {top_1_acc:.1f}%, Top-5 Acc: {top_5_acc:.1f}%)", fontsize=18, pad=20)
plt.xlabel("Predicted Class", fontsize=14, labelpad=10)
plt.ylabel("True Class", fontsize=14, labelpad=10)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()

cm_plot_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=200)
plt.close()
print(f"Confusion matrix plot saved to: {cm_plot_path}")

# ---- Step 5: Identify Top Classes & Common Confusions ----
class_f1 = []
for name in class_names:
    if name in report_dict:
        class_f1.append((name, report_dict[name]["f1-score"], report_dict[name]["support"]))

class_f1_sorted = sorted(class_f1, key=lambda x: -x[1])

print("\n--- TOP 10 BEST PERFORMING DISHES (by F1-score) ---")
for rank, (name, f1, sup) in enumerate(class_f1_sorted[:10], start=1):
    print(f"  {rank:2d}. {name:<35} F1: {f1:.2f} (samples: {int(sup)})")

# Find top confused pairs (where True != Pred)
confused_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((class_names[i], class_names[j], cm[i, j]))

confused_pairs_sorted = sorted(confused_pairs, key=lambda x: -x[2])

print("\n--- TOP 10 MOST CONFUSED DISH PAIRS ---")
for rank, (true_cls, pred_cls, count) in enumerate(confused_pairs_sorted[:10], start=1):
    print(f"  {rank:2d}. True: {true_cls:<30} -> Predicted: {pred_cls:<30} (Count: {count})")

# Save summary text file for project report
summary_text_path = RESULTS_DIR / "evaluation_summary.txt"
with open(summary_text_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("EVALUATION SUMMARY — BEST_MODEL_FINETUNED.KERAS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Test Images:     {len(y_true)}\n")
    f.write(f"Total Classes:   {NUM_CLASSES}\n")
    f.write(f"Top-1 Accuracy:  {top_1_acc:.2f}%\n")
    f.write(f"Top-5 Accuracy:  {top_5_acc:.2f}%\n")
    f.write(f"Macro F1-Score:  {macro_avg['f1-score']:.4f}\n")
    f.write(f"Weighted F1:     {weighted_avg['f1-score']:.4f}\n\n")
    f.write("TOP 10 BEST PERFORMING DISHES:\n")
    for rank, (name, f1, sup) in enumerate(class_f1_sorted[:10], start=1):
        f.write(f"  {rank:2d}. {name:<35} F1: {f1:.2f}\n")
    f.write("\nTOP 10 MOST CONFUSED PAIRS:\n")
    for rank, (true_cls, pred_cls, count) in enumerate(confused_pairs_sorted[:10], start=1):
        f.write(f"  {rank:2d}. True: {true_cls:<30} -> Predicted: {pred_cls:<30} (Count: {count})\n")

print(f"\nEvaluation summary saved to: {summary_text_path}")
