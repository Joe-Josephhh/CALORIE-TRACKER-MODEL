"""
07_evaluate_metrics.py — Comprehensive Evaluation & System Checking for best_model_finetuned.keras

What this file does:
1. Loads the fine-tuned model (best_model_finetuned.keras) and 80 class labels.
2. Runs system performance benchmarking (Parameters, Model Disk Size, Inference Latency, FPS).
3. Executes batch test inference on all 600 held-out test images.
4. Computes Top-1, Top-5 Accuracy, Precision, Recall, and F1-Scores (Macro & Weighted).
5. Generates the high-resolution Confusion Matrix heatmap plot (outputs/results/confusion_matrix.png).
6. Exports full per-class classification metrics (classification_report.csv).
7. Compiles an executive technical summary report (evaluation_summary.txt).
"""

from pathlib import Path
import time
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

# ---- Step 1: Model Loading & System Benchmark ----
print("=" * 70)
print("  EVALUATION & SYSTEM PERFORMANCE BENCHMARK")
print("=" * 70)
print(f"Loading model: {CHECKPOINT_PATH} ...")
model = tf.keras.models.load_model(CHECKPOINT_PATH)

test_df = pd.read_csv(CONFIG_DIR / "test_split.csv")
class_names = pd.read_csv(CONFIG_DIR / "class_names.csv")["class_name"].tolist()
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

# Parameter Counts
total_params = model.count_params()
trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
non_trainable_params = sum(int(np.prod(v.shape)) for v in model.non_trainable_variables)
model_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)

# Latency Benchmark (Warm-up + 30 runs)
print("Benchmarking single-image inference latency ...")
dummy_input = tf.random.normal([1, *IMG_SIZE, 3])
_ = model(dummy_input)  # warm-up
latencies = []
for _ in range(30):
    t0 = time.perf_counter()
    _ = model(dummy_input)
    latencies.append((time.perf_counter() - t0) * 1000)

avg_latency_ms = float(np.mean(latencies))
std_latency_ms = float(np.std(latencies))
throughput_fps = 1000.0 / avg_latency_ms

print(f"  • Model Size:           {model_size_mb:.2f} MB")
print(f"  • Total Parameters:     {total_params:,}")
print(f"  • Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
print(f"  • Non-Trainable:        {non_trainable_params:,} ({non_trainable_params/total_params*100:.1f}%)")
print(f"  • Average Latency:      {avg_latency_ms:.1f} ± {std_latency_ms:.1f} ms per image")
print(f"  • Inference Throughput: {throughput_fps:.1f} FPS (CPU)")

# ---- Step 2: Build Test Dataset & Predict ----
print(f"\nEvaluating on {len(test_df)} test images across {NUM_CLASSES} classes ...")


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

y_true = test_df["label"].map(class_to_idx).values
raw_preds = model.predict(test_ds, verbose=1)
y_pred = np.argmax(raw_preds, axis=1)

# Top-1 and Top-5 accuracy
top_1_correct = int(np.sum(y_pred == y_true))
top_5_correct = int(np.sum([y_true[i] in np.argsort(raw_preds[i])[::-1][:5] for i in range(len(y_true))]))

top_1_acc = (top_1_correct / len(y_true)) * 100
top_5_acc = (top_5_correct / len(y_true)) * 100

# ---- Step 3: Classification Metrics ----
report_dict = classification_report(
    y_true, y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = RESULTS_DIR / "classification_report.csv"
report_df.to_csv(report_csv_path)

macro_prec = report_dict["macro avg"]["precision"] * 100
macro_rec = report_dict["macro avg"]["recall"] * 100
macro_f1 = report_dict["macro avg"]["f1-score"] * 100

weighted_prec = report_dict["weighted avg"]["precision"] * 100
weighted_rec = report_dict["weighted avg"]["recall"] * 100
weighted_f1 = report_dict["weighted avg"]["f1-score"] * 100

print("\n" + "=" * 70)
print("  ACCURACY, PRECISION & RECALL SUMMARY")
print("=" * 70)
print(f"  • Top-1 Accuracy:     {top_1_acc:.2f}% ({top_1_correct} / {len(y_true)})")
print(f"  • Top-5 Accuracy:     {top_5_acc:.2f}% ({top_5_correct} / {len(y_true)})")
print(f"  • Macro Precision:    {macro_prec:.2f}%")
print(f"  • Macro Recall:       {macro_rec:.2f}%")
print(f"  • Macro F1-Score:     {macro_f1:.2f}%")
print(f"  • Weighted Precision: {weighted_prec:.2f}%")
print(f"  • Weighted Recall:    {weighted_rec:.2f}%")
print(f"  • Weighted F1-Score:  {weighted_f1:.2f}%")

# ---- Step 4: Confusion Matrix Plot ----
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
plt.title(f"Confusion Matrix — Fine-Tuned EfficientNetB0 (Top-1: {top_1_acc:.1f}%, Top-5: {top_5_acc:.1f}%)", fontsize=18, pad=20)
plt.xlabel("Predicted Class", fontsize=14, labelpad=10)
plt.ylabel("True Class", fontsize=14, labelpad=10)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()

cm_plot_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=200)
plt.close()

# ---- Step 5: Rank Classes & Confusions ----
class_stats = []
for name in class_names:
    if name in report_dict:
        class_stats.append({
            "name": name,
            "precision": report_dict[name]["precision"],
            "recall": report_dict[name]["recall"],
            "f1": report_dict[name]["f1-score"],
            "support": int(report_dict[name]["support"])
        })

top_classes = sorted(class_stats, key=lambda x: -x["f1"])[:10]
low_classes = sorted(class_stats, key=lambda x: x["f1"])[:10]

confused_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((class_names[i], class_names[j], int(cm[i, j])))

confused_pairs_sorted = sorted(confused_pairs, key=lambda x: -x[2])[:10]

# ---- Step 6: Generate Full Executive Summary Report ----
summary_text_path = RESULTS_DIR / "evaluation_summary.txt"
with open(summary_text_path, "w", encoding="utf-8") as f:
    f.write("=" * 76 + "\n")
    f.write("          SYSTEM CHECK & MODEL PERFORMANCE EVALUATION REPORT\n")
    f.write("=" * 76 + "\n\n")

    f.write("1. SYSTEM & ARCHITECTURE SPECIFICATIONS\n")
    f.write("-" * 76 + "\n")
    f.write(f"  • Model Architecture:          EfficientNetB0 (Fine-Tuned Transfer Learning)\n")
    f.write(f"  • Model Checkpoint File:       outputs/checkpoints/best_model_finetuned.keras\n")
    f.write(f"  • Model Checkpoint Size:       {model_size_mb:.2f} MB\n")
    f.write(f"  • Input Dimensions:            {IMG_SIZE[0]} x {IMG_SIZE[1]} x 3 (RGB)\n")
    f.write(f"  • Output Classes:              {NUM_CLASSES} Categories (Softmax probabilities)\n")
    f.write(f"  • Total Parameters:            {total_params:,}\n")
    f.write(f"  • Trainable Parameters:        {trainable_params:,} ({trainable_params/total_params*100:.1f}%)\n")
    f.write(f"  • Non-Trainable Parameters:    {non_trainable_params:,} ({non_trainable_params/total_params*100:.1f}%)\n")
    f.write(f"  • Inference Latency (CPU):     {avg_latency_ms:.2f} ms per image (± {std_latency_ms:.2f} ms)\n")
    f.write(f"  • Inference Throughput (CPU):  {throughput_fps:.1f} FPS (Images / Second)\n\n")

    f.write("2. OVERALL CLASSIFICATION PERFORMANCE (TEST SET: 600 IMAGES)\n")
    f.write("-" * 76 + "\n")
    f.write(f"  • Top-1 Accuracy:              {top_1_acc:.2f}% ({top_1_correct} / {len(y_true)} correct)\n")
    f.write(f"  • Top-5 Accuracy:              {top_5_acc:.2f}% ({top_5_correct} / {len(y_true)} correct)\n")
    f.write(f"  • Macro Precision:             {macro_prec:.2f}%\n")
    f.write(f"  • Macro Recall:                {macro_rec:.2f}%\n")
    f.write(f"  • Macro F1-Score:              {macro_f1:.2f}%\n")
    f.write(f"  • Weighted Precision:          {weighted_prec:.2f}%\n")
    f.write(f"  • Weighted Recall:             {weighted_rec:.2f}%\n")
    f.write(f"  • Weighted F1-Score:           {weighted_f1:.2f}%\n\n")

    f.write("3. TOP 10 HIGHEST PERFORMING FOOD CLASSES\n")
    f.write("-" * 76 + "\n")
    f.write(f"  {'Rank':<5} {'Food Class':<32} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}\n")
    f.write("-" * 76 + "\n")
    for r, item in enumerate(top_classes, start=1):
        f.write(f"  {r:<5} {item['name']:<32} {item['precision']*100:>6.1f}%      {item['recall']*100:>6.1f}%      {item['f1']*100:>6.1f}%\n")

    f.write("\n4. TOP 10 MOST CHALLENGING FOOD CLASSES\n")
    f.write("-" * 76 + "\n")
    f.write(f"  {'Rank':<5} {'Food Class':<32} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}\n")
    f.write("-" * 76 + "\n")
    for r, item in enumerate(low_classes, start=1):
        f.write(f"  {r:<5} {item['name']:<32} {item['precision']*100:>6.1f}%      {item['recall']*100:>6.1f}%      {item['f1']*100:>6.1f}%\n")

    f.write("\n5. TOP 10 MOST FREQUENT MISCLASSIFICATIONS (CONFUSED PAIRS)\n")
    f.write("-" * 76 + "\n")
    f.write(f"  {'Rank':<5} {'True Ground Truth':<30} {'Predicted Class':<30} {'Count'}\n")
    f.write("-" * 76 + "\n")
    for r, (t_cls, p_cls, cnt) in enumerate(confused_pairs_sorted, start=1):
        f.write(f"  {r:<5} {t_cls:<30} -> {p_cls:<30} {cnt} images\n")

    f.write("=" * 76 + "\n")

print(f"\nExecutive summary report saved to: {summary_text_path}")
print("Done!")
