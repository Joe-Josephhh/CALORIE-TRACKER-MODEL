"""
02_data_pipeline.py — Indian Food Images Stratified Pipeline & Preprocessing

What this file does:
1. Scans all class folders and builds a unified DataFrame of image filepaths and labels.
2. Performs stratified splitting into 70% Train, 15% Validation, and 15% Test subsets.
3. Saves the splits and class label indices into `configs/` for reproducible training.
4. Builds a data augmentation pipeline (rotation, zoom, contrast, flip) for training.
5. Constructs optimized tf.data.Dataset pipelines with batching and parallel prefetching.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---- Step 1: Configuration ----
DATASET_PATH = Path("Indian Food Images") if Path("Indian Food Images").exists() else Path(r"D:\JoJo\Indian Food Images")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_DIR = Path("configs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Step 2: Index Dataset Filepaths & Labels ----
class_folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()])
class_names = [f.name for f in class_folders]
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

records = []
for folder in class_folders:
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in VALID_EXT:
            records.append({"filepath": str(img_path), "label": folder.name})

df = pd.DataFrame(records)
print(f"Total valid images indexed: {len(df)}")
print(f"Total classes indexed: {len(class_names)}")

# ---- Step 3: Stratified 70 / 15 / 15 Train-Val-Test Split ----
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)

print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Save splits to configs for reproducibility across training and evaluation
train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val_split.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
pd.Series(class_names).to_csv(OUTPUT_DIR / "class_names.csv", index=False, header=["class_name"])
print("Split CSVs saved to configs/ directory.")

# ---- Step 4: Data Augmentation Pipeline (Train Only) ----
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")


# ---- Step 5: Preprocessing and tf.data Pipeline Construction ----
def load_and_preprocess(filepath, label, augment=False):
    image_bytes = tf.io.read_file(filepath)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image.set_shape([*IMG_SIZE, 3])
    if augment:
        image = data_augmentation(image)
    image = preprocess_input(image)
    return image, label


def make_dataset(dataframe, augment=False, shuffle=False):
    filepaths = dataframe["filepath"].values
    labels = dataframe["label"].map(class_to_idx).values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=SEED)
    ds = ds.map(
        lambda fp, lbl: load_and_preprocess(fp, lbl, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, augment=True, shuffle=True)
val_ds = make_dataset(val_df, augment=False, shuffle=False)
test_ds = make_dataset(test_df, augment=False, shuffle=False)

print("tf.data pipelines ready.")

# Sanity check: inspect one batch
for images, labels in train_ds.take(1):
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)