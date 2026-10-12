"""
03_model_build_and_train.py — EfficientNetB0 Transfer Learning & Model Training

What this file does:
1. Reloads saved train/val/test splits and class names from `configs/`.
2. Creates augmented tf.data pipelines for reproducible training.
3. Instantiates pretrained EfficientNetB0 (ImageNet weights) with its base frozen.
4. Adds classification head: GlobalAveragePooling2D -> Dropout(0.3) -> Dense(NUM_CLASSES, softmax).
5. Compiles with Adam optimizer (lr=1e-3) and sparse categorical crossentropy.
6. Trains with ModelCheckpoint (saving `best_model.keras`) and EarlyStopping.
7. Plots accuracy and loss curves to `outputs/results/training_curves.png`.
8. Evaluates performance on the held-out test split.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

# ---- Step 1: Configuration & Directory Setup ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 20

CONFIG_DIR = Path("configs")
CHECKPOINT_DIR = Path("outputs/checkpoints")
RESULTS_DIR = Path("outputs/results")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Step 2: Reload Saved Splits & Labels ----
train_df = pd.read_csv(CONFIG_DIR / "train_split.csv")
val_df = pd.read_csv(CONFIG_DIR / "val_split.csv")
test_df = pd.read_csv(CONFIG_DIR / "test_split.csv")
class_names = pd.read_csv(CONFIG_DIR / "class_names.csv")["class_name"].tolist()
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

print(f"Classes: {NUM_CLASSES} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---- Step 3: Build tf.data Input Pipeline ----
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="data_augmentation")


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
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, augment=True, shuffle=True)
val_ds = make_dataset(val_df, augment=False, shuffle=False)
test_ds = make_dataset(test_df, augment=False, shuffle=False)

# ---- Step 4: Build EfficientNetB0 with Frozen Backbone ----
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3),
)
base_model.trainable = False  # Freeze pretrained weights for transfer learning

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="efficientnetb0_food_classifier")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---- Step 5: Training with Checkpoints & Early Stopping ----
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_DIR / "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ---- Step 6: Plot Training History ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves.png")
plt.show()

# ---- Step 7: Final Test Set Evaluation ----
test_loss, test_acc = model.evaluate(test_ds)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss:     {test_loss:.4f}")
print(f"Best checkpoint saved at: {CHECKPOINT_DIR / 'best_model.keras'}")