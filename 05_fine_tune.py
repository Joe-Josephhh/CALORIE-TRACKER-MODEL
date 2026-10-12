"""
05_fine_tune.py — Fine-Tuning EfficientNetB0 Top Layers

What this file does:
1. Reloads saved train/val/test splits and class names from `configs/`.
2. Reconstructs augmented tf.data datasets matching the training pipeline.
3. Loads the previously trained `outputs/checkpoints/best_model.keras`.
4. Unfreezes the top 30 layers of the EfficientNetB0 backbone while keeping BatchNormalization frozen.
5. Recompiles the model with a low learning rate (1e-5) to prevent catastrophic forgetting.
6. Trains for up to 15 epochs with early stopping and saves `best_model_finetuned.keras`.
7. Plots fine-tuning curves to `outputs/results/fine_tuning_curves.png` and computes final test accuracy.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers

# ---- Step 1: Configuration & Directory Setup ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
FINE_TUNE_EPOCHS = 15
UNFREEZE_LAST_N_LAYERS = 30
FINE_TUNE_LR = 1e-5

CONFIG_DIR = Path("configs")
CHECKPOINT_DIR = Path("outputs/checkpoints")
RESULTS_DIR = Path("outputs/results")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Step 2: Reload Splits & Build Datasets ----
train_df = pd.read_csv(CONFIG_DIR / "train_split.csv")
val_df = pd.read_csv(CONFIG_DIR / "val_split.csv")
test_df = pd.read_csv(CONFIG_DIR / "test_split.csv")
class_names = pd.read_csv(CONFIG_DIR / "class_names.csv")["class_name"].tolist()
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

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

# ---- Step 3: Load Saved Pretrained Model ----
base_checkpoint = CHECKPOINT_DIR / "best_model.keras"
if not base_checkpoint.exists():
    raise FileNotFoundError(f"Base checkpoint not found at: {base_checkpoint}. Train 03 first.")

print(f"Loading checkpoint from: {base_checkpoint} ...")
model = tf.keras.models.load_model(base_checkpoint)

# Locate the EfficientNet backbone layer
base_model = next((l for l in model.layers if "efficientnet" in l.name.lower()), None)
if base_model is None:
    raise ValueError("EfficientNet backbone layer could not be found in loaded model.")

# ---- Step 4: Selective Unfreezing ----
base_model.trainable = True

# Freeze everything except the last N layers
for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
    layer.trainable = False

# Keep BatchNormalization layers frozen to stabilize batch statistics
for layer in base_model.layers[-UNFREEZE_LAST_N_LAYERS:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable layers in backbone: {trainable_count} / {len(base_model.layers)}")

# ---- Step 5: Recompile with Low Learning Rate ----
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---- Step 6: Fine-Tuning Execution ----
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_DIR / "best_model_finetuned.keras"),
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

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
)

# ---- Step 7: Plot Fine-Tuning History ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history_ft.history["accuracy"], label="Train Accuracy")
axes[0].plot(history_ft.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Fine-Tuning Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history_ft.history["loss"], label="Train Loss")
axes[1].plot(history_ft.history["val_loss"], label="Val Loss")
axes[1].set_title("Fine-Tuning Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fine_tuning_curves.png")
plt.show()

# ---- Step 8: Final Evaluation ----
test_loss, test_acc = model.evaluate(test_ds)
print(f"Fine-Tuned Test Accuracy: {test_acc:.4f}")
print(f"Fine-Tuned Test Loss:     {test_loss:.4f}")
print(f"Saved fine-tuned checkpoint to: {CHECKPOINT_DIR / 'best_model_finetuned.keras'}")