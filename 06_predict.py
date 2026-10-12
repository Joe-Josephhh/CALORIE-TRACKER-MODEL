"""
06_predict.py — Food Classification Inference

What this file does:
1. Automatically detects and loads the best available trained model checkpoint (prefers fine-tuned model).
2. Reads class category labels from `configs/class_names.csv`.
3. Preprocesses input images (decoding, resizing to 224x224, EfficientNet input scaling).
4. Computes softmax predictions and outputs Top-5 ranked classes with confidence percentages.
5. Supports both direct CLI argument mode and interactive command-line prompt mode.

USAGE:
  python 06_predict.py                                # Interactive mode: prompts for image path
  python 06_predict.py "test images/biriyani.jpg"     # Direct prediction via command line
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---- Configuration ----
IMG_SIZE = (224, 224)
CONFIG_DIR = Path("configs")
FINETUNED_MODEL_PATH = Path("outputs/checkpoints/best_model_finetuned.keras")
BASE_MODEL_PATH = Path("outputs/checkpoints/best_model.keras")
CLASS_NAMES_PATH = CONFIG_DIR / "class_names.csv"
TOP_K = 5
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_model_path():
    if FINETUNED_MODEL_PATH.exists():
        return FINETUNED_MODEL_PATH
    if BASE_MODEL_PATH.exists():
        print(f"Notice: Fine-tuned model not found. Using base model at {BASE_MODEL_PATH}")
        return BASE_MODEL_PATH
    raise FileNotFoundError(
        f"No model checkpoint found! Expected {FINETUNED_MODEL_PATH} or {BASE_MODEL_PATH}"
    )


def load_classes():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
    class_df = pd.read_csv(CLASS_NAMES_PATH)
    class_df.columns = class_df.columns.str.strip().str.replace('"', "")
    return class_df[class_df.columns[0]].astype(str).str.strip().str.replace('"', "").tolist()


def preprocess_image(image_path):
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def predict_image(model, class_names, image_path, top_k=TOP_K):
    image_tensor = preprocess_image(image_path)
    preds = model.predict(image_tensor, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((class_names[idx], float(preds[idx]) * 100))
    return results


def print_prediction_results(image_path, predictions):
    print("\n" + "=" * 65)
    print(f"IMAGE: {image_path}")
    print("=" * 65)
    for rank, (dish, confidence) in enumerate(predictions, start=1):
        print(f"  {rank}. {dish:<35} {confidence:>6.2f}%")
    print("-" * 65)
    best_dish, best_conf = predictions[0]
    print(f"  [TOP PREDICTION]: {best_dish.upper()} ({best_conf:.2f}%)")
    print("=" * 65 + "\n")


def main():
    print("=" * 65)
    print("      INDIAN FOOD IMAGE CLASSIFIER (EfficientNetB0)")
    print("=" * 65)

    model_path = get_model_path()
    print(f"* Loading model from: {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    class_names = load_classes()
    print(f"* Loaded {len(class_names)} classes.")

    # Direct CLI argument provided
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if not img_path.exists() or not img_path.is_file():
            print(f"[ERROR] File not found: {img_path}")
            sys.exit(1)
        preds = predict_image(model, class_names, img_path)
        print_prediction_results(img_path, preds)
        return

    # Interactive mode
    print("\n Enter image path to classify (or type 'quit' / 'q' to exit):")
    while True:
        user_input = input("\nEnter image path: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        # Remove surrounding quotes if user dragged & dropped file
        cleaned_path = user_input.strip("\"'")
        img_path = Path(cleaned_path)

        if not img_path.exists() or not img_path.is_file():
            print(f"[ERROR] File not found: {cleaned_path}. Please try again.")
            continue

        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            print(f"[ERROR] Unsupported format: {img_path.suffix}. Supported: {', '.join(sorted(VALID_EXTENSIONS))}")
            continue

        try:
            preds = predict_image(model, class_names, img_path)
            print_prediction_results(img_path, preds)
        except Exception as e:
            print(f"[ERROR] Processing image failed: {e}")


if __name__ == "__main__":
    main()