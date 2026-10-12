"""
01_data_exploration.py — Indian Food Images Dataset Exploration

What this file does:
1. Validates dataset directory path and identifies all 80 food categories.
2. Counts image quantities per class and identifies any class imbalances.
3. Generates a distribution bar chart of image counts per food category.
4. Samples image dimensions and file extensions (JPEG, PNG, BMP) to confirm data integrity.
5. Displays a 2x3 grid of random food sample images for visual inspection.
"""

from pathlib import Path
from collections import Counter
import random
import matplotlib.pyplot as plt
from PIL import Image

# ---- Step 1: Configuration & Dataset Discovery ----
DATASET_PATH = Path("Indian Food Images") if Path("Indian Food Images").exists() else Path(r"D:\JoJo\Indian Food Images")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset path not found at: {DATASET_PATH}")

print("Dataset found at:", DATASET_PATH.resolve())

class_folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()])
print(f"Total classes found: {len(class_folders)}")

# ---- Step 2: Count Images per Class ----
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
class_counts = {}

for folder in class_folders:
    images = [f for f in folder.iterdir() if f.suffix.lower() in valid_extensions]
    class_counts[folder.name] = len(images)

total_images = sum(class_counts.values())
print(f"Total valid images: {total_images}")
print("\nTop 5 largest classes:")
for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {cls}: {count}")

print("\nTop 5 smallest classes:")
for cls, count in sorted(class_counts.items(), key=lambda x: x[1])[:5]:
    print(f"  {cls}: {count}")

# ---- Step 3: Plot Class Distribution ----
plt.figure(figsize=(14, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=90, fontsize=8)
plt.title("Distribution of Images per Food Class")
plt.xlabel("Food Class")
plt.ylabel("Image Count")
plt.tight_layout()
plt.show()

# ---- Step 4: Inspect Image Formats and Dimensions ----
sizes = []
formats = Counter()

for folder in class_folders:
    images = [f for f in folder.iterdir() if f.suffix.lower() in valid_extensions][:5]
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                formats[img.format] += 1
        except Exception as e:
            print(f"[Warning] Could not open {img_path}: {e}")

print("\nImage formats distribution:", dict(formats))
print("Sample dimensions (width, height):", sizes[:5])

# ---- Step 5: Visualize Sample Images ----
sample_classes = random.sample(class_folders, min(6, len(class_folders)))
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, folder in zip(axes.flatten(), sample_classes):
    images = [f for f in folder.iterdir() if f.suffix.lower() in valid_extensions]
    if images:
        with Image.open(images[0]) as img:
            ax.imshow(img)
            ax.set_title(folder.name, fontsize=10)
            ax.axis("off")

plt.tight_layout()
plt.show()