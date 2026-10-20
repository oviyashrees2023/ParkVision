
from pathlib import Path
from PIL import Image
import imagehash
from itertools import combinations

ROOT = Path("datasets/parking_slots")
SPLITS = ["train", "valid", "test"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = {}

for split in SPLITS:
    folder = ROOT / split / "images"
    images[split] = []

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in EXTS:
            try:
                with Image.open(path) as img:
                    phash = imagehash.phash(img.convert("RGB"))
                images[split].append((path.name, phash))
            except Exception as e:
                print("Could not read:", path, e)

    print(f"{split}: {len(images[split])} images hashed")

# Compare images across different splits.
# Hamming distance <= 8 is a useful initial similarity threshold.
threshold = 8
matches = []

for split_a, split_b in combinations(SPLITS, 2):
    print(f"\nComparing {split_a} vs {split_b}...")

    for name_a, hash_a in images[split_a]:
        for name_b, hash_b in images[split_b]:
            distance = hash_a - hash_b

            if distance <= threshold:
                matches.append(
                    (split_a, name_a, split_b, name_b, distance)
                )

matches.sort(key=lambda x: x[4])

print(f"\nNear-duplicate pairs found: {len(matches)}")

for a, b, c, d, dist in matches[:100]:
    print(f"distance={dist}: [{a}] {b} <-> [{c}] {d}")

if len(matches) > 100:
    print("Only the first 100 matches are displayed.")
