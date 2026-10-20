from pathlib import Path
from collections import Counter
import re

root = Path(__file__).resolve().parent / "datasets" / "parking_slots_grouped"
date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})")

date_sets = {}
total_images = 0
total_boxes = 0
errors = []

for split in ("train", "valid", "test"):
    image_dir = root / split / "images"
    label_dir = root / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    dates = set()
    total_images += len(images)
    box_count = 0

    for image in images:
        match = date_pattern.match(image.name)
        if not match:
            errors.append(f"Missing date in filename: {image.name}")
        else:
            dates.add(match.group(1))

        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            errors.append(f"Missing label: {label}")
            continue

        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5 or parts[0] != "0":
                errors.append(f"Invalid one-class label: {label}")
                break
            box_count += 1

    date_sets[split] = dates
    total_boxes += box_count
    print(f"{split}: {len(images)} images, {len(dates)} dates, {box_count:,} boxes")

print(f"\nTotal images: {total_images:,}")
print(f"Total parking-slot boxes: {total_boxes:,}")

for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
    overlap = date_sets[a] & date_sets[b]
    print(f"Dates shared by {a} and {b}: {len(overlap)}")
    if overlap:
        errors.append(f"Date overlap between {a} and {b}: {sorted(overlap)}")

if errors:
    print(f"\nFAILED: {len(errors)} issue(s) found.")
    for error in errors[:20]:
        print(error)
else:
    print("\nPASSED: no date overlap, missing labels, or invalid label formats.")
