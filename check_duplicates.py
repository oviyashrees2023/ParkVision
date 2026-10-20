
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path("datasets/parking_slots")

splits = ["train", "valid", "test"]
hash_locations = defaultdict(list)

for split in splits:
    folder = ROOT / split / "images"

    if not folder.exists():
        print(f"Missing folder: {folder}")
        continue

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    print(f"{split}: {len(files)} images")

    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_locations[digest].append((split, path.name))

print("\nExact duplicate groups across splits:")

found = False
for digest, locations in hash_locations.items():
    split_names = {split for split, _ in locations}

    if len(split_names) > 1:
        found = True
        print("\n", locations)

if not found:
    print("No exact duplicate images found across splits.")
