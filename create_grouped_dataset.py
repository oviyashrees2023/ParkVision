from pathlib import Path
import random
import re
import shutil
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "datasets" / "parking_slots"
OUTPUT = ROOT / "datasets" / "parking_slots_grouped"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})")

# Collect every image across the original splits.
records = []
dates = set()
errors = []

for split in ("train", "valid", "test"):
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing images or labels folder: {split}")

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        match = DATE_PATTERN.match(image_path.name)
        if not match:
            errors.append(f"Cannot extract date: {image_path.name}")
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label: {label_path}")
            continue

        date = match.group(1)
        dates.add(date)
        records.append((image_path, label_path, date))

if errors:
    print("Problems found; fix these before continuing:")
    print("\n".join(errors[:30]))
    raise SystemExit(1)

if not records:
    raise RuntimeError("No images found. Check the SOURCE path.")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Rename or remove it before running again."
    )

# Assign whole dates to splits, using a reproducible shuffle.
date_list = sorted(dates)
random.Random(42).shuffle(date_list)

n_train = round(len(date_list) * 0.70)
n_valid = round(len(date_list) * 0.15)

date_split = {}
for date in date_list[:n_train]:
    date_split[date] = "train"
for date in date_list[n_train:n_train + n_valid]:
    date_split[date] = "valid"
for date in date_list[n_train + n_valid:]:
    date_split[date] = "test"

counts = {s: 0 for s in ("train", "valid", "test")}

for image_path, label_path, date in records:
    split = date_split[date]
    image_out = OUTPUT / split / "images" / image_path.name
    label_out = OUTPUT / split / "labels" / label_path.name

    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, image_out)

    # Convert both original classes into one detector class: parking_slot.
    converted_lines = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5 or parts[0] not in {"0", "1"}:
            raise ValueError(
                f"Unexpected label format in {label_path}, line {line_number}: {line}"
            )

        converted_lines.append("0 " + " ".join(parts[1:]))

    label_out.write_text(
        "\n".join(converted_lines) + ("\n" if converted_lines else ""),
        encoding="utf-8",
    )
    counts[split] += 1

yaml_data = {
    "path": OUTPUT.as_posix(),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["parking_slot"],
}

(OUTPUT / "data.yaml").write_text(
    yaml.safe_dump(yaml_data, sort_keys=False),
    encoding="utf-8",
)

print("\nDataset created successfully.")
print(f"Unique capture dates: {len(date_list)}")
print(f"Images processed: {len(records)}")
print(f"Training dates: {n_train}")
print(f"Validation dates: {n_valid}")
print(f"Test dates: {len(date_list) - n_train - n_valid}")
print(f"Training images: {counts['train']}")
print(f"Validation images: {counts['valid']}")
print(f"Test images: {counts['test']}")
print(f"Dataset YAML: {OUTPUT / 'data.yaml'}")