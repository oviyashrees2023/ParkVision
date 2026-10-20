from pathlib import Path
import random
import cv2

ROOT = Path("datasets/parking_slots")
OUTPUT = Path("datasets/annotation_preview")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "space-empty",
    1: "space-occupied",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

for split in ("train", "valid", "test"):
    images_dir = ROOT / split / "images"
    labels_dir = ROOT / split / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Missing images or labels for {split}")
        continue

    candidates = [
        p for p in labels_dir.glob("*.txt")
        if p.stat().st_size > 0
    ]
    random.shuffle(candidates)

    created = 0

    for label_path in candidates:
        image_path = next(
            (
                images_dir / (label_path.stem + ext)
                for ext in IMAGE_EXTS
                if (images_dir / (label_path.stem + ext)).exists()
            ),
            None,
        )

        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read: {image_path}")
            continue

        height, width = image.shape[:2]

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            x1 = round((cx - bw / 2) * width)
            y1 = round((cy - bh / 2) * height)
            x2 = round((cx + bw / 2) * width)
            y2 = round((cy + bh / 2) * height)

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            color = (0, 255, 0) if class_id == 0 else (0, 165, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            name = CLASS_NAMES.get(class_id, f"class-{class_id}")
            cv2.putText(
                image,
                name,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        output_path = OUTPUT / f"{split}_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"Created: {output_path}")

        created += 1
        if created >= 2:
            break

    print(f"{split}: created {created} preview(s)")

print(f"\nPreviews saved in: {OUTPUT.resolve()}")
