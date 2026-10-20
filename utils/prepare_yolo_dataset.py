from pathlib import Path
import os
import shutil

SOURCE = Path("datasets/parking_slots")
DEST = Path("datasets/parking_slots_oneclass")

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not (SOURCE / "data.yaml").exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE / 'data.yaml'}. "
            "Run this script from the ParkVision project folder."
        )

    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(
            f"{DEST} already contains files. "
            "Rename or move that folder before rerunning this script."
        )

    DEST.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0
    total_empty_labels = 0

    for split in SPLITS:
        source_images = SOURCE / split / "images"
        source_labels = SOURCE / split / "labels"

        dest_images = DEST / split / "images"
        dest_labels = DEST / split / "labels"

        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(
                f"Missing image or label directory for {split}"
            )

        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)

        images = sorted(
            p for p in source_images.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

        for image in images:
            label = source_labels / f"{image.stem}.txt"

            if not label.is_file():
                raise FileNotFoundError(
                    f"Missing label for image: {image}"
                )

            # Hard links save disk space when source and destination
            # are on the same Windows drive. Fall back to copying.
            dest_image = dest_images / image.name
            try:
                os.link(image, dest_image)
            except OSError:
                shutil.copy2(image, dest_image)

            output_lines = []

            for line_number, line in enumerate(
                label.read_text(encoding="utf-8-sig").splitlines(),
                start=1,
            ):
                parts = line.split()

                if not parts:
                    continue

                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid label: {label}, line {line_number}"
                    )

                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]

                if class_id not in (0, 1):
                    raise ValueError(
                        f"Unexpected class ID {class_id} in {label}"
                    )

                if not all(0 <= value <= 1 for value in coords):
                    raise ValueError(
                        f"Coordinates outside 0..1 in {label}"
                    )

                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError(
                        f"Invalid box size in {label}"
                    )

                # Both original classes represent parking slots.
                # Map both to the single detector class, ID 0.
                output_lines.append(
                    "0 " + " ".join(parts[1:])
                )
                total_boxes += 1

            # Preserve empty label files too.
            (dest_labels / label.name).write_text(
                "\n".join(output_lines) +
                ("\n" if output_lines else ""),
                encoding="utf-8",
            )

            if not output_lines:
                total_empty_labels += 1

            total_images += 1

        print(f"{split}: prepared {len(images)} images")

    # New paths are relative to the prepared dataset root.
    yaml_text = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['parking_slot']
"""
    (DEST / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("\nDataset preparation completed.")
    print(f"Prepared dataset: {DEST.resolve()}")
    print(f"Total images: {total_images}")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Empty label files: {total_empty_labels}")
    print(f"Configuration: {DEST / 'data.yaml'}")


if __name__ == "__main__":
    main()