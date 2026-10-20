from pathlib import Path
import zipfile

project = Path(__file__).resolve().parent.parent
dataset = project / "datasets" / "parking_slots_oneclass"
archive = project / "datasets" / "parking_slots_oneclass.zip"

if not (dataset / "data.yaml").is_file():
    raise FileNotFoundError("Prepared dataset data.yaml is missing.")

if archive.exists():
    raise FileExistsError(
        f"{archive} already exists. Rename or remove it first."
    )

with zipfile.ZipFile(
    archive,
    mode="w",
    compression=zipfile.ZIP_STORED,
    allowZip64=True,
) as zf:
    for file in dataset.rglob("*"):
        if file.is_file():
            zf.write(file, file.relative_to(dataset.parent))

print(f"Archive created: {archive}")
print(f"Size: {archive.stat().st_size / (1024**3):.2f} GB")