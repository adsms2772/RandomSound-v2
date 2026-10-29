
import csv
import os
import shutil
from pathlib import Path

# Groove MIDI dataset locations
downloads = Path.home() / "Downloads"
dataset_root = downloads / "RSound_MIDI" / "groove"
metadata_file = dataset_root / "info.csv"

# Save the organized datasets inside randomsound_model
output_root = Path(__file__).resolve().parent / "training_datasets"

# Approximate genre groupings
groups = {
    "EDM_TECH": ["dance-disco", "dance-breakbeat", "funk"],
    "HH_TRP": ["hiphop"],
    "LOFI_DRUMS": ["soul", "jazz"],
    "POP_ROK": ["pop", "rock", "rock-indie", "punk"],
}

if not metadata_file.exists():
    raise FileNotFoundError(f"Metadata not found: {metadata_file}")

# Create output folders
for group in groups:
    (output_root / group).mkdir(parents=True, exist_ok=True)

counts = {group: 0 for group in groups}
missing = 0

with metadata_file.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        style = row["style"].lower()
        filename = row["midi_filename"]

        # Only use the training split to avoid mixing evaluation examples
        if row.get("split", "").lower() != "train":
            continue

        source = (dataset_root / filename).resolve()

        # Ensure the file is inside the dataset folder
        if dataset_root.resolve() not in source.parents:
            continue

        if not source.is_file():
            missing += 1
            continue

        for group, styles in groups.items():
            if any(f"/{style_name}/" in f"/{style}/" for style_name in styles):
                destination = output_root / group / source.name
                shutil.copy2(source, destination)
                counts[group] += 1
                break

print("\nR-Sound dataset preparation complete!")
print(f"Output directory: {output_root}")
print(f"MIDI files missing: {missing}")

for group, count in counts.items():
    print(f"{group}: {count} MIDI files")
