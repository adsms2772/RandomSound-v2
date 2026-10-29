
import os
import zipfile

zip_path = os.path.join(
    os.environ["USERPROFILE"],
    "Downloads",
    "groove-v1.0.0-midionly.zip"
)

output_dir = os.path.join(
    os.environ["USERPROFILE"],
    "Downloads",
    "RSound_MIDI"
)

os.makedirs(output_dir, exist_ok=True)

extracted = 0
skipped = 0

with zipfile.ZipFile(zip_path, "r") as archive:
    for name in archive.namelist():
        if not name.lower().endswith((".mid", ".midi")):
            continue

        try:
            archive.extract(name, output_dir)
            extracted += 1
        except (OSError, ValueError) as error:
            skipped += 1
            print(f"Skipped: {name!r} — {error}")

print(f"\nMIDI files extracted: {extracted}")
print(f"Files skipped: {skipped}")
print(f"Destination: {output_dir}")
