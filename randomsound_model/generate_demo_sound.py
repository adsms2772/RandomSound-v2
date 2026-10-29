import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "reference_audio"

FLUIDSYNTH = Path(
    r"C:\Users\jijik\Downloads\fluidsynth-v2.6.1-win10-x64-cpp11"
    r"\fluidsynth-v2.6.1-win10-x64-cpp11\bin\fluidsynth.exe"
)

SOUNDFONT = Path(
    r"C:\Users\jijik\Downloads\GeneralUser_GS_v2.0.3--doc_r6"
    r"\GeneralUser-GS\GeneralUser-GS.sf2"
)

STYLES = {
    "medieval": "medieval_folk.mp3",
    "tropical": "tropical_chill.mp3",
    "rap": "rap_violin.mp3",
}


def run_command(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate an R-Sound demo track."
    )
    parser.add_argument(
        "--style",
        choices=STYLES.keys(),
        default="medieval",
        help="Reference style: medieval, tropical, or rap.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=8,
        help="Output duration from 3 to 13 seconds.",
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="Generated drum beat tempo.",
    )
    args = parser.parse_args()

    if not 3 <= args.seconds <= 13:
        parser.error("Duration must be between 3 and 13 seconds.")

    reference = REFERENCE_DIR / STYLES[args.style]

    if not reference.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference}")

    if not FLUIDSYNTH.is_file():
        raise FileNotFoundError(f"FluidSynth not found: {FLUIDSYNTH}")

    if not SOUNDFONT.is_file():
        raise FileNotFoundError(f"SoundFont not found: {SOUNDFONT}")

    if not (BASE_DIR / "generate_beat.py").is_file():
        raise FileNotFoundError("generate_beat.py was not found.")

    work_dir = BASE_DIR / "demo_work"
    work_dir.mkdir(exist_ok=True)

    token = str(random.randrange(100000, 999999))
    midi_file = work_dir / f"beat_{token}.mid"
    beat_wav = work_dir / f"beat_{token}.wav"
    reference_wav = work_dir / f"reference_{token}.wav"
    output_file = BASE_DIR / f"rsound_demo_{args.style}_{token}.wav"

    print(f"\nR-SOUND DEMO: {args.style.upper()}")
    print(f"Duration: {args.seconds} seconds")
    print(f"BPM: {args.bpm}")

    # 1. Generate a beat using the existing R-Sound model.
    run_command([
        sys.executable,
        str(BASE_DIR / "generate_beat.py"),
        "--seconds", str(args.seconds),
        "--bpm", str(args.bpm),
    ])

    # Locate the newest MIDI created by generate_beat.py.
    midi_candidates = list(BASE_DIR.glob("rsound_beat_*.mid"))

    if not midi_candidates:
        raise FileNotFoundError("The model did not create a MIDI file.")

    newest_midi = max(midi_candidates, key=lambda p: p.stat().st_mtime)
    midi_file.write_bytes(newest_midi.read_bytes())

    # 2. Render the MIDI beat as WAV.
    run_command([
        str(FLUIDSYNTH),
        "-ni",
        "-F", str(beat_wav),
        "-r", "44100",
        str(SOUNDFONT),
        str(midi_file),
    ])

    # 3. Convert and trim the reference track.
    run_command([
        "ffmpeg", "-y",
        "-i", str(reference),
        "-t", str(args.seconds),
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(reference_wav),
    ])

    # 4. Mix reference audio and generated beat.
    # These are starting gain weights, not guaranteed perceptual percentages.
    run_command([
        "ffmpeg", "-y",
        "-i", str(reference_wav),
        "-i", str(beat_wav),
        "-filter_complex",
        (
            f"[0:a]volume=0.7[ref];"
            f"[1:a]volume=0.3[beat];"
            f"[ref][beat]amix=inputs=2:duration=first:normalize=0,"
            f"alimiter=limit=0.95[out]"
        ),
        "-map", "[out]",
        "-t", str(args.seconds),
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(output_file),
    ])

    print("\nSUCCESS!")
    print(f"Style: {args.style}")
    print(f"Output: {output_file}")
    print(f"Duration limit: {args.seconds} seconds")
    print("Mix gains: reference 70%, generated beat 30%")
    print("\nPlay the result with:")
    print(f'Start-Process "{output_file}"')


if __name__ == "__main__":
    main()