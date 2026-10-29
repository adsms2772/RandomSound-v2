"""
R-Sound Procedural Audio Generator
----------------------------------
Creates varied sound effects by combining source recordings,
random snippets, playback-speed changes, filtering, echo,
distortion, and randomized layering.

No neural network or GPU required.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_DATA = "processed_dataset"
DEFAULT_OUTPUT = "generated.wav"

MIN_SECONDS = 3
MAX_SECONDS = 13

SUPPORTED_FORMATS = {".wav"}


# ---------------------------------------------------------
# Audio loading
# ---------------------------------------------------------

def load_audio(path, sample_rate):
    """Load audio, convert to mono, and resample if necessary."""

    audio, original_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {path}")

    audio = audio.mean(axis=1)

    if original_rate != sample_rate:
        new_length = max(
            1,
            round(len(audio) * sample_rate / original_rate),
        )

        old_positions = np.arange(len(audio), dtype=np.float64)
        new_positions = (
            np.arange(new_length, dtype=np.float64)
            * original_rate / sample_rate
        )

        audio = np.interp(
            new_positions,
            old_positions,
            audio,
        ).astype(np.float32)

    audio = np.nan_to_num(audio)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def find_audio_files(folder):
    """Find WAV recordings recursively, without duplicate paths."""

    root = Path(folder)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root.resolve()}"
        )

    files = sorted(
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
    )

    # Remove duplicate paths, not duplicate recordings.
    return list(dict.fromkeys(files))


# ---------------------------------------------------------
# Basic audio transformations
# ---------------------------------------------------------

def change_speed(audio, factor, rng):
    """Change playback speed while retaining the audio content."""

    if len(audio) < 2:
        return audio.copy()

    new_length = max(2, int(len(audio) / factor))

    positions = np.linspace(
        0,
        len(audio) - 1,
        new_length,
    )

    changed = np.interp(
        positions,
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)

    return changed


def apply_fade(audio, sample_rate, rng):
    """Apply randomized fades to avoid harsh segment boundaries."""

    audio = audio.copy()

    fade_in = min(
        int(rng.uniform(0.005, 0.10) * sample_rate),
        len(audio) // 2,
    )

    fade_out = min(
        int(rng.uniform(0.01, 0.15) * sample_rate),
        len(audio) // 2,
    )

    if fade_in > 0:
        audio[:fade_in] *= np.linspace(
            0.0, 1.0, fade_in, dtype=np.float32
        )

    if fade_out > 0:
        audio[-fade_out:] *= np.linspace(
            1.0, 0.0, fade_out, dtype=np.float32
        )

    return audio


def lowpass(audio, cutoff, sample_rate):
    """Simple one-pole low-pass filter."""

    alpha = 1.0 - np.exp(
        -2.0 * np.pi * cutoff / sample_rate
    )

    output = np.empty_like(audio)
    previous = 0.0

    for i, value in enumerate(audio):
        previous += alpha * (float(value) - previous)
        output[i] = previous

    return output


def apply_filter(audio, sample_rate, rng):
    """Randomly choose a low-pass, high-pass, or unchanged signal."""

    mode = rng.choice(
        ["none", "lowpass", "highpass", "lowpass", "highpass"]
    )

    if mode == "none":
        return audio

    cutoff = rng.uniform(250, min(6500, sample_rate * 0.40))

    filtered = lowpass(audio, cutoff, sample_rate)

    if mode == "lowpass":
        return filtered.astype(np.float32)

    return (audio - filtered).astype(np.float32)


def apply_echo(audio, sample_rate, rng):
    """Apply a randomized echo with several decaying repetitions."""

    if rng.random() > 0.65:
        return audio

    delay_seconds = rng.uniform(0.04, 0.32)
    decay = rng.uniform(0.15, 0.55)

    delay = max(1, int(delay_seconds * sample_rate))

    output = np.zeros_like(audio)
    output += audio

    # Keep the original duration; later echoes may be cut off.
    for repetition in range(1, 4):
        offset = delay * repetition

        if offset >= len(audio):
            break

        available = len(audio) - offset

        output[offset:] += (
            audio[:available] * (decay ** repetition)
        )

    return output.astype(np.float32)


def apply_distortion(audio, rng):
    """Add mild randomized saturation or distortion."""

    if rng.random() > 0.45:
        return audio

    drive = rng.uniform(1.2, 4.0)

    return (
        np.tanh(audio * drive) / np.tanh(drive)
    ).astype(np.float32)


# ---------------------------------------------------------
# Random snippet creation
# ---------------------------------------------------------

def random_snippet(files, sample_rate, target_samples, rng):
    """
    Select a random source recording and extract a random snippet.
    Each snippet can use a different recording, speed, and direction.
    """

    attempts = min(10, len(files))

    for _ in range(attempts):
        path = rng.choice(files)

        try:
            source = load_audio(path, sample_rate)
        except (OSError, RuntimeError, ValueError):
            continue

        if len(source) < 2:
            continue

        speed_factor = rng.uniform(0.55, 1.9)

        # Select enough source samples to approximately produce
        # the requested output duration after the speed change.
        needed = max(
            2,
            int(target_samples * speed_factor),
        )

        if len(source) > needed:
            start = rng.randrange(0, len(source) - needed + 1)
            source = source[start:start + needed]
        else:
            source = source.copy()

        if rng.random() < 0.25:
            source = source[::-1].copy()

        source = change_speed(source, speed_factor, rng)

        # Fit the segment to its requested output length.
        if len(source) < target_samples:
            source = np.pad(
                source,
                (0, target_samples - len(source)),
            )
        else:
            source = source[:target_samples]

        source = apply_fade(source, sample_rate, rng)

        return source.astype(np.float32), path

    raise RuntimeError("Could not load a usable source recording.")


# ---------------------------------------------------------
# Build a complete sound layer
# ---------------------------------------------------------

def create_layer(files, total_samples, sample_rate, rng):
    """
    Construct a layer from many randomized snippets.

    Snippets are crossfaded to reduce clicks and obvious seams.
    """

    output = np.zeros(total_samples, dtype=np.float32)

    position = 0
    previous_tail = None

    # Randomly choose segment lengths for each layer.
    while position < total_samples:
        remaining = total_samples - position

        segment_seconds = rng.uniform(0.20, 1.80)

        segment_samples = min(
            remaining,
            max(2, int(segment_seconds * sample_rate)),
        )

        segment, _ = random_snippet(
            files,
            sample_rate,
            segment_samples,
            rng,
        )

        segment *= rng.uniform(0.35, 1.0)

        # Crossfade adjacent snippets.
        if previous_tail is not None and position > 0:
            crossfade_length = min(
                len(previous_tail),
                len(segment),
                int(0.04 * sample_rate),
                position,
            )

            if crossfade_length > 0:
                fade_in = np.linspace(
                    0.0,
                    1.0,
                    crossfade_length,
                    dtype=np.float32,
                )

                fade_out = 1.0 - fade_in

                output[
                    position - crossfade_length:position
                ] *= fade_out

                segment[:crossfade_length] = (
                    segment[:crossfade_length] * fade_in
                    + output[
                        position - crossfade_length:position
                    ]
                )

        end = min(position + len(segment), total_samples)
        actual_length = end - position

        if actual_length <= 0:
            break

        output[position:end] = segment[:actual_length]

        previous_tail = segment[-int(0.04 * sample_rate):].copy()
        position = end

    # Add a randomized combination of effects.
    output = apply_filter(output, sample_rate, rng)
    output = apply_echo(output, sample_rate, rng)
    output = apply_distortion(output, rng)

    return output


# ---------------------------------------------------------
# Generate a complete sound
# ---------------------------------------------------------

def generate_sound(files, seconds, sample_rate, seed=None):
    """Mix randomized layers into one complete audio generation."""

    rng = random.Random(seed)

    if seed is not None:
        np.random.seed(seed)

    total_samples = int(seconds * sample_rate)

    # Choose a different number of layers for each generation.
    layer_count = rng.randint(2, 5)

    print(f"Source recordings: {len(files)}")
    print(f"Audio layers: {layer_count}")
    print("Creating randomized snippets and effects...")

    mix = np.zeros(total_samples, dtype=np.float32)

    for layer_index in range(layer_count):
        print(
            f"Generating layer {layer_index + 1}/{layer_count}..."
        )

        layer = create_layer(
            files,
            total_samples,
            sample_rate,
            rng,
        )

        # Each layer receives its own volume and random start.
        gain = rng.uniform(0.12, 0.48)

        max_offset = max(0, total_samples // 5)
        offset = rng.randint(0, max_offset) if max_offset else 0

        available = total_samples - offset

        if available > 0:
            mix[offset:] += layer[:available] * gain

    # Add a small amount of background texture sometimes.
    if rng.random() < 0.30:
        noise = np.random.normal(
            0.0,
            rng.uniform(0.0005, 0.004),
            total_samples,
        ).astype(np.float32)

        mix += noise

    # Gentle overall saturation.
    mix = np.tanh(mix * rng.uniform(1.0, 1.8)).astype(np.float32)

    # Normalize to a consistent, safe peak level.
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0

    if peak > 1e-8:
        mix *= 0.90 / peak

    return np.clip(mix, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------
# Save audio
# ---------------------------------------------------------

def save_audio(path, audio, sample_rate):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        str(output_path),
        audio,
        sample_rate,
        subtype="PCM_16",
    )

    return output_path


# ---------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="R-Sound Procedural Audio Generator"
    )

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Folder containing source WAV recordings.",
    )

    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help="Output WAV file.",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Output duration in seconds (1–60).",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output sample rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for repeatable generations.",
    )

    args = parser.parse_args()

    if not MIN_SECONDS <= args.seconds <= MAX_SECONDS:
        parser.error("--seconds must be between 1 and 60.")

    if args.sample_rate < 8000 or args.sample_rate > 96000:
        parser.error("--sample-rate must be between 8000 and 96000.")

    try:
        files = find_audio_files(args.data)

        if not files:
            parser.error(
                f"No WAV files found in '{args.data}'."
            )

        print("=" * 50)
        print("R-Sound Procedural Generator")
        print("=" * 50)
        print(f"Dataset: {Path(args.data).resolve()}")
        print(f"Output duration: {args.seconds:g} seconds")
        print(f"Sample rate: {args.sample_rate} Hz")

        audio = generate_sound(
            files=files,
            seconds=args.seconds,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )

        output_path = save_audio(
            args.out,
            audio,
            args.sample_rate,
        )

        print()
        print("Generation complete!")
        print(f"Saved: {output_path.resolve()}")
        print(f"Duration: {len(audio) / args.sample_rate:.2f} seconds")

    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()