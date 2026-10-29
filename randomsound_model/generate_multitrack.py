import argparse
import random
from pathlib import Path

import pretty_midi


BASE_DIR = Path(__file__).resolve().parent

MIN_SECONDS = 3
MAX_SECONDS = 13


def make_note(instrument, pitch, start, duration, velocity):
    start = max(0.0, start)
    end = min(start + duration, TOTAL_SECONDS)

    if end > start:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, velocity)),
                pitch=max(0, min(127, pitch)),
                start=start,
                end=end,
            )
        )


def choose_scale(root, scale_intervals):
    return [root + interval for interval in scale_intervals]


def create_composition(seconds, bpm, seed, style):
    global TOTAL_SECONDS
    TOTAL_SECONDS = seconds

    rng = random.Random(seed)

    # Musical scales
    scales = {
        "medieval": [0, 2, 3, 7, 8, 10],
        "tropical": [0, 2, 4, 5, 7, 9, 11],
        "rap": [0, 3, 5, 7, 10],
    }

    # Choose a different root and musical progression.
    root = rng.choice([45, 48, 50, 52, 55, 57, 60])
    scale = choose_scale(root, scales[style])

    # Chord progressions use scale degrees.
    progressions = [
        [0, 3, 4, 0],
        [0, 5, 3, 4],
        [0, 4, 5, 3],
        [0, 3, 0, 4],
        [5, 3, 0, 4],
    ]

    progression = rng.choice(progressions)

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Select instrument programs from General MIDI.
    if style == "medieval":
        melody_program = rng.choice([46, 48, 40, 73])
        chord_program = rng.choice([46, 48, 24, 46])
        bass_program = rng.choice([32, 43, 44])
        drum_density = 0.48

    elif style == "tropical":
        melody_program = rng.choice([10, 11, 46, 80])
        chord_program = rng.choice([4, 5, 89, 91])
        bass_program = rng.choice([33, 34, 38])
        drum_density = 0.60

    else:
        melody_program = 40  # Violin
        chord_program = rng.choice([4, 5, 48])
        bass_program = rng.choice([38, 39, 33])
        drum_density = 0.85

    # Instruments
    melody = pretty_midi.Instrument(
        program=melody_program,
        name="Original Melody",
    )

    chords = pretty_midi.Instrument(
        program=chord_program,
        name="Chord Accompaniment",
    )

    bass = pretty_midi.Instrument(
        program=bass_program,
        name="Bassline",
    )

    drums = pretty_midi.Instrument(
        program=0,
        is_drum=True,
        name="Procedural Drums",
    )

    beat = 60.0 / bpm
    bar_length = beat * 4
    chord_length = bar_length

    # Generate chords and bass.
    bar = 0

    while bar * bar_length < seconds:
        chord_start = bar * bar_length
        degree = progression[bar % len(progression)]

        chord_root = scale[degree % len(scale)]

        # Build a triad using scale degrees.
        chord_pitches = [
            root + chord_root,
            root + scale[(degree + 2) % len(scale)],
            root + scale[(degree + 4) % len(scale)],
        ]

        # Chords
        for pitch in chord_pitches:
            make_note(
                chords,
                pitch + 12,
                chord_start,
                chord_length * rng.uniform(0.65, 0.95),
                rng.randint(55, 82),
            )

        # Bass notes follow the chord root.
        for beat_index in range(4):
            bass_start = chord_start + beat_index * beat

            make_note(
                bass,
                max(28, root - 12 + chord_root),
                bass_start,
                beat * rng.uniform(0.45, 0.9),
                rng.randint(70, 105),
            )

        bar += 1

    # Generate a new melody from scratch.
    step = beat / 2
    position = 0.0
    previous_pitch = root + 12

    while position < seconds:
        if rng.random() < 0.72:
            pitch = rng.choice(scale) + 12

            # Move pitches into a comfortable melody range.
            while pitch < 60:
                pitch += 12

            while pitch > 84:
                pitch -= 12

            # Encourage some movement from the previous note.
            if rng.random() < 0.45:
                pitch = max(
                    60,
                    min(84, previous_pitch + rng.choice([-7, -5, -2, 2, 5, 7])),
                )

            # Adjust toward the selected scale.
            allowed = [
                candidate + 12
                for octave in range(4, 7)
                for candidate in scale
                if 60 <= candidate + 12 <= 84
            ]

            if allowed:
                pitch = min(allowed, key=lambda p: abs(p - pitch))

            duration = step * rng.choice([0.6, 0.9, 1.2, 1.8])

            make_note(
                melody,
                pitch,
                position,
                duration,
                rng.randint(65, 108),
            )

            previous_pitch = pitch

        position += step

    # Generate a new drum pattern.
    position = 0.0

    while position < seconds:
        beat_position = position % bar_length
        beat_number = round(beat_position / beat)

        # Kick
        if beat_number in [0, 2] and rng.random() < drum_density:
            make_note(
                drums, 36, position, 0.12, rng.randint(85, 120)
            )

        # Snare or clap
        if beat_number in [1, 3] and rng.random() < drum_density:
            drum_pitch = rng.choice([38, 39])
            make_note(
                drums, drum_pitch, position, 0.12, rng.randint(75, 110)
            )

        # Hi-hat
        if rng.random() < drum_density:
            make_note(
                drums, 42, position, 0.08, rng.randint(40, 85)
            )

        position += beat / 2

    midi.instruments.extend([melody, chords, bass, drums])

    return midi


def main():
    parser = argparse.ArgumentParser(
        description="R-Sound Original Procedural Multitrack Generator"
    )

    parser.add_argument("--seconds", type=float, default=13)
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--style",
        choices=["medieval", "tropical", "rap"],
        default="tropical",
    )

    args = parser.parse_args()

    if not MIN_SECONDS <= args.seconds <= MAX_SECONDS:
        parser.error("Duration must be between 3 and 13 seconds.")

    if not 40 <= args.bpm <= 240:
        parser.error("BPM must be between 40 and 240.")

    output = create_composition(
        args.seconds,
        args.bpm,
        args.seed,
        args.style,
    )

    output_path = BASE_DIR / "rsound_multitrack_test.mid"
    output.write(str(output_path))

    total_notes = sum(
        len(instrument.notes)
        for instrument in output.instruments
    )

    print("\nR-SOUND MULTITRACK GENERATOR V2")
    print("--------------------------------")
    print("Style:", args.style)
    print("Duration:", args.seconds, "seconds")
    print("Tempo:", args.bpm, "BPM")
    print("Instrument tracks:", len(output.instruments))
    print("Total notes:", total_notes)
    print("Random seed:", args.seed)
    print("Output:", output_path)
    print("\nThis version creates new procedural note sequences.")
    print("It does not use Dance.mid as a musical source.")


if __name__ == "__main__":
    main()