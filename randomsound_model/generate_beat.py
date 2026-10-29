
import os
import random
import argparse
from datetime import datetime

import pretty_midi
import torch
import torch.nn as nn

PITCH_COUNT = 128
STEPS_PER_BAR = 16
BARS_PER_PATTERN = 4
SEQUENCE_LENGTH = STEPS_PER_BAR * BARS_PER_PATTERN

MIN_SECONDS = 3
MAX_SECONDS = 13
DEFAULT_SECONDS = 8
DEFAULT_BPM = 120

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(
    BASE_DIR, "checkpoints", "rsound_edm_tech.pt"
)

DRUM_PITCHES = [
    35, 36, 38, 40, 37, 39, 42, 44, 46,
    41, 43, 45, 47, 48, 50, 49, 51, 52,
    55, 57, 59
]


class BeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=PITCH_COUNT,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.output = nn.Linear(128, PITCH_COUNT)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.activation(self.output(x))


def main():
    parser = argparse.ArgumentParser(
        description="Generate an R-Sound drum pattern."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="Duration in seconds (3–13).",
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=DEFAULT_BPM,
        help="Tempo in BPM.",
    )
    args = parser.parse_args()

    if not MIN_SECONDS <= args.seconds <= MAX_SECONDS:
        parser.error(
            f"Duration must be between {MIN_SECONDS} and {MAX_SECONDS} seconds."
        )

    if not 40 <= args.bpm <= 240:
        parser.error("BPM must be between 40 and 240.")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    random.seed()
    torch.manual_seed(random.randrange(0, 2**31))

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model = BeatModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    seed = torch.zeros(1, SEQUENCE_LENGTH, PITCH_COUNT)

    with torch.no_grad():
        probabilities = model(seed)[0]

    midi = pretty_midi.PrettyMIDI(initial_tempo=args.bpm)

    drums = pretty_midi.Instrument(
        program=0,
        is_drum=True,
        name="R-Sound EDM-TECH",
    )

    # Sixteenth-note spacing at the selected tempo.
    step_duration = 60.0 / args.bpm / 4.0

    # Generate enough steps to cover the requested duration.
    total_steps = max(
        1, int(args.seconds / step_duration)
    )

    for step in range(total_steps):
        # Reuse the learned 64-step probability pattern.
        row = probabilities[step % SEQUENCE_LENGTH]

        scores = torch.tensor(
            [row[pitch].item() for pitch in DRUM_PITCHES],
            dtype=torch.float32,
        )

        count = min(8, len(DRUM_PITCHES))
        top_indices = torch.topk(scores, k=count).indices.tolist()

        candidate_pitches = [
            DRUM_PITCHES[i] for i in top_indices
        ]
        candidate_scores = torch.tensor(
            [scores[i].item() for i in top_indices],
            dtype=torch.float32,
        )

        weights = candidate_scores.clamp(min=0.0001)

        hit_count = random.choices(
            [0, 1, 2],
            weights=[0.20, 0.60, 0.20],
            k=1,
        )[0]

        if hit_count == 0:
            continue

        selected_indices = torch.multinomial(
            weights,
            num_samples=hit_count,
            replacement=False,
        ).tolist()

        start = step * step_duration

        # Never create notes extending past the requested duration.
        end = min(start + 0.08, args.seconds)

        if end <= start:
            continue

        for index in selected_indices:
            drums.notes.append(
                pretty_midi.Note(
                    velocity=random.randint(75, 115),
                    pitch=candidate_pitches[index],
                    start=start,
                    end=end,
                )
            )

    midi.instruments.append(drums)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        BASE_DIR,
        f"rsound_beat_{timestamp}.mid",
    )

    midi.write(output_file)

    print("\nR-SOUND GENERATION COMPLETE!")
    print("MIDI file:", output_file)
    print("Drum notes:", len(drums.notes))
    print("Tempo:", args.bpm, "BPM")
    print("Requested duration:", args.seconds, "seconds")
    print("MIDI duration:", round(midi.get_end_time(), 3), "seconds")
    print("Note: Experimental AI-generated drum pattern.")


if __name__ == "__main__":
    main()