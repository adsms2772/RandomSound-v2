
import os
import glob
import json

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Configuration
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "training_datasets",
    "EDM_TECH",
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
STEPS_PER_BAR = 16
BARS_PER_PATTERN = 4
SEQUENCE_LENGTH = STEPS_PER_BAR * BARS_PER_PATTERN
PITCH_COUNT = 128
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Load MIDI files and convert them into drum-event grids.
def load_patterns():
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.mid"), recursive=True)
    patterns = []

    for path in files:
        try:
            midi = pretty_midi.PrettyMIDI(path)
            grid = np.zeros((SEQUENCE_LENGTH, PITCH_COUNT), dtype=np.float32)

            for instrument in midi.instruments:
                if not instrument.is_drum:
                    continue

                for note in instrument.notes:
                    step = int(round(note.start * 4))
                    step %= SEQUENCE_LENGTH

                    grid[step, note.pitch] = max(
                        grid[step, note.pitch],
                        note.velocity / 127.0,
                    )

            if grid.sum() > 0:
                patterns.append(grid)
        except Exception as error:
            print(f"Skipping {os.path.basename(path)}: {error}")

    if not patterns:
        raise RuntimeError(
            f"No usable drum patterns found in {DATA_DIR}"
        )

    return np.stack(patterns)


# Small LSTM neural network.
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
    print("Device:", DEVICE)
    print("Loading EDM-TECH MIDI patterns...")

    patterns = load_patterns()
    print("Usable patterns:", len(patterns))
    print("Pattern shape:", patterns.shape)

    # Predict the next drum-event grid from the preceding grid.
    inputs = torch.tensor(patterns, dtype=torch.float32)
    targets = torch.roll(inputs, shifts=-1, dims=1)

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
    )

    model = BeatModel().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    loss_function = nn.BCELoss()

    print("\nTraining Model 1: EDM-TECH")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(x)
            loss = loss_function(predictions, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} "
                f"- Loss: {average_loss:.5f}"
            )

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        "rsound_edm_tech.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pitch_count": PITCH_COUNT,
            "steps_per_bar": STEPS_PER_BAR,
            "bars_per_pattern": BARS_PER_PATTERN,
            "epochs": EPOCHS,
        },
        checkpoint_path,
    )

    print("\nTraining complete!")
    print("Model saved to:", checkpoint_path)


if __name__ == "__main__":
    main()
