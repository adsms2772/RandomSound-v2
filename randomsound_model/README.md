# RandomSound trainable Python starter

## What this is
A small PyTorch prototype that trains on WAV files and saves a checkpoint.
It is a starting point for experimentation, not a production-quality audio model.
Raw-sample generation can sound noisy or unstable and is slow because it predicts
samples one at a time. A serious production model will likely use a spectrogram,
codec tokens, or an existing pretrained audio model.

## 1. Install Python
Use Python 3.10 or 3.11 in a virtual environment.

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installing PyTorch fails, install the appropriate build from https://pytorch.org/get-started/locally/.

## 2. Prepare the dataset
Create a `dataset` folder and place your licensed/owned `.wav` audio files inside.
Use clean clips, preferably 1 second or longer. This baseline converts audio to mono
and resamples to 16 kHz. Start with a small test set to confirm the pipeline works.

Example:
```
randomsound_model/
  dataset/
    click.wav
    wind.wav
    drum_loop.wav
  train.py
  generate.py
  requirements.txt
```

Only train on audio you own or are licensed to use.

## 3. Train
```powershell
python train.py --data dataset --epochs 20 --batch-size 8
```
The checkpoint is written to `checkpoints/randomsound.pt`.

For a first test on a slower computer:
```powershell
python train.py --data dataset --epochs 2 --batch-size 2 --chunk-seconds 0.5 --channels 32 --layers 4
```

## 4. Generate
```powershell
python generate.py --checkpoint checkpoints/randomsound.pt --out generated.wav --seconds 5
```

## Important limitations
- This is a learning prototype. Its simple next-sample L1 objective is not enough
  for high-fidelity audio; output may be noisy, repetitive, or nearly silent.
- Training needs enough diverse data and compute. A laptop CPU can run a small test,
  but training and generation may be slow.
- The current generation loop is deliberately simple and inefficient.
- It does not yet implement the RandomSound 12% beat / 88% sound-effect routing,
  audio-library slicing, licensing metadata, a web API, or Stripe.
- For a real website, a better next step is to build a reliable procedural mixer
  (random slicing, layering, pitch/time changes, filters, fades, normalization),
  then compare its results with this learned model.
