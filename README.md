# Rig Reverse Engineer

Upload a guitar recording and get:

- **Sound profile**: spectrum and tonal balance, gain/saturation, dynamics, gate, cab roll-off, reverb, delay, tremolo, stereo width, tuning
- **Best-guess rig**: pedals → amp → cab → mic → post effects, with knob settings, drawn as a picture (`rig.svg`)
- **Logic Pro kit** (`logic_kit.zip`): a step-by-step setup using Logic's own Amp Designer / Pedalboard / Channel EQ / Tape Delay / ChromaVerb names, Space Designer IRs (cab+tone, reverb), Apple AU presets, and a DI + MIDI of the performance
- **Emulation**: your performance transcribed, re-played by a synthetic guitar through the estimated rig (`emulation.wav`) so you can A/B it against the original. Upload your own DI to hear *your* playing through the rig.

## How it works

1. **Transcribe**: onset detection + harmonic-sum polyphonic pitch → note events (with note-ends from mute detection).
2. **Measure** (`gre/analysis.py`): long-term spectrum, crest factor, note sustain, spectral flatness on attacks, noise floor/gate events, reverb tails at note stops, and **delay via a transient-residual autocorrelation**: a phase-vocoder predictor cancels sustained notes, so only attacks remain, and a delay pedal repeats those sample-exactly (a player repeating a riff never does).
3. **Analysis-by-synthesis** (`gre/infer.py`): re-play the transcription through candidate rigs (`gre/rig.py`), measure them *the same way*, and fit gain, amp model (all 13 archetypes auditioned), drive pedal, match EQ, delay mix/feedback, tremolo depth and reverb decay/mix until they match.
4. **Export** (`gre/logic_export.py`, `gre/draw.py`).

Validation on synthetic rigs with known settings: `python -m tests.validate`. Delay times, tremolo rates, reverb type/decay, drive type and gain range come back reliably. The amp model is often only top-3 (many amps sound alike once EQ'd), and the reverb mix tends to read low on clean tones.

## Run locally

```bash
pip install -r requirements.txt          # needs ffmpeg for mp3/m4a
uvicorn server:app --reload              # http://localhost:8000
# or CLI:
python analyze.py song.wav -o out --di my_di.wav --start 12
```

## Deploy on Railway

The repo includes a `Dockerfile` and `railway.json` (health check `/healthz`).

1. Push this folder to a GitHub repo.
2. Railway → New Project → Deploy from GitHub repo → pick it. It builds from the Dockerfile automatically.
3. Settings → Networking → Generate Domain.

Or from the CLI: `npm i -g @railway/cli && railway login && railway init && railway up`.

Optional environment variables: `MAX_UPLOAD_MB` (40), `WORKERS` (2 concurrent analyses), `JOB_TTL_HOURS` (12), `JOBS_DIR` (`/tmp/gre_jobs`; attach a Railway volume there to keep results across restarts). Each analysis uses ~1 CPU for about a minute and ~500 MB RAM, so 1 GB+ of memory is recommended.

## Logic Pro notes

Logic's channel-strip (`.cst`) and Amp Designer/Pedalboard (`.pst`) preset formats are proprietary, so the kit uses formats Logic opens directly:
- **WAV impulse responses** → Space Designer (exact cab/tone and reverb match)
- **`.aupreset`** for Apple's bundled AUs (AUNBandEQ, AUDelay, AUReverb2, AUDynamicsProcessor). Run `install_presets.command`, then choose them from the plug-in's Settings menu. These are built from the public AU state format and haven't been tested inside Logic yet, so treat them as experimental.
- **Setup sheet** with Amp Designer / Pedalboard model names and knob values.

## Limitations

- Isolated guitar works best; full mixes confuse the EQ and gain measures.
- Chorus/flanger/phaser are only inferred from stereo spread; wah and pitch effects aren't detected.
- The emulation uses a synthetic guitar, so judge tone and effects rather than feel.
