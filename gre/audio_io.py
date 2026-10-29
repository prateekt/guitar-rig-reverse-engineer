"""Audio loading/saving. Uses soundfile, falls back to ffmpeg for mp3/m4a/etc."""
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from . import SR


def _ffmpeg_decode(path):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "decoded.wav")
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path,
               "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", out]
        subprocess.run(cmd, check=True, timeout=120)
        y, sr = sf.read(out, dtype="float32", always_2d=True)
    return y, sr


def load_audio(path, max_seconds=60.0, start=0.0):
    """Return (stereo [n,2] float32 at SR, mono float32). Trims to max_seconds from start."""
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        y, sr = _ffmpeg_decode(path)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    if sr != SR:
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g, axis=0).astype(np.float32)
    s0 = int(max(0.0, start) * SR)
    y = y[s0:s0 + int(max_seconds * SR)]
    if len(y) < SR // 2:
        raise ValueError("Audio is too short (need at least 0.5 s after the start offset).")
    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak * 0.89).astype(np.float32)
    mono = y.mean(axis=1).astype(np.float32)
    return y, mono


def save_wav(path, y, sr=SR):
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 0.99:
        y = y / peak * 0.99
    sf.write(path, y, sr, subtype="PCM_24")
