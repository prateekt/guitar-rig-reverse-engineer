"""Rough polyphonic transcription: onsets + up to 4 pitches per onset.

Good enough to re-play the performance through a simulated rig; not a
notation-grade transcription.
"""
import numpy as np

from . import SR
from .dsp import stft_mag, smooth

N_FFT = 4096
HOP = 512


def onset_envelope(x):
    S = stft_mag(x, n_fft=2048, hop=HOP)
    L = np.log1p(100 * S)
    flux = np.maximum(0.0, np.diff(L, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    return flux


def detect_onsets(x, min_gap_s=0.07):
    flux = onset_envelope(x)
    if flux.max() <= 0:
        return np.array([0.0]), flux
    f = flux / (flux.max() + 1e-9)
    base = smooth(f, 16) * 1.4 + 0.04
    gap = int(min_gap_s * SR / HOP)
    peaks = []
    for i in range(1, len(f) - 1):
        if f[i] > base[i] and f[i] >= f[i - 1] and f[i] >= f[i + 1]:
            if peaks and i - peaks[-1] < gap:
                if f[i] > f[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    t = np.array(peaks) * HOP / SR
    if len(t) == 0:
        t = np.array([0.0])
    return t, flux


def midi_to_hz(m):
    return 440.0 * 2 ** ((np.asarray(m) - 69) / 12.0)


def _pitches_in_window(seg, max_notes=4):
    if len(seg) < 1024:
        return []
    w = np.hanning(len(seg))
    nfft = 1 << int(np.ceil(np.log2(max(len(seg) * 2, N_FFT))))
    X = np.abs(np.fft.rfft(seg * w, nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    # spectral whitening-ish: compress
    X = X ** 0.6
    cands = np.arange(36, 86)  # C2..C#6
    found = []
    first = None
    Xw = X.copy()
    for _ in range(max_notes):
        best, best_m = 0.0, None
        for m in cands:
            f0 = midi_to_hz(m)
            s = 0.0
            for h in range(1, 9):
                fh = f0 * h
                if fh > 8000:
                    break
                lo, hi = np.searchsorted(freqs, [fh * 0.985, fh * 1.015])
                if hi > lo:
                    s += (0.86 ** (h - 1)) * Xw[lo:hi].max()
            if s > best:
                best, best_m = s, m
        if best_m is None:
            break
        if first is None:
            first = best
        elif best < 0.45 * first:
            break
        # avoid octave/fifth duplicates of already-found notes' harmonics
        f0 = midi_to_hz(best_m)
        found.append(int(best_m))
        for h in range(1, 12):
            fh = f0 * h
            lo, hi = np.searchsorted(freqs, [fh * 0.97, fh * 1.03])
            Xw[lo:hi] *= 0.15
    return sorted(set(found))


def _note_end(env_db, t, t_next, fs):
    """First sharp drop (player mute) between onset t and the next onset."""
    a, b = int((t + 0.1) * fs), int(t_next * fs) - 1
    k = int(0.06 * fs)
    if b - a <= k:
        return t_next
    seg = env_db[a:b + 1]
    top = seg.max()
    for i in range(len(seg) - k):
        if seg[i] > top - 25 and seg[i] - seg[i + k] > 9:
            return (a + i) / fs + 0.02
    return t_next


def transcribe(x, max_events=400):
    onsets, _ = detect_onsets(x)
    events = []
    dur = len(x) / SR
    hop = 128
    from .dsp import rms_envelope, db
    env_db = db(smooth(rms_envelope(x, 512, hop), 3))
    fs = SR / hop
    for i, t in enumerate(onsets[:max_events]):
        t_next = onsets[i + 1] if i + 1 < len(onsets) else dur
        t_end = _note_end(env_db, t, t_next, fs)
        s0 = int((t + 0.04) * SR)
        s1 = int(min(t + 0.30, t_next - 0.005) * SR)
        if s1 - s0 < 1024:
            s1 = min(len(x), s0 + 2048)
        seg = x[s0:s1]
        notes = _pitches_in_window(seg)
        a0, a1 = int(t * SR), int(min(t_next, t + 0.25) * SR)
        vel = float(np.sqrt(np.mean(x[a0:a1] ** 2) + 1e-12)) if a1 > a0 else 0.0
        if notes:
            events.append({"t": float(t), "dur": float(max(0.05, t_end - t)), "notes": notes, "vel": vel,
                           "gap": float(max(0.0, t_next - t_end))})
    if events:
        vmax = max(e["vel"] for e in events) or 1.0
        for e in events:
            e["vel"] = float(np.clip(e["vel"] / vmax, 0.15, 1.0))
    return events


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"
