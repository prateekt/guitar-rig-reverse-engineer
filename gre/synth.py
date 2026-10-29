"""Karplus-Strong electric-guitar DI synthesizer (used to re-play a transcription)."""
import numpy as np
from scipy.signal import lfilter

from . import SR
from .dsp import apply_biquads
from .transcribe import midi_to_hz


def pluck(f0, dur, vel=1.0, brightness=0.6, decay=0.996, rng=None):
    rng = rng or np.random.default_rng(0)
    n = int(dur * SR)
    N = max(2, int(round(SR / f0)))
    burst = rng.uniform(-1, 1, N)
    # brightness: one-pole lowpass on excitation
    a = 1.0 - brightness
    burst = lfilter([1 - a], [1, -a], burst)
    burst -= burst.mean()
    out = np.zeros(n)
    buf = burst.copy()
    # vectorised KS in blocks of N
    pos = 0
    rho = decay ** (220.0 / max(f0, 40))  # lower notes sustain longer
    while pos < n:
        m = min(N, n - pos)
        out[pos:pos + m] = buf[:m]
        nb = rho * 0.5 * (buf + np.roll(buf, -1))
        buf = nb
        pos += m
    # picking-position comb (pick ~1/7 along the string)
    d = max(1, N // 7)
    out[d:] -= 0.5 * out[:-d]
    return out * vel


def render_di(events, duration, seed=1):
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR) + SR)
    for e in events:
        s0 = int(e["t"] * SR)
        # let notes ring until next event (+ small release)
        dur = min(e["dur"] + 0.03, 6.0)
        for k, m in enumerate(e["notes"]):
            f0 = midi_to_hz(m) * (1 + rng.normal(0, 0.0007))
            strum = int(k * 0.006 * SR)
            note = pluck(f0, dur, e["vel"], brightness=0.55 + 0.3 * e["vel"], rng=rng)
            # damp at end of event (palm/fret release)
            rel = int(0.04 * SR)
            if len(note) > rel:
                note[-rel:] *= np.linspace(1, 0, rel)
            a = s0 + strum
            if a < 0:
                note, a = note[-a:], 0
            b = a + len(note)
            if a >= len(y) or len(note) == 0:
                continue
            b = min(b, len(y))
            y[a:b] += note[: b - a]
    y = y[: int(duration * SR)]
    # magnetic pickup: resonant peak + rolloff, mild body
    y = apply_biquads(y, [("highpass", 60, 0.7, 0), ("peak", 3200, 1.6, 5.0), ("lowpass", 5200, 0.9, 0)])
    peak = np.max(np.abs(y)) or 1.0
    return (y / peak * 0.5).astype(np.float32)
