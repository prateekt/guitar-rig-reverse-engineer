"""Small DSP toolkit: RBJ biquads, envelopes, spectra."""
import numpy as np
from scipy.signal import lfilter, sosfilt, butter, sosfreqz, welch

from . import SR


# ---------------------------------------------------------------- biquads
def biquad(kind, f0, sr=SR, q=0.707, gain_db=0.0):
    """RBJ cookbook biquad -> (b, a) normalized."""
    f0 = float(np.clip(f0, 10.0, sr * 0.49))
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "highshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    elif kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def apply_biquads(x, filters, sr=SR):
    """filters: list of (kind, f0, q, gain_db)."""
    y = x
    for kind, f0, q, g in filters:
        b, a = biquad(kind, f0, sr, q, g)
        y = lfilter(b, a, y, axis=0)
    return y


def biquads_response_db(filters, freqs, sr=SR):
    w = 2 * np.pi * np.asarray(freqs) / sr
    z = np.exp(-1j * w)
    H = np.ones_like(z)
    for kind, f0, q, g in filters:
        b, a = biquad(kind, f0, sr, q, g)
        H *= (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    return 20 * np.log10(np.abs(H) + 1e-12)


def butter_sos(kind, f, order=4, sr=SR):
    return butter(order, np.clip(f, 10, sr * 0.49), btype=kind, fs=sr, output="sos")


def sos_response_db(sos, freqs, sr=SR):
    _, h = sosfreqz(sos, worN=np.asarray(freqs), fs=sr)
    return 20 * np.log10(np.abs(h) + 1e-12)


# ---------------------------------------------------------------- envelopes
def db(x, floor=1e-9):
    return 20 * np.log10(np.maximum(np.abs(x), floor))


def rms_envelope(x, win=1024, hop=256):
    n = 1 + max(0, (len(x) - win)) // hop
    if n <= 0:
        return np.zeros(1)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)


def smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(int(n)) / int(n)
    return np.convolve(x, k, mode="same")


# ---------------------------------------------------------------- spectra
PLOT_FREQS = np.geomspace(30, 18000, 240)


def ltas_db(x, sr=SR, freqs=PLOT_FREQS, mask_frames=None):
    """Long-term average spectrum in dB, 1/6-octave smoothed, sampled at freqs."""
    f, p = welch(x, fs=sr, nperseg=8192, noverlap=6144, window="hann", average="mean")
    p = np.maximum(p, 1e-20)
    out = np.empty(len(freqs))
    for i, fc in enumerate(freqs):
        lo, hi = fc * 2 ** (-1 / 12), fc * 2 ** (1 / 12)
        sel = (f >= lo) & (f <= hi)
        if not np.any(sel):
            sel = [np.argmin(np.abs(f - fc))]
        out[i] = 10 * np.log10(np.mean(p[sel]))
    return out


def band_mean(freqs, spec_db, lo, hi):
    sel = (freqs >= lo) & (freqs < hi)
    return float(np.mean(spec_db[sel])) if np.any(sel) else float("nan")


def stft_mag(x, n_fft=2048, hop=512):
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + max(0, len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
