"""Parametric guitar-rig renderer.

DI -> compressor -> drive pedal -> amp (preamp stages, tone stack, power amp)
   -> cab -> match EQ -> gate -> modulation -> delay -> reverb
"""
import copy

import numpy as np
from scipy.signal import lfilter, sosfilt, fftconvolve, resample_poly

from . import SR
from .dsp import apply_biquads, butter_sos, biquads_response_db, sos_response_db
from .models import AMPS


def default_rig():
    return {
        "compressor": {"on": False, "threshold_db": -24.0, "ratio": 4.0, "attack_ms": 8.0, "release_ms": 150.0},
        "drive": {"type": "none", "gain": 5.0, "tone": 5.0, "level": 5.0},
        "amp": {"archetype": "plexi", "gain": 5.0, "bass": 5.0, "mid": 5.0, "treble": 5.0, "presence": 5.0, "master": 5.0},
        "cab": dict(AMPS["plexi"]["cab"]),
        "eq": [],
        "gate": {"on": False, "threshold_db": -55.0},
        "mod": {"type": "none", "rate_hz": 4.0, "depth": 0.0},
        "delay": {"on": False, "time_ms": 375.0, "feedback": 0.3, "mix": 0.25, "tone_hz": 3500.0},
        "reverb": {"on": False, "type": "room", "decay_s": 1.2, "mix": 0.2, "predelay_ms": 12.0},
    }


# ------------------------------------------------------------- dynamics
def _env_follow(x, attack_ms, release_ms, hop=32):
    a = np.abs(x)
    n = len(a) // hop
    if n == 0:
        return np.ones_like(x)
    blk = a[: n * hop].reshape(n, hop).max(axis=1)
    ca = np.exp(-hop / (SR * attack_ms / 1000.0))
    cr = np.exp(-hop / (SR * release_ms / 1000.0))
    env = np.empty(n)
    e = 0.0
    for i in range(n):
        v = blk[i]
        c = ca if v > e else cr
        e = c * e + (1 - c) * v
        env[i] = e
    env = np.repeat(env, hop)
    if len(env) < len(x):
        env = np.concatenate([env, np.full(len(x) - len(env), env[-1])])
    return env


def compressor(x, p):
    env = _env_follow(x, p["attack_ms"], p["release_ms"])
    lv = 20 * np.log10(env + 1e-9)
    over = np.maximum(0.0, lv - p["threshold_db"])
    gr = over * (1 - 1 / max(1.0, p["ratio"]))
    g = 10 ** (-gr / 20)
    makeup = 10 ** ((gr.max() * 0.5) / 20) if len(gr) else 1.0
    return x * g * makeup


def gate_gain(detector, p):
    env = _env_follow(detector, 1.0, 60.0)
    lv = 20 * np.log10(env + 1e-9)
    open_ = (lv > p["threshold_db"]).astype(float)
    # smooth open/close (2 ms attack, 40 ms release)
    return np.clip(_env_follow(open_, 2.0, 40.0) * 1.05, 0, 1)


# ------------------------------------------------------------- nonlinear
def _drive_pedal(x, p):
    t = p["type"]
    if t == "none":
        return x
    g = 10 ** (p["gain"] / 10 * (36 if t != "fuzz" else 50) / 20)
    tone_hz = 900 + 400 * p["tone"] ** 1.35
    if t == "overdrive":  # TS-style: cut lows before clipping, mid hump
        y = apply_biquads(x, [("highpass", 720, 0.6, 0), ("peak", 720, 0.9, 4.0)], sr=SR * 2)
        y = np.tanh(g * 3 * y) / np.tanh(g * 3 * 0.5 + 1e-9)
        y = y * 0.6 + apply_biquads(x, [("lowpass", 300, 0.7, 0)], sr=SR * 2) * 0.4  # clean low blend
    elif t == "distortion":
        y = apply_biquads(x, [("highpass", 120, 0.7, 0)], sr=SR * 2)
        y = np.clip(g * 6 * y, -1, 1)
        y = apply_biquads(y, [("lowpass", 6000, 0.7, 0)], sr=SR * 2)
    else:  # fuzz: asymmetric hard clipping, Muff-ish scoop
        y = g * 10 * x + 0.15
        y = np.sign(y) * (1 - np.exp(-np.abs(y)))
        y = y - np.mean(y)
        y = apply_biquads(y, [("peak", 1000, 0.7, -6.0)], sr=SR * 2)
    y = apply_biquads(y, [("lowpass", tone_hz, 0.7, 0)], sr=SR * 2)
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * np.sqrt(np.mean(x ** 2)) * 10 ** ((p["level"] - 5) * 1.2 / 20)
    return y


def _stage(x, g, bias):
    return (np.tanh(g * x + bias) - np.tanh(bias)) / max(1.0, np.tanh(g * 0.5 + bias))


def amp(x, p, voicing):
    sr2 = SR * 2
    gain = float(np.clip(p["gain"], 0, 10))
    x = apply_biquads(x, [("highpass", voicing["tight_hz"], 0.7, 0)], sr=sr2)
    # bright/pre-emphasis
    x = apply_biquads(x, [("highshelf", 1500, 0.7, voicing["bright"] * 1.5)], sr=sr2)
    total_db = gain * 5.0  # 0..50 dB
    if voicing["stages"] == 2 and gain > 4:
        g1 = 10 ** (total_db * 0.55 / 20)
        g2 = 10 ** (total_db * 0.45 / 20)
        y = _stage(x, g1, voicing["bias"])
        y = apply_biquads(y, [("highpass", 160, 0.7, 0), ("lowpass", 7000, 0.7, 0)], sr=sr2)
        y = _stage(y / (np.max(np.abs(y)) + 1e-9) * 0.5, g2, -voicing["bias"] * 0.5)
    else:
        y = _stage(x, 10 ** (total_db / 20), voicing["bias"])
    # tone stack
    ts = [
        ("lowshelf", 120, 0.7, (p["bass"] - 5) * 1.6),
        ("peak", voicing["mid_hz"], 0.8, voicing["mid_db"] + (p["mid"] - 5) * 1.6),
        ("highshelf", 2800, 0.7, (p["treble"] - 5) * 1.6),
    ]
    y = apply_biquads(y, ts, sr=sr2)
    # power amp: gentle saturation driven by master
    m = 0.6 + p["master"] * 0.25
    y = np.tanh(m * y / (np.max(np.abs(y)) + 1e-9)) / np.tanh(m)
    y = apply_biquads(y, [("highshelf", 3800, 0.7, (p["presence"] - 5) * 1.5)], sr=sr2)
    return y


# ------------------------------------------------------------- cab & eq
def cab_filters(c):
    biq = [
        ("highpass", c["low_cut"], 0.85, 0),
        ("peak", c["res"], 1.4, 4.0),
        ("peak", c["pres_hz"], 1.2, c["pres_db"]),
    ]
    sos = butter_sos("lowpass", c["high_cut"], order=4)
    return biq, sos


def apply_cab(x, c):
    biq, sos = cab_filters(c)
    y = apply_biquads(x, biq)
    y = sosfilt(sos, y)
    y = apply_biquads(y, [("highshelf", min(c["high_cut"] * 1.5, 16000), 0.7, -9.0)])
    return y


def cab_response_db(c, freqs):
    biq, sos = cab_filters(c)
    r = biquads_response_db(biq, freqs) + sos_response_db(sos, freqs)
    r += biquads_response_db([("highshelf", min(c["high_cut"] * 1.5, 16000), 0.7, -9.0)], freqs)
    return r


def eq_filters(eq):
    return [("peak", b["f"], b["q"], b["g"]) for b in eq]


# ------------------------------------------------------------- time fx
def _frac_delay(x, d_samples):
    n = np.arange(len(x)) - d_samples
    i0 = np.floor(n).astype(int)
    fr = n - i0
    i0c = np.clip(i0, 0, len(x) - 1)
    i1c = np.clip(i0 + 1, 0, len(x) - 1)
    y = (1 - fr) * x[i0c] + fr * x[i1c]
    y[n < 0] = 0
    return y


def modulation(x, p):
    """x mono -> stereo [n,2]"""
    t = np.arange(len(x)) / SR
    if p["type"] == "tremolo" and p["depth"] > 0:
        lfo = 1 - p["depth"] * (0.5 + 0.5 * np.sin(2 * np.pi * p["rate_hz"] * t))
        y = x * lfo
        return np.stack([y, y], 1)
    if p["type"] == "chorus" and p["depth"] > 0:
        base, dev = 0.012 * SR, 0.003 * SR * p["depth"] * 2
        l = _frac_delay(x, base + dev * np.sin(2 * np.pi * p["rate_hz"] * t))
        r = _frac_delay(x, base + dev * np.sin(2 * np.pi * p["rate_hz"] * t + np.pi / 2))
        return np.stack([x + 0.7 * l, x + 0.7 * r], 1) / 1.4
    return np.stack([x, x], 1)


def delay(xs, p):
    if not p["on"] or p["mix"] <= 0:
        return xs
    L = int(p["time_ms"] / 1000 * SR)
    wet = np.zeros_like(xs)
    tap = xs.copy()
    g = 1.0
    b, a = np.array([1.0]), np.array([1.0])
    from .dsp import biquad
    bl, al = biquad("lowpass", p["tone_hz"], SR, 0.7)
    for k in range(1, 24):
        if g < 1e-3 or k * L >= len(xs):
            break
        tap = lfilter(bl, al, tap, axis=0)
        sh = np.zeros_like(xs)
        sh[k * L:] = tap[: len(xs) - k * L]
        # ping-pong-ish width
        if k % 2 == 0:
            sh = sh[:, ::-1] * np.array([0.85, 1.0])
        wet += g * sh
        g *= p["feedback"]
    return xs + p["mix"] * wet


def reverb_ir(p, seed=3):
    rng = np.random.default_rng(seed)
    T = float(np.clip(p["decay_s"], 0.1, 8.0))
    n = int(min(T * 1.2, 8.0) * SR)
    t = np.arange(n) / SR
    env = np.exp(-6.91 * t / T)
    ir = rng.standard_normal((n, 2)) * env[:, None]
    # darker as it decays
    for ch in range(2):
        hi = ir[:, ch]
        lo = lfilter(*_onepole(2500), hi)
        mixw = np.clip(t / T, 0, 1)
        ir[:, ch] = hi * (1 - mixw) + lo * mixw
    if p["type"] == "spring":
        ir = apply_biquads(ir, [("highpass", 250, 0.7, 0), ("lowpass", 4200, 0.7, 0), ("peak", 1800, 2.0, 4.0)])
        # spring 'boing': short comb
        d = int(0.033 * SR)
        ir[d:] += 0.35 * ir[:-d]
    else:
        ir = apply_biquads(ir, [("highpass", 150, 0.7, 0), ("lowpass", 9000, 0.7, 0)])
    pre = int(p["predelay_ms"] / 1000 * SR)
    ir = np.concatenate([np.zeros((pre, 2)), ir])
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True)) + 1e-9
    return ir


def _onepole(fc):
    a = np.exp(-2 * np.pi * fc / SR)
    return np.array([1 - a]), np.array([1, -a])


def reverb(xs, p):
    if not p["on"] or p["mix"] <= 0:
        return xs
    ir = reverb_ir(p)
    wet = np.stack([fftconvolve(xs[:, c], ir[:, c])[: len(xs)] for c in range(2)], 1)
    dry_rms = np.sqrt(np.mean(xs ** 2)) + 1e-9
    wet *= dry_rms / (np.sqrt(np.mean(wet ** 2)) + 1e-9)
    m = p["mix"]
    return xs * (1 - 0.5 * m) + wet * m


# ------------------------------------------------------------- top level
def render(di, rig, time_fx=True):
    """di: mono float array at SR -> stereo [n,2] float32 normalized."""
    rig = copy.deepcopy(rig)
    v = AMPS[rig["amp"]["archetype"]]
    x = di.astype(np.float64)
    if rig["compressor"]["on"]:
        x = compressor(x, rig["compressor"])
    x2 = resample_poly(x, 2, 1)
    x2 = _drive_pedal(x2, rig["drive"])
    y2 = amp(x2, rig["amp"], v)
    y = resample_poly(y2, 1, 2)[: len(x)]
    y = apply_cab(y, rig["cab"])
    if rig["eq"]:
        y = apply_biquads(y, eq_filters(rig["eq"]))
    if rig["gate"]["on"]:
        y = y * gate_gain(x, rig["gate"])
    else:
        # amp hiss when gain is high and no gate (realism + gate detectability)
        hiss = np.random.default_rng(7).standard_normal(len(y)) * 10 ** (-72 / 20) * (1 + rig["amp"]["gain"])
        y = y + apply_cab(hiss, rig["cab"])
    ys = modulation(y, rig["mod"])
    if time_fx:
        ys = delay(ys, rig["delay"])
        ys = reverb(ys, rig["reverb"])
    peak = np.max(np.abs(ys)) or 1.0
    return (ys / peak * 0.89).astype(np.float32)


def chain_ir(rig, n=4096):
    """Impulse response of cab + match EQ (for Space Designer / IR loaders)."""
    imp = np.zeros(n)
    imp[0] = 1.0
    y = apply_cab(imp, rig["cab"])
    if rig["eq"]:
        y = apply_biquads(y, eq_filters(rig["eq"]))
    fade = int(n * 0.25)
    y[-fade:] *= np.linspace(1, 0, fade) ** 2
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
