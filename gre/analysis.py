"""Feature extraction: the measurable 'sound profile' of a guitar recording."""
import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter, maximum_filter1d

from . import SR
from .dsp import (PLOT_FREQS, ltas_db, band_mean, rms_envelope, smooth, db, stft_mag,
                  apply_biquads)
from .transcribe import detect_onsets, midi_to_hz

HOP = 256
BANDS = [("Low", 80, 250), ("Low-mid", 250, 500), ("Mid", 500, 1500),
         ("High-mid", 1500, 3000), ("Presence", 3000, 6000), ("Air", 6000, 12000)]


# ---------------------------------------------------------------- tone / EQ
def tonal_profile(x):
    spec = ltas_db(x)
    ref = band_mean(PLOT_FREQS, spec, 200, 4000)
    rel = spec - ref
    bands = {name: round(band_mean(PLOT_FREQS, rel, lo, hi), 2) for name, lo, hi in BANDS}
    # power-weighted centroid
    f = PLOT_FREQS
    p = 10 ** (spec / 10)
    centroid = float(np.sum(f * p) / np.sum(p))
    # mid scoop index: mids relative to average of lows and highs
    scoop = bands["Mid"] - 0.5 * (bands["Low"] + bands["Presence"])
    # cab high cut: first freq above the 1-4k plateau where response is 15 dB under it
    sm = smooth(spec, 5)
    plateau = np.max(sm[(f > 800) & (f < 4000)])
    hc = None
    for fi, v in zip(f, sm):
        if fi > 2000 and v < plateau - 15:
            hc = float(fi)
            break
    hc = hc or 12000.0
    # slope above the cut (dB/oct)
    i0 = np.argmin(np.abs(f - hc))
    i1 = np.argmin(np.abs(f - min(hc * 2, 16000)))
    slope = float((sm[i1] - sm[i0]) / max(np.log2(f[i1] / f[i0]), 1e-3))
    # low cut
    lowref = band_mean(f, sm, 150, 400)
    lc = 40.0
    for fi, v in zip(f[::-1], sm[::-1]):
        if fi < 150 and v < lowref - 12:
            lc = float(fi)
            break
    # low resonance peak (cab thump)
    sel = (f > 70) & (f < 200)
    res = float(f[sel][np.argmax(sm[sel])])
    # mid peak / notch location
    selm = (f > 300) & (f < 3000)
    detr = sm - smooth(sm, 41)
    mid_peak = float(f[selm][np.argmax(detr[selm])])
    mid_dip = float(f[selm][np.argmin(detr[selm])])
    return {
        "spectrum_db": spec.tolist(),
        "bands_db": bands,
        "centroid_hz": round(centroid, 1),
        "mid_scoop_db": round(float(scoop), 2),
        "high_cut_hz": round(hc, 0),
        "rolloff_slope_db_oct": round(slope, 1),
        "low_cut_hz": round(lc, 0),
        "low_resonance_hz": round(res, 0),
        "mid_peak_hz": round(mid_peak, 0),
        "mid_dip_hz": round(mid_dip, 0),
    }


# ---------------------------------------------------------------- distortion
def _yin_f0(frame, fmin=70, fmax=1000):
    n = len(frame)
    maxlag = int(SR / fmin)
    minlag = int(SR / fmax)
    if n < 2 * maxlag:
        return None, 1.0
    x = frame - frame.mean()
    d = np.array([np.sum((x[:n - maxlag] - x[l:n - maxlag + l]) ** 2) for l in range(maxlag + 1)])
    cm = d[1:] * np.arange(1, maxlag + 1) / (np.cumsum(d[1:]) + 1e-12)
    cm = np.concatenate([[1.0], cm])
    seg = cm[minlag:maxlag]
    i = int(np.argmin(seg)) + minlag
    return SR / i, float(cm[i])


def harmonic_stats(x, onsets):
    """Measure harmonic richness and odd/even balance on clean-pitch frames."""
    rich, oddeven, f0s = [], [], []
    win = 4096
    for t in onsets[:120]:
        s0 = int((t + 0.08) * SR)
        fr = x[s0:s0 + win]
        if len(fr) < win:
            continue
        f0, ap = _yin_f0(fr[:2048 + 600])
        if f0 is None or ap > 0.25:
            continue
        X = np.abs(np.fft.rfft(fr * np.hanning(win), 4 * win))
        freqs = np.fft.rfftfreq(4 * win, 1 / SR)
        amps = []
        for h in range(1, 21):
            fh = f0 * h
            if fh > 10000:
                break
            lo, hi = np.searchsorted(freqs, [fh * 0.97, fh * 1.03])
            amps.append(X[lo:hi].max() if hi > lo else 0)
        amps = np.array(amps) ** 2
        if len(amps) < 8 or amps[:5].sum() <= 0:
            continue
        rich.append(10 * np.log10(amps[5:].sum() / amps[:5].sum() + 1e-9))
        odd = amps[2:10:2].sum()   # harmonics 3,5,7,9
        even = amps[1:10:2].sum()  # 2,4,6,8,10
        oddeven.append(10 * np.log10((odd + 1e-12) / (even + 1e-12)))
        f0s.append(f0)
    return rich, oddeven, f0s


def distortion_profile(x, onsets):
    env = rms_envelope(x, 2048, 512)
    act = env > np.max(env) * 10 ** (-30 / 20)
    # crest factor in 50 ms blocks on active audio
    blk = int(0.05 * SR)
    n = len(x) // blk
    xb = x[: n * blk].reshape(n, blk)
    rms = np.sqrt(np.mean(xb ** 2, 1)) + 1e-9
    pk = np.max(np.abs(xb), 1)
    good = rms > np.max(rms) * 10 ** (-24 / 20)
    crest = float(np.median(20 * np.log10(pk[good] / rms[good]))) if np.any(good) else 12.0
    # spectral flatness 500-5000 Hz on active frames
    S = stft_mag(x, 2048, 512) ** 2
    freqs = np.fft.rfftfreq(2048, 1 / SR)
    band = (freqs > 500) & (freqs < 5000)
    Sb = S[:, band] + 1e-12
    fl = np.exp(np.mean(np.log(Sb), 1)) / np.mean(Sb, 1)
    m = min(len(fl), len(act))
    # only frames shortly after an onset, where the direct sound dominates any reverb
    att = np.zeros(m, bool)
    for t in onsets:
        a, b = int((t + 0.02) * SR / 512), int((t + 0.15) * SR / 512)
        att[max(0, a):min(m, b)] = True
    sel = act[:m] & att
    if sel.sum() < 5:
        sel = act[:m]
    flat = float(np.median(fl[:m][sel])) if np.any(sel) else 0.0
    # sustain: decay rate after each onset (dB/s over 0.08..0.5s)
    e_db = db(rms_envelope(x, 1024, HOP))
    decays = []
    for i, t in enumerate(onsets):
        t_next = onsets[i + 1] if i + 1 < len(onsets) else len(x) / SR
        if t_next - t < 0.45:
            continue
        a = int((t + 0.08) * SR / HOP)
        b = int(min(t + 0.6, t_next - 0.02) * SR / HOP)
        if b - a < 10 or b >= len(e_db):
            continue
        seg = e_db[a:b]
        tt = np.arange(len(seg)) * HOP / SR
        slope = np.polyfit(tt, seg, 1)[0]
        decays.append(-slope)
    decay = float(np.median(decays)) if decays else float("nan")
    rich, oe, f0s = harmonic_stats(x, onsets)
    return {
        "crest_db": round(crest, 2),
        "flatness": round(flat, 4),
        "note_decay_db_s": None if np.isnan(decay) else round(decay, 1),
        "harmonic_richness_db": round(float(np.median(rich)), 2) if rich else None,
        "odd_even_db": round(float(np.median(oe)), 2) if oe else None,
        "pitched_frames": len(rich),
        "_f0s": f0s,
    }


# ---------------------------------------------------------------- dynamics & gate
def dynamics_profile(x, onsets):
    e = rms_envelope(x, 4410, 1102)  # 100 ms
    edb = db(e)
    act = edb > edb.max() - 35
    a = edb[act]
    dr = float(np.percentile(a, 95) - np.percentile(a, 15)) if len(a) > 5 else 0.0
    # attack overshoot: peak of first 30 ms vs mean of 60-200 ms after onset
    env = rms_envelope(x, 256, 64)
    ov = []
    for t in onsets[:200]:
        i = int(t * SR / 64)
        p = env[i:i + int(0.03 * SR / 64)]
        s = env[i + int(0.06 * SR / 64): i + int(0.2 * SR / 64)]
        if len(p) and len(s):
            ov.append(20 * np.log10((p.max() + 1e-9) / (s.mean() + 1e-9)))
    overshoot = float(np.median(ov)) if ov else 0.0
    # noise floor & gate
    e20 = db(rms_envelope(x, 882, 441))
    floor = float(np.percentile(e20, 5))
    fast_drops = 0
    edb_f = e20
    for i in range(len(edb_f) - 4):
        if edb_f[i] > edb_f.max() - 30 and edb_f[i + 3] < floor + 6 and edb_f[i] - edb_f[i + 3] > 25:
            fast_drops += 1
    return {
        "dynamic_range_db": round(dr, 2),
        "attack_overshoot_db": round(overshoot, 2),
        "noise_floor_dbfs": round(floor, 1),
        "gate_events": int(fast_drops),
    }


# ---------------------------------------------------------------- reverb
def reverb_profile(x):
    """Reverb from 'stops': where the player mutes, the direct sound vanishes and
    what's left is the reverb tail. Tail slope -> RT60, tail level -> wetness."""
    hop = 128
    fs = SR / hop
    e = db(smooth(rms_envelope(x, 512, hop), 3))
    top = e.max()
    floor = np.percentile(e, 3)
    k30, k60 = int(0.03 * fs), int(0.06 * fs)
    rts, levels = [], []
    j = 0
    while j < len(e) - int(0.5 * fs):
        if e[j] > top - 22 and e[j] - e[j + k60] > 9:
            # tail starts after the drop settles
            a = j + k60
            # follow tail until rise (new note) or floor
            b = a
            while b < len(e) - 1 and b - a < int(3.0 * fs) and e[b + 1] <= e[b] + 1.0 and e[b] > floor + 5:
                b += 1
            lvl = float(np.mean(e[a:a + int(0.1 * fs)]) - e[j]) if b > a + 3 else -60.0
            levels.append(lvl)
            if b - a > int(0.15 * fs) and e[a] - e[b] > 6:
                tt = np.arange(b - a) / fs
                s = -np.polyfit(tt, e[a:b], 1)[0]
                if s > 1:
                    rts.append(60.0 / s)
            j = max(b, j + k30) + 1
            continue
        j += 1
    rt = float(np.median(rts)) if rts else None
    wet = None
    if levels:
        lv = float(np.median(levels))
        # tail 10 dB under the note -> very wet; 40+ dB under -> dry
        wet = float(np.clip((lv + 40) / 30, 0, 1))
    if rt is not None and wet is not None and wet < 0.08:
        rt = None
    return {"rt60_s": None if rt is None else round(min(rt, 8.0), 2),
            "wet_estimate": None if wet is None else round(wet, 2),
            "stops_found": len(levels)}


def reverb_at_stops(x, stops):
    """Reverb measured at known note-stop times (used when fitting: same stops in the
    original and in every candidate render). Returns (tail_db, rt60_s)."""
    hop = 128
    fs = SR / hop
    e = db(smooth(rms_envelope(x, 512, hop), 3))
    floor = np.percentile(e, 3)
    tails, rts = [], []
    for t, gap in stops:
        i = int(t * fs)
        pre = e[max(0, i - int(0.1 * fs)): max(1, i - int(0.02 * fs))]
        post = e[i + int(0.08 * fs): i + int(0.25 * fs)]
        if len(pre) == 0 or len(post) == 0:
            continue
        tails.append(float(np.mean(post) - np.mean(pre)))
        a, b = i + int(0.06 * fs), i + int(min(gap, 2.5) * fs)
        seg = e[a:b]
        if len(seg) > int(0.15 * fs):
            keep = seg > floor + 6
            if keep.sum() > int(0.12 * fs) and seg[keep][0] - seg[keep][-1] > 5:
                tt = np.arange(len(seg))[keep] / fs
                s = -np.polyfit(tt, seg[keep], 1)[0]
                if s > 1:
                    rts.append(60.0 / s)
    tail = float(np.median(tails)) if tails else None
    rt = float(np.median(rts)) if rts else None
    return tail, rt


# ---------------------------------------------------------------- delay
def transient_residual(x, n_fft=2048, hop=256):
    """Audio of what a phase-vocoder predictor can't predict from the previous two frames.
    Steady notes cancel; attacks, pick noise and note changes remain."""
    x = np.asarray(x, dtype=np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    nfr = 1 + max(0, len(x) - n_fft) // hop
    out = np.zeros(len(x) + n_fft, dtype=np.float32)
    norm = np.zeros_like(out)
    prev2 = prev1 = None
    block = 512
    for s in range(0, nfr, block):
        idx = np.arange(n_fft)[None, :] + hop * np.arange(s, min(nfr, s + block))[:, None]
        X = np.fft.rfft(x[idx] * win, axis=1).astype(np.complex64)
        if prev1 is not None:
            Xc = np.vstack([prev2[None], prev1[None], X])
        else:
            Xc = np.vstack([X[:1], X[:1], X])
        mag1 = np.abs(Xc[1:-1])
        ph = 2 * np.angle(Xc[1:-1]) - np.angle(Xc[:-2])
        E = Xc[2:] - mag1 * np.exp(1j * ph)
        if prev1 is None:
            E[:2] = X[:2]
        prev2, prev1 = Xc[-2], Xc[-1]
        fr = np.fft.irfft(E, n_fft, axis=1).astype(np.float32) * win
        for k in range(fr.shape[0]):
            p0 = (s + k) * hop
            out[p0:p0 + n_fft] += fr[k]
            norm[p0:p0 + n_fft] += win ** 2
    return (out / np.maximum(norm, 1e-3))[: len(x)]


def delay_profile(x, min_ms=60, max_ms=1300):
    """Echo detection: autocorrelate the transient residual. A delay pedal repeats
    attacks sample-exactly; a player repeating a riff never does."""
    e = transient_residual(x).astype(np.float64)
    nfft = 1 << int(np.ceil(np.log2(2 * len(e))))
    P = np.abs(np.fft.rfft(e, nfft)) ** 2
    k = max(31, int(4.0 / (SR / nfft)))  # ~4 Hz smoothing keeps echo comb ripple
    Ps = uniform_filter1d(P, k) + 1e-20
    r = np.fft.irfft(P / Ps)[: len(e)]
    r = r / (r[0] + 1e-12)
    lo, hi = int(min_ms / 1000 * SR), min(int(max_ms / 1000 * SR), len(r) - 1)
    seg = r[lo:hi]
    if len(seg) < 10:
        return {"detected": False}
    # 'excess' = how far each lag sticks out above its neighbourhood (excluding +-1.5 ms).
    # Sustained notes make a comb of near-equal peaks (excess ~ 0); an echo adds an isolated peak.
    ex, nb = int(0.0015 * SR), int(0.025 * SR)
    w = nb - ex
    M = maximum_filter1d(r, size=w)
    idx = np.arange(len(r))
    left = M[np.clip(idx - ex - w // 2, 0, len(r) - 1)]
    right = M[np.clip(idx + ex + w // 2, 0, len(r) - 1)]
    excess = r - np.maximum(left, right)
    exs = excess[lo:hi]
    i = int(np.argmax(exs)) + lo
    peak = float(r[i])
    exc = float(excess[i])
    others = np.maximum(exs[np.abs(np.arange(lo, hi) - i) > nb], 0)
    noise = float(np.max(others)) + 1e-4 if len(others) else 1e-4
    iso = peak / (peak - exc + 1e-6)
    prominence = exc / noise
    edge = i > hi - int(0.03 * SR)
    detected = exc > 0.008 and prominence > 2.0 and iso > 3.0 and not edge
    out = {"detected": bool(detected), "time_ms": round(i / SR * 1000, 1), "corr": round(peak, 4),
           "excess": round(exc, 4), "prominence": round(prominence, 2), "isolation": round(iso, 2)}
    j = 2 * i
    out["corr2"] = round(float(np.max(r[j - 60:j + 60])), 4) if j + 60 < len(r) else 0.0
    return out


def corr_at(x, time_ms):
    """Residual autocorrelation at a given delay time (and at 2x) - used for fitting."""
    e = transient_residual(x).astype(np.float64)
    nfft = 1 << int(np.ceil(np.log2(2 * len(e))))
    P = np.abs(np.fft.rfft(e, nfft)) ** 2
    k = max(31, int(4.0 / (SR / nfft)))
    r = np.fft.irfft(P / (uniform_filter1d(P, k) + 1e-20))[: len(e)]
    r = r / (r[0] + 1e-12)
    i = int(time_ms / 1000 * SR)
    c1 = float(np.max(r[max(0, i - 30): i + 30])) if i + 30 < len(r) else 0.0
    j = 2 * i
    c2 = float(np.max(r[j - 60: j + 60])) if j + 60 < len(r) else 0.0
    return c1, c2


def remove_echo(x, time_ms, mix, feedback):
    """Invert y = x + mix * sum fb^k x(t-(k+1)L):  x = y(1 - fb z^-L) / (1 + (mix - fb) z^-L)."""
    L = int(time_ms / 1000 * SR)
    if L <= 0:
        return x
    y = np.asarray(x, dtype=np.float64)
    num = y.copy()
    num[L:] -= feedback * y[:-L]
    out = np.zeros_like(y)
    c = mix - feedback
    for s in range(0, len(y), L):
        e = min(len(y), s + L)
        out[s:e] = num[s:e]
        if s >= L:
            out[s:e] -= c * out[s - L:e - L]
    return out


# ---------------------------------------------------------------- tremolo
def tremolo_profile(x):
    """Tremolo = a *sinusoidal* level modulation. Picked rhythms modulate the level too,
    but with sharp attacks (rich in harmonics of the modulation rate)."""
    hop = 128
    fs = SR / hop
    e = db(smooth(rms_envelope(x, 1024, hop), 3))
    top = e.max()
    act = e > top - 30
    if act.mean() < 0.3:
        return None
    e = np.clip(e, top - 30, None)
    hp = e - uniform_filter1d(e, int(0.6 * fs))
    hp = hp * act
    F = np.abs(np.fft.rfft(hp * np.hanning(len(hp)), 1 << 17))
    fr = np.fft.rfftfreq(1 << 17, 1 / fs)
    band = (fr > 2.5) & (fr < 12)
    k = int(np.argmax(np.where(band, F, 0)))
    f0 = fr[k]
    def en(f):
        sel = np.abs(fr - f) < 0.15
        return float(np.sum(F[sel] ** 2))
    e1 = en(f0)
    harm = sum(en(f0 * h) for h in (2, 3, 4))
    sub = en(f0 / 2)
    sinus = e1 / (e1 + harm + sub + 1e-12)
    floor = float(np.median(F[(fr > 1) & (fr < 20)]) + 1e-9)
    prom = F[k] / floor
    depth = float(np.percentile(hp[act], 95) - np.percentile(hp[act], 5))
    ok = sinus > 0.55 and prom > 12 and 2.0 < depth < 30
    return {"rate_hz": round(float(f0), 2), "depth_db": round(depth, 1), "sinusoidality": round(sinus, 2),
            "prominence": round(float(prom), 1), "detected": bool(ok)}


# ---------------------------------------------------------------- modulation
def modulation_profile(x, stereo):
    trem = tremolo_profile(x)
    # stereo width / decorrelation (chorus, stereo delay & reverb)
    L, R = stereo[:, 0].astype(np.float64), stereo[:, 1].astype(np.float64)
    corr = float(np.dot(L, R) / (np.sqrt(np.dot(L, L) * np.dot(R, R)) + 1e-12))
    side = float(np.sqrt(np.mean((L - R) ** 2) / (np.mean((L + R) ** 2) + 1e-12)))
    return {"tremolo": trem, "stereo_correlation": round(corr, 3), "side_to_mid": round(side, 3)}


# ---------------------------------------------------------------- tuning
def tuning_profile(f0s, events):
    notes = [m for e in events for m in e["notes"]]
    out = {"lowest_note_midi": int(min(notes)) if notes else None}
    if f0s:
        cents = 1200 * np.log2(np.array(f0s) / 440.0)
        dev = float(np.median(((cents + 50) % 100) - 50))
        out["a4_reference_hz"] = round(440 * 2 ** (dev / 1200), 1)
    low = out["lowest_note_midi"]
    if low is None:
        out["tuning_guess"] = "unknown"
    elif low >= 40:
        out["tuning_guess"] = "E standard (or capo/higher)"
    elif low == 39:
        out["tuning_guess"] = "Eb standard (half-step down)"
    elif low == 38:
        out["tuning_guess"] = "Drop D or D standard"
    elif low in (36, 37):
        out["tuning_guess"] = "Drop C / C standard"
    else:
        out["tuning_guess"] = "7-string or very low tuning"
    return out


# ---------------------------------------------------------------- warnings
def mix_warnings(tone, dist):
    w = []
    if tone["bands_db"]["Low"] > 3 and tone["low_resonance_hz"] < 90:
        w.append("Strong sub/low energy: this may be a full mix (bass/drums). Isolated guitar works best.")
    if tone["high_cut_hz"] >= 12000 and dist["crest_db"] < 9:
        w.append("Lots of energy above 10 kHz with heavy saturation: possibly cymbals in the mix, or a DI'd distortion with no cab.")
    return w


def analyze(stereo, mono, events=None, with_spectrum=True, echo=None):
    """echo: optional (time_ms, mix, feedback) to cancel before reverb/tremolo analysis."""
    onsets, _ = detect_onsets(mono)
    tone = tonal_profile(mono)
    dist = distortion_profile(mono, onsets)
    dyn = dynamics_profile(mono, onsets)
    dly = delay_profile(mono)
    dry = mono
    if echo is not None:
        dry = remove_echo(mono, *echo)
    rev = reverb_profile(dry)
    mod = modulation_profile(dry, stereo)
    out = {"duration_s": round(len(mono) / SR, 2), "onsets": len(onsets), "tone": tone,
           "distortion": dist, "dynamics": dyn, "reverb": rev, "delay": dly, "modulation": mod}
    if events is not None:
        out["tuning"] = tuning_profile(dist.get("_f0s", []), events)
    out["warnings"] = mix_warnings(tone, dist)
    dist.pop("_f0s", None)
    if not with_spectrum:
        tone.pop("spectrum_db", None)
    return out
