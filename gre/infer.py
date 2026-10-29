"""Estimate a rig from a recording by analysis-by-synthesis.

1. transcribe the performance and re-play it as a synthetic DI
2. pick an amp archetype + drive from the measured features
3. fit gain so distortion features match, then a match-EQ so the spectrum matches
4. fit delay / reverb / tremolo amounts by rendering and re-measuring the same way
"""
import copy

import numpy as np

from . import SR
from .analysis import (distortion_profile, tonal_profile, reverb_profile, delay_profile, corr_at,
                       tremolo_profile, remove_echo, dynamics_profile, reverb_at_stops)
from .dsp import PLOT_FREQS, ltas_db, smooth, biquads_response_db
from .models import AMPS
from .rig import default_rig, render, delay as fx_delay, reverb as fx_reverb, modulation as fx_mod
from .synth import render_di
from .transcribe import detect_onsets

FIT_SECONDS = 16.0


def _excerpt(mono, seconds=FIT_SECONDS):
    """Most active window of the clip (by RMS)."""
    n = int(seconds * SR)
    if len(mono) <= n:
        return 0, len(mono)
    hop = SR // 2
    best, b0 = -1, 0
    for s in range(0, len(mono) - n, hop):
        e = float(np.mean(mono[s:s + n] ** 2))
        if e > best:
            best, b0 = e, s
    return b0, b0 + n


def _shift_events(events, t0, t1):
    out = []
    for e in events:
        if t0 <= e["t"] < t1:
            d = dict(e)
            d["t"] = e["t"] - t0
            out.append(d)
    return out


def _dist_feats(y):
    on, _ = detect_onsets(y)
    d = distortion_profile(y, on)
    return d


def _dist_distance(a, b):
    d = ((a["crest_db"] - b["crest_db"]) / 0.8) ** 2
    if a["note_decay_db_s"] is not None and b["note_decay_db_s"] is not None:
        d += ((np.log1p(max(a["note_decay_db_s"], 0)) - np.log1p(max(b["note_decay_db_s"], 0))) / 0.35) ** 2
    d += ((np.log(a["flatness"] + 1e-4) - np.log(b["flatness"] + 1e-4)) / 0.35) ** 2
    return float(d)


def choose_amp(gain, tone, exclude=()):
    """Score archetypes against fitted gain and measured tone."""
    scoop = tone["mid_scoop_db"]
    bands = tone["bands_db"]
    bright = bands["Presence"] - bands["Low-mid"]
    scores = {}
    for k, a in AMPS.items():
        if k in exclude:
            continue
        lo, hi = a["gain"]
        s = 0.0
        if gain < lo:
            s += ((lo - gain) / 1.2) ** 2
        elif gain > hi:
            s += ((gain - hi) / 1.2) ** 2
        # mid character: archetype mid_db vs measured scoop (scaled)
        s += ((a["mid_db"] - np.clip(scoop, -10, 10) * 0.8) / 4.0) ** 2
        s += ((a["bright"] * 3 - np.clip(bright, -15, 15) * 0.3) / 6.0) ** 2
        # cab: lo-fi narrow band strongly suggests the small-speaker amp
        if tone["high_cut_hz"] < 3800:
            s -= 1.0 if k == "pawnshop" else 0.0
        elif k == "pawnshop":
            s += 2.0
        scores[k] = s
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    tot = sum(np.exp(-v) for _, v in ranked)
    return [(k, float(np.exp(-v) / tot)) for k, v in ranked]


def fit_match_eq(target_db, render_db, max_bands=8):
    """Greedy peaking-EQ fit of (target - render) over 80 Hz - 9 kHz."""
    f = PLOT_FREQS
    sel = (f > 80) & (f < 9000)
    diff = smooth(target_db - render_db, 7)
    diff -= np.mean(diff[(f > 300) & (f < 3000)])
    resid = np.clip(diff, -15, 15)
    bands = []
    for _ in range(max_bands):
        r = np.where(sel, resid, 0)
        i = int(np.argmax(np.abs(r)))
        g = float(np.clip(r[i], -12, 12))
        if abs(g) < 1.0:
            break
        # width: how far the deviation stays above half
        half = abs(g) / 2
        lo = i
        while lo > 0 and abs(r[lo]) > half and np.sign(r[lo]) == np.sign(g):
            lo -= 1
        hi = i
        while hi < len(f) - 1 and abs(r[hi]) > half and np.sign(r[hi]) == np.sign(g):
            hi += 1
        bw_oct = max(np.log2(f[hi] / f[lo]), 0.3)
        q = float(np.clip(1.0 / (2 ** (bw_oct / 2) - 2 ** (-bw_oct / 2)), 0.4, 4.0))
        b = {"f": float(round(f[i])), "g": round(g, 1), "q": round(q, 2)}
        bands.append(b)
        resid = resid - biquads_response_db([("peak", b["f"], b["q"], b["g"])], f)
    return bands


def _secant_fit(fun, target, x0, x1, lo, hi, iters=4):
    """Find x with fun(x) ~= target (fun roughly monotonic)."""
    f0, f1 = fun(x0) - target, fun(x1) - target
    best = (abs(f0), x0) if abs(f0) < abs(f1) else (abs(f1), x1)
    for _ in range(iters):
        if abs(f1 - f0) < 1e-9:
            break
        x2 = float(np.clip(x1 - f1 * (x1 - x0) / (f1 - f0), lo, hi))
        x0, f0 = x1, f1
        x1, f1 = x2, fun(x2) - target
        if abs(f1) < best[0]:
            best = (abs(f1), x1)
    return best[1]


def _spec_cost(target_db, render_db):
    """How much correction EQ a candidate needs (std of smoothed difference, 90 Hz-7 kHz)."""
    f = PLOT_FREQS
    sel = (f > 90) & (f < 7000)
    d = smooth(target_db - render_db, 9)[sel]
    return float(np.std(d - np.mean(d)))


def estimate_rig(stereo, mono, events, target, progress=lambda *a: None):
    """Return (rig, report) where report documents choices and confidences."""
    a0, a1 = _excerpt(mono)
    t0, t1 = a0 / SR, a1 / SR
    seg = mono[a0:a1]
    seg_st = stereo[a0:a1]
    ev = _shift_events(events, t0, t1)
    if not ev:
        raise ValueError("Could not find any notes to analyze. Is this an isolated guitar recording?")
    di = render_di(ev, (a1 - a0) / SR)
    stops = [(e["t"] + e["dur"], e.get("gap", 0.0)) for e in ev if e.get("gap", 0.0) >= 0.3]
    notes = {"stops": len(stops)}

    progress("Measuring the original", 0.30)
    tgt_d = _dist_feats(seg)
    tgt_tone = tonal_profile(seg)
    tgt_spec = np.array(tgt_tone["spectrum_db"])
    tgt_dly = delay_profile(seg)
    dyn_t = dynamics_profile(seg, detect_onsets(seg)[0])

    rig = default_rig()
    cache = {}

    def core(r, fx=False):
        keys = ("compressor", "drive", "amp", "cab", "eq", "gate") + (("delay", "reverb", "mod") if fx else ())
        key = repr({k: r[k] for k in keys}) + str(fx)
        if key not in cache:
            if len(cache) > 60:
                cache.clear()
            cache[key] = render(di, r, time_fx=fx)[:, 0]
        return cache[key]

    def dist_err(r):
        return _dist_distance(_dist_feats(core(r, fx=True)), tgt_d)

    # ---------------- 0. rough time effects + gate, so candidates are compared like with like
    if tgt_dly.get("detected"):
        rig["delay"].update(on=True, time_ms=tgt_dly["time_ms"], mix=0.25, feedback=0.3)
    tail_t, rt_t = reverb_at_stops(seg, stops) if stops else (None, None)
    rough = reverb_profile(seg)
    has_rev = (tail_t is not None and tail_t > -30) or ((rough["wet_estimate"] or 0) > 0.35 and not stops)
    if has_rev:
        rig["reverb"].update(on=True, type="room", decay_s=float(rt_t or rough["rt60_s"] or 1.2), mix=0.25)
    if dyn_t["noise_floor_dbfs"] < -85 or (dyn_t["gate_events"] >= 3 and dyn_t["noise_floor_dbfs"] < -75):
        rig["gate"].update(on=True, threshold_db=-50.0)
    notes["rough"] = {"tail_db": tail_t, "rt60": rt_t, "gate": rig["gate"]["on"]}

    # ---------------- 1. gain staging on a neutral voicing
    progress("Fitting gain", 0.36)
    base = copy.deepcopy(rig)
    base["amp"]["archetype"] = "plexi"
    grid = [0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5]
    errs = []
    for g in grid:
        r = copy.deepcopy(base)
        r["amp"]["gain"] = g
        errs.append(dist_err(r))
    g_best = grid[int(np.argmin(errs))]
    notes["gain_grid"] = dict(zip(grid, [round(e, 2) for e in errs]))

    # ---------------- 2. amp archetype: render every amp at noon knobs, keep the closest
    progress("Auditioning amp models", 0.44)
    scores = {}
    for k, a in AMPS.items():
        r = copy.deepcopy(rig)
        lo, hi = a["gain"]
        r["amp"].update(archetype=k, gain=float(np.clip(g_best, lo - 1.0, hi + 0.5)))
        r["cab"] = dict(a["cab"])
        y = core(r, fx=True)
        sc = _spec_cost(tgt_spec, ltas_db(y))
        de = _dist_distance(_dist_feats(y), tgt_d)
        prior = ((lo - g_best) / 1.5) ** 2 if g_best < lo else (((g_best - hi) / 1.5) ** 2 if g_best > hi else 0.0)
        scores[k] = sc / 1.2 + 0.15 * de + prior
    ranked_raw = sorted(scores.items(), key=lambda kv: kv[1])
    best_s = ranked_raw[0][1]
    w = {k: np.exp(-(v - best_s) / 0.6) for k, v in ranked_raw}
    tot = sum(w.values())
    ranked = [(k, float(w[k] / tot)) for k, _ in ranked_raw]
    arche = ranked[0][0]
    rig["amp"]["archetype"] = arche
    rig["cab"] = dict(AMPS[arche]["cab"])

    # ---------------- 3. refine gain for this voicing; test drive pedals
    progress("Comparing pedal combinations", 0.52)
    lo_g, hi_g = AMPS[arche]["gain"]
    cands = []
    for g in sorted(set(np.clip([g_best - 1.5, g_best - 0.75, g_best, g_best + 0.75, g_best + 1.5], 0, 10))):
        r = copy.deepcopy(rig)
        r["amp"]["gain"] = float(g)
        cands.append((dist_err(r), "none", r))
    if g_best >= 5.0:
        for dtype, dg, ag in (("overdrive", 6.0, max(lo_g, g_best - 2.5)),
                              ("fuzz", 7.5, min(3.0, g_best)),
                              ("distortion", 6.0, max(1.0, g_best - 3.0))):
            r = copy.deepcopy(rig)
            r["drive"].update(type=dtype, gain=dg, tone=5.0, level=6.0)
            r["amp"]["gain"] = float(ag)
            cands.append((dist_err(r) + 1.5, dtype, r))  # pedals must clearly earn their place
    cands.sort(key=lambda c: c[0])
    rig = cands[0][2]
    notes["drive_candidates"] = [(c[1], round(c[2]["amp"]["gain"], 2), round(c[0], 2)) for c in cands[:6]]

    # ---------------- 5. delay amounts (measured the same way on candidate renders)
    dry_core = core(rig)
    st = np.stack([dry_core, dry_core], 1)
    echo = None
    if rig["delay"]["on"]:
        progress("Fitting delay", 0.60)
        tc1, tc2 = tgt_dly["corr"], tgt_dly.get("corr2", 0.0)
        tm = rig["delay"]["time_ms"]
        if rig["reverb"]["on"]:
            st_r = fx_reverb(st, rig["reverb"])
        else:
            st_r = st

        def c_of(mix, fb=None):
            p = dict(rig["delay"], mix=mix, feedback=rig["delay"]["feedback"] if fb is None else fb)
            return corr_at(fx_delay(st, p)[:, 0], tm)

        mix = _secant_fit(lambda m: c_of(m)[0], tc1, 0.15, 0.4, 0.03, 1.0)
        rig["delay"]["mix"] = round(float(mix), 3)
        if tc1 > 0:
            ratio_t = tc2 / tc1
            fb = _secant_fit(lambda b: (lambda c: c[1] / max(c[0], 1e-6))(c_of(mix, b)), ratio_t,
                             0.2, 0.5, 0.0, 0.85, iters=3)
            rig["delay"]["feedback"] = round(float(fb), 3)
        echo = (tm, rig["delay"]["mix"], rig["delay"]["feedback"])
        rig["delay"]["tone_hz"] = 3200.0

    # ---------------- 6. tremolo
    seg_dry = remove_echo(seg, *echo) if echo else seg
    trem_t = tremolo_profile(seg_dry)
    if trem_t and trem_t["detected"]:
        progress("Fitting tremolo", 0.66)
        rig["mod"].update(type="tremolo", rate_hz=trem_t["rate_hz"])

        raw_t = tremolo_profile(seg) or trem_t

        def depth_of(d):
            ys = fx_mod(dry_core, dict(rig["mod"], depth=d))
            if rig["delay"]["on"]:
                ys = fx_delay(ys, rig["delay"])
            t = tremolo_profile(ys[:, 0])
            return t["depth_db"] if t else 0.0
        rig["mod"]["depth"] = round(float(_secant_fit(depth_of, raw_t["depth_db"], 0.3, 0.6, 0.05, 1.0, 3)), 3)

    # ---------------- 7. reverb: fit tail level + RT60 at the same stop times
    rev_t = {"tail_db": None, "rt60_s": None}
    if stops:
        _, rt_t = reverb_at_stops(seg_dry, stops)
        tail_t, _ = reverb_at_stops(seg, stops)
        stm = fx_mod(dry_core, rig["mod"]) if rig["mod"]["type"] != "none" else st
        if rig["delay"]["on"]:
            stm = fx_delay(stm, rig["delay"])
        tail_dry, _ = reverb_at_stops(stm[:, 0], stops)
        rev_t = {"tail_db": tail_t, "rt60_s": rt_t, "dry_render_tail_db": tail_dry}
        if tail_t is not None and tail_dry is not None and tail_t > tail_dry + 4 and rt_t:
            progress("Fitting reverb", 0.72)
            clean = rig["amp"]["gain"] < 4.5 and rig["drive"]["type"] == "none"
            rtype = "spring" if (clean and rt_t < 3.0 and arche in ("blackface", "tweed", "ac30", "pawnshop", "hiwatt")) \
                else ("room" if rt_t < 1.2 else ("hall" if rt_t > 2.6 else "plate"))
            rig["reverb"].update(on=True, type=rtype, decay_s=float(np.clip(rt_t, 0.2, 8.0)))

            def meas(mix, dec):
                y = fx_reverb(stm, dict(rig["reverb"], mix=mix, decay_s=dec))[:, 0]
                return reverb_at_stops(y, stops)
            dec = rig["reverb"]["decay_s"]
            mix = 0.3
            for _ in range(2):
                mix = _secant_fit(lambda m: (meas(m, dec)[0] or -60.0), tail_t, 0.15, 0.45, 0.02, 1.0, iters=3)
                _, rt_m = meas(mix, dec)
                if rt_m:
                    dec = float(np.clip(dec * rt_t / rt_m, 0.2, 8.0))
            rig["reverb"].update(decay_s=round(dec, 2), mix=round(float(mix), 3))
        else:
            rig["reverb"]["on"] = False
    elif not has_rev:
        rig["reverb"]["on"] = False

    # ---------------- 8. final gain check with every effect in place, then tone
    progress("Final gain + tone match", 0.80)
    cands = []
    for dg in (-0.75, 0.0, 0.75):
        r = copy.deepcopy(rig)
        r["amp"]["gain"] = float(np.clip(rig["amp"]["gain"] + dg, 0, 10))
        cands.append((dist_err(r), r))
    rig = min(cands, key=lambda c: c[0])[1]

    full = core(rig, fx=True)
    dyn_r = dynamics_profile(full, detect_onsets(full)[0])
    notes["dynamics_render"] = dyn_r
    if rig["amp"]["gain"] < 4.5 and rig["drive"]["type"] == "none" and \
            dyn_t["dynamic_range_db"] < dyn_r["dynamic_range_db"] - 4 and \
            dyn_t["attack_overshoot_db"] < dyn_r["attack_overshoot_db"] - 1.5:
        rig["compressor"].update(on=True, threshold_db=-28.0, ratio=4.0)
    rspec = ltas_db(core(rig, fx=True))
    diff = smooth(tgt_spec - rspec, 9)
    f = PLOT_FREQS
    ref = np.mean(diff[(f > 300) & (f < 3000)])

    def band(lo, hi):
        sel = (f >= lo) & (f < hi)
        return float(np.mean(diff[sel]) - ref)
    rig["amp"]["bass"] = round(float(np.clip(5 + band(70, 250) / 1.6, 0, 10)), 1)
    rig["amp"]["mid"] = round(float(np.clip(5 + band(400, 1500) / 1.6, 0, 10)), 1)
    rig["amp"]["treble"] = round(float(np.clip(5 + band(2000, 4500) / 1.6, 0, 10)), 1)
    rig["amp"]["presence"] = round(float(np.clip(5 + band(4000, 7000) / 1.5, 0, 10)), 1)
    rig["amp"]["gain"] = round(rig["amp"]["gain"], 1)
    rig["eq"] = fit_match_eq(tgt_spec, ltas_db(core(rig, fx=True)))

    # chorus hint: stereo spread not explained by reverb/delay
    side = float(np.sqrt(np.mean((seg_st[:, 0] - seg_st[:, 1]) ** 2) /
                         (np.mean((seg_st[:, 0] + seg_st[:, 1]) ** 2) + 1e-12)))
    if side > 0.25 and not rig["reverb"]["on"] and not rig["delay"]["on"] and rig["mod"]["type"] == "none":
        rig["mod"].update(type="chorus", rate_hz=0.8, depth=0.5)
        notes["chorus_from_stereo"] = round(side, 3)

    report = {
        "excerpt_s": [round(t0, 2), round(t1, 2)],
        "amp_candidates": [{"key": k, "label": AMPS[k]["label"], "logic_amp": AMPS[k]["logic_amp"],
                            "real": AMPS[k]["real"], "confidence": round(p, 3)} for k, p in ranked[:4]],
        "target_excerpt": {"distortion": tgt_d, "delay": tgt_dly, "reverb": rev_t, "tremolo": trem_t,
                           "dynamics": dyn_t},
        "notes": notes,
    }
    return rig, report
