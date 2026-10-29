"""Blind test: render a riff through known rigs, estimate the rig back, compare.

    python -m tests.validate
"""
import json
import sys
import time

import numpy as np

from gre import SR
from gre.synth import render_di
from gre.rig import default_rig, render
from gre.analysis import analyze
from gre.transcribe import transcribe
from gre.infer import estimate_rig

PATTERN = [([40, 47, 52], 0.5), ([43, 50], 0.5), ([45, 52], 0.25), ([45, 52], 0.25), ([47], 0.5),
           ([52], 1.2), ([], 0.6), ([40, 47, 52, 56, 59, 64], 1.6), ([], 0.8), ([57], 0.4), ([59], 0.4),
           ([62], 1.0), ([], 0.7)]

CASES = {
    "clean_spring": dict(amp=dict(archetype="blackface", gain=1.5),
                         reverb=dict(on=True, type="spring", decay_s=2.0, mix=0.35, predelay_ms=10)),
    "plexi_delay": dict(amp=dict(archetype="plexi", gain=6.0),
                        delay=dict(on=True, time_ms=420, feedback=0.35, mix=0.3, tone_hz=3000)),
    "recto_gate": dict(amp=dict(archetype="recto", gain=9.5), gate=dict(on=True, threshold_db=-50)),
    "ac30_dly_trem": dict(amp=dict(archetype="ac30", gain=2.5),
                          delay=dict(on=True, time_ms=280, feedback=0.45, mix=0.35, tone_hz=4000),
                          mod=dict(type="tremolo", rate_hz=5.5, depth=0.5)),
    "jcm800_dry": dict(amp=dict(archetype="jcm800", gain=7.0)),
    "fuzz_room": dict(drive=dict(type="fuzz", gain=8), amp=dict(archetype="blackface", gain=3),
                      reverb=dict(on=True, type="room", decay_s=1.0, mix=0.25)),
}


def make_events(reps=3, seed=0):
    rng = np.random.default_rng(seed)
    ev, t = [], 0.0
    for _ in range(reps):
        for notes, d in PATTERN:
            if notes:
                ev.append(dict(t=t + rng.normal(0, 0.004), dur=d, notes=notes, vel=float(rng.uniform(0.75, 1.0))))
            t += d
    return ev, t + 1.5


def summarize(r):
    out = {"amp": f'{r["amp"]["archetype"]} g={r["amp"]["gain"]:.1f}', "drive": r["drive"]["type"]}
    if r["delay"]["on"]:
        d = r["delay"]
        out["delay"] = f'{d["time_ms"]:.0f}ms fb={d["feedback"]:.2f} mix={d["mix"]:.2f}'
    if r["reverb"]["on"]:
        v = r["reverb"]
        out["reverb"] = f'{v["type"]} {v["decay_s"]:.2f}s mix={v["mix"]:.2f}'
    if r["mod"]["type"] != "none":
        m = r["mod"]
        out["mod"] = f'{m["type"]} {m["rate_hz"]:.2f}Hz d={m["depth"]:.2f}'
    if r["gate"]["on"]:
        out["gate"] = "on"
    if r["compressor"]["on"]:
        out["comp"] = "on"
    return out


def main(only=None):
    ev, dur = make_events()
    di = render_di(ev, dur, seed=11)
    for name, over in CASES.items():
        if only and name not in only:
            continue
        truth = default_rig()
        for k, v in over.items():
            truth[k].update(v)
        y = render(di, truth)
        mono = y.mean(1)
        t = time.time()
        events = transcribe(mono)
        target = analyze(y, mono, events)
        est, rep = estimate_rig(y, mono, events, target)
        print(f"\n=== {name}  ({time.time() - t:.1f}s)")
        print("  truth:", json.dumps(summarize(truth)))
        print("  est  :", json.dumps(summarize(est)))
        print("  amps :", [(c["key"], c["confidence"]) for c in rep["amp_candidates"][:3]])


if __name__ == "__main__":
    main(sys.argv[1:] or None)
