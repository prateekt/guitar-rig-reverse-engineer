"""End-to-end: audio file -> sound profile, rig estimate, picture, Logic kit, emulation."""
import json
import os
import time

import numpy as np

from . import SR, __version__
from .audio_io import load_audio
from .analysis import analyze
from .draw import draw_rig
from .dsp import PLOT_FREQS, ltas_db
from .infer import estimate_rig
from .logic_export import build_kit
from .models import AMPS, PEDALS
from .rig import render
from .synth import render_di
from .transcribe import transcribe, note_name

MAX_SECONDS = 45.0


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def _match_scores(orig_spec, emu_spec, target, emu):
    f = PLOT_FREQS
    sel = (f > 80) & (f < 8000)
    d = (np.array(orig_spec) - np.mean(np.array(orig_spec)[sel])) - (np.array(emu_spec) - np.mean(np.array(emu_spec)[sel]))
    tone_err = float(np.sqrt(np.mean(d[sel] ** 2)))
    return {
        "tone_rms_db": round(tone_err, 2),
        "crest_db": [target["distortion"]["crest_db"], emu["distortion"]["crest_db"]],
        "note_decay_db_s": [target["distortion"]["note_decay_db_s"], emu["distortion"]["note_decay_db_s"]],
        "tone_score": round(float(np.clip(100 - tone_err * 12, 0, 100)), 0),
    }


def run(audio_path, out_dir, di_path=None, start=0.0, progress=lambda msg, frac: None):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    progress("Loading audio", 0.03)
    stereo, mono = load_audio(audio_path, max_seconds=MAX_SECONDS, start=start)
    dur = len(mono) / SR

    progress("Transcribing the performance", 0.10)
    events = transcribe(mono)

    progress("Building the sound profile", 0.18)
    profile = analyze(stereo, mono, events)

    rig, report = estimate_rig(stereo, mono, events, profile, progress=progress)

    progress("Rendering the emulation", 0.86)
    di = render_di(events, dur)
    emulation = render(di, rig)

    emu_profile = analyze(emulation, emulation.mean(1), with_spectrum=True)
    scores = _match_scores(profile["tone"]["spectrum_db"], emu_profile["tone"]["spectrum_db"], profile, emu_profile)

    user_di = None
    if di_path:
        progress("Running your DI through the rig", 0.90)
        _, dmono = load_audio(di_path, max_seconds=120.0)
        user_di = render(dmono * 0.56, rig)

    progress("Drawing the rig & building the Logic kit", 0.93)
    svg = draw_rig(rig, report)
    with open(os.path.join(out_dir, "rig.svg"), "w") as fh:
        fh.write(svg)
    rig_json = json.dumps(_clean(rig), indent=2)
    md, steps = build_kit(out_dir, rig, report, profile, stereo, emulation, di, events, rig_json)
    if user_di is not None:
        from .audio_io import save_wav
        save_wav(os.path.join(out_dir, "your_di_through_rig.wav"), user_di)

    a = AMPS[rig["amp"]["archetype"]]
    chain = []
    if rig["gate"]["on"]:
        chain.append({"kind": "gate", "name": "Noise Gate", "logic": "Noise Gate", "real": "Noise gate",
                      "settings": {"threshold_db": rig["gate"]["threshold_db"]}})
    if rig["compressor"]["on"]:
        chain.append({"kind": "compressor", "name": "Compressor", "logic": PEDALS["compressor"]["logic"],
                      "real": PEDALS["compressor"]["real"], "settings": rig["compressor"]})
    if rig["drive"]["type"] != "none":
        p = PEDALS[rig["drive"]["type"]]
        chain.append({"kind": "drive", "name": p["label"], "logic": p["logic"], "real": p["real"], "settings": rig["drive"]})
    if rig["mod"]["type"] != "none":
        p = PEDALS[rig["mod"]["type"]]
        chain.append({"kind": "mod", "name": p["label"], "logic": p["logic"], "real": p["real"], "settings": rig["mod"]})
    chain.append({"kind": "amp", "name": a["label"], "logic": a["logic_amp"], "real": a["real"], "settings": rig["amp"]})
    chain.append({"kind": "cab", "name": a["logic_cab"], "logic": a["logic_cab"], "real": f"{a['speakers'][0]}x{a['speakers'][1]}",
                  "settings": rig["cab"]})
    if rig["eq"]:
        chain.append({"kind": "eq", "name": "Match EQ", "logic": "Channel EQ", "real": "Studio EQ", "settings": {"bands": rig["eq"]}})
    if rig["delay"]["on"]:
        chain.append({"kind": "delay", "name": "Delay", "logic": "Tape Delay", "real": PEDALS["delay"]["real"], "settings": rig["delay"]})
    if rig["reverb"]["on"]:
        chain.append({"kind": "reverb", "name": f"{rig['reverb']['type'].title()} reverb", "logic": "ChromaVerb / Space Designer",
                      "real": "", "settings": rig["reverb"]})

    notes_hist = {}
    for e in events:
        for m in e["notes"]:
            notes_hist[note_name(m)] = notes_hist.get(note_name(m), 0) + 1

    result = _clean({
        "version": __version__,
        "duration_s": round(dur, 2),
        "elapsed_s": round(time.time() - t_start, 1),
        "summary": {
            "amp": a["label"], "amp_real": a["real"], "logic_amp": a["logic_amp"], "logic_cab": a["logic_cab"],
            "gain_character": _gain_word(rig),
            "effects": [c["name"] for c in chain if c["kind"] in ("drive", "mod", "delay", "reverb", "compressor", "gate")],
        },
        "chain": chain,
        "rig": rig,
        "estimate_report": report,
        "profile": profile,
        "emulation_profile": {"spectrum_db": emu_profile["tone"]["spectrum_db"]},
        "match": scores,
        "freqs": [round(float(f), 1) for f in PLOT_FREQS],
        "logic_steps": steps,
        "logic_markdown": md,
        "events": len(events),
        "notes_histogram": dict(sorted(notes_hist.items(), key=lambda kv: -kv[1])[:16]),
        "files": sorted(fn for fn in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, fn))),
    })
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    progress("Done", 1.0)
    return result


def _gain_word(rig):
    g = rig["amp"]["gain"] + (2.5 if rig["drive"]["type"] in ("overdrive", "distortion") else 0) + \
        (4 if rig["drive"]["type"] == "fuzz" else 0)
    if g < 2.5:
        return "Clean"
    if g < 4.5:
        return "Edge of breakup"
    if g < 6.5:
        return "Crunch"
    if g < 8.5:
        return "High gain"
    return "Saturated / metal"
