"""Logic Pro kit: setup sheet, IRs for Space Designer, Apple AU presets, DI + MIDI.

Logic's own channel-strip (.cst) and Amp Designer/Pedalboard (.pst) formats are
proprietary and undocumented, so the kit uses formats Logic loads openly:
  * WAV impulse responses  -> Space Designer (cab+tone IR, reverb IR)
  * .aupreset (plist)      -> Apple's bundled Audio Units (AUNBandEQ, AUDelay, ...)
  * WAV DI + MIDI           -> tracks to drop into a project
plus a step-by-step sheet for Amp Designer / Pedalboard using Logic's model names.
"""
import io
import math
import os
import plistlib
import struct
import zipfile

import numpy as np

from . import SR
from .audio_io import save_wav
from .models import AMPS, PEDALS
from .rig import chain_ir, reverb_ir

APPL = 0x6170706C  # 'appl'
AUFX = 0x61756678  # 'aufx'


def _fourcc(s):
    return struct.unpack(">I", s.encode("ascii"))[0]


# ------------------------------------------------------------- .aupreset
def aupreset(name, subtype, params):
    """Classic AU state: per scope/element: [scope][element][count] then (id, float) pairs, big-endian."""
    body = struct.pack(">III", 0, 0, len(params))
    for pid, val in params:
        body += struct.pack(">If", int(pid), float(val))
    d = {"manufacturer": APPL, "name": name, "subtype": _fourcc(subtype), "type": AUFX, "version": 0, "data": body}
    return plistlib.dumps(d, fmt=plistlib.FMT_XML)


def _q_to_bw_oct(q):
    return 2.0 / math.log(2) * math.asinh(1.0 / (2.0 * q))


def nbandeq_preset(eq, name="GRE Match EQ"):
    params = [(0, 0.0)]  # global gain
    for i in range(8):
        if i < len(eq):
            b = eq[i]
            params += [(1000 + i, 0.0), (2000 + i, 0.0), (3000 + i, b["f"]), (4000 + i, b["g"]),
                       (5000 + i, float(np.clip(_q_to_bw_oct(b["q"]), 0.05, 5.0)))]
        else:
            params += [(1000 + i, 1.0)]  # bypass unused bands
    return aupreset(name, "nbeq", params)


def delay_preset(d, name="GRE Delay"):
    # 0 wet/dry %, 1 time s, 2 feedback %, 3 lowpass Hz
    wet = 100.0 * d["mix"] / (1.0 + d["mix"])
    return aupreset(name, "dely", [(0, wet), (1, d["time_ms"] / 1000.0), (2, 100.0 * d["feedback"]), (3, d["tone_hz"])])


def reverb2_preset(r, name="GRE Reverb"):
    # AUReverb2: 0 dry/wet %, 1 gain dB, 2 min delay s, 3 max delay s, 4 decay@0Hz s, 5 decay@Nyquist s, 6 randomize
    wet = 100.0 * r["mix"] / (1.0 + r["mix"])
    return aupreset(name, "rvb2", [(0, wet), (1, 0.0), (2, 0.008), (3, 0.05),
                                   (4, r["decay_s"]), (5, max(0.2, r["decay_s"] * 0.35)), (6, 1.0)])


def dynamics_preset(rig, name="GRE Gate+Comp"):
    c, g = rig["compressor"], rig["gate"]
    thr = c["threshold_db"] if c["on"] else 20.0
    head = 5.0 if c["on"] else 40.0
    exp_ratio = 20.0 if g["on"] else 1.0
    exp_thr = g["threshold_db"] if g["on"] else -100.0
    return aupreset(name, "dcmp", [(0, thr), (1, head), (2, exp_ratio), (3, exp_thr),
                                   (4, c["attack_ms"] / 1000.0), (5, c["release_ms"] / 1000.0), (6, 0.0)])


# ------------------------------------------------------------- MIDI
def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def midi_file(events, bpm=120.0):
    tpq = 480
    tick = lambda t: int(round(t * bpm / 60.0 * tpq))
    msgs = []
    for e in events:
        v = int(np.clip(30 + 97 * e["vel"], 1, 127))
        for m in e["notes"]:
            msgs.append((tick(e["t"]), 1, bytes([0x90, m, v])))
            msgs.append((tick(e["t"] + e["dur"]), 0, bytes([0x80, m, 0])))
    msgs.sort(key=lambda x: (x[0], x[1]))
    trk = bytearray()
    trk += b"\x00\xff\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big")
    trk += b"\x00\xff\x03" + _vlq(len(b"Guitar (transcribed)")) + b"Guitar (transcribed)"
    trk += b"\x00\xc0\x1d"  # GM overdriven guitar
    last = 0
    for t, _, m in msgs:
        trk += _vlq(max(0, t - last)) + m
        last = t
    trk += b"\x00\xff\x2f\x00"
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) + b"MTrk" + struct.pack(">I", len(trk)) + bytes(trk)


# ------------------------------------------------------------- setup sheet
def clock(v):
    """0-10 knob value -> clock position (7 o'clock .. 5 o'clock)."""
    hours = 7 + v * 1.0  # 0->7:00, 5->12:00, 10->17:00
    h = int(hours) % 12 or 12
    m = int(round((hours - int(hours)) * 60 / 15) * 15) % 60
    return f"{h}:{m:02d}"


def setup_steps(rig, report):
    a = AMPS[rig["amp"]["archetype"]]
    amp = rig["amp"]
    gain_total = amp["gain"] + (3 if rig["drive"]["type"] != "none" else 0)
    mic = "Dynamic" if gain_total >= 6 else "Condenser"
    steps = []
    steps.append({"plugin": "Audio track", "settings": [
        "Create an audio track, input = your guitar DI (or import di_resynth.wav from this kit to hear the emulation).",
        "Record/monitor at 44.1 or 48 kHz; aim for peaks around -12 dBFS."]})
    pb = []
    if rig["compressor"]["on"]:
        c = rig["compressor"]
        pb.append(f'{PEDALS["compressor"]["logic"]}: Sustain ~{clock(min(10, c["ratio"] * 1.4))}, Level to unity')
    if rig["drive"]["type"] != "none":
        d = rig["drive"]
        p = PEDALS[d["type"]]
        pb.append(f'{p["logic"]} (≈ {p["real"]}): Drive {d["gain"]:.1f}/10 ({clock(d["gain"])}), '
                  f'Tone {d["tone"]:.1f}/10, Level {d["level"]:.1f}/10')
    if rig["mod"]["type"] == "tremolo":
        m = rig["mod"]
        pb.append(f'{PEDALS["tremolo"]["logic"]}: Rate {m["rate_hz"]:.2f} Hz, Depth {m["depth"] * 100:.0f}%')
    elif rig["mod"]["type"] == "chorus":
        m = rig["mod"]
        pb.append(f'{PEDALS["chorus"]["logic"]}: Rate ~{m["rate_hz"]:.1f} Hz, Depth ~{m["depth"] * 100:.0f}% (low confidence)')
    if pb:
        steps.append({"plugin": "Pedalboard (Amps and Pedals > Pedalboard)", "settings": pb})
    steps.append({"plugin": "Amp Designer (Amps and Pedals > Amp Designer)", "settings": [
        f'Amp: {a["logic_amp"]}   (modeled on {a["real"]})',
        f'Cabinet: {a["logic_cab"]}',
        f'Mic: {mic}, centered; move off-axis if too bright',
        f'Gain {amp["gain"]:.1f} · Bass {amp["bass"]:.1f} · Mids {amp["mid"]:.1f} · Treble {amp["treble"]:.1f} · '
        f'Presence {amp["presence"]:.1f} · Master {amp["master"]:.1f}',
        "Turn Amp Designer's own Reverb and Effects off (the kit handles them below).",
        "Alternative: set Cabinet = Direct and load cab_tone_ir.wav in Space Designer right after (exact tone match)."]})
    if rig["gate"]["on"]:
        steps.append({"plugin": "Noise Gate (Dynamics > Noise Gate)", "settings": [
            f'Threshold {rig["gate"]["threshold_db"]:.0f} dB, Reduction -100 dB, Attack 0.5 ms, Release ~40 ms',
            "Place it first in the chain (before Pedalboard) so it keys off the clean DI."]})
    if rig["eq"]:
        steps.append({"plugin": "Channel EQ (EQ > Channel EQ), after the amp", "settings":
                      [f'Band {i + 1}: {b["f"]:.0f} Hz, {b["g"]:+.1f} dB, Q {b["q"]:.2f}' for i, b in enumerate(rig["eq"])]
                      + ["Or: Match EQ - learn Reference from original.wav, learn Current from your track, Match."]})
    if rig["delay"]["on"]:
        d = rig["delay"]
        steps.append({"plugin": "Tape Delay (Delay > Tape Delay)", "settings": [
            f'Delay {d["time_ms"]:.0f} ms (sync off), Feedback {d["feedback"] * 100:.0f}%',
            f'Low-pass ~{d["tone_hz"]:.0f} Hz, Wet {100 * d["mix"] / (1 + d["mix"]):.0f}% / Dry 100%',
            f'Pedalboard alternative: {PEDALS["delay"]["logic"]} with the same time/feedback/mix.']})
    if rig["reverb"]["on"]:
        r = rig["reverb"]
        rtype = {"spring": "Amp Designer Spring reverb or Pedalboard Spring Box", "room": "ChromaVerb: Room",
                 "plate": "ChromaVerb: Vocal/Plate", "hall": "ChromaVerb: Concert Hall"}[r["type"]]
        steps.append({"plugin": "Reverb (ChromaVerb or Space Designer)", "settings": [
            f'{rtype}; Decay {r["decay_s"]:.2f} s, Pre-delay {r["predelay_ms"]:.0f} ms, Mix {100 * r["mix"] / (1 + r["mix"]):.0f}%',
            "Exact match: Space Designer > load reverb_ir.wav, Dry 100%, Wet to taste (start at the mix above)."]})
    return steps


def setup_markdown(rig, report, analysis):
    a = AMPS[rig["amp"]["archetype"]]
    lines = ["# Logic Pro setup - estimated guitar rig", "",
             f"Best guess: **{a['label']}** (≈ {a['real']}) -> Logic **{a['logic_amp']}** + **{a['logic_cab']}**", ""]
    alts = report.get("amp_candidates", [])[1:3]
    if alts:
        lines.append("Close alternatives: " + ", ".join(f"{c['logic_amp']} ({c['real']}, {c['confidence']:.0%})" for c in alts))
        lines.append("")
    lines.append("Signal chain, in channel-strip order:")
    lines.append("")
    for i, s in enumerate(setup_steps(rig, report), 1):
        lines.append(f"## {i}. {s['plugin']}")
        lines += [f"- {x}" for x in s["settings"]]
        lines.append("")
    t = analysis.get("tuning", {})
    if t:
        lines.append(f"Tuning guess: {t.get('tuning_guess')} (A4 ≈ {t.get('a4_reference_hz', 440)} Hz)")
    lines += ["", "## Files in this kit",
              "- original.wav - the analysed clip (A/B reference, Match EQ reference)",
              "- emulation.wav - the transcribed performance played through the estimated rig",
              "- di_resynth.wav - dry synthetic DI of the performance: put it on a track and build the chain above",
              "- performance.mid - the transcription as MIDI",
              "- cab_tone_ir.wav - cabinet + tone-match impulse response (Space Designer, Wet 100 / Dry 0)",
              "- reverb_ir.wav - reverb impulse response (Space Designer, Dry 100)",
              "- presets/*.aupreset - Apple AU presets (AUNBandEQ, AUDelay, AUReverb2, AUDynamicsProcessor).",
              "  Run install_presets.command, then pick them from the plug-in's Settings menu under",
              "  Audio Units > Apple. These are generated from the public AU state format and are experimental.",
              "- rig.json - every estimated parameter"]
    return "\n".join(lines)


INSTALL_SH = """#!/bin/bash
# Copies the Apple Audio Unit presets into ~/Library/Audio/Presets so Logic can list them.
cd "$(dirname "$0")"
for d in presets/*/; do
  name=$(basename "$d")
  mkdir -p "$HOME/Library/Audio/Presets/Apple/$name"
  cp "$d"*.aupreset "$HOME/Library/Audio/Presets/Apple/$name/"
  echo "Installed presets for $name"
done
echo "Done. In Logic: insert the Apple AU (Audio Units > Apple > ...), open its Settings menu."
"""


def build_kit(out_dir, rig, report, analysis, original_stereo, emulation, di, events, rig_json):
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    save_wav(os.path.join(out_dir, "original.wav"), original_stereo)
    save_wav(os.path.join(out_dir, "emulation.wav"), emulation)
    save_wav(os.path.join(out_dir, "di_resynth.wav"), di)
    save_wav(os.path.join(out_dir, "cab_tone_ir.wav"), chain_ir(rig))
    if rig["reverb"]["on"]:
        save_wav(os.path.join(out_dir, "reverb_ir.wav"), reverb_ir(rig["reverb"]).astype(np.float32))
    with open(os.path.join(out_dir, "performance.mid"), "wb") as fh:
        fh.write(midi_file(events))
    md = setup_markdown(rig, report, analysis)
    with open(os.path.join(out_dir, "LOGIC_SETUP.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(out_dir, "rig.json"), "w") as fh:
        fh.write(rig_json)
    presets = {"AUNBandEQ": ("GRE Match EQ", nbandeq_preset(rig["eq"]))}
    if rig["delay"]["on"]:
        presets["AUDelay"] = ("GRE Delay", delay_preset(rig["delay"]))
    if rig["reverb"]["on"]:
        presets["AUReverb2"] = ("GRE Reverb", reverb2_preset(rig["reverb"]))
    if rig["gate"]["on"] or rig["compressor"]["on"]:
        presets["AUDynamicsProcessor"] = ("GRE Gate+Comp", dynamics_preset(rig))
    zpath = os.path.join(out_dir, "logic_kit.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out_dir)):
            p = os.path.join(out_dir, fn)
            if fn.endswith(".zip") or fn.endswith(".svg") or fn.endswith(".png") or not os.path.isfile(p):
                continue
            z.write(p, f"logic_kit/{fn}")
        for plug, (nm, data) in presets.items():
            z.writestr(f"logic_kit/presets/{plug}/{nm}.aupreset", data)
        zi = zipfile.ZipInfo("logic_kit/install_presets.command")
        zi.external_attr = (0o100755 << 16)
        z.writestr(zi, INSTALL_SH)
        if os.path.exists(os.path.join(out_dir, "rig.svg")):
            z.write(os.path.join(out_dir, "rig.svg"), "logic_kit/rig.svg")
    files["zip"] = zpath
    return md, setup_steps(rig, report)
