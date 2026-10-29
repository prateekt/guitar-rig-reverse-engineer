"""SVG illustration of the estimated rig, knobs drawn at the estimated settings."""
import math
from xml.sax.saxutils import escape

from .models import AMPS, PEDALS

W, H = 1500, 640
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"


def _t(x, y, s, size=12, fill="#eee", anchor="middle", weight=400, extra=""):
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}' {extra}>{escape(str(s))}</text>")


def knob(cx, cy, r, v, label=None, cap="#1b1b1b", ring="#bbb", txt="#111", value=True):
    ang = math.radians(-135 + 27 * max(0, min(10, v)) - 90)
    x2, y2 = cx + math.cos(ang) * r * 0.8, cy + math.sin(ang) * r * 0.8
    out = [f"<circle cx='{cx}' cy='{cy}' r='{r + 2}' fill='{ring}' opacity='0.5'/>",
           f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{cap}' stroke='#000' stroke-width='1'/>",
           f"<line x1='{cx}' y1='{cy}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#fff' stroke-width='2.2' stroke-linecap='round'/>"]
    if label:
        out.append(_t(cx, cy + r + 13, label, 9, txt, weight=600))
    if value:
        out.append(_t(cx, cy - r - 5, f"{v:.1f}", 9, txt))
    return "".join(out)


def pedal(x, y, key, title, knobs, sub):
    p = PEDALS[key]
    w, h = 112, 170
    c = p["color"]
    out = [f"<g filter='url(#sh)'><rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{c}' stroke='#000' stroke-width='1.5'/></g>",
           f"<rect x='{x + 8}' y='{y + 8}' width='{w - 16}' height='{h - 16}' rx='6' fill='none' stroke='rgba(255,255,255,.18)'/>"]
    n = len(knobs)
    for i, (lab, v) in enumerate(knobs):
        kx = x + w * (i + 1) / (n + 1)
        out.append(knob(kx, y + 38, 10 if n > 2 else 13, v, None, cap="#f2f2f2" if c in ("#2b2b2b",) else "#151515",
                        txt="#fff", value=True))
        out.append(_t(kx, y + 62, lab, 7.5 if n > 2 else 8.5, "#fff", weight=700))
    out.append(f"<circle cx='{x + w / 2}' cy='{y + 82}' r='4' fill='#ff3b30'/>")
    out.append(_t(x + w / 2, y + 108, title, 12, "#fff", weight=800))
    out.append(_t(x + w / 2, y + 122, sub, 8.5, "rgba(255,255,255,.75)"))
    out.append(f"<circle cx='{x + w / 2}' cy='{y + 148}' r='12' fill='#c9c9c9' stroke='#555' stroke-width='2'/>")
    return "".join(out), (x, y + 100), (x + w, y + 100)


def speakers(x, y, w, h, n, size):
    out = []
    if n == 4:
        pos = [(x + w * 0.28, y + h * 0.28), (x + w * 0.72, y + h * 0.28), (x + w * 0.28, y + h * 0.72), (x + w * 0.72, y + h * 0.72)]
    elif n == 2:
        pos = [(x + w * 0.27, y + h * 0.5), (x + w * 0.73, y + h * 0.5)]
    else:
        pos = [(x + w * 0.5, y + h * 0.5)]
    base = min(w / (2.4 if n > 1 else 1.4), h / (2.4 if n == 4 else 1.25)) / 2
    r = base * (size / 12) ** 0.5
    for cx, cy in pos:
        out.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{r:.1f}' fill='url(#cone)' opacity='.55'/>")
        out.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{r * 0.25:.1f}' fill='rgba(0,0,0,.35)'/>")
    return "".join(out)


def amp_block(x, y, rig):
    a = AMPS[rig["amp"]["archetype"]]
    amp = rig["amp"]
    n, size = a["speakers"]
    combo = n <= 2 or a["logic_cab"].startswith(("Blackface", "Tweed"))
    out = []
    W_ = 380
    labels = [("GAIN", amp["gain"]), ("BASS", amp["bass"]), ("MID", amp["mid"]), ("TREBLE", amp["treble"]),
              ("PRES", amp["presence"]), ("MASTER", amp["master"])]
    if combo:
        h = 330
        out.append(f"<g filter='url(#sh)'><rect x='{x}' y='{y}' width='{W_}' height='{h}' rx='14' fill='{a['color']}' stroke='#000' stroke-width='2'/></g>")
        out.append(f"<rect x='{x + 14}' y='{y + 12}' width='{W_ - 28}' height='74' rx='6' fill='{a['panel']}' stroke='rgba(0,0,0,.5)'/>")
        for i, (lab, v) in enumerate(labels):
            out.append(knob(x + 50 + i * 56, y + 46, 13, v, lab, txt="#111"))
        out.append(f"<rect x='{x + 14}' y='{y + 98}' width='{W_ - 28}' height='{h - 112}' rx='6' fill='url(#grille)' stroke='rgba(0,0,0,.6)'/>")
        out.append(speakers(x + 14, y + 98, W_ - 28, h - 112, n, size))
        out.append(f"<rect x='{x + W_ / 2 - 60}' y='{y + 104}' width='120' height='24' rx='4' fill='rgba(0,0,0,.55)'/>")
        out.append(_t(x + W_ / 2, y + 121, a["logo"], 14, "#f0e2b6", weight=800, extra="font-style='italic'"))
        cab_box = (x + 14, y + 98, W_ - 28, h - 112)
        top = y
        bottom = y + h
    else:
        hh = 112
        out.append(f"<g filter='url(#sh)'><rect x='{x}' y='{y}' width='{W_}' height='{hh}' rx='10' fill='{a['color']}' stroke='#000' stroke-width='2'/></g>")
        out.append(f"<rect x='{x + 12}' y='{y + 30}' width='{W_ - 24}' height='70' rx='5' fill='{a['panel']}' stroke='rgba(0,0,0,.5)'/>")
        out.append(_t(x + W_ / 2, y + 22, a["logo"], 15, "#f0e2b6", weight=800, extra="font-style='italic'"))
        for i, (lab, v) in enumerate(labels):
            out.append(knob(x + 50 + i * 56, y + 60, 13, v, lab, txt="#111"))
        cy0 = y + hh + 6
        ch = 300
        out.append(f"<g filter='url(#sh)'><rect x='{x}' y='{cy0}' width='{W_}' height='{ch}' rx='10' fill='{a['color']}' stroke='#000' stroke-width='2'/></g>")
        out.append(f"<rect x='{x + 14}' y='{cy0 + 14}' width='{W_ - 28}' height='{ch - 28}' rx='6' fill='url(#grille)' stroke='rgba(0,0,0,.6)'/>")
        out.append(speakers(x + 14, cy0 + 14, W_ - 28, ch - 28, n, size))
        cab_box = (x + 14, cy0 + 14, W_ - 28, ch - 28)
        top = y
        bottom = cy0 + ch
    out.append(_t(x + W_ / 2, bottom + 22, f"{a['logic_amp']}  ·  {a['logic_cab']}", 13, "#fff", weight=700))
    out.append(_t(x + W_ / 2, bottom + 38, f"≈ {a['real']}", 11, "#aab"))
    return "".join(out), (x, y + 60), cab_box, bottom, W_


def mic(x, y, dyn=True):
    """Side-view mic aimed right (at the speaker)."""
    col = "#3a3a3a" if dyn else "#b8b8b8"
    return (f"<g><rect x='{x}' y='{y}' width='62' height='18' rx='8' fill='{col}' stroke='#000'/>"
            f"<rect x='{x + 56}' y='{y + 1}' width='18' height='16' rx='7' fill='#8a8a8a' stroke='#000'/>"
            f"<path d='M{x + 60},{y + 4} h10 M{x + 60},{y + 9} h10 M{x + 60},{y + 14} h10' stroke='#555'/></g>")


def rack_unit(x, y, title, lines, accent):
    w, h = 230, 70
    out = [f"<g filter='url(#sh)'><rect x='{x}' y='{y}' width='{w}' height='{h}' rx='4' fill='#22252b' stroke='#000'/></g>",
           f"<rect x='{x}' y='{y}' width='8' height='{h}' fill='{accent}'/>",
           f"<circle cx='{x + w - 14}' cy='{y + 14}' r='3' fill='#4cd964'/>",
           _t(x + 18, y + 20, title, 12, "#fff", anchor="start", weight=800)]
    for i, l in enumerate(lines[:3]):
        out.append(_t(x + 18, y + 38 + i * 13, l, 10, "#c9cdd6", anchor="start"))
    return "".join(out)


def guitar(x, y):
    # generic offset-body solid guitar, drawn from primitives
    return f"""
<g transform='translate({x},{y}) rotate(-28)'>
  <path d='M0,40 C-10,0 40,-10 70,10 C95,0 130,5 135,35 C140,60 120,75 125,95 C130,125 95,150 60,140 C20,150 -15,125 -5,95 C0,75 -12,60 0,40 Z' fill='#b3202a' stroke='#111' stroke-width='2'/>
  <path d='M20,52 C30,40 70,38 95,48 L92,110 C70,120 35,118 22,105 Z' fill='#f4efe6' opacity='.9'/>
  <rect x='40' y='60' width='34' height='9' rx='3' fill='#222'/><rect x='40' y='88' width='34' height='9' rx='3' fill='#222'/>
  <rect x='130' y='62' width='190' height='18' rx='3' fill='#8a5a2b' stroke='#111'/>
  <path d='M318,58 L360,52 L366,88 L318,84 Z' fill='#222' stroke='#111'/>
  <circle cx='105' cy='120' r='5' fill='#ddd'/><circle cx='88' cy='130' r='5' fill='#ddd'/>
</g>"""


def cable(p0, p1, sag=40):
    (x0, y0), (x1, y1) = p0, p1
    mx = (x0 + x1) / 2
    return (f"<path d='M{x0:.1f},{y0:.1f} C{mx:.1f},{max(y0, y1) + sag:.1f} {mx:.1f},{max(y0, y1) + sag:.1f} {x1:.1f},{y1:.1f}' "
            f"fill='none' stroke='#0d0d0d' stroke-width='5' stroke-linecap='round'/>"
            f"<path d='M{x0:.1f},{y0:.1f} C{mx:.1f},{max(y0, y1) + sag:.1f} {mx:.1f},{max(y0, y1) + sag:.1f} {x1:.1f},{y1:.1f}' "
            f"fill='none' stroke='#3a3a3a' stroke-width='2' stroke-linecap='round'/>")


def draw_rig(rig, report=None, title=None):
    a = AMPS[rig["amp"]["archetype"]]
    conf = None
    if report and report.get("amp_candidates"):
        conf = report["amp_candidates"][0]["confidence"]
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
             """<defs>
  <linearGradient id='bg' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#1a1d24'/><stop offset='.72' stop-color='#11131a'/><stop offset='1' stop-color='#2a2118'/></linearGradient>
  <pattern id='grille' width='6' height='6' patternUnits='userSpaceOnUse'><rect width='6' height='6' fill='#2b2620'/><path d='M0,6 L6,0' stroke='#4a4035' stroke-width='1.2'/></pattern>
  <radialGradient id='cone'><stop offset='0' stop-color='#000'/><stop offset='.7' stop-color='#111'/><stop offset='1' stop-color='#555'/></radialGradient>
  <filter id='sh' x='-10%' y='-10%' width='130%' height='140%'><feDropShadow dx='0' dy='6' stdDeviation='6' flood-color='#000' flood-opacity='.55'/></filter>
</defs>""",
             f"<rect width='{W}' height='{H}' fill='url(#bg)'/>",
             f"<rect y='{H - 120}' width='{W}' height='120' fill='#2b2118' opacity='.8'/>",
             f"<path d='M0,{H - 120} L{W},{H - 120}' stroke='#5a4630' stroke-width='2'/>"]
    head = title or f"Estimated rig: {a['label']}"
    parts.append(_t(36, 44, head, 24, "#fff", anchor="start", weight=800))
    sub = f"Logic Pro: {a['logic_amp']} + {a['logic_cab']}"
    if conf is not None:
        sub += f"   ·   amp-model confidence {conf:.0%}"
    parts.append(_t(36, 68, sub, 13, "#9aa3b2", anchor="start"))

    # guitar
    parts.append(guitar(70, 350))
    jack = (250, 400)

    # pedals
    peds = []
    if rig["gate"]["on"]:
        peds.append(("compressor", "Noise Gate", [("THRESH", (rig["gate"]["threshold_db"] + 80) / 8)], "Logic: Noise Gate"))
    if rig["compressor"]["on"]:
        c = rig["compressor"]
        peds.append(("compressor", "Squash Comp", [("SUSTAIN", min(10, c["ratio"] * 1.4)), ("LEVEL", 5)], PEDALS["compressor"]["short"]))
    if rig["drive"]["type"] != "none":
        d = rig["drive"]
        p = PEDALS[d["type"]]
        peds.append((d["type"], p["logic"], [("DRIVE", d["gain"]), ("TONE", d["tone"]), ("LEVEL", d["level"])], p["short"]))
    if rig["mod"]["type"] in ("tremolo", "chorus"):
        m = rig["mod"]
        k = m["type"]
        peds.append((k, PEDALS[k]["logic"], [("RATE", min(10, m["rate_hz"])), ("DEPTH", m["depth"] * 10)], f"{m['rate_hz']:.1f} Hz"))
    x = 300
    prev = jack
    for key, ttl, kn, sub_ in peds:
        svg, inp, outp = pedal(x, 360, key, ttl, kn, sub_)
        parts.append(cable(prev, inp, 50))
        parts.append(svg)
        prev = outp
        x += 132
    amp_x = max(x + 150, 700)
    n_spk = a["speakers"][0]
    combo = n_spk <= 2 or a["logic_cab"].startswith(("Blackface", "Tweed"))
    amp_y = (H - 125 - 330) if combo else (H - 125 - 418)
    svg, amp_in, cab_box, bottom, aw = amp_block(amp_x, amp_y, rig)
    parts.append(cable(prev, (amp_x + 8, amp_y + 50), 30))
    parts.append(svg)
    gain_total = rig["amp"]["gain"] + (3 if rig["drive"]["type"] != "none" else 0)
    cx, cy, cw, ch = cab_box
    mx, my = amp_x - 100, cy + ch * 0.72
    parts.append(f"<line x1='{mx + 20}' y1='{my + 9}' x2='{mx + 20}' y2='{H - 128}' stroke='#555' stroke-width='3'/>")
    parts.append(f"<line x1='{mx}' y1='{H - 128}' x2='{mx + 40}' y2='{H - 128}' stroke='#555' stroke-width='4'/>")
    parts.append(mic(mx + 12, my, dyn=gain_total >= 6))
    parts.append(_t(mx + 40, my - 8, "Dynamic mic" if gain_total >= 6 else "Condenser mic", 9, "#9aa3b2"))

    # post effects rack
    rx = amp_x + aw + 50
    ry = 120
    racks = []
    if rig["eq"]:
        racks.append(("Channel EQ (match)", [", ".join(f"{b['f']:.0f}Hz {b['g']:+.0f}" for b in rig["eq"][:3]),
                                             ", ".join(f"{b['f']:.0f}Hz {b['g']:+.0f}" for b in rig["eq"][3:6]),
                                             f"{len(rig['eq'])} bands"], "#8e8e93"))
    if rig["delay"]["on"]:
        d = rig["delay"]
        racks.append(("Tape Delay", [f"Time {d['time_ms']:.0f} ms", f"Feedback {d['feedback'] * 100:.0f}%",
                                     f"Mix {100 * d['mix'] / (1 + d['mix']):.0f}%"], PEDALS["delay"]["color"]))
    if rig["reverb"]["on"]:
        r = rig["reverb"]
        racks.append((f"Reverb · {r['type'].title()}", [f"Decay {r['decay_s']:.2f} s", f"Pre-delay {r['predelay_ms']:.0f} ms",
                                                         f"Mix {100 * r['mix'] / (1 + r['mix']):.0f}%"], PEDALS["reverb"]["color"]))
    if racks:
        parts.append(_t(rx, ry - 14, "POST-AMP (Logic plug-ins)", 11, "#9aa3b2", anchor="start", weight=700))
        parts.append(f"<rect x='{rx - 10}' y='{ry - 4}' width='250' height='{len(racks) * 82 + 6}' rx='6' fill='#0c0d10' stroke='#333'/>")
        for i, (t_, ls, col) in enumerate(racks):
            parts.append(rack_unit(rx, ry + 4 + i * 82, t_, ls, col))
        parts.insert(5, cable((mx + 20, H - 128), (rx + 115, ry + len(racks) * 82 + 2), 110))

    # signal flow caption
    flow = ["Guitar"] + [p[1] for p in peds] + [a["logic_amp"], a["logic_cab"]] + [r[0] for r in racks]
    parts.append(_t(36, H - 22, "Signal flow:  " + "  →  ".join(flow), 12, "#c9b89a", anchor="start"))
    parts.append("</svg>")
    return "".join(parts)
