"""Amp/pedal archetypes and their Logic Pro equivalents.

`gain` ranges are on this tool's 0-10 amp-gain scale. `voicing` drives the DSP
renderer. Logic names follow Logic Pro's Amp Designer / Pedalboard.
"""

AMPS = {
    "blackface": dict(
        label="Blackface Clean", real="Fender Twin Reverb / Super Reverb",
        logic_amp="Large Blackface Amp", logic_cab="Blackface 4 x 10",
        speakers=(4, 10), gain=(0.0, 3.5), mid_hz=450, mid_db=-4.0, bright=1.0,
        tight_hz=70, stages=1, bias=0.05, cab=dict(low_cut=75, high_cut=5600, res=105, pres_hz=2600, pres_db=3.0),
        color="#1c1c1c", panel="#c9c9c9", logo="Tolex Twin"),
    "tweed": dict(
        label="Tweed Breakup", real="Fender Tweed Deluxe / Bassman",
        logic_amp="Small Tweed Amp", logic_cab="Tweed 1 x 12",
        speakers=(1, 12), gain=(2.5, 6.0), mid_hz=750, mid_db=1.5, bright=-1.5,
        tight_hz=55, stages=1, bias=0.18, cab=dict(low_cut=85, high_cut=4600, res=120, pres_hz=2200, pres_db=2.0),
        color="#c8a86a", panel="#7a5a2a", logo="Tweed"),
    "ac30": dict(
        label="British Chime", real="Vox AC30",
        logic_amp="British Amp", logic_cab="British 2 x 12",
        speakers=(2, 12), gain=(1.5, 6.0), mid_hz=2200, mid_db=3.0, bright=1.5,
        tight_hz=95, stages=1, bias=0.12, cab=dict(low_cut=95, high_cut=5200, res=115, pres_hz=2600, pres_db=4.0),
        color="#2a2a2a", panel="#7b2e2a", logo="Chime 30"),
    "bluesbreaker": dict(
        label="British Blues", real="Marshall Bluesbreaker",
        logic_amp="British Blues Amp", logic_cab="British Blues 2 x 12",
        speakers=(2, 12), gain=(3.0, 6.0), mid_hz=800, mid_db=2.0, bright=0.5,
        tight_hz=80, stages=1, bias=0.12, cab=dict(low_cut=85, high_cut=5000, res=110, pres_hz=2400, pres_db=3.0),
        color="#1e1e1e", panel="#c8a24a", logo="Blues 45"),
    "plexi": dict(
        label="Plexi Crunch", real="Marshall Super Lead 'Plexi'",
        logic_amp="Vintage British Head", logic_cab="Vintage British 4 x 12",
        speakers=(4, 12), gain=(4.0, 7.5), mid_hz=850, mid_db=3.0, bright=0.5,
        tight_hz=90, stages=1, bias=0.10, cab=dict(low_cut=80, high_cut=4800, res=100, pres_hz=2300, pres_db=3.0),
        color="#151515", panel="#d4b04c", logo="Plexi 100"),
    "jcm800": dict(
        label="British Hot Rod", real="Marshall JCM800",
        logic_amp="Modern British Head", logic_cab="Modern British 4 x 12",
        speakers=(4, 12), gain=(5.5, 8.5), mid_hz=1000, mid_db=2.5, bright=1.0,
        tight_hz=120, stages=2, bias=0.08, cab=dict(low_cut=85, high_cut=5200, res=100, pres_hz=2600, pres_db=4.0),
        color="#151515", panel="#d4b04c", logo="Lead 800"),
    "orange": dict(
        label="Thick British", real="Orange AD30 / Rockerverb",
        logic_amp="Sunshine Head", logic_cab="Sunshine 4 x 12",
        speakers=(4, 12), gain=(5.0, 8.0), mid_hz=500, mid_db=2.5, bright=-1.0,
        tight_hz=85, stages=2, bias=0.12, cab=dict(low_cut=80, high_cut=4500, res=95, pres_hz=2000, pres_db=2.0),
        color="#e07a1f", panel="#f2f0ea", logo="Sunshine"),
    "hiwatt": dict(
        label="Stadium Clean-Crunch", real="Hiwatt DR103",
        logic_amp="Stadium Head", logic_cab="Stadium 4 x 12",
        speakers=(4, 12), gain=(2.0, 5.5), mid_hz=1500, mid_db=1.5, bright=1.5,
        tight_hz=100, stages=1, bias=0.06, cab=dict(low_cut=85, high_cut=5600, res=100, pres_hz=2800, pres_db=3.0),
        color="#1a1a1a", panel="#d9d9d9", logo="Stadium"),
    "mark": dict(
        label="California Lead", real="Mesa/Boogie Mark IV",
        logic_amp="Studio Amp", logic_cab="Studio 1 x 12",
        speakers=(1, 12), gain=(6.0, 9.0), mid_hz=750, mid_db=-3.0, bright=1.0,
        tight_hz=130, stages=2, bias=0.08, cab=dict(low_cut=90, high_cut=5000, res=115, pres_hz=2600, pres_db=3.0),
        color="#202020", panel="#9a9a9a", logo="Studio IV"),
    "recto": dict(
        label="Modern High Gain", real="Mesa/Boogie Dual Rectifier",
        logic_amp="Modern American Head", logic_cab="Modern American 4 x 12",
        speakers=(4, 12), gain=(7.5, 10.0), mid_hz=650, mid_db=-6.0, bright=1.0,
        tight_hz=150, stages=2, bias=0.06, cab=dict(low_cut=75, high_cut=5000, res=90, pres_hz=2700, pres_db=4.0),
        color="#111111", panel="#8a8f96", logo="RECTIFIED"),
    "high_octane": dict(
        label="Boutique High Gain", real="Soldano SLO-100 / Diezel",
        logic_amp="High Octane Head", logic_cab="High Octane 4 x 12",
        speakers=(4, 12), gain=(7.0, 10.0), mid_hz=900, mid_db=1.0, bright=0.5,
        tight_hz=140, stages=2, bias=0.07, cab=dict(low_cut=80, high_cut=5000, res=95, pres_hz=2500, pres_db=3.0),
        color="#101010", panel="#3a62b0", logo="Octane"),
    "turbo": dict(
        label="Scooped Metal", real="Bogner Uberschall",
        logic_amp="Turbo Head", logic_cab="Turbo 4 x 12",
        speakers=(4, 12), gain=(8.5, 10.0), mid_hz=700, mid_db=-8.0, bright=2.0,
        tight_hz=160, stages=2, bias=0.05, cab=dict(low_cut=75, high_cut=5400, res=90, pres_hz=3000, pres_db=5.0),
        color="#0c0c0c", panel="#b0161b", logo="TURBO"),
    "pawnshop": dict(
        label="Lo-Fi Pawnshop", real="Supro S6616",
        logic_amp="Pawnshop Amp", logic_cab="Pawnshop 1 x 8",
        speakers=(1, 8), gain=(4.0, 8.0), mid_hz=1100, mid_db=4.0, bright=-2.0,
        tight_hz=120, stages=1, bias=0.2, cab=dict(low_cut=130, high_cut=3600, res=160, pres_hz=1800, pres_db=4.0),
        color="#e9e4d6", panel="#2d5d8a", logo="Pawnshop"),
}

PEDALS = {
    "overdrive": dict(short="≈ Tube Screamer", label="Overdrive", real="Ibanez Tube Screamer-style mid boost", logic="Vintage Drive", color="#3fa34d"),
    "distortion": dict(short="≈ ProCo RAT", label="Distortion", real="ProCo RAT", logic="Grit", color="#2b2b2b"),
    "fuzz": dict(short="≈ Fuzz Face / Muff", label="Fuzz", real="Fuzz Face / Big Muff", logic="Happy Face Fuzz", color="#c73b2f"),
    "compressor": dict(short="≈ Dyna Comp", label="Compressor", real="MXR Dyna Comp", logic="Squash Compressor", color="#2f6fc7"),
    "tremolo": dict(short="Amp-style tremolo", label="Tremolo", real="Amp tremolo", logic="Trem-o-Tone", color="#7d4bc2"),
    "chorus": dict(short="≈ Boss CE-2", label="Chorus", real="Boss CE-2", logic="Retro Chorus", color="#36b3c4"),
    "delay": dict(short="Tape / analog echo", label="Delay", real="Analog / tape echo", logic="Tru-Tape Delay", color="#d8a21c"),
    "reverb": dict(short="Spring / room", label="Reverb", real="Spring / room reverb", logic="Spring Box", color="#3a7bd5"),
}
