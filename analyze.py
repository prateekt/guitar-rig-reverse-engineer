#!/usr/bin/env python3
"""Command-line: python analyze.py song.wav [-o out_dir] [--di my_di.wav] [--start 12.5]"""
import argparse
import json
import sys

from gre.pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Reverse-engineer a guitar rig from audio.")
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="gre_output")
    ap.add_argument("--di", help="optional dry DI recording to run through the estimated rig")
    ap.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    a = ap.parse_args()
    res = run(a.audio, a.out, di_path=a.di, start=a.start,
              progress=lambda m, f: print(f"[{f * 100:5.1f}%] {m}", file=sys.stderr))
    print(json.dumps(res["summary"], indent=2))
    print(f"\nOutputs in {a.out}/  (logic_kit.zip, rig.svg, emulation.wav, LOGIC_SETUP.md)")


if __name__ == "__main__":
    main()
