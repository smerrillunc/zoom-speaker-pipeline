#!/usr/bin/env python
"""
Merge: attribute every transcribed segment to a named speaker.

Order: run last, after ``transcribe.py`` and ``clean_ocr.py``.

Joins each meeting's transcript (``<id>.asr.json`` from ``transcribe.py``) with its
cleaned on-screen speaker track (``<id>.speakers.json`` from ``clean_ocr.py``):

* The screen track is shifted back ``--lag`` seconds: Zoom moves the highlight
  about 0.6 s after a new voice starts.
* A voice cluster that is at least ``--pure-threshold`` one person on screen is named
  as a whole, so a few misread frames cannot rename a speaker.
* In a mixed cluster (pyannote put two people together), a segment takes the name
  covering at least ``--screen-threshold`` of it on screen, provided that person
  holds at least ``--minor-share`` of the cluster; otherwise it keeps the cluster's
  name. A cluster heard mostly under a room, device or screen-share tile is ``Other``.

    python pipeline/merge.py --asr work/ --speakers work/ --out transcripts/ --formats json,txt,srt

Writes per meeting:

    <out>/<id>.json               [{"start", "end", "speaker", "text"}, ...]   (the transcript)
    <out>/<id>.attribution.json   per turn: named from "cluster" or "screen", on-screen share,
                                  voice clusters -- plus the display name of every speaker
    <out>/<id>.stats.json         grid fraction, layout, turns, speakers
    <out>/<id>.txt|.srt|.vtt      optional renderings (--formats)
"""

import argparse
import os
import sys

from helpers import __version__
from helpers.attribution import HIGHLIGHT_LAG_SECONDS, attribute_segments, merge_turns
from helpers.files import collect, read_json, stem, write_json


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asr", nargs="+", required=True, help="*.asr.json files or directories (from transcribe.py)")
    p.add_argument("--speakers", nargs="+", required=True, help="*.speakers.json files or directories (from clean_ocr.py)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--formats", default="json", help="comma list of json,txt,srt,vtt (default json)")
    p.add_argument("--lag", type=float, default=HIGHLIGHT_LAG_SECONDS,
                   help=f"seconds the Zoom highlight trails the voice (default {HIGHLIGHT_LAG_SECONDS})")
    p.add_argument("--pure-threshold", type=float, default=0.8,
                   help="a voice cluster this much one person is named as a whole (default 0.8)")
    p.add_argument("--screen-threshold", type=float, default=0.7,
                   help="in a mixed cluster, on-screen share a name needs to claim a segment (default 0.7)")
    p.add_argument("--minor-share", type=float, default=0.1,
                   help="...and the share of the cluster that person must hold (default 0.1)")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    unknown = formats - {"json", "txt", "srt", "vtt"}
    if unknown:
        print(f"Unknown format(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    from helpers import render

    asr = {stem(p): p for p in collect(args.asr, (".asr.json",))}
    speakers = {stem(p): p for p in collect(args.speakers, (".speakers.json",))}
    meetings = sorted(set(asr) & set(speakers))
    for missing, what in ((sorted(set(speakers) - set(asr)), "transcript"), (sorted(set(asr) - set(speakers)), "speaker track")):
        if missing:
            print(f"{len(missing)} meeting(s) have no {what} yet, skipped: {', '.join(missing[:5])}"
                  + (" ..." if len(missing) > 5 else ""), file=sys.stderr)
    if not meetings:
        print("Nothing to merge.", file=sys.stderr)
        return 1

    written = 0
    for meeting in meetings:
        target = os.path.join(args.out, f"{meeting}.json")
        if os.path.exists(target) and not args.overwrite:
            continue
        speech = read_json(asr[meeting])
        track = read_json(speakers[meeting])
        segments = attribute_segments(
            speech["segments"], [tuple(c) for c in track["changes"]], lag=args.lag,
            pure_threshold=args.pure_threshold, screen_threshold=args.screen_threshold,
            minor_share=args.minor_share)
        turns, provenance = merge_turns(segments)

        write_json(target, turns, indent=2)
        names = {s: v["display"] for s, v in track["speakers"].items()}
        write_json(os.path.join(args.out, f"{meeting}.attribution.json"), {
            "version": __version__,
            "parameters": {"lag": args.lag, "pure_threshold": args.pure_threshold,
                           "screen_threshold": args.screen_threshold, "minor_share": args.minor_share,
                           "voice_clusters": bool(speech.get("clustered"))},
            "display_names": {t["speaker"]: names.get(t["speaker"], t["speaker"]) for t in turns},
            "turns": provenance,
        })
        write_json(os.path.join(args.out, f"{meeting}.stats.json"), {
            **track["stats"],
            "duration_minutes": round(track["duration"] / 60, 1),
            "turns": len(turns),
            "turns_named_from_screen": sum(1 for p in provenance if p["source"] == "screen"),
            "speakers": sorted({t["speaker"] for t in turns}),
            "audio_clustering": bool(speech.get("clustered")),
            "clustering_error": speech.get("error"),
        }, indent=2)
        if "txt" in formats:
            open(os.path.join(args.out, f"{meeting}.txt"), "w", encoding="utf-8").write(render.to_text(turns))
        if "srt" in formats:
            open(os.path.join(args.out, f"{meeting}.srt"), "w", encoding="utf-8").write(render.to_srt(turns))
        if "vtt" in formats:
            open(os.path.join(args.out, f"{meeting}.vtt"), "w", encoding="utf-8").write(render.to_vtt(turns))
        written += 1
        print(f"{meeting}: {len(turns)} turns, {len({t['speaker'] for t in turns})} speakers, "
              f"grid {track['stats']['grid_fraction']:.0%}", flush=True)
    print(f"Wrote {written} transcript(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
