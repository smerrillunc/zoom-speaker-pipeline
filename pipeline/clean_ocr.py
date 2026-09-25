#!/usr/bin/env python
"""
Clean the OCR: turn raw tile reads into stable person identities.

Run once per **collection** (one body's meetings: a council, a court). Two passes:

1. **Within each meeting**, every raw read is parsed (titles, roles, pronouns, device
   names and punctuation split from the name) and spelling variants of one person
   are linked: "Pat Benavides - CC..", "Pat Benaivides", "Pat BenavidesCC6" all
   become ``patbenavides``. Room, device and organisation tiles become ``Other``.
2. **Across meetings**, the same rules link each meeting's identities into one
   person per collection ("Judge Patricia O'Caña-Oli.." in one meeting, "Patricia
   O'Cana-Olivarez" in another). Every merge and every refused merge is recorded in
   ``registry.json`` with the rule that decided it.

    python pipeline/clean_ocr.py work/ --out work/ --context "Montague Township Council, Ontario"

Order: run after ``ocr.py`` has finished every meeting of the collection, and before
``merge.py``.

Reads ``*.ocr.json`` (from ``ocr.py``; ``*.video.json`` from earlier runs also works) and writes:

    <out>/<id>.speakers.json   the cleaned on-screen speaker track of one meeting
    <out>/registry.json        who is who across the collection, with evidence

``--context`` names the collection (institution, place). Its words mark tiles such as
"Montague Township" as the organisation's own account, not a person. Without it, the
directory name is used.

Seconds of work per meeting; no GPU, no models. Deterministic: the same inputs give
byte-identical outputs. See docs/METHOD.md for the rules and guards.
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

from helpers import __version__
from helpers.files import collect, read_json, stem, write_json
from helpers.identity import (
    context_words,
    display_name,
    link_identities,
    merge_pools,
    parse_label,
    resolve_meeting,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="*.ocr.json files, or directories of them (one collection)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--context", default=None,
                   help="institution and place of the collection, e.g. 'Bexar County Court at Law No. 6, Texas'")
    p.add_argument("--no-cross-meeting", action="store_true",
                   help="clean each meeting on its own; skip linking identities across meetings")
    p.add_argument("--variant-threshold", type=float, default=0.88,
                   help="minimum similarity for two spellings to be one person (default 0.88)")
    p.add_argument("--cannot-link-seconds", type=float, default=5.0,
                   help="two names each on screen this long in one meeting are different people (default 5)")
    p.add_argument("--fill-gap-seconds", type=float, default=2.0,
                   help="an unnamed gap this short between two reads of one person is filled (default 2)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = collect(args.inputs, (".ocr.json", ".video.json"))
    # Prefer .ocr.json when both forms of one meeting are present.
    by_meeting = {}
    for path in paths:
        if path.endswith(".ocr.json") or stem(path) not in by_meeting:
            by_meeting[stem(path)] = path
    if not by_meeting:
        print("No *.ocr.json files found.", file=sys.stderr)
        return 1

    folder = os.path.basename(os.path.abspath(args.inputs[0] if os.path.isdir(args.inputs[0])
                                              else os.path.dirname(args.inputs[0])))
    context = context_words(args.context or folder.replace("_", " "))

    # Pass 1: each meeting on its own.
    meetings, pools, kinds = {}, [], Counter()
    for meeting in sorted(by_meeting):
        ocr = read_json(by_meeting[meeting])
        changes, pool, mapping, evidence = resolve_meeting(
            ocr["raw_speaker_changes"], ocr["duration"], meeting=meeting,
            variant_threshold=args.variant_threshold, fill_gap_seconds=args.fill_gap_seconds,
            context=context)
        meetings[meeting] = (ocr, changes, pool, mapping)
        pools.append((pool, mapping))
        kinds.update(parse_label(raw, context).kind for _, raw in ocr["raw_speaker_changes"])

    # Pass 2: across meetings.
    body = merge_pools(pools)
    if args.no_cross_meeting:
        alias, evidence = {k: k for k in body}, []
    else:
        alias, evidence = link_identities(body, args.variant_threshold, args.cannot_link_seconds)

    clusters = defaultdict(list)
    for key, canonical in alias.items():
        clusters[canonical].append(key)
    merged_by = {e["merge"]: e for e in evidence if "merge" in e}
    identities = []
    for canonical, keys in clusters.items():
        roles = Counter()
        for key in keys:
            roles.update(body[key].roles)
        identities.append({
            "speaker": canonical,
            "display": display_name(body, keys, canonical),
            "seconds": round(sum(body[k].seconds for k in keys), 1),
            "meetings": len({m for k in keys for m in body[k].meetings}),
            "titles": sorted({t for k in keys for t in body[k].titles}),
            "roles": [r for r, _ in roles.most_common(5)],
            "variants": [
                {"key": k, "seconds": round(body[k].seconds, 1), "meetings": len(body[k].meetings),
                 "names": [n for n, _ in body[k].names.most_common(3)],
                 **({"rule": merged_by[k]["rule"], "score": merged_by[k]["score"], "via": merged_by[k]["into"]}
                    if k in merged_by else {})}
                for k in sorted(keys, key=lambda k: (-body[k].seconds, k))
            ],
        })
    identities.sort(key=lambda r: (-r["seconds"], r["speaker"]))
    display = {r["speaker"]: r["display"] for r in identities}

    parameters = {k: getattr(args, k) for k in ("variant_threshold", "cannot_link_seconds", "fill_gap_seconds",
                                                 "no_cross_meeting")}
    parameters["context"] = sorted(context)

    for meeting, (ocr, changes, pool, mapping) in meetings.items():
        track = []
        for t, key in changes:
            speaker = key if key == "Other" else alias.get(key, key)
            if not track or track[-1][1] != speaker:
                track.append([float(t), speaker])
        sampled = ocr.get("frames_sampled") or 1
        grid = ocr.get("frames_grid", 0) / sampled
        on_screen = Counter()
        for (t, speaker), nxt in zip(track, track[1:] + [[ocr["duration"], None]]):
            on_screen[speaker] += max(0.0, nxt[0] - t)
        write_json(os.path.join(args.out, f"{meeting}.speakers.json"), {
            "meeting": meeting,
            "video": ocr.get("video"),
            "duration": ocr["duration"],
            "changes": track,
            "speakers": {s: {"display": display.get(s, "Other" if s == "Other" else s),
                             "seconds_on_screen": round(v, 1)} for s, v in sorted(on_screen.items())},
            "stats": {
                "frames_sampled": ocr.get("frames_sampled"),
                "frames_named": ocr.get("frames_named"),
                "frames_grid": ocr.get("frames_grid"),
                "frames_overlay": ocr.get("frames_overlay"),
                "grid_fraction": round(grid, 3),
                "named_fraction": round((ocr.get("frames_named") or 0) / sampled, 3),
                "layout": "gallery" if grid >= 0.8 else "mixed" if grid >= 0.3 else "screen-share",
                "sampling_rate": ocr.get("sampling_rate"),
            },
        })

    write_json(os.path.join(args.out, "registry.json"), {
        "version": __version__,
        "parameters": parameters,
        "meetings": sorted(meetings),
        "label_kinds": dict(sorted(kinds.items())),
        "merges": dict(sorted(Counter(e["rule"] for e in evidence if "merge" in e).items())),
        "identities": identities,
        "refused": [e for e in evidence if "kept_apart" in e],
    }, indent=1)

    people = sum(1 for r in identities)
    print(f"{len(meetings)} meetings: {len(body)} per-meeting names -> {people} people "
          f"({len(merged_by)} cross-meeting merges, "
          f"{sum(1 for e in evidence if 'kept_apart' in e)} refused). Wrote {args.out}/registry.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
