#!/usr/bin/env python
"""
Step 1 of 4 -- OCR: read who is on screen, once per sampled frame.

For every video, sample frames (1 per second by default), find Zoom's highlighted
active-speaker tile, crop its name strip and read it with PaddleOCR. The raw reads are
kept exactly as OCR returned them; cleaning them is step 3 (``clean_ocr.py``).

    python ocr.py meeting.mp4 --out work/
    python ocr.py videos/ --out work/ --shard 3/8        # one task of an 8-way array

Writes ``<out>/<id>.ocr.json``:

    {"video": "abc.mp4", "duration": 3745.0, "sampling_rate": 1.0,
     "frames_sampled": 3746, "frames_named": 3739,
     "frames_grid": 3739,      # frames where the highlight ring was found
     "frames_overlay": 7,      # frames named from the screen-share thumbnail instead
     "raw_speaker_changes": [[0.0, "Pat Benavides - CC..."], [3.0, "Nicholas Mahrou AD.."], ...]}

``raw_speaker_changes`` lists a timestamp each time the read changes. Check
``frames_grid / frames_sampled`` (the grid fraction): near 1 means a real gallery view;
near 0 means the names came from a fixed corner thumbnail and cannot be trusted to be
the speaker.

CPU-bound: about 1 s per frame on 4 cores. Must not run in the same process as
``transcribe.py`` (Paddle and CTranslate2 conflict).
"""

import argparse
import os
import sys
import time
import traceback

from zoompipe.files import collect, shard, stem, write_json


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", help="video files or directories of them")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--sampling-rate", type=float, default=1.0, help="frames read per second of video (default 1)")
    p.add_argument("--no-super-resolution", action="store_true",
                   help="skip EDSR upscaling of low-confidence crops (faster, misses small labels)")
    p.add_argument("--max-minutes", type=float, default=None, help="only read this much of each video (for tests)")
    p.add_argument("--shard", default="0/1", help="'i/n': process every n-th video starting at i")
    p.add_argument("--overwrite", action="store_true", help="redo videos that already have output")
    return p


def preflight(reader):
    """
    Fail in seconds, not hours, when OCR is broken.

    The reader swallows OCR exceptions and returns no text, so a broken Paddle install
    would otherwise produce a folder of meetings where nobody is ever named.
    """
    import cv2
    import numpy as np

    reader.reader  # constructs PaddleOCR outside the exception guard
    canvas = np.zeros((40, 260, 3), dtype=np.uint8)
    cv2.putText(canvas, "Jane Doe", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    text, confidence, _ = reader.read_detailed(canvas)
    if not text:
        raise RuntimeError("OCR returned nothing for a synthetic name label. Check the PaddleOCR install "
                           "(see requirements.txt: paddlex[ocr] must be pinned).")
    print(f"preflight OCR: {text!r} at {confidence:.2f}", flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    videos = shard(collect(args.videos), args.shard)
    if not videos:
        print("No videos to process.", file=sys.stderr)
        return 0

    from zoompipe.highlight import ZoomSpeakerDiarization
    from zoompipe.namereader import NameReader

    reader = NameReader(use_super_resolution=not args.no_super_resolution)
    preflight(reader)

    failures = 0
    for n, video in enumerate(videos, 1):
        meeting = stem(video)
        target = os.path.join(args.out, f"{meeting}.ocr.json")
        if os.path.exists(target) and not args.overwrite:
            print(f"[{n}/{len(videos)}] {meeting}: exists, skipping", flush=True)
            continue
        started = time.time()
        try:
            diarizer = ZoomSpeakerDiarization(video, reader=reader)
            try:
                result = diarizer.process_video(
                    sampling_rate=args.sampling_rate,
                    max_seconds=None if args.max_minutes is None else args.max_minutes * 60)
            finally:
                diarizer.close()
        except Exception as error:
            failures += 1
            print(f"[{n}/{len(videos)}] {meeting}: FAILED - {error}", file=sys.stderr)
            traceback.print_exc()
            continue

        write_json(target, {
            "video": os.path.basename(video),
            "duration": result.duration,
            "sampling_rate": args.sampling_rate,
            "super_resolution": not args.no_super_resolution,
            "frames_sampled": result.frames_sampled,
            "frames_named": result.frames_named,
            "frames_grid": result.frames_grid,
            "frames_overlay": result.frames_overlay,
            "raw_speaker_changes": [[float(t), name] for t, name in result.speaker_changes],
            "video_seconds": round(time.time() - started, 1),
        })
        print(f"[{n}/{len(videos)}] {meeting}: {result.frames_sampled} frames, grid {result.grid_fraction:.0%}, "
              f"named {result.named_fraction:.0%}, {time.time() - started:.0f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
