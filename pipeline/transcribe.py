#!/usr/bin/env python
"""
Transcription: what was said, when, and in which voice.

For every video (or audio file), extract 16 kHz mono audio, transcribe it with
faster-whisper, and group the segments into voice clusters with pyannote.

    export HF_TOKEN=...            # access to the gated pyannote/speaker-diarization-3.1
    python pipeline/transcribe.py meeting.mp4 --out work/
    python pipeline/transcribe.py videos/ --out work/ --whisper large-v2 --shard 0/4

Writes ``<out>/<id>.asr.json``:

    {"segments": [{"start": 1.62, "end": 7.17, "text": "Good afternoon ...", "speaker": "SPEAKER_11"}, ...],
     "clustered": true,     # false when pyannote did not run; "speaker" is then absent
     "error": null,         # why clustering failed, if it did
     "asr_seconds": 176.3}

``--clustering auto`` (default) clusters when ``HF_TOKEN`` is set; ``on`` treats a
clustering failure as an error; ``off`` never clusters. Without clusters, ``merge.py``
names each segment from screen time alone, which is noisier.

Order: run before ``merge.py``. Independent of ``ocr.py``.

GPU-bound: Whisper large-v2 runs ~22x real time on an L40. Must not run in the same
process as ``ocr.py``.
"""

import argparse
import os
import sys
import tempfile
import time
import traceback

from helpers.files import collect, shard, stem, write_json

AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="video or audio files, or directories of them")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--whisper", default="large-v2", help="Whisper model (default large-v2, as released)")
    p.add_argument("--language", default="en", help="language code, or 'auto' to detect")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    p.add_argument("--compute-type", default=None, help="CTranslate2 precision (default float16 on GPU, int8 on CPU)")
    p.add_argument("--clustering", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--shard", default="0/1", help="'i/n': process every n-th input starting at i")
    p.add_argument("--overwrite", action="store_true", help="redo inputs that already have output")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from helpers import files
    inputs = shard(collect(args.inputs, files.VIDEO_SUFFIXES + AUDIO_SUFFIXES), args.shard)
    if not inputs:
        print("No inputs to process.", file=sys.stderr)
        return 0

    from helpers import speech

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    cluster = args.clustering == "on" or (args.clustering == "auto" and bool(token))
    if args.clustering == "on" and not token:
        print("--clustering on needs HF_TOKEN.", file=sys.stderr)
        return 2

    failures = 0
    for n, path in enumerate(inputs, 1):
        meeting = stem(path)
        target = os.path.join(args.out, f"{meeting}.asr.json")
        if os.path.exists(target) and not args.overwrite:
            print(f"[{n}/{len(inputs)}] {meeting}: exists, skipping", flush=True)
            continue
        started = time.time()
        audio = path
        scratch = None
        try:
            if not path.lower().endswith(AUDIO_SUFFIXES):
                scratch = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                audio = speech.extract_audio(path, scratch)
            result = speech.run_asr(
                audio, model_size=args.whisper, device=args.device,
                language=None if args.language == "auto" else args.language,
                compute_type=args.compute_type, cluster=cluster, hf_token=token,
                min_speakers=args.min_speakers, max_speakers=args.max_speakers)
        except Exception as error:
            failures += 1
            print(f"[{n}/{len(inputs)}] {meeting}: FAILED - {error}", file=sys.stderr)
            traceback.print_exc()
            continue
        finally:
            if scratch and os.path.exists(scratch):
                os.unlink(scratch)

        if args.clustering == "on" and result.get("error"):
            failures += 1
            print(f"[{n}/{len(inputs)}] {meeting}: clustering required but failed - {result['error']}",
                  file=sys.stderr)
            continue
        result["asr_seconds"] = round(time.time() - started, 1)
        write_json(target, result)
        print(f"[{n}/{len(inputs)}] {meeting}: {len(result['segments'])} segments, "
              f"clustered={result['clustered']}, {time.time() - started:.0f}s"
              + (f" ({result['error']})" if result.get("error") else ""), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
