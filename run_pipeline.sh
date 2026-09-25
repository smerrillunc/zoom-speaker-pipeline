#!/usr/bin/env bash
# Run the four pipeline scripts, in order, over one collection of meeting videos.
#
#   ./run_pipeline.sh VIDEO_DIR WORK_DIR ["Institution, Place"]
#
# Intermediate files go to WORK_DIR/intermediate, transcripts to WORK_DIR/transcripts.
# Order: ocr.py and transcribe.py (independent; here one after the other), then
# clean_ocr.py over the whole folder, then merge.py. On a cluster, run ocr.py and
# transcribe.py as separate jobs; both take --shard i/n for job arrays.
set -euo pipefail
if [ $# -lt 2 ]; then sed -n 2,9p "$0"; exit 2; fi
VIDEOS=$1; WORK=$2; CONTEXT=${3:-$(basename "$(cd "$VIDEOS" && pwd)" | tr '_' ' ')}
HERE=$(cd "$(dirname "$0")" && pwd)/pipeline
PY=${PYTHON:-python}

"$PY" "$HERE/ocr.py"        "$VIDEOS" --out "$WORK/intermediate"
"$PY" "$HERE/transcribe.py" "$VIDEOS" --out "$WORK/intermediate"
"$PY" "$HERE/clean_ocr.py"  "$WORK/intermediate" --out "$WORK/intermediate" --context "$CONTEXT"
"$PY" "$HERE/merge.py" --asr "$WORK/intermediate" --speakers "$WORK/intermediate" --out "$WORK/transcripts" --formats json,txt
