#!/usr/bin/env bash
# Run all four steps over one collection of meeting videos.
#
#   ./run_pipeline.sh VIDEO_DIR WORK_DIR ["Institution, Place"]
#
# Intermediate files go to WORK_DIR/steps, transcripts to WORK_DIR/transcripts.
# Steps 1 and 2 run one after the other here; on a cluster, run them as separate
# jobs (they are independent) and take --shard i/n for job arrays.
set -euo pipefail
if [ $# -lt 2 ]; then sed -n 2,8p "$0"; exit 2; fi
VIDEOS=$1; WORK=$2; CONTEXT=${3:-$(basename "$(cd "$VIDEOS" && pwd)" | tr '_' ' ')}
HERE=$(cd "$(dirname "$0")" && pwd)/pipeline
PY=${PYTHON:-python}

"$PY" "$HERE/step1_ocr.py"        "$VIDEOS" --out "$WORK/steps"
"$PY" "$HERE/step2_transcribe.py" "$VIDEOS" --out "$WORK/steps"
"$PY" "$HERE/step3_clean_ocr.py"  "$WORK/steps" --out "$WORK/steps" --context "$CONTEXT"
"$PY" "$HERE/step4_merge.py" --asr "$WORK/steps" --speakers "$WORK/steps" --out "$WORK/transcripts" --formats json,txt
