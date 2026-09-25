#!/usr/bin/env bash
# One collection on a Slurm cluster: OCR on CPU and transcription on GPU as two
# independent job arrays, then cleaning and merging once both finish.
# Adjust partitions, sizes and paths to your cluster.
#
#   bash scripts/slurm_example.sh /data/riverton_council /scratch/riverton "Riverton City Council, Oregon"
set -euo pipefail
VIDEOS=$1; WORK=$2; CONTEXT=$3
REPO=$(cd "$(dirname "$0")/.." && pwd)
N_OCR=${N_OCR:-16}   # OCR tasks (each processes every 16th video)
N_ASR=${N_ASR:-2}    # transcription tasks
mkdir -p "$WORK/steps" "$WORK/logs"

ocr=$(sbatch --parsable --array=0-$((N_OCR - 1)) -c 4 --mem 16G -t 2-00:00:00 \
  -o "$WORK/logs/ocr_%a.log" --wrap \
  "cd $REPO && python ocr.py $VIDEOS --out $WORK/steps --shard \$SLURM_ARRAY_TASK_ID/$N_OCR")

asr=$(sbatch --parsable --array=0-$((N_ASR - 1)) -c 8 --mem 32G --gres=gpu:1 -t 1-00:00:00 \
  -o "$WORK/logs/asr_%a.log" --wrap \
  "cd $REPO && python transcribe.py $VIDEOS --out $WORK/steps --shard \$SLURM_ARRAY_TASK_ID/$N_ASR")

sbatch --dependency=afterok:$ocr:$asr -c 1 --mem 8G -t 2:00:00 -o "$WORK/logs/merge.log" --wrap \
  "cd $REPO && python clean_ocr.py $WORK/steps --out $WORK/steps --context '$CONTEXT' && \
   python merge.py --asr $WORK/steps --speakers $WORK/steps --out $WORK/transcripts --formats json,txt"
