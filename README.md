# Zoom speaker pipeline

Named-speaker transcripts from Zoom-style meeting recordings.

Audio diarization can tell you that *someone else* started talking. It cannot tell you
*who*. In a Zoom gallery view, Zoom draws a highlight around the active speaker's
tile and prints their name in it. This pipeline reads that name once a second, cleans
the reads into stable person identities, and aligns them onto a Whisper transcript:

```json
[{"start": 4.89, "end": 29.59, "speaker": "waynechristian", "text": "Good afternoon, ladies and gentlemen..."},
 {"start": 29.59, "end": 41.20, "speaker": "susangross", "text": "Thank you, Judge..."}]
```

## The four steps

```
                ┌── step1_ocr ────────► <id>.ocr.json ──► step3_clean_ocr ──► <id>.speakers.json ─┐
 meeting video ─┤                                          (per collection)      registry.json       ├─► step4_merge ──► <id>.json
                └── step2_transcribe ─► <id>.asr.json ───────────────────────────────────────────────┘                  (transcript)
```

| Step | Script | Input → output | Cost |
|---|---|---|---|
| 1 | `step1_ocr.py` | video → raw on-screen names, one read per second | CPU, ~1 s per frame on 4 cores |
| 2 | `step2_transcribe.py` | video → Whisper segments + pyannote voice clusters | GPU, ~22× real time (large-v2, L40) |
| 3 | `step3_clean_ocr.py` | all of a collection's OCR → cleaned speaker tracks + who-is-who registry | seconds per meeting |
| 4 | `step4_merge.py` | transcript + cleaned track → speaker-attributed transcript | seconds per meeting |

Steps 1 and 2 are independent: run them in parallel, on different machines, or as
separate job arrays. They must not share a process, because PaddleOCR and CTranslate2
load conflicting OpenMP runtimes. Steps 3 and 4 need only `rapidfuzz`, so the
expensive outputs can be re-cleaned and re-merged anywhere in minutes.

## Layout

```
pipeline/
    step1_ocr.py            video → raw on-screen names
    step2_transcribe.py     video → Whisper segments + voice clusters
    step3_clean_ocr.py      raw names → people, per collection
    step4_merge.py          transcript + people → speaker-attributed transcript
    helpers/                code the steps call; nothing here is run directly
        highlight.py        find the highlighted tile, crop its name strip
        namereader.py       read the name strip (PaddleOCR + EDSR)
        speech.py           audio extraction, Whisper, pyannote
        identity.py         parse names, link them into people
        attribution.py      name each transcribed segment
        render.py           txt / srt / vtt output
        files.py            file discovery, sharding, safe writes
run_pipeline.sh             all four steps over one folder of videos
docs/METHOD.md              how names are cleaned and linked, with measurements
docs/FORMATS.md             every output file and field
```

Steps 1 and 2 do not depend on each other; the numbers give the reading order.

## Quick start

```bash
conda create -n zoomspeakers python=3.11 && conda activate zoomspeakers
pip install -r requirements.txt            # exact versions; see Installation
export HF_TOKEN=hf_...                     # for pyannote voice clusters (step 2)

python pipeline/step1_ocr.py        meetings/ --out work/
python pipeline/step2_transcribe.py meetings/ --out work/
python pipeline/step3_clean_ocr.py  work/ --out work/ --context "Riverton City Council, Oregon"
python pipeline/step4_merge.py --asr work/ --speakers work/ --out transcripts/ --formats json,txt
```

Or all four at once: `./run_pipeline.sh meetings/ work/ "Riverton City Council, Oregon"`.

Every step skips meetings that already have output (`--overwrite` redoes them), writes
through a temporary file so an interrupted run never leaves half a result, and accepts
`--shard i/n` (steps 1–2), so a job array can split a folder across tasks.

## What your video needs

The recording must show the **Zoom gallery view with the active-speaker highlight**
and legible name labels. A fixed camera on a council chamber carries no identity
signal. Neither does a layout that names the speaker only in a "X is talking" banner.
After step 1, check each meeting's **grid fraction** (`frames_grid / frames_sampled`
in `<id>.ocr.json`, also in `<id>.stats.json` after step 4):

| grid fraction | meaning |
|---|---|
| ≥ 0.8 (`layout: gallery`) | the highlight was found; names are the active speaker |
| 0.3 – 0.8 (`mixed`) | part gallery, part screen-share; check the meeting |
| < 0.3 (`screen-share`) | names came from a fixed corner thumbnail and may not be the speaker; exclude |

## Step by step

### 1. `step1_ocr.py`: who is on screen

```bash
python pipeline/step1_ocr.py VIDEO_OR_DIR... --out DIR [--sampling-rate 1.0] [--no-super-resolution] [--shard i/n]
```

Each sampled frame is searched for the highlight ring, a thin, saturated green
outline tested for geometry and not just colour, so a green shirt or a lawn does not
qualify. The bottom-left name strip of that tile is cropped and read with PaddleOCR. Crops
read with low confidence are upscaled 4× with EDSR and re-read. The raw reads are
stored unchanged. A preflight reads a synthetic label first, so a broken OCR install
fails in seconds instead of producing a folder of unnamed meetings.

### 2. `step2_transcribe.py`: what was said, and in which voice

```bash
python pipeline/step2_transcribe.py VIDEO_OR_DIR... --out DIR [--whisper large-v2] [--clustering auto|on|off] [--shard i/n]
```

Extracts 16 kHz mono audio, transcribes with faster-whisper (VAD on), and clusters the
segments by voice with `pyannote/speaker-diarization-3.1`. That model is gated: accept
its terms on the Hugging Face Hub and set `HF_TOKEN`. With `--clustering auto` and no
token, segments are left unclustered and step 4 names each one from screen time alone.

### 3. `step3_clean_ocr.py`: raw reads → people

```bash
python pipeline/step3_clean_ocr.py DIR --out DIR [--context "Institution, Place"] [--no-cross-meeting]
```

Run it once per **collection**: the meetings of one council, court or board, in one
folder. Identities are linked across the meetings of a collection and never between
collections.

- **Parse** each read into name, titles, roles, pronouns and device:
  `Appellant - Niles Illich` → `nilesillich`, `JudgePatricia O'Caña-Oli..` →
  `patriciaocanaoli` (clipped), `Pat Benavides CC6 Crt Coordinator` → `patbenavides`.
  Device, room and organisation tiles (`Michael's iPad (2)`, `Council Chambers`,
  `Montague Townshi..`) become `Other`.
- **Link** spellings of one person within each meeting, then across meetings, with
  named rules: OCR variant, clipped end, clipped start, debris after the name, initial
  + surname, titled surname. Two guards keep a merge from inventing attribution. A
  clipped or partial form links only when exactly one person could complete it. Two
  names that each hold the screen for 5 s in one meeting are never merged.
- `--context` gives the collection's institution and place. Tiles made of those words
  (the body's own Zoom account) are organisation tiles, not people. It defaults to the
  folder name.

`registry.json` lists every person: canonical key, display name, each variant with the
rule and score that linked it, and every **refused** link with its reason. Merges are
auditable and reversible. See [docs/METHOD.md](docs/METHOD.md) for the rules.

### 4. `step4_merge.py`: attribute the text

```bash
python pipeline/step4_merge.py --asr DIR --speakers DIR --out DIR [--formats json,txt,srt,vtt] [--lag 0.6]
```

The screen track is shifted back 0.6 s, because Zoom moves the highlight after a new
voice starts. A voice cluster that is ≥80% one person on screen is named as a whole, so
a few misread frames cannot rename anyone. When pyannote put two people in one
cluster, a segment takes the name covering ≥70% of it on screen, as long as that
person holds ≥10% of the cluster. A cluster heard mostly under a room, device or
screen-share tile stays `Other`. Output formats: [docs/FORMATS.md](docs/FORMATS.md).

## Installation

Use `requirements.txt` as is. Several packages fail **silently** when unpinned:

| Package | Pin | Symptom without it |
|---|---|---|
| `paddlex[ocr]` | `3.2.1` with the `[ocr]` extra | every frame reads as no name (step 1's preflight catches this) |
| `pyannote.audio` | `3.3.2` | voice clustering quietly skipped |
| `huggingface_hub` | `0.36.2` | reported as "gated model, supply HF_TOKEN" when it is not |
| `torch` / `torchaudio` | `2.5.1` | 2.6 breaks pyannote checkpoints; 2.9 drops `AudioMetaData` |
| `opencv-contrib-python-headless` | `4.10.0.84` | EDSR (`cv2.dnn_superres`) exists only in the contrib build |

Also needed: `ffmpeg`/`ffprobe` on `PATH`. For a GPU install use `paddlepaddle-gpu` and
a CUDA build of torch (`torch==2.5.1+cu124`).

Useful environment variables:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | access to the gated pyannote model (step 2) |
| `EDSR_CACHE_DIR` / `EDSR_MODEL_PATH` | where the 38 MB EDSR model is cached, or a local copy; otherwise it is downloaded to `./models/` on first use |
| `PADDLE_PDX_CACHE_HOME` | where PaddleOCR caches its models; point it at a writable disk |

## Reproducibility

- Steps 3 and 4 are deterministic: the same inputs give byte-identical outputs,
  whatever the input order. Step 2 always uses faster-whisper, never an optional
  backend that might happen to be installed.
- Every parameter that affects the result is written into `registry.json` and
  `<id>.attribution.json`.

## Limitations

- **Two people sharing one tile** (`Joyce Johnson / Jennifer Michalik`) are attributed
  to the first name. The video cannot separate them.
- **A partial name that more than one person could complete** stays a separate
  identity. `Judge Patricia` never merges when two judges named Patricia sit on the
  same bench.
- **Given names alone** (`Brandy`) are linked to a full name only within a meeting.
- The thresholds (0.6 s lag; 0.8 / 0.7 / 0.1 attribution) were set from measured
  distributions over 2,224 meetings, not tuned against labels.

## License

MIT. See [LICENSE](LICENSE).
