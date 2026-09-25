"""
Audio extraction, transcription and (optional) audio speaker clustering.

The video pipeline supplies *who* is speaking; this module supplies *what* was said
and *when*.  Transcription always runs through faster-whisper, so a given model and
input give the same segments on every machine.

Audio clustering with pyannote groups segments into voice clusters, which
``attribution.attribute_segments`` then names from the on-screen signal. It needs a
Hugging Face token with access to the gated ``pyannote/speaker-diarization-3.1``;
without one every segment is named from screen time alone.

PaddlePaddle (OCR) and CTranslate2 (Whisper) each load their own OpenMP runtime and
crash when both are live in one process. That is why OCR and transcription are
separate steps: never import this module and ``namereader`` in the same process.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Callable, List, Optional


def has_audio_stream(video_path: str) -> bool:
    """
    Report whether a media file carries an audio stream.

    Worth checking up front: a video-only download (``yt-dlp -f bv*``) or a silent
    screen capture otherwise fails deep inside ffmpeg with an opaque message.

    Args:
        video_path (str): Path to the media file.

    Returns:
        bool: ``True`` when at least one audio stream is present. Returns ``True``
        when ffprobe is unavailable, so the check never blocks a valid file.
    """
    if shutil.which("ffprobe") is None:
        return True

    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", video_path,
        ],
        capture_output=True,
        text=True,
    )
    return bool(process.stdout.strip())


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract a 16 kHz mono WAV track from a video file.

    Args:
        video_path (str): Source video.
        output_path (str | None): Destination WAV; a temporary file is used if omitted.

    Returns:
        str: Path to the extracted audio.

    Raises:
        RuntimeError: If ffmpeg is unavailable, the file carries no audio, or the
            extraction fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract audio but was not found on PATH")

    if not has_audio_stream(video_path):
        raise RuntimeError(
            "This file has no audio track, so there is nothing to transcribe. "
            "If it was downloaded with yt-dlp, request audio too "
            "(for example -f 'bv*[height<=720]+ba/b')."
        )

    if output_path is None:
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        output_path = handle.name

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        output_path,
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{process.stderr.strip()[-1000:]}")

    return output_path


def _torch_device(requested: Optional[str] = None) -> str:
    """Resolve the compute device, preferring CUDA when it is actually usable."""
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def transcribe(
    audio_path: str,
    model_size: str = "large-v2",
    device: Optional[str] = None,
    language: Optional[str] = "en",
    compute_type: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[dict]:
    """
    Transcribe audio into time-stamped segments.

    Args:
        audio_path (str): Path to a WAV file.
        model_size (str): Whisper checkpoint name, e.g. ``"large-v2"`` or ``"small"``.
        device (str | None): ``"cuda"``/``"cpu"``; auto-detected when omitted.
        language (str | None): Force a language code, or ``None`` to auto-detect.
        compute_type (str | None): CTranslate2 precision; defaults to ``float16`` on
            CUDA and ``int8`` on CPU.
        progress (callable | None): Called as ``progress(fraction, message)``.

    Returns:
        list[dict]: Segments with ``start``, ``end`` and ``text``.
    """
    device = _torch_device(device)
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")

    if progress:
        progress(0.0, f"Loading {model_size} on {device}")

    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(audio_path, language=language, vad_filter=True)

    segments = []
    total = getattr(info, "duration", 0.0) or 0.0
    for segment in raw_segments:
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
        )
        if progress and total:
            progress(min(segment.end / total, 1.0), f"Transcribed {segment.end / 60:.1f} min")

    return segments


def cluster_speakers(
    audio_path: str,
    segments: List[dict],
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[str] = None,
) -> List[dict]:
    """
    Label each segment with an anonymous audio speaker cluster via pyannote.

    Every segment is assigned the cluster it overlaps with most, rather than the
    first cluster that happens to contain either endpoint.  Segments overlapping no
    cluster are labelled ``"unknown"``.

    Args:
        audio_path (str): Path to a WAV file.
        segments (list[dict]): Segments with ``start`` and ``end``.
        hf_token (str | None): Token with access to the gated pyannote pipeline.
        min_speakers (int | None): Lower bound on speaker count.
        max_speakers (int | None): Upper bound on speaker count.
        device (str | None): Compute device.

    Returns:
        list[dict]: The segments, each with a ``speaker`` key such as ``"SPEAKER_00"``.

    Raises:
        RuntimeError: If the pipeline cannot be loaded, typically because the model
            is gated and no token with access was supplied.
    """
    from collections import defaultdict

    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
    except Exception as error:
        raise RuntimeError(
            "Could not load pyannote/speaker-diarization-3.1. It is a gated model: "
            "accept its terms on the Hub and supply a token via HF_TOKEN."
        ) from error

    device = _torch_device(device)
    if device == "cuda":
        try:
            import torch

            pipeline.to(torch.device("cuda"))
        except Exception:
            pass

    diarization = pipeline(
        audio_path, min_speakers=min_speakers, max_speakers=max_speakers
    )
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    labelled = []
    for segment in segments:
        tally = defaultdict(float)
        for start, end, speaker in turns:
            overlap = max(0.0, min(segment["end"], end) - max(segment["start"], start))
            if overlap > 0:
                tally[speaker] += overlap
        segment = dict(segment)
        segment["speaker"] = (
            max(tally.items(), key=lambda item: item[1])[0] if tally else "unknown"
        )
        labelled.append(segment)

    return labelled


def run_asr(
    audio_path: str,
    model_size: str = "large-v2",
    device: Optional[str] = None,
    language: Optional[str] = "en",
    compute_type: Optional[str] = None,
    cluster: bool = False,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Transcribe and optionally cluster voices.

    Args:
        audio_path (str): Path to a WAV file.
        model_size (str): Whisper checkpoint.
        device (str | None): Compute device; auto-detected when omitted.
        language (str | None): Language code, or ``None`` to auto-detect.
        compute_type (str | None): CTranslate2 precision.
        cluster (bool): Run pyannote speaker clustering as well.
        hf_token (str | None): Token for the gated pyannote pipeline.
        min_speakers (int | None): Lower bound on speaker count.
        max_speakers (int | None): Upper bound on speaker count.
        progress (callable | None): Called as ``progress(fraction, message)``.

    Returns:
        dict: ``{"segments": [...], "clustered": bool, "error": str | None}``.
    """
    segments = transcribe(
        audio_path,
        model_size=model_size,
        device=device,
        language=language,
        compute_type=compute_type,
        progress=progress,
    )
    clustered = False
    error = None
    if cluster:
        try:
            segments = cluster_speakers(
                audio_path, segments, hf_token=hf_token,
                min_speakers=min_speakers, max_speakers=max_speakers, device=device,
            )
            clustered = True
        except RuntimeError as failure:
            error = str(failure)
    return {"segments": segments, "clustered": clustered, "error": error}

