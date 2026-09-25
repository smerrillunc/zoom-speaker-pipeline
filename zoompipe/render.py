"""Output formats for a speaker-attributed transcript."""

import json
from typing import Sequence


def _timestamp(seconds: float, separator: str = ",") -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (WebVTT)."""
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def to_json(segments: Sequence[dict], indent: int = 2) -> str:
    """
    Serialize to the JSON layout used by ``Datasets/*/transcripts_json``.

    Args:
        segments (Sequence[dict]): Speaker-attributed turns.
        indent (int): JSON indentation.

    Returns:
        str: JSON text -- a list of ``{start, end, speaker, text}`` objects.
    """
    records = [
        {
            "start": round(float(s["start"]), 3),
            "end": round(float(s["end"]), 3),
            "speaker": s.get("speaker", "unknown"),
            "text": s.get("text", "").strip(),
        }
        for s in segments
    ]
    return json.dumps(records, indent=indent, ensure_ascii=False)


def to_text(segments: Sequence[dict], timestamps: bool = True) -> str:
    """
    Render a plain-text transcript, one turn per block.

    Args:
        segments (Sequence[dict]): Speaker-attributed turns.
        timestamps (bool): Prefix each turn with its start and end time.

    Returns:
        str: The transcript.
    """
    lines = []
    for segment in segments:
        speaker = segment.get("speaker", "unknown")
        text = segment.get("text", "").strip()
        if timestamps:
            stamp = f"({_timestamp(segment['start'], '.')} - {_timestamp(segment['end'], '.')}) "
        else:
            stamp = ""
        lines.append(f"{stamp}{speaker}: {text}")
    return "\n\n".join(lines)


def to_markdown(segments: Sequence[dict], limit: int = 0) -> str:
    """
    Render the transcript as Markdown for display in the app.

    Args:
        segments (Sequence[dict]): Speaker-attributed turns.
        limit (int): Show at most this many turns; ``0`` shows all.

    Returns:
        str: Markdown text with bolded speaker names and muted timestamps.
    """
    shown = segments[:limit] if limit else segments
    blocks = []
    for segment in shown:
        speaker = segment.get("speaker", "unknown")
        stamp = _timestamp(segment["start"], ".")[:-4]
        blocks.append(f"**{speaker}**  `{stamp}`\n\n{segment.get('text', '').strip()}")
    if limit and len(segments) > limit:
        blocks.append(f"_...{len(segments) - limit} more turns in the downloads above._")
    return "\n\n---\n\n".join(blocks)


def to_srt(segments: Sequence[dict]) -> str:
    """Render SubRip subtitles with the speaker name inlined in each cue."""
    cues = []
    for index, segment in enumerate(segments, start=1):
        start = _timestamp(segment["start"], ",")
        end = _timestamp(segment["end"], ",")
        speaker = segment.get("speaker", "unknown")
        cues.append(f"{index}\n{start} --> {end}\n{speaker}: {segment.get('text','').strip()}")
    return "\n\n".join(cues) + "\n"


def to_vtt(segments: Sequence[dict]) -> str:
    """Render WebVTT subtitles with the speaker name as a cue voice tag."""
    cues = ["WEBVTT", ""]
    for segment in segments:
        start = _timestamp(segment["start"], ".")
        end = _timestamp(segment["end"], ".")
        speaker = segment.get("speaker", "unknown")
        cues.append(f"{start} --> {end}\n<v {speaker}>{segment.get('text','').strip()}")
        cues.append("")
    return "\n".join(cues)


def speaker_summary(segments: Sequence[dict]) -> list:
    """
    Tabulate speaking time and turn counts per speaker.

    Args:
        segments (Sequence[dict]): Speaker-attributed turns.

    Returns:
        list: Rows of ``[speaker, turns, minutes, share_of_speech]`` sorted by time.
    """
    from collections import defaultdict

    seconds = defaultdict(float)
    turns = defaultdict(int)
    words = defaultdict(int)
    for segment in segments:
        speaker = segment.get("speaker", "unknown")
        seconds[speaker] += float(segment["end"]) - float(segment["start"])
        turns[speaker] += 1
        words[speaker] += len(segment.get("text", "").split())

    total = sum(seconds.values()) or 1.0
    rows = [
        [speaker, turns[speaker], round(value / 60, 1), words[speaker], f"{value / total:.1%}"]
        for speaker, value in seconds.items()
    ]
    rows.sort(key=lambda row: row[2], reverse=True)
    return rows
