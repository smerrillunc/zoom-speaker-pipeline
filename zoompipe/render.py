"""Output formats for a speaker-attributed transcript."""

from typing import Sequence


def _timestamp(seconds: float, separator: str = ",") -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (WebVTT)."""
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


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
