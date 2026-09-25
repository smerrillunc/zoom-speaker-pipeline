"""Small file-handling helpers shared by the step scripts."""

import json
import os
import re
from typing import Iterable, List

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")

# yt-dlp leaves "<id>.f<format>.<ext>" stream fragments and a "<id>.temp.<ext>" merge
# intermediate behind. Both are openable video files; the merge intermediate would be
# processed successfully as a duplicate of the real meeting.
_FRAGMENT = re.compile(r"\.(?:f\d+|temp)\.[^.]+$")


def collect(paths: Iterable[str], suffixes=VIDEO_SUFFIXES) -> List[str]:
    """Expand files and directories into matching files, in a stable order."""
    found = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(suffixes) and not _FRAGMENT.search(name):
                    found.append(os.path.join(path, name))
        elif path.lower().endswith(suffixes) and not _FRAGMENT.search(path):
            found.append(path)
    return found


def shard(items: List[str], spec: str) -> List[str]:
    """
    Keep the items of shard ``"i/n"`` (index mod n == i), so a job array can split
    a folder without a work queue.

    Example:
        >>> shard(["a", "b", "c", "d"], "1/2")
        ['b', 'd']
    """
    index, count = (int(part) for part in spec.split("/"))
    return [item for position, item in enumerate(items) if position % count == index]


_OUTPUT_SUFFIXES = (".ocr.json", ".video.json", ".asr.json", ".speakers.json",
                    ".attribution.json", ".json")


def stem(path: str) -> str:
    """
    Meeting id from a video or step-output path.

    Example:
        >>> stem("council.2021-05-04.mp4"), stem("out/council.2021-05-04.ocr.json")
        ('council.2021-05-04', 'council.2021-05-04')
    """
    name = os.path.basename(path)
    for suffix in _OUTPUT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def write_json(path: str, payload, indent=None) -> None:
    """Write JSON through a temporary file, so an interrupted run never leaves a
    half-written result that a rerun would mistake for finished work."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
    os.replace(temporary, path)


def read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
