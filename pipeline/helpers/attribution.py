"""
Attribution: which cleaned on-screen identity said each transcribed segment.

``attribute_segments`` combines the two signals the pipeline has for every stretch of
speech: the pyannote **voice cluster** (from ``transcribe.py``) and the **on-screen
identity** (from ``clean_ocr.py``). ``merge_turns`` then joins consecutive segments of
one speaker into turns and records, per turn, how the name was decided.
"""

import bisect
from collections import Counter, defaultdict
from typing import List, Sequence, Tuple

# Zoom moves the highlight about this long after a new voice starts. Measured on 2,224
# meetings as the median offset between on-screen changes and voice-cluster changes.
HIGHLIGHT_LAG_SECONDS = 0.6


def attribute_segments(
    segments: Sequence[dict],
    changes: Sequence[Tuple[float, str]],
    lag: float = HIGHLIGHT_LAG_SECONDS,
    pure_threshold: float = 0.8,
    screen_threshold: float = 0.7,
    minor_share: float = 0.1,
) -> List[dict]:
    """
    Name every ASR segment, combining the voice cluster with the on-screen tile.

    Naming a whole voice cluster after the tile it overlaps most (the earlier approach)
    is robust when a cluster is one person, and wrong for every segment of the minority
    speaker when pyannote has put two people in one cluster. Here:

    * The screen signal is shifted back by ``lag`` seconds, since Zoom moves the
      highlight after the voice starts.
    * A cluster whose top name holds >= ``pure_threshold`` of its overlap is named as a
      whole, as before.
    * In a mixed cluster, a segment takes the name that covers >= ``screen_threshold``
      of it on screen, provided that name holds >= ``minor_share`` of the cluster (so
      the voice evidence still agrees it is one of the cluster's speakers); otherwise it
      takes the cluster's name.

    Each returned segment carries ``speaker`` plus ``source`` (``cluster`` or
    ``screen``) and ``screen_share`` (how much of it the chosen name covers on screen).

    Example:
        >>> segs = [{"start": 0, "end": 4, "speaker": "S0", "text": "a"},
        ...         {"start": 4, "end": 8, "speaker": "S0", "text": "b"},
        ...         {"start": 8, "end": 12, "speaker": "S0", "text": "c"}]
        >>> out = attribute_segments(segs, [(0.0, "ann"), (8.0, "bob")], lag=0.0)
        >>> [(s["speaker"], s["source"]) for s in out]
        [('ann', 'cluster'), ('ann', 'cluster'), ('bob', 'screen')]
    """
    intervals = []
    for i, (t, name) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else float("inf")
        intervals.append((t - lag, end - lag, name))
    starts = [iv[0] for iv in intervals]

    def screen(seg_start, seg_end):
        tally = defaultdict(float)
        i = max(0, bisect.bisect_right(starts, seg_start) - 1)
        while i < len(intervals) and intervals[i][0] < seg_end:
            a, b, name = intervals[i]
            overlap = min(seg_end, b) - max(seg_start, a)
            if overlap > 0:
                tally[name] += overlap
            i += 1
        return tally

    per_segment = [screen(s["start"], s["end"]) for s in segments]
    cluster_tally = defaultdict(lambda: defaultdict(float))
    for seg, tally in zip(segments, per_segment):
        for name, v in tally.items():
            cluster_tally[seg.get("speaker")][name] += v

    cluster_name, cluster_share, cluster_dist = {}, {}, {}
    for label, tally in cluster_tally.items():
        total = sum(tally.values())
        # Plurality, "Other" included: a cluster heard mostly while a room, device or
        # shared-screen tile is up stays unattributed rather than borrowing a name.
        best = max(sorted(tally), key=lambda n: tally[n])
        cluster_name[label] = best
        cluster_share[label] = tally[best] / total if total else 0.0
        cluster_dist[label] = {n: v / total for n, v in tally.items()} if total else {}

    out = []
    for seg, tally in zip(segments, per_segment):
        label = seg.get("speaker")
        name = cluster_name.get(label, "Other")
        source = "cluster"
        length = max(seg["end"] - seg["start"], 1e-6)
        if cluster_share.get(label, 1.0) < pure_threshold and tally:
            top = max(sorted(tally), key=lambda n: tally[n])
            if top != "Other" and top != name and tally[top] / length >= screen_threshold \
                    and cluster_dist[label].get(top, 0.0) >= minor_share:
                name, source = top, "screen"
        row = dict(seg)
        row["cluster"] = label
        row["speaker"] = name
        row["source"] = source
        row["screen_share"] = round(tally.get(name, 0.0) / length, 3)
        out.append(row)
    return out


def merge_turns(segments: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Join consecutive segments with the same speaker into turns.

    Returns ``(turns, provenance)``: ``turns`` in the release format
    ``{start, end, speaker, text}``, and per turn how it was named -- ``source``
    (``screen`` if any segment was named from the screen, else ``cluster``),
    ``screen_share`` (share of the turn the chosen name covered on screen), and the
    voice ``clusters`` it spans.

    Example:
        >>> turns, how = merge_turns([
        ...     {"start": 0, "end": 2, "speaker": "ann", "text": "Hello", "source": "cluster",
        ...      "screen_share": 1.0, "cluster": "S0"},
        ...     {"start": 2, "end": 4, "speaker": "ann", "text": "world", "source": "cluster",
        ...      "screen_share": 0.5, "cluster": "S0"}])
        >>> turns, how[0]["screen_share"]
        ([{'start': 0, 'end': 4, 'speaker': 'ann', 'text': 'Hello world'}], 0.75)
    """
    turns: List[dict] = []
    provenance: List[dict] = []
    for seg in segments:
        seconds = seg["end"] - seg["start"]
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turn, how = turns[-1], provenance[-1]
            turn["end"] = seg["end"]
            turn["text"] = f"{turn['text']} {seg['text'].strip()}".strip()
        else:
            turns.append({"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"],
                          "text": seg["text"].strip()})
            how = {"segments": 0, "screen_segments": 0, "_screen_seconds": 0.0, "clusters": Counter()}
            provenance.append(how)
        how["segments"] += 1
        how["screen_segments"] += seg.get("source") == "screen"
        how["_screen_seconds"] += seg.get("screen_share", 0.0) * seconds
        how["clusters"][seg.get("cluster") or ""] += 1
    for turn, how in zip(turns, provenance):
        length = max(turn["end"] - turn["start"], 1e-6)
        how["screen_share"] = round(how.pop("_screen_seconds") / length, 3)
        how["source"] = "screen" if how["screen_segments"] else "cluster"
        how["clusters"] = dict(how["clusters"])
    return turns, provenance


NO_READ = "No Speaker"


def label_turns(
    turns: Sequence[dict],
    raw_changes: Sequence[Sequence],
    raw_to_speaker: dict,
    display_names: dict,
    duration: float,
    lag: float = HIGHLIGHT_LAG_SECONDS,
) -> List[dict]:
    """
    Add ``speaker_name`` and ``ocr_label`` to each turn.

    ``speaker`` is the linked identity key and ``speaker_name`` its display name.
    ``ocr_label`` is the label as OCR read it: of the raw reads that resolved to this
    turn's speaker, the one on screen longest during the turn. When none was on screen
    during the turn (it was named from its voice cluster), the speaker's most-seen read
    in the meeting is used. ``Other`` turns keep the room, device or organisation text
    that was on screen, and ``None`` when nothing was read at all.

    Example:
        >>> raw = [[0, "Pat Benavides - CC.."], [10, "Pat Benaivides"], [20, "Council Chambers"]]
        >>> to = {"Pat Benavides - CC..": "patbenavides", "Pat Benaivides": "patbenavides",
        ...       "Council Chambers": "Other"}
        >>> out = label_turns([{"start": 1, "end": 8, "speaker": "patbenavides", "text": "a"},
        ...                    {"start": 21, "end": 25, "speaker": "Other", "text": "b"}],
        ...                   raw, to, {"patbenavides": "Pat Benavides"}, 30, lag=0.0)
        >>> [(t["speaker_name"], t["ocr_label"]) for t in out]
        [('Pat Benavides', 'Pat Benavides - CC..'), (None, 'Council Chambers')]
    """
    rows = sorted((float(t), s) for t, s in raw_changes)
    intervals = [(t - lag, (rows[i + 1][0] if i + 1 < len(rows) else duration) - lag, s)
                 for i, (t, s) in enumerate(rows)]
    starts = [iv[0] for iv in intervals]

    most_seen = defaultdict(Counter)
    for a, b, raw in intervals:
        if raw != NO_READ:
            most_seen[raw_to_speaker.get(raw, "Other")][raw] += b - a

    out = []
    for turn in turns:
        tally = Counter()
        i = max(0, bisect.bisect_right(starts, turn["start"]) - 1)
        while i < len(intervals) and intervals[i][0] < turn["end"]:
            a, b, raw = intervals[i]
            overlap = min(turn["end"], b) - max(turn["start"], a)
            if overlap > 0 and raw != NO_READ and raw_to_speaker.get(raw, "Other") == turn["speaker"]:
                tally[raw] += overlap
            i += 1
        pool = tally
        if not pool and turn["speaker"] != "Other":
            pool = most_seen.get(turn["speaker"], Counter())
        label = max(sorted(pool), key=lambda r: pool[r]) if pool else None
        name = None if turn["speaker"] == "Other" else display_names.get(turn["speaker"], turn["speaker"])
        out.append({"start": turn["start"], "end": turn["end"], "speaker": turn["speaker"],
                    "speaker_name": name, "ocr_label": label, "text": turn["text"]})
    return out
