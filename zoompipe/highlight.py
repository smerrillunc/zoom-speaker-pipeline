"""
Video-based speaker diarization for Zoom-style meeting recordings.

Zoom highlights the active speaker's tile with a coloured border and prints that
participant's name in the corner of the tile.  Sampling frames, locating the
highlighted tile, cropping the name region and running OCR therefore yields a
per-second signal of speaker *identity* -- something audio diarization cannot
recover on its own.

This is a refactor of ``Data Processing/zoomDiarization.py``: the detection geometry
and thresholds are unchanged, but model loading is lazy, debug plotting is replaced
by returnable artifacts, and frame sampling reads the stream sequentially instead of
seeking per frame (a large speed-up on multi-hour recordings).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .namereader import NameReader

NO_SPEAKER = "No Speaker"


@dataclass
class FrameProbe:
    """One sampled frame, retained for inspection in the UI or a compatibility check."""

    timestamp: float
    name: str
    confidence: float
    detection: str
    annotated: Optional[np.ndarray] = None
    name_crop: Optional[np.ndarray] = None
    box: Optional[Tuple[int, int, int, int]] = None


@dataclass
class DiarizationResult:
    """Output of a video diarization pass."""

    speaker_changes: List[Tuple[float, str]] = field(default_factory=list)
    frames_sampled: int = 0
    frames_named: int = 0
    frames_grid: int = 0
    frames_overlay: int = 0
    duration: float = 0.0
    probes: List[FrameProbe] = field(default_factory=list)

    @property
    def named_fraction(self) -> float:
        """Share of sampled frames that produced a legible name."""
        return self.frames_named / self.frames_sampled if self.frames_sampled else 0.0

    @property
    def grid_fraction(self) -> float:
        """
        Share of sampled frames where a highlighted active-speaker tile was found.

        This separates true gallery view -- Zoom drawing its coloured border around
        the speaking participant -- from the screen-share fallback, where the speaker
        thumbnail is read from a fixed corner instead. Both can yield names, but only
        the former carries the highlight signal the pipeline is designed around, so a
        low value is worth knowing even when naming succeeds.
        """
        return self.frames_grid / self.frames_sampled if self.frames_sampled else 0.0

    @property
    def speakers(self) -> List[str]:
        """Distinct speaker names observed, excluding the no-speaker sentinel."""
        seen = []
        for _, name in self.speaker_changes:
            if name != NO_SPEAKER and name not in seen:
                seen.append(name)
        return seen


# Zoom draws the active-speaker highlight as a thin, strongly saturated border. The
# measured value on a 720p DC Court of Appeals recording is BGR (2, 248, 107) --
# HSV (47, 253, 248) -- so the mask demands high saturation *and* high value. A
# loose green mask matches shirts, foliage and slides; those are what made an
# earlier version of this code report "grid detected" on recordings that have no
# highlight at all.
HIGHLIGHT_LOWER = np.array([25, 150, 150])
HIGHLIGHT_UPPER = np.array([90, 255, 255])


# Saturation floors tried in order, strictest first.
#
# 150 alone was losing real highlights to video compression rather than to any
# property of the meeting. A tile border is a line one or two pixels thick, and
# 4:2:0 chroma subsampling damages thin *horizontal* lines much harder than vertical
# ones -- so on a YouTube re-encode the ring's top and bottom edges routinely fall
# under S=150 while its sides survive. What reaches the structural test below is then
# two disconnected slivers, which can never satisfy "three of four sides covered",
# and a textbook gallery view scores no-highlight. Measured against 50 hand-labelled
# meetings this cost roughly a fifth of all ring-present videos, and wrongly retired
# five whole organisations.
#
# The floors are tried in order and the first that yields a box wins, because looser
# is not uniformly better: a bigger mask also trips the interior-fill and side-coverage
# gates, and frames were observed detecting at S=80 and going dark again at S=60.
# Laddering keeps every detection the strict floor already made -- on the labelled set
# it recovered 8 of 9 missed videos with zero regressions and zero false positives
# against 36 confirmed ring-free meetings.
HIGHLIGHT_SATURATION_LADDER = (150, 100, 70)


def highlight_mask(frame: np.ndarray, s_min: Optional[int] = None) -> np.ndarray:
    """Return a binary mask of saturated green/yellow pixels.

    Args:
        frame (np.ndarray): BGR frame.
        s_min (int | None): Saturation floor. Defaults to ``HIGHLIGHT_LOWER``'s.
            Pass the value reported by :func:`find_highlight_box_scaled` to measure a
            border with the same floor that found it -- scoring a box located at a
            looser floor against the strict mask reads as near-zero coverage.
    """
    import cv2

    lower = (HIGHLIGHT_LOWER if s_min is None
             else np.array([HIGHLIGHT_LOWER[0], s_min, HIGHLIGHT_LOWER[2]]))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, HIGHLIGHT_UPPER)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _largest_outline_box(mask: np.ndarray, width: int, height: int
                         ) -> Optional[Tuple[int, int, int, int]]:
    """Largest rectangular *outline* in ``mask``, or None. See find_highlight_box."""
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 60 or h < 40:
            continue
        area = w * h
        if not (0.01 <= area / float(width * height) <= 0.75):
            continue
        if not (0.9 <= w / float(h) <= 2.8):
            continue

        # Zoom draws the ring at a fixed pixel width -- about 4px in a 720p recording --
        # not as a proportion of the tile. Deriving thickness purely as 2% of the tile
        # holds for a crowded gallery, where tiles are small, but overshoots badly when
        # a meeting has only two or three participants: on a 358px tile it assumes 7px,
        # so the perimeter strips sampled below are mostly tile content rather than ring
        # and a textbook highlight scores 0.58 against a 0.60 threshold. That rejects
        # the frame outright. Capping the estimate keeps the crowded-gallery behaviour
        # (2% is already <= 5px for tiles up to 250px) while making sparse layouts work.
        thickness = max(2, min(5, int(round(min(w, h) * 0.02))))
        window = mask[y : y + h, x : x + w] > 0
        interior = window[thickness * 2 : h - thickness * 2,
                          thickness * 2 : w - thickness * 2]
        if interior.size and interior.mean() > 0.06:
            continue

        sides = sorted([
            window[:thickness, :].mean(), window[h - thickness :, :].mean(),
            window[:, :thickness].mean(), window[:, w - thickness :].mean(),
        ])
        # Three of four sides must be covered: a contour's bounding box can overshoot
        # the tile by a few pixels, emptying one side even on a textbook highlight.
        if sides[1] < 0.5 or float(np.mean(sides[1:])) < 0.6:
            continue

        if area > best_area:
            best_area, best = area, (x, y, w, h)

    return best


def find_highlight_box_scaled(
    frame: np.ndarray,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[int]]:
    """:func:`find_highlight_box`, also reporting which saturation floor found the box.

    Returns:
        tuple: ``((x, y, w, h), s_min)``, or ``(None, None)``. Pass ``s_min`` to
        :func:`highlight_mask` when measuring the border's colour, so the ring is
        scored with the floor that located it.
    """
    height, width = frame.shape[:2]
    for s_min in HIGHLIGHT_SATURATION_LADDER:
        mask = highlight_mask(frame, s_min)
        if not mask.any():
            continue
        box = _largest_outline_box(mask, width, height)
        if box is not None:
            return box, s_min
    return None, None


def find_highlight_box(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate the Zoom active-speaker tile by its highlighted border.

    The test is structural, not merely chromatic: the matched pixels must form a
    thin rectangular *outline* -- dense along the perimeter, empty inside -- with a
    plausible tile aspect ratio. A filled green region (a shirt, a lawn, a slide)
    fails the interior test, and a coloured banner fails the aspect test.

    Saturation floors are tried strictest first; see ``HIGHLIGHT_SATURATION_LADDER``.

    Args:
        frame (np.ndarray): BGR frame.

    Returns:
        tuple | None: ``(x, y, w, h)`` of the highlighted tile, or ``None`` when the
        frame carries no active-speaker highlight.
    """
    return find_highlight_box_scaled(frame)[0]


def crop_black_border(frame: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop letterboxing from a frame.

    Args:
        frame (np.ndarray): Input image, colour or grayscale.
        threshold (int): Pixel values at or below this are considered black.

    Returns:
        np.ndarray: The frame with uniform black borders removed.  The input is
        returned unchanged when the frame is entirely black.
    """
    if frame.ndim == 3:
        mask = np.any(frame > threshold, axis=2)
    else:
        mask = frame > threshold

    coords = np.argwhere(mask)
    if coords.size == 0:
        return frame

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return frame[y0:y1, x0:x1]


def find_speaker_tile(
    frame: np.ndarray, annotate: bool = False
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int], str, Optional[np.ndarray]]:
    """
    Locate the active-speaker tile in a Zoom frame.

    Two layouts are handled:

    1. **Participant grid** -- the active speaker's tile carries a highlighted
       border, located by :func:`find_highlight_box`.
    2. **Screen share** -- no border exists, so the speaker thumbnail is taken from
       its fixed position in the upper-right corner (18% of frame width and height).

    Args:
        frame (np.ndarray): BGR frame.
        annotate (bool): Also return a copy of the frame with the tile outlined.

    Returns:
        tuple: ``(annotated_frame, (x, y, w, h), detection_mode, tile_crop)``.
        ``annotated_frame`` is ``None`` unless ``annotate`` is set.
    """
    import cv2

    frame = crop_black_border(frame)
    height, width = frame.shape[:2]

    best_rect = find_highlight_box(frame)

    if best_rect is not None:
        box = best_rect
        detection = "grid"
        colour = (0, 255, 0)
    else:
        overlay_w = int(width * 0.18)
        overlay_h = int(height * 0.18)
        box = (width - overlay_w, 0, overlay_w, overlay_h)
        detection = "overlay"
        colour = (255, 0, 0)

    x, y, w, h = box
    tile = frame[y : y + h, x : x + w]

    annotated = None
    if annotate:
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 3)

    return annotated, box, detection, tile


def crop_name_only(
    tile: np.ndarray,
    height_ratio: float = 0.22,
    width_ratio: float = 0.5,
    bottom_padding: int = 2,
) -> np.ndarray:
    """
    Crop the name-label region from an active-speaker tile.

    Zoom prints the participant name in the lower-left of the tile.  The default
    ratios take the bottom 22% and left half, trimming a couple of pixels off the
    bottom so the highlight border does not bleed into the crop.

    Args:
        tile (np.ndarray): The active-speaker tile.
        height_ratio (float): Fraction of tile height occupied by the label strip.
        width_ratio (float): Fraction of tile width occupied by the label strip.
        bottom_padding (int): Pixels trimmed from the bottom edge.

    Returns:
        np.ndarray: The name-label crop.
    """
    height, width = tile.shape[:2]
    name_h = int(height * height_ratio)
    name_w = int(width * width_ratio)
    return tile[height - name_h : height - bottom_padding, 0:name_w]


class ZoomSpeakerDiarization:
    """
    Turns a Zoom recording into a list of ``(timestamp, speaker_name)`` changes.

    Args:
        video_path (str): Path to the recording.
        reader (NameReader | None): OCR reader; one is constructed if omitted.
        height_ratio (float): Name-crop height ratio, see :func:`crop_name_only`.
        width_ratio (float): Name-crop width ratio in grid layouts.
        overlay_width_ratio (float): Name-crop width ratio in screen-share layouts,
            where the speaker thumbnail is small and a half-width crop truncates
            longer names.
        truncation_extent (float): If OCR text reaches this fraction of the crop
            width, the crop is widened to the full tile and re-read. Catches long
            names cut short in either layout.
    """

    def __init__(
        self,
        video_path: str,
        reader: Optional[NameReader] = None,
        height_ratio: float = 0.22,
        width_ratio: float = 0.5,
        overlay_width_ratio: float = 0.9,
        truncation_extent: float = 0.92,
    ):
        import cv2

        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps else 0.0
        self.reader = reader or NameReader()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio
        self.overlay_width_ratio = overlay_width_ratio
        self.truncation_extent = truncation_extent
        self.speaker_changes: List[Tuple[float, str]] = []

    def close(self) -> None:
        """Release the underlying video capture."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def read_frame_name(
        self, frame: np.ndarray, annotate: bool = False
    ) -> Tuple[str, float, str, Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detect the active-speaker tile in one frame and OCR its name label.

        Returns:
            tuple: ``(name, confidence, detection_mode, annotated_frame, name_crop,
            box)``.
        """
        annotated, box, detection, tile = find_speaker_tile(frame, annotate=annotate)
        if tile is None or tile.size == 0:
            return NO_SPEAKER, 0.0, detection, annotated, None, box

        width_ratio = (
            self.overlay_width_ratio if detection == "overlay" else self.width_ratio
        )
        crop = crop_name_only(tile, self.height_ratio, width_ratio)
        name, confidence, extent = self.reader.read_detailed(crop)

        # A label running to the right edge of the crop is probably cut short
        # ("Mayor Carol Pritch" for "Mayor Carol Pritchett"), so widen and re-read.
        if name and extent >= self.truncation_extent and width_ratio < 1.0:
            wider = crop_name_only(tile, self.height_ratio, 1.0)
            wider_name, wider_confidence, _ = self.reader.read_detailed(wider)
            if len(wider_name) > len(name) and wider_confidence >= confidence - 0.1:
                return wider_name, wider_confidence, detection, annotated, wider, box

        return name or NO_SPEAKER, confidence, detection, annotated, crop, box

    def process_video(
        self,
        sampling_rate: float = 1.0,
        max_seconds: Optional[float] = None,
        start_seconds: float = 0.0,
        progress: Optional[Callable[[float, str], None]] = None,
        probe_every: int = 0,
    ) -> DiarizationResult:
        """
        Sample the recording and record every change of on-screen speaker.

        Args:
            sampling_rate (float): Frames inspected per second of video.
            max_seconds (float | None): Stop after this much video; ``None`` for all.
            start_seconds (float): Skip this much video before sampling.
            progress (callable | None): Called as ``progress(fraction, message)``.
            probe_every (int): Retain an annotated frame every N samples for display.
                ``0`` disables retention.

        Returns:
            DiarizationResult: Speaker changes plus sampling statistics.
        """
        import cv2

        frame_interval = max(1, int(round(self.fps / max(sampling_rate, 1e-6))))
        start_frame = int(start_seconds * self.fps)
        end_frame = self.total_frames
        if max_seconds is not None:
            end_frame = min(end_frame, start_frame + int(max_seconds * self.fps))

        if start_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        result = DiarizationResult(duration=self.duration)
        last_speaker = None
        current_frame = start_frame
        sample_index = 0

        while current_frame < end_frame:
            ok, frame = self.capture.read()
            if not ok:
                break

            timestamp = current_frame / self.fps
            annotate = probe_every and sample_index % probe_every == 0
            try:
                name, confidence, detection, annotated, crop, box = self.read_frame_name(
                    frame, annotate=bool(annotate)
                )
            except Exception:
                name, confidence, detection, annotated, crop, box = (
                    NO_SPEAKER,
                    0.0,
                    "error",
                    None,
                    None,
                    None,
                )

            result.frames_sampled += 1
            if name != NO_SPEAKER:
                result.frames_named += 1
            if detection == "grid":
                result.frames_grid += 1
            elif detection == "overlay":
                result.frames_overlay += 1
            if annotate:
                result.probes.append(
                    FrameProbe(timestamp, name, confidence, detection, annotated,
                               crop, box)
                )

            if name != last_speaker:
                result.speaker_changes.append((timestamp, name))
                last_speaker = name

            if progress is not None and sample_index % 10 == 0:
                span = max(end_frame - start_frame, 1)
                progress(
                    (current_frame - start_frame) / span,
                    f"{timestamp / 60:.1f} min - {name}",
                )

            # Skip ahead by grabbing (decode-free) rather than seeking per frame.
            for _ in range(frame_interval - 1):
                if not self.capture.grab():
                    current_frame = end_frame
                    break
                current_frame += 1
            current_frame += 1
            sample_index += 1

        self.speaker_changes = result.speaker_changes
        return result


def probe_video(
    video_path: str,
    samples: int = 6,
    reader: Optional[NameReader] = None,
    start_fraction: float = 0.15,
    end_fraction: float = 0.85,
) -> List[FrameProbe]:
    """
    Sample a handful of frames spread through a recording to judge OCR compatibility.

    This is the cheap pre-flight check used both by the app (to warn a user their
    upload is not a Zoom-style grid) and by ``Data Processing/check_zoom_video.py``
    when screening candidate meeting sources.

    Args:
        video_path (str): Path to the recording.
        samples (int): Number of frames to inspect.
        reader (NameReader | None): OCR reader to reuse across calls.
        start_fraction (float): Where in the video to begin sampling.
        end_fraction (float): Where to stop, avoiding title cards and sign-offs.

    Returns:
        list[FrameProbe]: One probe per sampled frame, with annotated imagery.
    """
    import cv2

    diarizer = ZoomSpeakerDiarization(video_path, reader=reader)
    try:
        probes: List[FrameProbe] = []
        total = diarizer.total_frames
        if total <= 0:
            return probes

        positions = np.linspace(
            int(total * start_fraction), int(total * end_fraction), samples
        ).astype(int)

        for position in positions:
            diarizer.capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = diarizer.capture.read()
            if not ok:
                continue
            try:
                name, confidence, detection, annotated, crop, box = diarizer.read_frame_name(
                    frame, annotate=True
                )
            except Exception:
                continue
            probes.append(
                FrameProbe(
                    timestamp=position / diarizer.fps,
                    name=name,
                    confidence=confidence,
                    detection=detection,
                    annotated=annotated,
                    name_crop=crop,
                    box=box,
                )
            )
        return probes
    finally:
        diarizer.close()


def compatibility_report(probes: Sequence[FrameProbe]) -> dict:
    """
    Judge whether a recording carries a usable active-speaker identity signal.

    The decisive question is not "can OCR read a name" but "can we tell who is
    speaking". Three things must hold, and each rules out a real failure mode seen
    in public meeting video:

    * **A highlight is present.** Zoom must be drawing the active-speaker border.
      A gallery with no highlight (the speaker named only in a "X is talking"
      banner) reads names perfectly and attributes every one of them to the wrong
      person.
    * **The highlight moves.** If it never leaves one tile across the sampled span,
      either one person spoke throughout or the detection is latching onto
      something static.
    * **More than one name is read.** A recording published from a single account
      shows that account's name on every tile.

    Args:
        probes (Sequence[FrameProbe]): Output of :func:`probe_video`.

    Returns:
        dict: ``highlight_fraction``, ``highlight_positions``, ``named_fraction``,
        ``mean_confidence``, ``distinct_names``, ``names`` and a ``verdict``.

    Verdicts:
        ``good``
            Gallery view with a moving highlight and several names read.
        ``static-highlight``
            A highlight is found but never moves between tiles.
        ``single-name``
            One name everywhere; usually the publishing account.
        ``no-highlight``
            No active-speaker border. Unusable, whatever OCR reports.
        ``poor``
            Highlight present but names are not legible.
    """
    empty = {
        "highlight_fraction": 0.0,
        "highlight_positions": 0,
        "named_fraction": 0.0,
        "mean_confidence": 0.0,
        "distinct_names": 0,
        "names": [],
        "verdict": "no-highlight",
    }
    if not probes:
        return empty

    highlighted = [p for p in probes if p.detection == "grid" and p.box]
    highlight_fraction = len(highlighted) / len(probes)

    # Count distinct tile positions the highlight occupied. Boxes are snapped to a
    # coarse grid so that a few pixels of jitter on one tile is not read as motion.
    positions = set()
    for probe in highlighted:
        x, y, w, h = probe.box
        positions.add((round((x + w / 2) / 80), round((y + h / 2) / 80)))

    named = [p for p in probes if p.name != NO_SPEAKER]
    named_fraction = len(named) / len(probes)
    mean_confidence = float(np.mean([p.confidence for p in named])) if named else 0.0

    from .identity import parse_label

    parsed = [parse_label(p.name) for p in named]
    distinct = {label.key for label in parsed if label.kind == "person"}
    names = sorted({p.name for p in named})

    if highlight_fraction < 0.25:
        verdict = "no-highlight"
    elif len(positions) < 2:
        verdict = "static-highlight"
    elif named_fraction < 0.3:
        verdict = "poor"
    elif len(distinct) < 2:
        verdict = "single-name"
    elif mean_confidence >= 0.7:
        verdict = "good"
    else:
        verdict = "marginal"

    return {
        "highlight_fraction": highlight_fraction,
        "highlight_positions": len(positions),
        "named_fraction": named_fraction,
        "mean_confidence": mean_confidence,
        "distinct_names": len(distinct),
        "names": names,
        "verdict": verdict,
    }
