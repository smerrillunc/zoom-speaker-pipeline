"""
OCR of Zoom name labels, with optional super-resolution for low-confidence crops.

Name labels in a Zoom grid are small -- often 12-16px tall in a 720p recording --
which is where off-the-shelf OCR fails.  Following the paper, a crop whose OCR
confidence falls below a threshold is upscaled 4x with EDSR and re-read.

The PaddleOCR wrapper here is deliberately version-tolerant: PaddleOCR 2.x returns
nested ``[[box, (text, score)], ...]`` lists, while 3.x returns ``OCRResult`` mappings
carrying ``rec_texts``/``rec_scores``.  Both are normalized to the same shape.
"""

import os
import urllib.request
from typing import List, Optional, Tuple

import numpy as np

EDSR_URL = (
    "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
)


def _paddle_results_to_boxes(result) -> List[Tuple[str, float, float, float]]:
    """
    Normalize a PaddleOCR result into ``(text, score, x_left, x_right)`` tuples.

    Args:
        result: Whatever ``PaddleOCR.ocr`` returned (2.x or 3.x shape).

    Returns:
        list[tuple]: One tuple per detected text box. ``x_left`` orders boxes
        left-to-right, since a name is frequently split across boxes; ``x_right``
        reveals a label that runs off the edge of the crop.
    """
    boxes: List[Tuple[str, float, float, float]] = []
    if not result:
        return boxes

    page = result[0]
    if page is None:
        return boxes

    # PaddleOCR 3.x: a mapping with parallel text/score/polygon lists.
    if hasattr(page, "get") and page.get("rec_texts") is not None:
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys")
        if polys is None:
            polys = page.get("dt_polys")
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            x_left = x_right = 0.0
            if polys is not None and index < len(polys):
                xs = np.asarray(polys[index])[:, 0]
                x_left, x_right = float(xs.min()), float(xs.max())
            boxes.append((str(text), score, x_left, x_right))
        return boxes

    # PaddleOCR 2.x: [[polygon, (text, score)], ...]
    try:
        for entry in page:
            polygon, (text, score) = entry[0], entry[1]
            xs = np.asarray(polygon)[:, 0]
            boxes.append((str(text), float(score), float(xs.min()), float(xs.max())))
    except (TypeError, ValueError, IndexError):
        return []

    return boxes


class NameReader:
    """
    Reads a speaker name out of a cropped Zoom name-label image.

    Args:
        lang (str): PaddleOCR language code.
        confidence_threshold (float): Below this mean confidence, the crop is
            upscaled with EDSR and re-read.
        noise_floor (float): Below this confidence the crop is treated as containing
            no text at all, so super-resolution is skipped as pointless.
        edsr_path (str | None): Path to ``EDSR_x4.pb``.  When ``None`` the model is
            looked up in ``EDSR_MODEL_PATH``/``models/`` and downloaded on first use.
        use_super_resolution (bool): Disable to run OCR only on the raw crop.
    """

    def __init__(
        self,
        lang: str = "en",
        confidence_threshold: float = 0.8,
        noise_floor: float = 0.1,
        edsr_path: Optional[str] = None,
        use_super_resolution: bool = True,
    ):
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self.noise_floor = noise_floor
        self.use_super_resolution = use_super_resolution
        self._edsr_path = edsr_path or os.environ.get("EDSR_MODEL_PATH")
        self._reader = None
        self._sr = None

    # -- lazy model loading -------------------------------------------------

    @property
    def reader(self):
        """The PaddleOCR instance, constructed on first use."""
        if self._reader is None:
            from paddleocr import PaddleOCR

            try:
                # PaddleOCR 3.x: document pre-processing is off by default here
                # because a name label is a single cropped line, not a page.
                self._reader = PaddleOCR(
                    lang=self.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except (TypeError, ValueError):
                # PaddleOCR 2.x keyword set.
                self._reader = PaddleOCR(use_angle_cls=True, lang=self.lang)
        return self._reader

    def _resolve_edsr(self) -> Optional[str]:
        """Return a local path to ``EDSR_x4.pb``, downloading it if necessary."""
        if self._edsr_path and os.path.exists(self._edsr_path):
            return self._edsr_path

        cache_dir = os.environ.get("EDSR_CACHE_DIR", os.path.join(os.getcwd(), "models"))
        target = os.path.join(cache_dir, "EDSR_x4.pb")
        if os.path.exists(target):
            self._edsr_path = target
            return target

        try:
            os.makedirs(cache_dir, exist_ok=True)
            urllib.request.urlretrieve(EDSR_URL, target)
            self._edsr_path = target
            return target
        except Exception:
            return None

    @property
    def super_resolver(self):
        """
        The EDSR x4 upsampler, or ``None`` when unavailable.

        ``cv2.dnn_superres`` ships only with ``opencv-contrib-python``; when it or the
        model file is missing, callers fall back to bicubic interpolation.
        """
        if not self.use_super_resolution:
            return None
        if self._sr is None:
            import cv2

            if not hasattr(cv2, "dnn_superres"):
                self.use_super_resolution = False
                return None
            path = self._resolve_edsr()
            if not path:
                self.use_super_resolution = False
                return None
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(path)
            sr.setModel("edsr", 4)
            self._sr = sr
        return self._sr

    # -- reading ------------------------------------------------------------

    def _read_once(self, image: np.ndarray) -> Tuple[str, float]:
        """Run OCR on one image and join its boxes into a single left-to-right name."""
        text, confidence, _ = self._read_with_extent(image)
        return text, confidence

    def _read_with_extent(self, image: np.ndarray) -> Tuple[str, float, float]:
        """
        Run OCR and also report how far right the text reaches.

        Returns:
            tuple: ``(text, confidence, right_extent)`` where ``right_extent`` is the
            rightmost text pixel as a fraction of image width. A value close to 1.0
            means the label runs off the edge of the crop and is probably cut short.
        """
        try:
            result = self.reader.ocr(image)
        except Exception:
            return "", 0.0, 0.0

        boxes = _paddle_results_to_boxes(result)
        if not boxes:
            return "", 0.0, 0.0

        boxes.sort(key=lambda box: box[2])
        text = " ".join(box[0].strip() for box in boxes if box[0].strip())
        confidence = float(np.mean([box[1] for box in boxes]))
        width = image.shape[1] or 1
        right_extent = max(box[3] for box in boxes) / width
        return text.strip(), confidence, right_extent

    def read(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Read a name from a cropped label, escalating to super-resolution if needed.

        Args:
            crop (np.ndarray): BGR image of the name-label region.

        Returns:
            tuple: ``(name, confidence)``.  ``name`` is ``""`` when nothing legible
            was found, which callers record as "No Speaker".
        """
        text, confidence, _ = self.read_detailed(crop)
        return text, confidence

    def read_detailed(self, crop: np.ndarray) -> Tuple[str, float, float]:
        """
        Read a name and report whether it reached the right edge of the crop.

        Returns:
            tuple: ``(name, confidence, right_extent)``. See
            :meth:`_read_with_extent` for the meaning of ``right_extent``.
        """
        if crop is None or crop.size == 0:
            return "", 0.0, 0.0

        text, confidence, extent = self._read_with_extent(crop)
        if confidence >= self.confidence_threshold or confidence <= self.noise_floor:
            return text, confidence, extent

        upscaled = self.upscale(crop)
        if upscaled is None:
            return text, confidence, extent

        upscaled_text, upscaled_confidence, upscaled_extent = self._read_with_extent(upscaled)
        if upscaled_confidence > confidence:
            return upscaled_text, upscaled_confidence, upscaled_extent
        return text, confidence, extent

    def upscale(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Upscale a crop 4x with EDSR, falling back to bicubic interpolation."""
        import cv2

        resolver = self.super_resolver
        if resolver is not None:
            try:
                return resolver.upsample(crop)
            except Exception:
                pass
        try:
            return cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        except Exception:
            return None
