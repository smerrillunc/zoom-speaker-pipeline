"""
Named-speaker transcripts from Zoom-style meeting recordings.

Zoom draws a highlight around the active speaker's tile and prints their name in it.
Reading that name once a second gives an identity signal audio diarization cannot
recover; this package aligns it onto a Whisper transcript.

The pipeline is the four step scripts next to this folder (``step1_ocr.py``,
``step2_transcribe.py``, ``step3_clean_ocr.py``, ``step4_merge.py``). The modules here
are what they call:

``highlight``    find the highlighted tile, crop its name strip (OpenCV)
``namereader``   read the name strip (PaddleOCR, optional EDSR super-resolution)
``speech``       audio extraction, Whisper transcription, pyannote voice clusters
``identity``     parse raw OCR labels and link them into person identities
``attribution``  name each transcribed segment from voice cluster + screen
``render``       plain-text / SRT / VTT renderings of a transcript
``files``        video discovery, sharding, atomic JSON writes

``highlight``/``namereader`` (Paddle) and ``speech`` (CTranslate2) must not be loaded
in one process: their OpenMP runtimes conflict.
"""

__version__ = "1.0.0"
