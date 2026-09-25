"""
End-to-end tests of steps 3 and 4 on two synthetic meetings (no video, no models).

Run with ``pytest`` or directly: ``python tests/test_steps.py``. The fixture is a
fictional council; every name is invented.
"""

import doctest
import filecmp
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import clean_ocr  # noqa: E402
import merge  # noqa: E402

# Meeting A: full names, a role prefix, a device tile, a room tile.
OCR_A = [
    [0.0, "Mayor Dana Whitfield"],
    [30.0, "Appellant - Omar Castillo"],
    [60.0, "Mayor Dana Whitfield"],
    [90.0, "Bea's iPad (2)"],
    [100.0, "Riverton Council Chambers"],
    [120.0, "Mayor Dana Whitfield"],
]
# Meeting B: the same people, read badly -- an OCR slip, a clipped label, the
# organisation's own tile, and a one-frame dropout inside one person's run.
OCR_B = [
    [0.0, "Dana Whitfeld"],
    [30.0, "Omar Casti..."],
    [60.0, "No Speaker"],
    [61.0, "Omar Casti..."],
    [90.0, "Riverton City"],
    [120.0, "Dana Whitfeld"],
]


def segments(spec):
    return [{"start": s, "end": e, "text": t, "speaker": c} for s, e, t, c in spec]


# Meeting A: one clean voice per person. Meeting B: pyannote merged Dana and Omar into
# one cluster (S0), so step 4 has to split it by the screen.
ASR_A = segments([(1, 29, "Good evening, this meeting is called to order.", "S0"),
                  (31, 59, "Thank you, I represent the appellant.", "S1"),
                  (61, 89, "Please continue.", "S0"),
                  (101, 119, "Staff report follows.", "S2"),
                  (121, 150, "Motion carries.", "S0")])
ASR_B = segments([(1, 29, "Next item on the agenda.", "S0"),
                  (31, 59, "We ask the council to reconsider.", "S0"),
                  (91, 119, "The clerk has the documents.", "S2"),
                  (121, 150, "Adjourned.", "S0")])


def _write_fixture(folder):
    for meeting, ocr, asr in (("meetingA", OCR_A, ASR_A), ("meetingB", OCR_B, ASR_B)):
        duration = asr[-1]["end"]
        with open(os.path.join(folder, f"{meeting}.ocr.json"), "w") as h:
            json.dump({"video": f"{meeting}.mp4", "duration": duration, "sampling_rate": 1.0,
                       "frames_sampled": int(duration), "frames_named": int(duration) - 1,
                       "frames_grid": int(duration), "frames_overlay": 0, "raw_speaker_changes": ocr}, h)
        with open(os.path.join(folder, f"{meeting}.asr.json"), "w") as h:
            json.dump({"segments": asr, "clustered": True, "error": None}, h)


def _run(folder):
    work = os.path.join(folder, "work")
    out = os.path.join(folder, "out")
    assert clean_ocr.main([folder, "--out", work, "--context", "Riverton City Council"]) == 0
    assert merge.main(["--asr", folder, "--speakers", work, "--out", out, "--formats", "json,txt,srt"]) == 0
    return work, out


def _load(path):
    with open(path) as h:
        return json.load(h)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as folder:
        _write_fixture(folder)
        work, out = _run(folder)
        registry = _load(os.path.join(work, "registry.json"))
        people = {r["speaker"]: r for r in registry["identities"]}

        # Cross-meeting linking: the OCR slip and the clipped label joined the full names.
        assert set(people) == {"danawhitfield", "omarcastillo"}, sorted(people)
        dana_rules = {v["key"]: v.get("rule") for v in people["danawhitfield"]["variants"]}
        assert dana_rules == {"danawhitfield": None, "danawhitfeld": "ocr_variant"}
        # The clipped read joins the full name, and never names the person.
        omar = {v["key"] for v in people["omarcastillo"]["variants"]}
        assert omar == {"omarcastillo", "omarcasti"}
        assert people["danawhitfield"]["display"] == "Mayor Dana Whitfield"
        assert "appellant" in people["omarcastillo"]["roles"]

        a = _load(os.path.join(out, "meetingA.json"))
        b = _load(os.path.join(out, "meetingB.json"))
        assert [t["speaker"] for t in a] == ["danawhitfield", "omarcastillo", "danawhitfield", "Other",
                                             "danawhitfield"]
        # Meeting B's mixed voice cluster is split by the screen; the organisation tile
        # ("Riverton City") is not a person.
        assert [t["speaker"] for t in b] == ["danawhitfield", "omarcastillo", "Other", "danawhitfield"]
        how = _load(os.path.join(out, "meetingB.attribution.json"))
        assert [t["source"] for t in how["turns"]] == ["cluster", "screen", "cluster", "cluster"]
        assert how["display_names"]["omarcastillo"] == "Omar Castillo"

        stats = _load(os.path.join(out, "meetingA.stats.json"))
        assert stats["layout"] == "gallery" and stats["turns"] == 5
        assert os.path.exists(os.path.join(out, "meetingA.srt"))


def test_deterministic():
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        for folder in (one, two):
            _write_fixture(folder)
            _run(folder)
        for sub, names in (("work", ["registry.json", "meetingA.speakers.json", "meetingB.speakers.json"]),
                           ("out", ["meetingA.json", "meetingB.json", "meetingB.attribution.json"])):
            match, mismatch, errors = filecmp.cmpfiles(os.path.join(one, sub), os.path.join(two, sub),
                                                       names, shallow=False)
            assert not mismatch and not errors, (mismatch, errors)


def test_doctests():
    import zoompipe.attribution
    import zoompipe.files
    import zoompipe.identity
    for module in (zoompipe.identity, zoompipe.attribution, zoompipe.files):
        failed, _ = doctest.testmod(module)
        assert failed == 0, module.__name__


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
