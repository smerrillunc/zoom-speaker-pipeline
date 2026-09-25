"""
Speaker identity resolution: raw Zoom-tile OCR strings -> stable person identities.

The original pipeline turned a label into a key by cutting at the first ``-``, ``|``
or ``,`` and deleting spaces. That loses the structure a label carries, and three
failures follow from it:

* **The wrong half is kept.** ``"Appellant - Niles Illich"`` and ``"Appellant - John
  Wall"`` both become ``appellant``, so different people merge into one identity;
  ``"CLO - Susan Gross"`` becomes ``clo``.
* **Compound names are cut.** ``"O'Caña-Olivarez"`` becomes ``o'caña``, and OCR
  variants of the cut half (``o'cañia``) then split the person again.
* **Variants never meet.** Titles, middle initials, diacritics, punctuation and
  truncation (``"Robert J. Torr.."``) all live inside the key, so one person has many
  keys that no fuzzy threshold reconciles safely.

This module parses a label into structure first (:func:`parse_label`), then links
identities with a small set of named rules (:func:`link_identities`). Every link records
the rule and the score that produced it, so a merge can be audited and reversed.

The rules, in priority order:

``same_key``
    Identical identity keys. The key drops titles, roles, pronouns, diacritics,
    punctuation and middle initials, so ``"Judge Patricia O'Caña-Olivarez"`` and
    ``"Patricia O'Cana-Olivarez"`` meet here.
``ocr_variant``
    Near-identical OCR skeletons (``i``/``l``/``1``, ``0``/``o``, doubled letters
    folded), edit similarity >= ``variant_threshold``.
``truncation``
    One key is a prefix of exactly one maximal longer key. Zoom clips labels to the
    tile width, and ``"Robert J. Torr.."`` is a prefix of ``"Robert J. Torres"``.
``initial``
    ``a chen`` / ``achen`` against ``alice chen``, when exactly one identity in the
    pool has that initial and surname.
``surname``
    A titled single surname (``"Councillor Gardi"``) against the one multi-token
    identity with that surname. Untitled given names link only within a meeting.

Two guards apply to every merge. **Uniqueness**: a partial form links only when exactly
one candidate could complete it. **Cannot-link**: identities that each hold the screen
for at least ``cannot_link_seconds`` in the same meeting are asserted to be different
people and are never merged, directly or through a chain.

Everything here is deterministic: the same labels give the same identities, in any
input order.
"""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # pragma: no cover - rapidfuzz is required (requirements.txt)
    Levenshtein = None


# --------------------------------------------------------------------------- lexicon
# One generic lexicon for every collection. A word belongs here when it describes a
# role, party, place, organisation or device rather than naming a person. Nothing in
# it names a particular body.

# Honorifics that precede a person's name and are dropped from the identity key.
TITLES = frozenset("""
judge justice honorable honourable hon mayor deputy councillor councilor councilman
councilwoman councilmember councilperson cllr cr commissioner comm chair chairman
chairwoman chairperson vice president trustee alderman alderperson supervisor dr mr
mrs ms miss mx prof rev magistrate marshal sheriff officer sgt sergeant det detective
lt captain capt chief director senator sen rep representative reeve selectman
selectwoman superintendent atty attorney ada adas da clerk secretary treasurer
member presiding associate senior assistant asst pro tem ret retired esq aag ausa
""".split())

# Titles OCR commonly misreads ("Juage", "JudgeI"); one edit away still counts, for
# these long, name-unlike words only (never "mayor", which is one edit from "Mayer").
FUZZY_TITLES = ("judge", "justice", "honorable", "councillor", "councilor", "commissioner",
                "trustee", "magistrate")

# Generational suffixes are part of a person's name but not of the identity key.
SUFFIXES = frozenset("jr jrs sr srs ii iii iv".split())
_SUFFIX_SKELETONS = frozenset({"l", "lv", "jr", "sr"})
# Titles whose clipped tail is still unmistakable, with the shortest tail accepted.
_CLIPPED_TITLES = (("judge", 3), ("justice", 4), ("councillor", 5), ("commissioner", 5))

# Words that describe a role, party, venue or organisation. A segment made only of
# these (plus titles) is not a person.
ROLE_WORDS = frozenset("""
appellant appellants appellee appellees appelant appelle petitioner petitioners
respondent respondents relator relators plaintiff plaintiffs defendant defendants
intervenor amicus counsel prosecutor prosecution defense defence defender public
state states people county city town township village district borough parish
municipal council committee commission board boardroom courtroom court crt courts
chambers chamber room rooms hall floor podium lectern studio av audio video sound
system systems recording record broadcast stream streaming live meeting meetings
conference planning works police fire library ward office offices department dept
services service legal aid probation pretrial coordinator coordinators reporter
interpreter interpreting translator bailiff staff host cohost admin administrator
moderator zoom webinar panelist presenter speaker participant participants guest
guests user audience visitor caller phone call dial telephone viewer the of and for
at in on to by llc llp inc pc pllc pa corp co company group firm law associates
association university college school schools education unified isd usd authority
agency government gov org com net edu tenth first second third fourth fifth sixth
seventh eighth ninth eleventh twelfth thirteenth fourteenth appeals appellate supreme
circuit division clo cc victim witness jury juror clerks secretary treasurer manager
engineer planner principal teacher student students parent parents community
""".split())

# Role words that are also common surnames. They count as roles only next to an
# unambiguous role word ("Council Chambers") or when nothing else is left.
AMBIGUOUS_ROLE_WORDS = frozenset("""
chambers hall floor ward law king white may will public court chief clerk mayor
marshal sheriff judge justice bishop parish major deputy park street
""".split())

DEVICE_WORDS = frozenset("""
iphone ipad ipod galaxy android samsung pixel macbook imac laptop desktop pc
chromebook thinkpad surface dell lenovo hp huawei motorola moto oneplus lg thinq
kindle tablet polycom crestron logitech owl device computer zoomroom zoomrooms
""".split())

# Labels that are Zoom chrome, not a participant.
_NOISE = frozenset({
    "", "no speaker", "unmute", "mute", "muted", "you", "recording", "stop recording",
    "screen share", "sharing screen", "view options",
})

_STATE_PREFIXES = ("talking", "speaking", "now speaking", "active speaker")

_PRONOUN = re.compile(
    r"[\(\[]?\s*\b(?:she|he|they|ze|xe)\s*/\s*(?:her|hers|him|his|them|theirs|they|she|he)?"
    r"(?:\s*/\s*[a-z]*)?\.*\s*[\)\]]?",
    re.IGNORECASE,
)
_TRUNCATION_TAIL = re.compile(r"(\s*[.…]){2,}\s*$|…\s*$")
_SEGMENT_SPLIT = re.compile(r"\s+[-–—]+\s+|[–—]+|\||,|;|\(|\)|\[|\]|\s+/\s+| {2,}")
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+", re.UNICODE)

MIN_NAME_LENGTH = 3
MAX_NAME_TOKENS = 4
MAX_NAME_LENGTH = 30


def fold(text: str) -> str:
    """
    Lowercase, strip diacritics and keep letters and digits only.

    Example:
        >>> fold("O'Caña-Olivarez")
        'ocanaolivarez'
        >>> fold("Pérez")
        'perez'
    """
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", ascii_only.lower())


def skeleton(key: str) -> str:
    """
    OCR-confusable skeleton of a key: the characters OCR mistakes for each other are
    mapped to one representative and doubled letters are collapsed.

    Example:
        >>> skeleton("michaellsirignano") == skeleton("michaelsirignano")
        True
        >>> skeleton("corina1elozano") == skeleton("corinalelozano")
        True
    """
    mapped = key.translate(str.maketrans({"i": "l", "1": "l", "|": "l", "0": "o", "5": "s"}))
    return re.sub(r"(.)\1+", r"\1", mapped)


# --------------------------------------------------------------------------- parsing
@dataclass(frozen=True)
class ParsedLabel:
    """
    What one raw OCR string says about the tile's owner.

    Attributes:
        raw: the OCR string as read.
        kind: ``person``, ``device``, ``role`` (role/room/organisation only),
            ``runon`` (the crop caught on-screen text) or ``noise``.
        name: the person's name as displayed (original case), titles removed.
        tokens: folded name tokens, e.g. ``("patricia", "ocanaolivarez")``.
        key: identity key: folded name tokens with middle initials dropped.
        titles: folded honorifics found (``judge``, ``councillor``).
        roles: folded role words found (``appellant``, ``ada``).
        truncated: the label was ellipsised or clipped by the tile.
    """

    raw: str
    kind: str
    name: str = ""
    tokens: Tuple[str, ...] = ()
    key: str = ""
    titles: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()
    truncated: bool = False


_NAME_PARTICLES = ("mc", "mac", "de", "da", "di", "du", "la", "le", "van", "von", "o", "st")


def _split_camel(word: str) -> List[str]:
    """
    "JeffreyCarroll" -> ["Jeffrey", "Carroll"], "ADAAdrian" -> ["ADA", "Adrian"],
    "BenavidesCC" -> ["Benavides", "CC"]; name particles stay attached, so "McGinn",
    "DeAngelo" and "MacDonald" are one word.
    """
    parts = []
    for part in re.split(r"(?<=[a-z]{2})(?=[A-Z])|(?<=[A-Z]{2})(?=[A-Z][a-z])", word):
        if parts and fold(parts[-1]) in _NAME_PARTICLES:
            parts[-1] += part
        elif part:
            parts.append(part)
    return parts


def _token_class(folded: str, context: frozenset = frozenset(), last: bool = False) -> str:
    if not folded:
        return "empty"
    if folded in DEVICE_WORDS or any(
        len(folded) > len(word) + 2 and folded.endswith(word) for word in ("iphone", "ipad", "galaxy")
    ):
        return "device"
    if any(char.isdigit() for char in folded):
        return "digit"
    if folded in SUFFIXES:
        return "suffix"
    if folded in TITLES:
        return "title"
    if folded in ROLE_WORDS or folded in context:
        return "role"
    # A word clipped by the tile edge ("Townshi", "Coordinat") is still a role word.
    if last and len(folded) >= 5 and any(w.startswith(folded) for w in ROLE_WORDS | context if len(w) > len(folded)):
        return "role"
    return "name"


def _segment_tokens(segment: str, context: frozenset = frozenset(), clipped: bool = False) -> List[Tuple[str, str, str]]:
    """(display, folded, class) for each token of a segment."""
    out = []
    # A dot or colon between letters is OCR punctuation, not structure: "F.Philip",
    # "michael:sirignano". An apostrophe is part of the name: "O'Connor".
    segment = re.sub(r"(?<=\w)[.:_/](?=\w)", " ", segment)
    words = [w for m in _TOKEN.finditer(segment) for w in _split_camel(m.group(0))]
    shouting = all(not c.islower() for c in segment)
    names_seen = 0
    for i, word in enumerate(words):
        folded = fold(word)
        cls = _token_class(folded, context, clipped and i == len(words) - 1)
        # A leading title may be misread ("Juage") or clipped by the tile's left edge
        # ("udge Person", "stice Evans"): one edit from a title or from its tail.
        if cls == "name" and i == 0 and len(words) > 1 and (
                (len(folded) >= 5 and any(Levenshtein.distance(folded, t) <= 1 for t in FUZZY_TITLES))
                or any(t.endswith(folded) and len(folded) >= n for t, n in _CLIPPED_TITLES)
                or (len(folded) >= 4 and any(Levenshtein.distance(folded, t[-len(folded):]) <= 1
                                             for t in ("judge",) if len(t) > len(folded)))):
            cls = "title"
        # "Ill" and "Il" are OCR for the suffix "III"/"II".
        if cls == "name" and names_seen >= 2 and skeleton(folded) in _SUFFIX_SKELETONS:
            cls = "suffix"
        # After a full name, a short capitalised acronym is a role or office tag
        # ("Nicholas Mahrou AD", "Bianca Rivera CCg"), not a third name.
        if cls == "name" and not shouting and (
                (names_seen >= 2 and len(word) <= 4 and sum(c.isupper() for c in word) >= 2)
                or (names_seen >= 1 and 2 <= len(word) <= 3 and word.isupper())):
            cls = "role"
        # A clipped role word ending a long name: "Connie Soto Probat".
        if cls == "name" and names_seen >= 2 and i == len(words) - 1 and len(folded) >= 5 \
                and any(w.startswith(folded) and len(w) > len(folded) + 1 for w in ROLE_WORDS):
            cls = "role"
        names_seen += cls == "name"
        out.append((word, folded, cls))
    return out


def _is_role_only(tokens: Sequence[Tuple[str, str, str]]) -> bool:
    """True when a segment's tokens are all roles, titles, digits or devices."""
    return bool(tokens) and all(cls in ("role", "title", "digit", "device") for _, _, cls in tokens)


def _resolve_ambiguous(tokens: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    A word such as "Chambers" or "Hall" is a surname unless the segment holds an
    unambiguous role word, in which case it is part of a place name.
    """
    strong_role = any(cls == "role" and folded not in AMBIGUOUS_ROLE_WORDS for _, folded, cls in tokens)
    resolved = []
    for display, folded, cls in tokens:
        if folded in AMBIGUOUS_ROLE_WORDS and cls in ("role", "title") and not strong_role:
            # Keep a leading honorific ("Judge Hall") as a title, otherwise a name.
            is_leading = all(c == "title" for _, _, c in resolved)
            if cls == "title" and is_leading and len(tokens) > 1:
                resolved.append((display, folded, cls))
            else:
                resolved.append((display, folded, "name"))
        else:
            resolved.append((display, folded, cls))
    return resolved


def _join_compound(segments: List[str], context: frozenset = frozenset()) -> List[str]:
    """
    Undo splits that were not structure. "Rojas-Moore" and "O'Caña-Olivarez" are
    compound surnames; "Appellant-John Thetford" is a role and a name. A hyphen or
    slash with no surrounding spaces separates only when one side is role-only.
    """
    out = []
    for segment in segments:
        pieces = re.split(r"(?<=[^\W\d_])[-/](?=[^\W\d_])", segment)
        if len(pieces) == 1:
            out.append(segment)
            continue
        current = pieces[0]
        for piece in pieces[1:]:
            left_tail = _segment_tokens(current, context)[-1:]
            right_head = _segment_tokens(piece, context)[:1]
            if _is_role_only(_segment_tokens(current, context)) or _is_role_only(_segment_tokens(piece, context)) \
                    or (left_tail and left_tail[0][2] in ("role", "device")) \
                    or (right_head and right_head[0][2] in ("role", "device", "title")):
                out.append(current)
                current = piece
            else:
                current = current + piece  # compound word: hyphen dropped
        out.append(current)
    return out


# A device name appended to a person's name: "Ian Streight iPad", "Michael's iPhone (2)".
_DEVICE_TAIL = re.compile(
    r"(?:'s|’s)?\s*(?:i\s?phone|i\s?pad|galaxy[\w ]*|android|macbook[\w ]*|chromebook|laptop|thinq)"
    r"\s*(?:\(\d+\)|\d+)?\s*$",
    re.IGNORECASE,
)


def context_words(*texts: str) -> frozenset:
    """
    Organisation words for one collection, from its own metadata (institution name,
    region): a tile reading "Montague" or "Montague Townshi" in the Montague Township
    council is the body's own account, not a person. Derived, never hand-listed.

    Example:
        >>> sorted(context_words("Montague Township Council", "Ontario, Canada"))
        ['canada', 'council', 'montague', 'ontario', 'township']
    """
    words = set()
    for text in texts:
        for word in re.findall(r"[^\W\d_]+", text or ""):
            folded = fold(word)
            if len(folded) >= 4:
                words.add(folded)
    return frozenset(words)


def _restore_hyphens(display: str, raw: str) -> str:
    """
    Put back the hyphen of compound names in the displayed name only.

    Parsing joins "O'Caña-Olivarez" and "Rojas-Moore" into single words for the key;
    the displayed name should still read as written.

    Example:
        >>> _restore_hyphens("Patricia O'CañaOlivarez", "Judge Patricia O'Caña-Olivarez")
        "Patricia O'Caña-Olivarez"
        >>> _restore_hyphens("Aida Rojas Moore", "Aida Rojas-Moore")
        'Aida Rojas-Moore'
    """
    for compound in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?(?:-[^\W\d_]+(?:'[^\W\d_]+)?)+", str(raw or "")):
        parts = compound.split("-")
        for joined in ("".join(parts), " ".join(parts)):
            if joined in display:
                display = display.replace(joined, compound)
                break
    return display


def parse_label(raw: str, context: frozenset = frozenset()) -> ParsedLabel:
    """
    Parse one raw OCR label into a person identity (or say why it is not one).

    ``context`` holds the collection's own organisation words (:func:`context_words`).

    Examples:
        >>> parse_label("Appellant - Niles Illich").key
        'nilesillich'
        >>> parse_label("CLO - Susan Gross").key
        'susangross'
        >>> p = parse_label("Judge Patricia O'Caña-Olivarez"); p.key, p.titles, p.name
        ('patriciaocanaolivarez', ('judge',), "Patricia O'Caña-Olivarez")
        >>> p = parse_label("JudgePatricia O'Caña-Oli.."); p.key, p.truncated
        ('patriciaocanaoli', True)
        >>> parse_label("Pat Benavides CC6 Crt Coordinator").key
        'patbenavides'
        >>> parse_label("Katie McGinn (she/her)").key
        'katiemcginn'
        >>> parse_label("Appellee (Angela Sullivan)").key
        'angelasullivan'
        >>> parse_label("F. Philip Carbullido").key == parse_label("Honorable Philip Carbullido").key
        True
        >>> parse_label("A. Chen").key
        'achen'
        >>> parse_label("yolanda/aguilar").key
        'yolandaaguilar'
        >>> parse_label("michael:sirignano").key
        'michaelsirignano'
        >>> parse_label("Kim Hall").key
        'kimhall'
        >>> parse_label("Council Chambers").kind
        'role'
        >>> parse_label("Michael's iPad (2)").kind
        'device'
        >>> parse_label("Courtroom Clerk").kind
        'role'
        >>> parse_label("Talking: Suzanne Swope").key
        'suzanneswope'
        >>> parse_label("Jacklyn.Martin@cincinnati-oh.gov").key
        'jacklynmartin'
        >>> parse_label("No Speaker").kind
        'noise'
        >>> parse_label("Judge Danielle Edy Evidence must be kept until end of trial").kind
        'runon'
        >>> parse_label("lanStreightiPad").key
        'lanstreight'
        >>> parse_label("Ian Streight iPad").key
        'ianstreight'
        >>> parse_label("JeffreyCarroll").tokens
        ('jeffrey', 'carroll')
        >>> parse_label("Katie McGinn").tokens
        ('katie', 'mcginn')
        >>> ctx = context_words("Montague Township Council")
        >>> parse_label("Montague Townshi..", ctx).kind, parse_label("Montague", ctx).kind
        ('role', 'role')
        >>> parse_label("Pat Benavides CC6 Crt Coordinat..").key
        'patbenavides'
        >>> parse_label("ADAAdrian Lozano").key, parse_label("Nicholas Mahrou AD..").key
        ('adrianlozano', 'nicholasmahrou')
        >>> parse_label("Bianca Rivera CCg Coordi").key, parse_label("Connie Soto Probat").key
        ('biancarivera', 'conniesoto')
        >>> parse_label("Pat BenavidesCC6").key, parse_label("Isidro Montoya Jr").key
        ('patbenavides', 'isidromontoya')
        >>> parse_label("Juage Patricia").titles, parse_label("Pat Benavides Crt").titles
        (('juage',), ())
        >>> parse_label("Mayer Smith").key, parse_label("AJ Garcia").key
        ('mayersmith', 'ajgarcia')
        >>> parse_label("Justice Bill Pedersen Ill").key, parse_label("udge Person").key
        ('billpedersen', 'person')
        >>> parse_label("stice Evans").key, parse_label("LG Escape Plus").kind
        ('evans', 'device')
        >>> parse_label("Dge Smith").key, parse_label("Uday Smith").key
        ('smith', 'udaysmith')
        >>> parse_label("Kate Bierman").key, parse_label("Rice Johnson").key
        ('katebierman', 'ricejohnson')
        >>> parse_label("Idge Person").key, parse_label("Person CE").key, parse_label("Lee Smith").key
        ('person', 'person', 'leesmith')
    """
    if raw is None:
        return ParsedLabel("", "noise")
    text = unicodedata.normalize("NFKC", " ".join(str(raw).split()))
    if text.lower().strip(" .") in _NOISE:
        return ParsedLabel(raw, "noise")

    lowered = text.lower()
    for word in _STATE_PREFIXES:
        if lowered.startswith(word):
            remainder = text[len(word):]
            if not remainder.strip() or remainder[:1] in ":;.,|":
                text = remainder.lstrip(":;.,|").strip()
                break

    if "@" in text:  # an e-mail address as display name: the local part is the name
        text = re.sub(r"(\S+)@\S+", lambda m: m.group(1).replace(".", " ").replace("_", " "), text)

    truncated = bool(_TRUNCATION_TAIL.search(text))
    text = _TRUNCATION_TAIL.sub("", text)
    text = _PRONOUN.sub(" ", text)
    device = False
    stripped_device = _DEVICE_TAIL.sub("", text)
    if stripped_device != text:
        device, text = True, stripped_device.strip()

    segments = [s.strip() for s in _SEGMENT_SPLIT.split(text) if s and s.strip()]
    segments = _join_compound(segments, context)

    titles: List[str] = []
    roles: List[str] = []
    candidates = []
    device_first = False  # (score, order, display tokens, folded tokens)
    for order, segment in enumerate(segments):
        clipped = truncated and order == len(segments) - 1
        tokens = _resolve_ambiguous(_segment_tokens(segment, context, clipped))
        device = device or any(cls == "device" for _, _, cls in tokens)
        # Only honorifics that lead the segment are titles ("Judge X"); a title-like
        # word elsewhere ("Pat Benavides Crt Coordinator") describes a role.
        leading = 0
        while leading < len(tokens) and tokens[leading][2] == "title":
            leading += 1
        titles += [f for _, f, _ in tokens[:leading]]
        roles += [f for _, f, cls in tokens[leading:] if cls in ("role", "digit", "title")]
        # The name is the first run of name tokens; leading titles are skipped and the
        # run ends at the first role, digit or device token ("Pat Benavides CC6 Crt").
        run: List[Tuple[str, str]] = []
        for display, folded, cls in tokens:
            if cls == "name":
                run.append((display, folded))
            elif run:
                break
            elif cls == "device":
                # A brand before the words names the hardware ("LG Escape Plus",
                # "Galaxy Note9"); the words after it are a model, not a person.
                device_first = True
                break
        if run:
            letters = sum(len(f) for _, f in run)
            multi = sum(1 for _, f in run if len(f) > 1)
            candidates.append(((min(multi, 3), letters > 3), -order, run))

    if not candidates:
        if device:
            return ParsedLabel(raw, "device", titles=tuple(titles), roles=tuple(roles), truncated=truncated)
        if titles or roles:
            return ParsedLabel(raw, "role", titles=tuple(titles), roles=tuple(roles), truncated=truncated)
        return ParsedLabel(raw, "noise")

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    run = candidates[0][2]
    display = _restore_hyphens(" ".join(d for d, _ in run), raw)
    tokens = tuple(f for _, f in run if f)
    # Middle and leading initials are dropped from the key once two full words remain,
    # so "F. Philip Carbullido", "Philip Carbullido" and "Wade J. Hedtke"/"Wade Hedtke"
    # meet. A lone initial is kept: "A. Chen" stays "achen".
    full = [t for t in tokens if len(t) > 1]
    key = "".join(full) if len(full) >= 2 else "".join(tokens)

    common = dict(raw=raw, name=display, tokens=tokens, key=key,
                  titles=tuple(titles), roles=tuple(roles), truncated=truncated)
    if device and (len(full) <= 1 or device_first):
        return ParsedLabel(kind="device", **common)
    if len(key) < MIN_NAME_LENGTH:
        return ParsedLabel(kind="noise", **common)
    if len(tokens) > MAX_NAME_TOKENS or len(key) > MAX_NAME_LENGTH:
        return ParsedLabel(kind="runon", **common)
    return ParsedLabel(kind="person", **common)


# --------------------------------------------------------------------------- linking
@dataclass
class Identity:
    """
    One distinct key in a pool, with the evidence gathered for it.

    ``seconds`` is on-screen dwell; ``meetings`` maps meeting id -> dwell there;
    ``names`` counts the displayed renderings (for choosing a display name).
    """

    key: str
    tokens: Tuple[str, ...] = ()
    truncated_seconds: float = 0.0
    seconds: float = 0.0
    meetings: Dict[str, float] = field(default_factory=dict)
    names: Counter = field(default_factory=Counter)
    titles: Counter = field(default_factory=Counter)
    roles: Counter = field(default_factory=Counter)

    @property
    def truncated(self) -> bool:
        """Mostly seen clipped: a prefix of the real name rather than the name."""
        return self.truncated_seconds > 0.5 * self.seconds

    @property
    def boundaries(self) -> frozenset:
        """Character offsets in ``key`` where one name word ends and the next begins."""
        full = [t for t in self.tokens if len(t) > 1]
        words = full if len(full) >= 2 and "".join(full) == self.key else list(self.tokens)
        if "".join(words) != self.key:
            return frozenset()
        cuts, total = set(), 0
        for word in words[:-1]:
            total += len(word)
            cuts.add(total)
        return frozenset(cuts)

    @property
    def surname(self) -> str:
        full = [t for t in self.tokens if len(t) > 1]
        return full[-1] if len(full) >= 2 else ""

    @property
    def given(self) -> str:
        return self.tokens[0] if len(self.tokens) >= 2 else ""


def _similarity(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(a, b)


def _prefix_match(short: str, long: str) -> bool:
    """``short`` is ``long`` clipped on the right, allowing one OCR slip in the kept part."""
    if len(short) >= len(long):
        return False
    head = long[: len(short)]
    return head == short or (len(short) >= 10 and _similarity(short, head) >= 0.9)


def _suffix_match(short: str, long: str) -> bool:
    """``short`` is ``long`` clipped on the left (the crop cut the first letters)."""
    if len(short) >= len(long):
        return False
    tail = long[-len(short):]
    return tail == short or (len(short) >= 10 and _similarity(short, tail) >= 0.9)


# Rule priorities: lower is stronger. "Partial" rules absorb an incomplete form into a
# complete one and are subject to the uniqueness guard.
RULES = {"ocr_variant": 1, "truncation": 2, "tail_noise": 2, "left_clip": 3, "initial": 4, "surname": 5,
         "given_name": 6}
PARTIAL_RULES = {"truncation", "tail_noise", "left_clip", "initial", "surname", "given_name"}
SPELLING_RULES = ("ocr_variant", "truncation", "tail_noise", "left_clip")
NAME_RULES = ("initial", "surname", "given_name")


def pair_rule(x: Identity, y: Identity, variant_threshold: float = 0.88) -> Optional[Tuple[str, float, str, str]]:
    """
    The strongest rule under which ``x`` and ``y`` could be one person, ignoring the
    guards, as ``(rule, score, partial_key, full_key)``; ``None`` when no rule applies.

    Examples:
        >>> I = lambda key, tokens, trunc=False: Identity(key, tokens, truncated_seconds=float(trunc), seconds=1.0)
        >>> pair_rule(I("billdobson", ("bill", "dobson")), I("lldobson", ("ll", "dobson")))[0]
        'left_clip'
        >>> pair_rule(I("robertjtorr", ("robert", "j", "torr"), True), I("robertjtorres", ("robert", "j", "torres")))[0]
        'truncation'
        >>> pair_rule(I("achen", ("a", "chen")), I("alicechen", ("alice", "chen")))[:2]
        ('initial', 1.0)
        >>> pair_rule(I("johnsmith", ("john", "smith")), I("joansmith", ("joan", "smith"))) is None
        True
        >>> pair_rule(I("waynechristian", ("wayne", "christian")), I("waynechristians", ("wayne", "christians")))[0]
        'ocr_variant'
        >>> big = Identity("waynechristian", ("wayne", "christian"), seconds=9e4)
        >>> rule = pair_rule(big, I("waynechristiasu", ("wayne", "christiasu"))); rule[0], rule[2], rule[3]
        ('tail_noise', 'waynechristiasu', 'waynechristian')
        >>> pair_rule(I("ichaelgrana", ("ichael", "grana")), I("michaelgrana", ("michael", "grana")))[0]
        'left_clip'
    """
    sx, sy = skeleton(x.key), skeleton(y.key)
    if sx == sy:
        lo, hi = sorted((x.key, y.key))
        return ("ocr_variant", 1.0, lo, hi)

    # Edits beyond the confusable characters are allowed in proportion to length: none
    # under 10 characters ("johnsmith" / "joansmith" are two people), one under 18,
    # two beyond. A difference away from the start is a spelling variant, and on-screen
    # dwell -- not length -- decides which spelling names the person.
    longest = max(len(sx), len(sy))
    allowed = 0 if longest < 10 else 1 if longest < 18 else 2
    variant = bool(allowed) and Levenshtein.distance(sx, sy) <= allowed \
        and _similarity(sx, sy) >= variant_threshold
    if variant and x.key[:3] == y.key[:3]:
        lo, hi = sorted((x.key, y.key))
        return ("ocr_variant", _similarity(sx, sy), lo, hi)

    # Clipping: one form is the other with its end (or start) cut off.
    short, full = sorted((x, y), key=lambda i: (len(i.key), i.key))
    if len(short.key) >= 6 and len(short.key) < len(full.key):
        ss, sf = skeleton(short.key), skeleton(full.key)
        ratio = len(short.key) / len(full.key)
        if _prefix_match(ss, sf) and short.key[:3] == full.key[:3]:
            cut = len(short.key)
            mid_word = cut not in full.boundaries
            # One whole trailing word may be missing ("Jeffrey Carroll" / "Jeffrey Carroll
            # VEMUE"): the short form must itself be a full name.
            one_word = sum(1 for b in full.boundaries if b >= cut) == 1 and len(short.tokens) >= 2 \
                and ratio >= 0.6
            if short.truncated or mid_word or one_word:
                if short.truncated or short.seconds <= full.seconds:
                    return ("truncation", ratio, short.key, full.key)
                # The clean, better-attested form is the name; the longer one carries
                # OCR debris past its end ("waynechristian" / "waynechristiasu").
                return ("tail_noise", ratio, full.key, short.key)
        cut = len(full.key) - len(short.key)
        if ratio >= 0.6 and _suffix_match(ss, sf) and full.tokens and len(full.tokens[0]) > cut:
            # A left clip cuts *into* the first word ("ll Dobson" for "Bill Dobson").
            return ("left_clip", ratio, short.key, full.key)
        # Clipped at both ends ("aynechristi" for "waynechristian"): an exact interior
        # substring that starts inside the first word.
        at = sf.find(ss)
        if len(short.key) >= 8 and ratio >= 0.6 and 0 < at and full.tokens \
                and len(full.tokens[0]) > at and at + len(ss) < len(sf):
            return ("left_clip", ratio, short.key, full.key)

    if variant:
        lo, hi = sorted((x.key, y.key))
        return ("ocr_variant", _similarity(sx, sy), lo, hi)

    for a, b in ((x, y), (y, x)):
        if not (b.surname and b.given) or b.truncated or len(a.key) < 5:
            continue
        if len(b.surname) >= 4 and a.key[0] == b.given[0] and len(a.key) < len(b.key) \
                and skeleton(a.key[1:]) == skeleton(b.surname):
            return ("initial", 1.0, a.key, b.key)
        if len(a.tokens) == 1 and skeleton(a.key) == skeleton(b.surname):
            return ("surname", 1.0, a.key, b.key)
        if len(a.tokens) == 1 and skeleton(a.key) == skeleton(b.given):
            return ("given_name", 1.0, a.key, b.key)
    return None


def _candidate_pairs(pool: Dict[str, Identity]) -> Iterable[Tuple[str, str]]:
    """Pairs worth testing: blocked on skeleton head, skeleton tail, and name parts."""
    keys = sorted(pool)
    skel = {k: skeleton(k) for k in keys}
    blocks = defaultdict(set)
    for k in keys:
        s = skel[k]
        blocks["h" + s[:2]].add(k)
        blocks["t" + s[-4:]].add(k)
        ident = pool[k]
        if ident.surname:
            blocks["s" + skeleton(ident.surname)].add(k)
        if ident.given:
            blocks["g" + skeleton(ident.given)].add(k)
        if len(ident.tokens) == 1:
            blocks["s" + s].add(k)
            blocks["g" + s].add(k)
            if len(k) >= 5:
                blocks["s" + skeleton(k[1:])].add(k)
        elif len(ident.tokens) == 2 and len(ident.tokens[0]) == 1:
            blocks["s" + skeleton(ident.tokens[1])].add(k)
    seen = set()
    for block in blocks.values():
        members = sorted(block)
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if (a, b) not in seen:
                    seen.add((a, b))
                    yield a, b


def candidate_links(
    pool: Dict[str, Identity],
    rules: Iterable[str],
    variant_threshold: float = 0.88,
    within_meeting: bool = False,
) -> Tuple[List[Tuple[int, float, str, str, str]], List[dict]]:
    """
    Rule-backed links in a pool as ``(priority, score, partial, full, rule)``, strongest
    first, plus the links refused by the uniqueness guard.

    A partial form (clipped, initial, lone surname) links only when every full form it
    could complete is one person -- i.e. the candidates are all spelling variants of
    each other. ``john`` inside both ``johnsimpson`` and ``johnmoore`` links to neither.
    Across meetings a lone surname needs a title ("Councillor Gardi"), and a lone given
    name never links.

    Example:
        >>> pool = {k: Identity(k, t, seconds=1.0) for k, t in [
        ...     ("achen", ("a", "chen")), ("alicechen", ("alice", "chen")), ("adamchen", ("adam", "chen"))]}
        >>> links, refused = candidate_links(pool, NAME_RULES)
        >>> links, [r["reason"] for r in refused]
        ([], ['ambiguous: 2 candidates'])
    """
    rules = set(rules)
    by_partial = defaultdict(list)
    symmetric = []
    for a, b in _candidate_pairs(pool):
        found = pair_rule(pool[a], pool[b], variant_threshold)
        if not found or found[0] not in rules:
            continue
        rule, score, part, full = found
        if rule == "surname" and not within_meeting and not sum(pool[part].titles.values()):
            continue
        if rule == "given_name" and not within_meeting:
            continue
        if rule in PARTIAL_RULES:
            by_partial[part].append((RULES[rule], score, part, full, rule))
        else:
            symmetric.append((RULES[rule], score, part, full, rule))

    links = list(symmetric)
    refused = []
    skel = {k: skeleton(k) for k in pool}
    for part, options in sorted(by_partial.items()):
        fulls = {o[3] for o in options}
        # Uniqueness counts every key the partial form could be clipped from, not only
        # those that passed the length-ratio test: "carroll" ends both "evcarroll" and
        # "jeffreycarroll", so it is ambiguous even though only one is close in length.
        sp = skel[part]
        if any(o[4] == "truncation" for o in options):
            fulls |= {k for k in pool if k != part and len(skel[k]) > len(sp) and skel[k].startswith(sp)}
        if any(o[4] == "tail_noise" for o in options):
            fulls |= {k for k in pool if k != part and 6 <= len(skel[k]) < len(sp) and sp.startswith(skel[k])}
        if any(o[4] == "left_clip" for o in options):
            fulls |= {k for k in pool if k != part and len(skel[k]) > len(sp) and skel[k].endswith(sp)}
        fulls = sorted(fulls)
        # The completions must all be one person: pairwise spelling variants.
        coherent = all(
            (pair_rule(pool[p], pool[q], variant_threshold) or ("",))[0] in SPELLING_RULES
            for i, p in enumerate(fulls) for q in fulls[i + 1:]
        )
        if not coherent:
            refused.append({"kept_apart": [part] + fulls[:5], "rule": options[0][4],
                            "reason": f"ambiguous: {len(fulls)} candidates"})
            continue
        best = min(options, key=lambda o: (o[0], -o[1], -pool[o[3]].seconds, o[3]))
        links.append(best)
    links.sort(key=lambda link: (link[0], -link[1], link[2], link[3]))
    return links, refused


def credibility(ident: Identity) -> Tuple:
    """
    How credible a key is as the name for its cluster: not clipped, then a full
    (multi-word) name, then on-screen dwell. The most credible key names the cluster.
    """
    return (not ident.truncated, len([t for t in ident.tokens if len(t) > 1]) >= 2,
            round(ident.seconds, 3), ident.key)


def aggregate(pool: Dict[str, Identity], mapping: Dict[str, str]) -> Dict[str, Identity]:
    """Collapse a pool by ``mapping`` into one identity per canonical key."""
    out: Dict[str, Identity] = {}
    for key in sorted(pool):
        ident = pool[key]
        target = mapping.get(key, key)
        agg = out.setdefault(target, Identity(target, pool[target].tokens))
        agg.seconds += ident.seconds
        agg.truncated_seconds += ident.truncated_seconds
        for m, s in ident.meetings.items():
            agg.meetings[m] = agg.meetings.get(m, 0.0) + s
        if key == target:
            agg.names.update(ident.names)
        agg.titles.update(ident.titles)
        agg.roles.update(ident.roles)
    return out


def _cluster(pool, links, variant_threshold, cannot_link_seconds):
    parent = {k: k for k in pool}
    members = {k: {k} for k in pool}
    meetings = {k: {m for m, s in pool[k].meetings.items() if s >= cannot_link_seconds} for k in pool}
    evidence: List[dict] = []

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def representative(root):
        return max(members[root], key=lambda k: credibility(pool[k]))

    for _, score, a, b, rule in links:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        shared = sorted(meetings[ra] & meetings[rb])
        if shared:
            evidence.append({"kept_apart": [a, b], "rule": rule, "score": round(score, 3),
                             "reason": "both on screen in the same meeting", "meetings": shared[:5]})
            continue
        rep_a, rep_b = representative(ra), representative(rb)
        if rep_a != a or rep_b != b:
            # Chains must not drift: the names that will represent the two clusters
            # have to be linkable on their own.
            if pair_rule(pool[rep_a], pool[rep_b], variant_threshold) is None:
                evidence.append({"kept_apart": [a, b], "rule": rule, "score": round(score, 3),
                                 "reason": f"cluster names disagree ({rep_a} / {rep_b})"})
                continue
        parent[ra] = rb
        members[rb] |= members.pop(ra)
        meetings[rb] |= meetings.pop(ra)
        evidence.append({"merge": a, "into": b, "rule": rule, "score": round(score, 3),
                         "seconds": [round(pool[a].seconds, 1), round(pool[b].seconds, 1)],
                         "meetings": [len(pool[a].meetings), len(pool[b].meetings)]})

    canonical = {}
    for root, keys in members.items():
        rep = representative(root)
        for k in keys:
            canonical[k] = rep
    return canonical, evidence


def link_identities(
    pool: Dict[str, Identity],
    variant_threshold: float = 0.88,
    cannot_link_seconds: float = 5.0,
    within_meeting: bool = False,
    max_rounds: int = 6,
) -> Tuple[Dict[str, str], List[dict]]:
    """
    Cluster a pool of identities and return ``(key -> canonical key, evidence)``.

    Two stages. Spelling rules (``ocr_variant``, ``truncation``, ``left_clip``) run
    first, so that the name rules (``initial``, ``surname``, ``given_name``) judge
    uniqueness among people rather than among spellings of one person.

    Within a stage, links apply strongest first. A merge is refused when both clusters
    hold the screen for ``cannot_link_seconds`` in a shared meeting (cannot-link), or
    when the names that would represent the two clusters are not themselves linkable
    (so ``a~b`` and ``b~c`` cannot drag ``a`` and ``c`` together).

    Example:
        >>> pool = {}
        >>> for key, tokens, secs, meeting in [
        ...     ("patriciaocanaolivarez", ("patricia", "ocanaolivarez"), 900, "m1"),
        ...     ("patriciaocanaoli", ("patricia", "ocanaoli"), 300, "m2"),
        ...     ("patriciaocanaoliv", ("patricia", "ocanaoliv"), 200, "m3"),
        ...     ("johnwall", ("john", "wall"), 60, "m1"),
        ...     ("johnwali", ("john", "wali"), 60, "m1")]:
        ...     pool[key] = Identity(key, tokens, seconds=secs, meetings={meeting: secs})
        >>> for k in ("patriciaocanaoli", "patriciaocanaoliv"):
        ...     pool[k].truncated_seconds = pool[k].seconds
        >>> mapping, evidence = link_identities(pool)
        >>> sorted(set(mapping.values()))
        ['johnwali', 'johnwall', 'patriciaocanaolivarez']
        >>> [e["reason"] for e in evidence if "reason" in e]
        ['both on screen in the same meeting']
    """
    # Spelling rules run to a fixed point: each round re-judges uniqueness among the
    # clusters the previous round formed, so "waynechristia" becomes unambiguous once
    # "waynechristians" and "waynechristiasu" have joined "waynechristian".
    first = {k: k for k in pool}
    evidence: List[dict] = []
    reduced = pool
    for _ in range(max_rounds):
        links, refused = candidate_links(reduced, SPELLING_RULES, variant_threshold, within_meeting)
        step, round_evidence = _cluster(reduced, links, variant_threshold, cannot_link_seconds)
        merged = [e for e in round_evidence if "merge" in e]
        evidence += [e for e in round_evidence if "merge" in e]
        if not merged:
            evidence += refused + round_evidence
            break
        first = {k: step[v] for k, v in first.items()}
        reduced = aggregate(pool, first)

    reduced = aggregate(pool, first)
    links, refused = candidate_links(reduced, NAME_RULES, variant_threshold, within_meeting)
    second, more = _cluster(reduced, links, variant_threshold, cannot_link_seconds)
    evidence += refused + more

    # The final name of each cluster is chosen over all its original keys. A key that
    # was absorbed as a partial form (clipped, initial, lone surname) never names it.
    partial = {e["merge"] for e in evidence if "merge" in e and e["rule"] in PARTIAL_RULES}
    clusters = defaultdict(list)
    for key in pool:
        clusters[second[first[key]]].append(key)
    mapping = {}
    for keys in clusters.values():
        rep = max(keys, key=lambda k: (k not in partial,) + credibility(pool[k]))
        for k in keys:
            mapping[k] = rep
    return mapping, evidence


def display_name(pool: Dict[str, Identity], keys: Iterable[str], canonical: str) -> str:
    """
    The most credible displayed rendering for a cluster: the canonical key's most
    frequent rendering, prefixed by its most frequent title if one was ever shown.
    """
    ident = pool[canonical]
    name = ident.names.most_common(1)[0][0] if ident.names else canonical
    titles = Counter()
    for k in keys:
        titles.update(pool[k].titles)
    if titles:
        title, _ = titles.most_common(1)[0]
        if title in ("judge", "justice", "mayor", "councillor", "councilor", "commissioner",
                     "honorable", "dr", "cr", "trustee", "magistrate", "chair"):
            name = f"{title.capitalize()} {name}"
    return name


# ------------------------------------------------------------------ one meeting
def dwell_intervals(raw_changes: Sequence[Sequence], duration: float) -> List[Tuple[float, float, str]]:
    """``[[t, label], ...]`` -> ``[(start, end, label)]`` with each label held until the next."""
    rows = sorted(((float(t), s) for t, s in raw_changes), key=lambda r: r[0])
    out = []
    for i, (t, s) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else max(duration, t + 1.0)
        if end > t:
            out.append((t, end, s))
    return out


def resolve_meeting(
    raw_changes: Sequence[Sequence],
    duration: float,
    meeting: str = "",
    variant_threshold: float = 0.88,
    fill_gap_seconds: float = 2.0,
    context: frozenset = frozenset(),
) -> Tuple[List[Tuple[float, str]], Dict[str, Identity], Dict[str, str], List[dict]]:
    """
    Resolve one meeting's raw OCR stream into identity-keyed speaker changes.

    Returns ``(changes, pool, mapping, evidence)``: ``changes`` is ``[(t, key)]`` with
    non-person tiles as ``"Other"``; ``pool`` the meeting's identities with dwell;
    ``mapping`` the within-meeting merges.

    A gap of at most ``fill_gap_seconds`` of ``Other`` between two runs of the same
    identity is one misread frame of an unchanged tile and is filled.

    Example:
        >>> changes, pool, mapping, _ = resolve_meeting(
        ...     [[0, "Susan Gross - Pro.."], [5, "Susan Gross - Pro..."], [9, "No Speaker"],
        ...      [10, "Susan Gross"], [20, "Appellant - John Wall"], [30, "iPad (7)"]], 40)
        >>> changes
        [(0.0, 'susangross'), (20.0, 'johnwall'), (30.0, 'Other')]
    """
    parsed_cache: Dict[str, ParsedLabel] = {}
    pool: Dict[str, Identity] = {}
    labelled = []
    for start, end, raw in dwell_intervals(raw_changes, duration):
        p = parsed_cache.get(raw) or parsed_cache.setdefault(raw, parse_label(raw, context))
        seconds = end - start
        if p.kind != "person":
            labelled.append((start, end, "Other"))
            continue
        ident = pool.setdefault(p.key, Identity(p.key, p.tokens))
        ident.seconds += seconds
        ident.meetings[meeting] = ident.meetings.get(meeting, 0.0) + seconds
        if p.truncated:
            ident.truncated_seconds += seconds
        ident.names[p.name] += seconds
        ident.titles.update(p.titles)
        ident.roles.update(p.roles)
        labelled.append((start, end, p.key))

    mapping, evidence = link_identities(pool, variant_threshold, cannot_link_seconds=float("inf"),
                                        within_meeting=True)
    labelled = [(s, e, mapping.get(k, k)) for s, e, k in labelled]

    # Fill a short Other between two runs of the same identity.
    for i in range(1, len(labelled) - 1):
        s, e, k = labelled[i]
        if k == "Other" and e - s <= fill_gap_seconds and labelled[i - 1][2] == labelled[i + 1][2] != "Other":
            labelled[i] = (s, e, labelled[i - 1][2])

    changes: List[Tuple[float, str]] = []
    for s, _, k in labelled:
        if not changes or changes[-1][1] != k:
            changes.append((s, k))
    return changes, pool, mapping, evidence


def merge_pools(pools: Iterable[Tuple[Dict[str, Identity], Dict[str, str]]]) -> Dict[str, Identity]:
    """
    Combine per-meeting pools into a body pool keyed by each meeting's canonical key,
    so the cross-meeting linker sees one identity per person per meeting.
    """
    body: Dict[str, Identity] = {}
    for pool, mapping in pools:
        for key, ident in aggregate(pool, mapping).items():
            agg = body.setdefault(key, Identity(key, ident.tokens))
            agg.seconds += ident.seconds
            agg.truncated_seconds += ident.truncated_seconds
            for m, sec in ident.meetings.items():
                agg.meetings[m] = agg.meetings.get(m, 0.0) + sec
            agg.names.update(ident.names)
            agg.titles.update(ident.titles)
            agg.roles.update(ident.roles)
    return body
