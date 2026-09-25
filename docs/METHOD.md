# Method

How a raw tile read becomes a person (steps 3–4). Steps 1–2 are described in the
README. The code is `zoompipe/identity.py` (parsing and linking) and
`zoompipe/attribution.py` (attribution). Their doctests are the executable version of
this page.

## Why cleaning is needed

OCR of a 12–16 px name label, once a second for hours, yields many spellings of each
person. They come from clipping at the tile edge (`Robert J. Torr..`,
`ll Dobson` for "Bill Dobson"), misreads (`Pat Benaivides`, `michael:sirignano`),
role annotations (`Appellant - Niles Illich`, `Pat Benavides CC6 Crt Coordinator`),
pronouns, device names and room accounts. Measured on 2,224 meetings, 94.6% of reads
name a person. The rest are noise (3.7%), device tiles (0.8%), role, room or
organisation tiles (0.7%) and run-on reads of on-screen text (0.1%).

An earlier version of this pipeline cut every label at its first `-` and deleted
spaces. That kept the wrong half of role-prefixed labels, so every lawyer in one
appellate court became `appellant`. It also split compound surnames and left titles,
initials and diacritics inside the identity key, where no fuzzy threshold could safely
reconcile them.

## 1. Parsing a label (`parse_label`)

1. Normalise Unicode. Strip a leading state word (`Talking:`), a trailing ellipsis
   (recorded as *clipped*), pronoun groups (`(she/her)`), and a trailing device name
   (`iPad (2)`, recorded as *device*). Replace an e-mail address with its local part.
2. Split into segments at spaced dashes, em-dashes, pipes, commas and brackets. A
   hyphen or slash with no spaces around it separates only when one side is a role
   (`Appellant-John Thetford`). Otherwise it joins a compound (`O'Caña-Olivarez`,
   `Rojas-Moore`).
3. Split each segment into words, including camel-case (`JeffreyCarroll`,
   `ADAAdrian`). Name particles stay attached (`McGinn`, `DeAngelo`).
4. Class each word as **title** (judge, mayor, councillor, …; also OCR-damaged or
   left-clipped forms such as `Juage`, `udge`), **role** (appellant, clerk, chambers,
   …, plus the collection's own institution and place words from `--context`),
   **device**, **digit**, **suffix** (Jr, III) or **name**. Words that are also
   surnames (Hall, Chambers, King) count as roles only next to an unambiguous role
   word. After a full name, a short capitalised tag (`AD`, `CCg`, `CE`) is a role.
5. The person's name is the first run of name words in the segment with the most of
   them. The **key** is those words folded to lowercase ASCII, with middle initials
   dropped once two full words remain: `F. Philip Carbullido` and `Philip Carbullido`
   share `philipcarbullido`, while `A. Chen` stays `achen`.
6. The label's **kind** is `person`, `device` (a device name with at most one name
   word, or a brand first: `LG Escape Plus`), `role` (no name words), `runon` (more
   than 4 words or 30 characters) or `noise`. Everything but `person` becomes `Other`.

## 2. Linking identities (`link_identities`)

This runs twice: over the keys of one meeting, then over the per-meeting identities of
a whole collection. Every candidate link is backed by a named rule.

| Rule | Example | Kind |
|---|---|---|
| `ocr_variant` | `patbenaivides` ~ `patbenavides` | symmetric |
| `truncation` | `savannahgon` → `savannahgonzalez` | partial |
| `tail_noise` | `waynechristiasu` → `waynechristian` | partial |
| `left_clip` | `lldobson` → `billdobson`; `aynechristi` → `waynechristian` | partial |
| `initial` | `achen` → `alicechen` | partial |
| `surname` | `Councillor Gardi` → `markgardi` (only with a title across meetings) | partial |
| `given_name` | `miranda` → `mirandabouchard` (only within one meeting) | partial |

Keys are compared as OCR **skeletons**, with the confusable characters `i`/`l`/`1` and
`0`/`o` mapped together and doubled letters collapsed. `ocr_variant` then allows no
further edit under 10 characters, one under 18, and two beyond. `johnsmith` and
`joansmith` are two people. A clip must cut *into* a word (`Kerrisa Chelko` →
`Chelkowski`). A form missing whole words is a different claim, except for one
trailing word after a full name (`Jeffrey Carroll` / `Jeffrey Carroll VEMUE`). When a
clean, well-attested form is a prefix of a rarer longer one, the longer one is the
noise (`tail_noise`), not the completion.

**Guards.** A candidate link is applied only if all of these hold:

- **Uniqueness.** A partial form links only when every full form it could complete
  is one person. `john` inside both `johnsimpson` and `johnmoore` links to neither.
- **Cannot-link.** Two identities that each hold the screen ≥5 s in one meeting are
  different people. No chain of links may join them.
- **No drift.** When two clusters merge, the names that will represent them must be
  linkable on their own, so `a~b` and `b~c` cannot drag `a` and `c` together.
- **Spelling before names.** The spelling rules run to a fixed point first. The name
  rules (`initial`, `surname`, `given_name`) then judge uniqueness among people rather
  than among spellings of one person.
- **Partial forms never name a cluster.** The canonical key is the most credible
  complete form: not clipped, multi-word, most time on screen. The display name is its
  most frequent rendering, with a leading title if one was shown.

Links are applied strongest rule first, ties broken by score and then by key, so the
result does not depend on input order.

Within a meeting, a gap of ≤2 s with no name between two reads of the same person is
one misread frame, and is filled.

## 3. Attribution (`attribute_segments`)

Measured over 2,224 meetings before designing this step:

- The highlight moves a **median 0.6 s after** the voice-cluster change (IQR −0.4 to
  +1.3 s). The screen track is shifted back by 0.6 s.
- pyannote voice clusters are **83% pure** (speech-weighted). **9.6%** of speech sits in
  clusters where no on-screen name reaches 60%. Naming each cluster after its majority
  gave all of that speech to one person, and **7.3%** of people with ≥30 s on screen
  received no text at all.

Rules:

1. Tally each cluster's overlap with each on-screen name. The plurality names the
   cluster, and `Other` counts: a cluster heard mostly under a room, device or
   screen-share tile stays unattributed rather than borrowing a name.
2. If the cluster's name holds ≥80% of it, every segment takes that name.
3. Otherwise, a segment takes the name covering ≥70% of it on screen, if that name
   holds ≥10% of the cluster (so the voice evidence agrees it is one of the cluster's
   speakers). Failing that, it takes the cluster's name.
4. Consecutive segments with one speaker are joined into turns. Each turn records
   whether any segment was named from the screen (`source: screen`, 9.1% of turns
   on the evaluation corpus).

## Evaluation corpus

The rules were developed on retained raw OCR from 2,224 meetings in 48 collections.
They were evaluated on 1,266 released meetings in 28 collections, against the earlier
key-cutting version:

| | earlier version | this version |
|---|---:|---:|
| distinct speaker labels | 7,014 | 5,488 |
| label pairs ≥85% similar left unmerged | 2,740 | 145 |
| … ≥95% similar | 856 | 3 |
| turns on a label in such a pair | 66.7% | 4.6% |
| turns attributed to `Other` | 4.8% | 5.8% |

`Other` rises because device and room tiles are no longer counted as people. The
similar-pairs measure counts missed merges only. Wrong merges and wrong screen
attributions need human judgement. A review of sampled turns is under way, and its
accuracy figures are not reported here yet.
