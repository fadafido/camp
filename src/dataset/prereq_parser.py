"""Shared prerequisite-string parser for the four real-catalogue institutions
(Khalifa, AUS, UNC).

The four catalogues expose *real* prerequisites as free-text strings
(``prereq_raw``). This module parses one such string into the CAP-Bench
prerequisite form: an AND-of-OR-groups array of prefixed ``course_id`` values
(an AND-of-OR-groups array of prefixed ``course_id`` values).

Design — faithful, never fabricated:

* Non-course tokens (grade thresholds, standing/classification, permission,
  fees, placement tests, "recommended", header bleed) are stripped into a
  separate ``prereq_notes`` list, never into the structured prerequisites.
* "credit or enrollment in X" and "prerequisite/concurrent: X"
  (AUS) are treated as *corequisite* notes, not hard prerequisites.
* References to courses outside the scoped set are dropped from the structured
  prerequisites and counted (``n_out_of_set``).
* A string whose course-logic cannot be reduced to AND-of-OR (e.g. a genuine
  OR-of-AND such as "(A and B) or (C and D)" that survives
  out-of-set pruning) is returned with ``unparsed=True`` and empty structured
  prerequisites, so the caller can log it rather than silently mangle it.

Boolean grammar (precedence: OR binds tighter than AND; parentheses group):
    expr      := and_expr
    and_expr  := or_expr ( (AND | ',') or_expr )*
    or_expr   := atom ( OR atom )*
    atom      := CODE | '(' expr ')'

Module: AI503. British English in prose. Deterministic; no randomness.
"""

from __future__ import annotations

import re
from typing import Any

# Institution → course_id prefix.
PREFIX = {"khalifa": "KHAL", "aus": "AUS", "unc": "UNC"}

# A course code: subject (2–5 letters) + number (2–4 digits, optional letter
# suffix such as 129P, 3116X, 57H).
_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s*(\d{2,4}[A-Z]?)\b")
# A bare course number (subject inherited from the preceding code) — UNC style
# "ECON 400 and 410".
_BARE_NUM_RE = re.compile(r"\b(\d{2,4}[A-Z]?)\b")

# Note phrases removed from the text and recorded in ``prereq_notes``. Order
# matters: longer / more specific patterns first. Each match is replaced by a
# space; its text is appended to the notes list.
_NOTE_PATTERNS = [
    r"prerequisite/concurrent\s*:\s*[^.;]*",          # handled as coreq below too
    r"a grade of\s*[A-Z][+-]?\s*or better(?:\s*is required)?",
    r"with a grade of\s*[A-Z][+-]?\s*or better",
    r"both with a grade of\s*[A-Z][+-]?\s*or better",
    r"a grade of\s*[A-Z][+-]?\s*or better in[^.;]*",
    r"a grade\s*[A-Z][+-]?\s*or better",
    r"grade of\s*[A-Z][+-]?\s*or better",
    r"Requires a grade of[^.;]*",
    r"is required in[^.;]*",
    r"is required",
    r"permission (?:of|from)(?: the)? instructor[^.;]*",
    r"permission of the instructor for students lacking[^.;]*",
    r"permission from instructor for students lacking[^.;]*",
    r"Instructor Permission for Course",
    r"Department Approval",
    r"business senior standing",
    r"(?:junior|senior|sophomore|freshman)(?:\s*I{1,2})?\s*standing",
    r"[A-Za-z ]*?classification(?:\s+in the Ivy College of Business)?",
    r"Ivy College of Business[^.;]*",
    r"Graduate (?:Classification|Student|classification)[^.;]*",
    r"Master of[^.;]*",
    r"Lab/Tech fee rate\s*[A-Z]\s*applies\.?",
    r"Registration fees? appl(?:ies|y)\.?",
    r"Students must have a CGPA[^.;]*",
    r"placement by the department",
    r"placement into\s*[A-Z]+\s*\d+",
    r"or placement[^.;]*",
    r"an internship approved[^.;]*",
    r"or equivalent",
    r"or higher",
    r"EPT score[^.;]*",
    r"ELPT score[^.;]*",
    r"SAT[^.;]*",
    r"any AUS math placement[^.;]*",
    r"exemption from the placement test",
    r"any preparatory math course",
    r"any one of",                                    # connective kept as note marker
    r"School of Business Administration",
    r"College of Arts and Sciences",
    r"\(\s*Typically Offered[^)]*\)?",
]

# "X recommended" is advisory, not a prerequisite.
_RECOMMENDED_RE = re.compile(r"[A-Z]{2,5}\s*\d{2,4}[A-Z]?\s+recommended")


class _NotFlattenable(Exception):
    """Raised when the parsed tree is a genuine OR-of-AND (deeper than the
    AND-of-OR the schema supports)."""


def make_course_id(subject: str, number: str, university: str) -> str:
    """Build a prefixed course_id, e.g. ('COSC','310','khalifa') -> KHAL_COSC_310."""
    return f"{PREFIX[university]}_{subject.upper()}_{number.upper()}"


def code_to_id(code: str, university: str) -> str | None:
    """Normalise a raw code string ('COSC230' or 'COSC 230') to a course_id."""
    m = _CODE_RE.match(code.strip())
    if not m:
        return None
    return make_course_id(m.group(1), m.group(2), university)


# --------------------------------------------------------------------------- #
# Note / coreq extraction
# --------------------------------------------------------------------------- #
def _extract_coreqs(text: str, university: str) -> tuple[str, list[str]]:
    """Pull out corequisite clauses (treated as notes, not hard prereqs).

    Handles "credit or enrollment in X" and
    "prerequisite/concurrent: X". X may be a parenthesised list, or run to the
    next ';'/'.'/end. Returns (remaining_text, coreq_note_strings)."""
    notes: list[str] = []

    def grab(pattern: str, label: str, s: str) -> str:
        out = []
        last = 0
        result = []
        for m in re.finditer(pattern, s, flags=re.IGNORECASE):
            clause = m.group(1).strip()
            notes.append(f"{label}: {clause}")
        # remove all matches
        return re.sub(pattern, " ", s, flags=re.IGNORECASE)

    text = grab(r"credit or enrollment in\s+(\(.*\)|[^;.]+)", "corequisite", text)
    text = grab(r"prerequisite/concurrent\s*:\s*([^.;]*)", "prerequisite/concurrent", text)
    return text, notes


def _extract_notes(text: str) -> tuple[str, list[str]]:
    """Strip note phrases; return (remaining_text, note_strings)."""
    notes: list[str] = []
    # "recommended" courses first (advisory, not required).
    for m in _RECOMMENDED_RE.finditer(text):
        notes.append(m.group(0).strip())
    text = _RECOMMENDED_RE.sub(" ", text)
    for pat in _NOTE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            frag = m.group(0).strip()
            if frag and frag.lower() not in ("any one of",):
                notes.append(frag)
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text, notes


# --------------------------------------------------------------------------- #
# Tokenisation + parse
# --------------------------------------------------------------------------- #
def _tokenise(text: str, university: str) -> list[Any]:
    """Tokenise into CODE ids, operators ('AND'/'OR'), and parens ('('/')').

    Recognises both glued codes ("COSC101", Khalifa) and spaced codes
    ("ISA 201", AUS/UNC/ISU). Bare numbers inherit the most recent subject
    (UNC "ECON 400 and 410"). Commas become AND; ', or' is normalised to ' or'
    upstream. Stray words (leftover note fragments) are ignored.
    """
    tokens: list[Any] = []
    last_subject: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        if ch == ",":
            tokens.append("AND")
            i += 1
            continue
        # Keyword 'and' / 'or' (whole word).
        kw = re.match(r"(and|or)\b", text[i:], flags=re.IGNORECASE)
        if kw:
            tokens.append("AND" if kw.group(1).lower() == "and" else "OR")
            i += len(kw.group(1))
            continue
        # Course code: subject letters + optional space(s) + number.
        cm = re.match(r"([A-Z]{2,5})\s*(\d{2,4}[A-Z]?)\b", text[i:])
        if cm:
            last_subject = cm.group(1)
            tokens.append(make_course_id(cm.group(1), cm.group(2), university))
            i += cm.end()
            continue
        # Bare number → inherit subject.
        bare = re.match(r"(\d{2,4}[A-Z]?)\b", text[i:])
        if bare and last_subject:
            tokens.append(make_course_id(last_subject, bare.group(1), university))
            i += bare.end()
            continue
        # Stray word/char → skip the run.
        skip = re.match(r"[^\s(),]+", text[i:])
        i += skip.end() if skip else 1
    return tokens


def _parse_tokens(tokens: list[Any]) -> Any:
    """Recursive-descent parse → nested tree of ('AND', [..]) / ('OR', [..]) / id."""
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def parse_expr():
        return parse_and()

    def parse_and():
        nonlocal pos
        groups = [parse_or()]
        while peek() == "AND":
            pos += 1
            if peek() in (None, ")"):
                break
            groups.append(parse_or())
        groups = [g for g in groups if g is not None]
        if len(groups) == 1:
            return groups[0]
        return ("AND", groups)

    def parse_or():
        nonlocal pos
        members = [parse_atom()]
        while peek() == "OR":
            pos += 1
            if peek() in (None, ")"):
                break
            members.append(parse_atom())
        members = [m for m in members if m is not None]
        if len(members) == 1:
            return members[0]
        return ("OR", members)

    def parse_atom():
        nonlocal pos
        tok = peek()
        if tok == "(":
            pos += 1
            inner = parse_expr()
            if peek() == ")":
                pos += 1
            return inner
        if tok in ("AND", "OR", ")", None):
            return None
        pos += 1  # a course_id leaf
        return tok

    tree = parse_expr()
    return tree


def _prune(tree: Any, scoped: set[str], counter: list[int]) -> Any:
    """Drop out-of-set leaves; count them. Returns pruned tree or None if empty."""
    if tree is None:
        return None
    if isinstance(tree, str):
        if tree in scoped:
            return tree
        counter[0] += 1
        return None
    kind, children = tree
    pruned = [_prune(c, scoped, counter) for c in children]
    pruned = [c for c in pruned if c is not None]
    if not pruned:
        return None
    if len(pruned) == 1:
        return pruned[0]
    return (kind, pruned)


def _flatten(tree: Any) -> list[list[str]]:
    """Flatten a pruned tree to AND-of-OR. Raise _NotFlattenable on OR-of-AND."""
    if tree is None:
        return []
    if isinstance(tree, str):
        return [[tree]]
    kind, children = tree
    if kind == "AND":
        groups: list[list[str]] = []
        for c in children:
            groups.extend(_flatten(c))
        return groups
    # OR node: every child must be a leaf (or an OR of leaves) to stay 2-level.
    or_members: list[str] = []
    for c in children:
        if isinstance(c, str):
            or_members.append(c)
        elif c[0] == "OR":
            for cc in c[1]:
                if not isinstance(cc, str):
                    raise _NotFlattenable()
                or_members.append(cc)
        else:  # AND inside OR → genuine OR-of-AND
            raise _NotFlattenable()
    # dedupe preserving order
    seen: set[str] = set()
    deduped = [m for m in or_members if not (m in seen or seen.add(m))]
    return [deduped]


def _preprocess(raw: str) -> str:
    """Shared text clean-up before note extraction and tokenisation."""
    text = raw.strip()
    # Strip a leading "Prerequisite(s)" label with ':' or ','.
    text = re.sub(r"^\s*Pre-?requisites?\s*[:,]\s*", "", text, flags=re.IGNORECASE)
    # Cross-listed code "MATH/ STOR 235" → "MATH 235 or STOR 235".
    text = re.sub(r"([A-Z]{2,5})\s*/\s*([A-Z]{2,5})\s*(\d{2,4}[A-Z]?)",
                  r"\1 \3 or \2 \3", text)
    # "any one of A, B, C or D" → make the listed commas OR (the comma-list is a
    # set of alternatives). Applied to the tail after the phrase.
    def _anyone(m):
        head, tail = m.group(1), m.group(2)
        return head + tail.replace(",", " or ")
    text = re.sub(r"(any one of\s+)([^.;]*)", _anyone, text, flags=re.IGNORECASE)
    text = re.sub(r"(one of the following(?:\s+courses)?\s*:?\s*)([^.;]*)", _anyone,
                  text, flags=re.IGNORECASE)
    # Absorb ', or' into the OR (comma is only a separator there).
    text = re.sub(r",\s*or\b", " or ", text)
    return text


def parse_prereq(raw: str, university: str, scoped_ids: set[str]) -> dict[str, Any]:
    """Parse one ``prereq_raw`` string.

    Returns a dict with:
      * ``prerequisites``  — AND-of-OR list of course_ids (possibly empty)
      * ``prereq_notes``   — list of non-course note strings (incl. coreqs)
      * ``n_out_of_set``   — count of course refs dropped as out-of-scope
      * ``unparsed``       — True if the course-logic could not be flattened
    """
    result = {"prerequisites": [], "prereq_notes": [], "n_out_of_set": 0,
              "unparsed": False}
    if not raw or not raw.strip():
        return result

    text = _preprocess(raw)
    text, coreq_notes = _extract_coreqs(text, university)
    text, notes = _extract_notes(text)
    result["prereq_notes"] = coreq_notes + notes

    # Only the parts of the text up to the first sentence break matter for the
    # hard-prereq logic; later sentences are description/header bleed.
    text = text.split(". ")[0]
    # Semicolon: keep clauses that are course requirements; the rest already
    # went to notes. Re-join remaining clauses with AND only if they contain a
    # code.
    clauses = [c for c in re.split(r";", text) if _CODE_RE.search(c)]
    text = " and ".join(clauses) if clauses else text

    tokens = _tokenise(text, university)
    if not any(isinstance(t, str) and t.startswith(tuple(PREFIX.values())) for t in tokens):
        return result  # no in-text course codes → genuinely no structured prereq

    tree = _parse_tokens(tokens)
    counter = [0]
    pruned = _prune(tree, scoped_ids, counter)
    result["n_out_of_set"] = counter[0]
    if pruned is None:
        return result
    try:
        flat = _flatten(pruned)
    except _NotFlattenable:
        result["unparsed"] = True
        return result
    # Drop empty groups and dedupe identical AND-groups. Duplicates arise when a
    # grade-condition clause re-lists the same courses (UNC "...; a grade of C
    # or better in ECON 400 and 410 is required"); the grade phrase is stripped
    # to notes but the residual course list would otherwise repeat the group.
    seen_groups: set[tuple[str, ...]] = set()
    deduped_groups: list[list[str]] = []
    for g in flat:
        if not g:
            continue
        key = tuple(g)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        deduped_groups.append(g)
    result["prerequisites"] = deduped_groups
    return result
