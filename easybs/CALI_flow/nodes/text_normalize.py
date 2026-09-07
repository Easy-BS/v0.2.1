# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:07:04 2026

@author: Xiguan Liang @SKKU
"""


# ./CALI_flow/nodes/text_normalize.py



from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MAX_INPUT_CHARS = 8000
MIN_INPUT_CHARS = 3
MAX_TARGETS = 50

_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ", "\u3000": " ",
}

_INVISIBLE_RE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_FLOOR_TOKEN_RE = re.compile(r"_f\d+$")

ALLOWED_INTENTS = ("add_rfh", "ask_clarification", "unknown")


# --------------------------------------------------------------------------
# N1..N6  Text normalization
# --------------------------------------------------------------------------

def normalize_input(text: Optional[str]) -> Tuple[str, List[str]]:
    """Apply N1..N6 to raw user input.

    Returns (normalized_text, applied_rules). applied_rules lists the rules
    that actually changed the text, so the transformation is auditable.
    """
    applied: List[str] = []
    s = text if isinstance(text, str) else ""

    # N1  Unicode NFKC. Folds full-width Latin and digits, which CJK
    #     keyboards emit by default, plus ligatures and compatibility forms.
    t = unicodedata.normalize("NFKC", s)
    if t != s:
        applied.append("N1")
    s = t

    # N2  Remove zero-width, bidi-control and C0 control characters,
    #     preserving newline and tab.
    t = _INVISIBLE_RE.sub("", s)
    if t != s:
        applied.append("N2")
    s = t

    # N3  Map typographic punctuation to ASCII equivalents.
    t = s
    for src, dst in _PUNCT_MAP.items():
        t = t.replace(src, dst)
    if t != s:
        applied.append("N3")
    s = t

    # N4  Collapse horizontal whitespace to one space and 3+ newlines to 2.
    #     Newlines are preserved because the calibration prompt carries
    #     monthly values line by line.
    t = _NL_RE.sub("\n\n", _WS_RE.sub(" ", s))
    if t != s:
        applied.append("N4")
    s = t

    # N5  Strip whitespace from the whole string and from each line.
    t = "\n".join(line.strip() for line in s.split("\n")).strip()
    if t != s:
        applied.append("N5")
    s = t

    # N6  Truncate to MAX_INPUT_CHARS on a whitespace boundary.
    if len(s) > MAX_INPUT_CHARS:
        cut = s.rfind(" ", 0, MAX_INPUT_CHARS)
        s = s[: cut if cut > 0 else MAX_INPUT_CHARS]
        applied.append("N6")

    return s, applied


# --------------------------------------------------------------------------
# L1..L4  Label canonicalization
# --------------------------------------------------------------------------

def canonical_label(label: Optional[str]) -> str:
    """Apply L1..L4 so user-supplied and IDF-derived names compare equally.

    L1 NFKC and invisible-character removal.
    L2 Case folding.
    L3 Spaces, hyphens and dots mapped to underscore.
    L4 Repeated underscores collapsed, leading/trailing underscores dropped.
    """
    s = unicodedata.normalize("NFKC", label if isinstance(label, str) else "")
    s = _INVISIBLE_RE.sub("", s)
    s = s.casefold()
    s = re.sub(r"[\s\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# --------------------------------------------------------------------------
# V1..V5  Validation
# --------------------------------------------------------------------------

def validate_input_text(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """V1, V2. Returns (outcome, code, message)."""
    if not text:
        return "reject", "V1", "No user input received."
    if len(text) < MIN_INPUT_CHARS:
        return "reject", "V1", "Input is too short to interpret."
    return "accept", None, None


def validate_agent_json(agent: object) -> Tuple[str, Optional[str], Optional[str]]:
    """V3, V4. Structural validation of the LLM output."""
    if not isinstance(agent, dict):
        return "reject", "V3", "Agent output is not a JSON object."

    intent = agent.get("intent")
    if intent not in ALLOWED_INTENTS:
        return "reject", "V3", f"Unrecognized intent: {intent!r}."

    targets = agent.get("rfh_targets", [])
    if targets is not None and not isinstance(targets, list):
        return "reject", "V4", "'rfh_targets' must be a list."
    if isinstance(targets, list) and not all(
        isinstance(x, (str, int, float)) for x in targets
    ):
        return "reject", "V4", "'rfh_targets' must contain scalar labels only."

    cq = agent.get("clarification_question", "")
    if cq is not None and not isinstance(cq, str):
        return "reject", "V4", "'clarification_question' must be a string."

    return "accept", None, None


def clean_targets(raw: Iterable) -> Tuple[List[str], List[str]]:
    """V5. Normalize, drop empties, de-duplicate, preserve order.

    Returns (kept_labels, dropped_raw_values).
    """
    kept: List[str] = []
    dropped: List[str] = []
    seen = set()
    for item in raw or []:
        label = str(item).strip()
        key = canonical_label(label)
        if not key:
            dropped.append(str(item))
            continue
        if key in seen:
            dropped.append(label)
            continue
        seen.add(key)
        kept.append(label)
        if len(kept) >= MAX_TARGETS:
            break
    return kept, dropped


# --------------------------------------------------------------------------
# R1..R4  Zone resolution
# --------------------------------------------------------------------------

class ZoneResolution:
    """Outcome of resolving one user label against the zones in an IDF."""

    __slots__ = ("label", "status", "zone_name", "candidates", "rule")

    def __init__(self, label: str, status: str, zone_name: Optional[str] = None,
                 candidates: Optional[List[str]] = None, rule: str = ""):
        self.label = label
        self.status = status              # resolved | ambiguous | unresolved
        self.zone_name = zone_name
        self.candidates = candidates or []
        self.rule = rule

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "status": self.status,
            "zone_name": self.zone_name,
            "candidates": self.candidates,
            "rule": self.rule,
        }

    def __repr__(self) -> str:
        return f"<ZoneResolution {self.label!r} {self.status} {self.zone_name!r}>"


def resolve_zone_label(label: str, zone_map: Dict[str, str]) -> ZoneResolution:
    """Apply R1..R4.

    R1 Exact match on the canonical key.
    R2 Unique prefix match.
    R3 Multiple prefix matches -> ambiguous, candidates returned so the
       caller can ask for clarification. Prevents 'Room_1' silently
       binding to 'Room_10'.
    R4 No match -> unresolved, available zones returned.
    """
    key = canonical_label(label)
    if not key:
        return ZoneResolution(label, "unresolved", rule="R4",
                              candidates=sorted(set(zone_map.values())))

    if key in zone_map:
        return ZoneResolution(label, "resolved", zone_map[key], rule="R1")

    hits = sorted(k for k in zone_map if k.startswith(key))
    if len(hits) == 1:
        return ZoneResolution(label, "resolved", zone_map[hits[0]], rule="R2")
    if len(hits) > 1:
        resolved = sorted({zone_map[h] for h in hits})
        if len(resolved) == 1:
            return ZoneResolution(label, "resolved", resolved[0], rule="R2")
        return ZoneResolution(label, "ambiguous", None,
                              candidates=resolved, rule="R3")

    return ZoneResolution(label, "unresolved", None,
                          candidates=sorted(set(zone_map.values())), rule="R4")


def resolve_all(labels: Iterable[str], zone_map: Dict[str, str]) -> List[ZoneResolution]:
    return [resolve_zone_label(lbl, zone_map) for lbl in (labels or [])]


def build_canonical_zone_map(
    zone_names: Iterable[str],
    strip_prefixes: Iterable[str] = ("block",),
    strip_suffixes: Iterable[str] = ("storey_0",),
) -> Dict[str, str]:
    """Build canonical key -> Zone.Name.

    For a zone named 'Block Room_1_F1 Storey 0' these keys are registered:

        block_room_1_f1_storey_0    full canonical form
        room_1_f1_storey_0          prefix stripped
        room_1_f1                   suffix stripped
        room_1                      floor token stripped

    The floor-token key matters: without it, 'Room_1' would be a prefix of
    both 'room_1_f1' and 'room_10_f1' and would be reported as ambiguous.

    setdefault() is used so a shorter key never overwrites an earlier
    binding. A label that matches no key is reported unresolved (R4)
    rather than guessed.
    """
    mapping: Dict[str, str] = {}
    for name in zone_names:
        full = canonical_label(name)
        keys = [full]

        trimmed = full
        for p in strip_prefixes:
            if trimmed.startswith(p + "_"):
                trimmed = trimmed[len(p) + 1:]
        if trimmed and trimmed not in keys:
            keys.append(trimmed)

        for s in strip_suffixes:
            if trimmed.endswith("_" + s):
                trimmed = trimmed[: -(len(s) + 1)]
        if trimmed and trimmed not in keys:
            keys.append(trimmed)

        no_floor = _FLOOR_TOKEN_RE.sub("", trimmed)
        if no_floor and no_floor not in keys:
            keys.append(no_floor)

        for k in keys:
            mapping.setdefault(k, name)
    return mapping