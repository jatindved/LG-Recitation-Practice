from __future__ import annotations

import re
from dataclasses import dataclass


INDEPENDENT_VOWELS = {
    "अ": ("a", "hrasva"),
    "इ": ("i", "hrasva"),
    "उ": ("u", "hrasva"),
    "ऋ": ("r", "hrasva"),
    "ऌ": ("l", "hrasva"),
    "आ": ("ā", "dirgha"),
    "ई": ("ī", "dirgha"),
    "ऊ": ("ū", "dirgha"),
    "ॠ": ("ṝ", "dirgha"),
    "ए": ("e", "dirgha"),
    "ऐ": ("ai", "dirgha"),
    "ओ": ("o", "dirgha"),
    "औ": ("au", "dirgha"),
}

VOWEL_SIGNS = {
    "": ("a", "hrasva"),
    "ि": ("i", "hrasva"),
    "ु": ("u", "hrasva"),
    "ृ": ("r", "hrasva"),
    "ॢ": ("l", "hrasva"),
    "ा": ("ā", "dirgha"),
    "ी": ("ī", "dirgha"),
    "ू": ("ū", "dirgha"),
    "ॄ": ("ṝ", "dirgha"),
    "े": ("e", "dirgha"),
    "ै": ("ai", "dirgha"),
    "ो": ("o", "dirgha"),
    "ौ": ("au", "dirgha"),
}

CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
VIRAMA = "्"
ANUSVARA = "ं"
CHANDRABINDU = "ँ"
VISARGA = "ः"
DEVANAGARI_DIGITS = set("०१२३४५६७८९")


@dataclass(frozen=True)
class SyllableRule:
    text: str
    vowel: str
    vowel_length: str
    weight: str
    reason: str


def _clean_text(text: str) -> str:
    text = re.sub(r"[।॥,;:()\[\]{}\"'“”‘’\-–—]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def syllable_rules(text: str) -> list[SyllableRule]:
    """Best-effort Sanskrit orthographic akshara rules for learner guidance.

    This is a text-rule engine, not an audio judgement. It identifies the
    visible vowel length and laghu/guru weight in Devanagari text.
    """
    text = _clean_text(text)
    result: list[SyllableRule] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue

        if ch in INDEPENDENT_VOWELS:
            start = i
            vowel, length = INDEPENDENT_VOWELS[ch]
            i += 1
            marks = ""
            while i < len(text) and text[i] in {ANUSVARA, CHANDRABINDU, VISARGA} | DEVANAGARI_DIGITS:
                marks += text[i]
                i += 1
            syll = text[start:i]
            weight, reason = _weight(length, marks, next_text=text[i:i + 4])
            result.append(SyllableRule(syll, vowel, _pluta_if_marked(length, marks), weight, reason))
            continue

        if ch in CONSONANTS:
            start = i
            i += 1
            while i + 1 < len(text) and text[i] == VIRAMA and text[i + 1] in CONSONANTS:
                i += 2
            sign = ""
            if i < len(text) and text[i] in VOWEL_SIGNS and text[i] != "":
                sign = text[i]
                i += 1
            elif i < len(text) and text[i] == VIRAMA:
                i += 1
                continue
            vowel, length = VOWEL_SIGNS.get(sign, VOWEL_SIGNS[""])
            marks = ""
            while i < len(text) and text[i] in {ANUSVARA, CHANDRABINDU, VISARGA} | DEVANAGARI_DIGITS:
                marks += text[i]
                i += 1
            syll = text[start:i]
            weight, reason = _weight(length, marks, next_text=text[i:i + 4])
            result.append(SyllableRule(syll, vowel, _pluta_if_marked(length, marks), weight, reason))
            continue

        i += 1
    return result


def _pluta_if_marked(length: str, marks: str) -> str:
    if any(ch in DEVANAGARI_DIGITS or ch == "3" for ch in marks):
        return "pluta"
    return length


def _weight(length: str, marks: str, next_text: str) -> tuple[str, str]:
    if any(ch in DEVANAGARI_DIGITS or ch == "3" for ch in marks):
        return "guru", "pluta vowel"
    if length == "dirgha":
        return "guru", "long vowel"
    if ANUSVARA in marks or CHANDRABINDU in marks:
        return "guru", "followed by nasal mark"
    if VISARGA in marks:
        return "guru", "followed by visarga"
    if _next_starts_with_conjunct(next_text):
        return "guru", "short vowel followed by conjunct consonants"
    return "laghu", "short vowel"


def _next_starts_with_conjunct(text: str) -> bool:
    text = text.lstrip()
    return len(text) >= 3 and text[0] in CONSONANTS and text[1] == VIRAMA and text[2] in CONSONANTS
