from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio_analysis import AudioData
from .sanskrit_rules import syllable_rules


MODEL_ID = "OpenVoiceOS/ai4bharat-indicconformer-sa-onnx"


@dataclass(frozen=True)
class AksharaIssue:
    shloka: int
    charan: str
    expected: str
    heard: str
    kind: str
    confidence: str


@dataclass(frozen=True)
class AksharaMatch:
    transcript: str
    similarity: float
    issues: tuple[AksharaIssue, ...]
    available: bool
    message: str = ""


def _normalise(text: str) -> str:
    """Normalise only for comparison; never alter the displayed canonical text."""
    value = unicodedata.normalize("NFC", str(text or ""))
    # LearnGeeta CSV contains explanatory parenthetical sounds such as (म्).
    # They belong to the rule layer, not the basic word/akshara identity check.
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[\d०-९।॥,.;:!?\-–—'\"“”‘’]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _units(text: str) -> list[str]:
    clean = _normalise(text)
    out: list[str] = []
    for word in clean.split():
        rules = syllable_rules(word)
        out.extend(rule.text for rule in rules if rule.text.strip())
    return out


def _align(expected: list[str], heard: list[str]) -> tuple[list[tuple[str, str, str]], float]:
    """Levenshtein alignment returning (operation, expected, heard)."""
    n, m = len(expected), len(heard)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if expected[i - 1] == heard[j - 1] else 1
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + sub)
    aligned: list[tuple[str, str, str]] = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (expected[i - 1] != heard[j - 1]):
            op = "match" if expected[i - 1] == heard[j - 1] else "substitute"
            aligned.append((op, expected[i - 1], heard[j - 1]))
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            aligned.append(("missing", expected[i - 1], ""))
            i -= 1
        else:
            aligned.append(("extra", "", heard[j - 1]))
            j -= 1
    aligned.reverse()
    similarity = 1.0 - cost[n][m] / max(1, n, m)
    return aligned, max(0.0, similarity)


@lru_cache(maxsize=1)
def _model():
    import onnx_asr

    cache_dir = Path.home() / ".cache" / "learngeeta" / "sanskrit_asr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return onnx_asr.load_model(MODEL_ID, str(cache_dir))


def transcribe(audio: AudioData) -> str:
    model = _model()
    samples = np.asarray(audio.samples, dtype=np.float32)
    result = model.recognize(samples, sample_rate=audio.sample_rate, channel="mean")
    if isinstance(result, str):
        return _normalise(result)
    if hasattr(result, "text"):
        return _normalise(str(result.text))
    return _normalise(str(result))


def compare_aksharas(
    audio: AudioData,
    charans: list[tuple[int, str, str]],
) -> AksharaMatch:
    """Compare a Sanskrit ASR decode with known selected text.

    Individual highlights are withheld when the whole decode is too dissimilar;
    in that case the result is treated as a selection/recognition mismatch.
    """
    try:
        transcript = transcribe(audio)
    except Exception as exc:
        return AksharaMatch("", 0.0, (), False, str(exc))

    expected_units: list[str] = []
    owners: list[tuple[int, str]] = []
    for shloka, charan, text in charans:
        units = _units(text)
        expected_units.extend(units)
        owners.extend([(shloka, charan)] * len(units))
    heard_units = _units(transcript)
    aligned, similarity = _align(expected_units, heard_units)
    if similarity < 0.58:
        return AksharaMatch(transcript, similarity, (), True, "SELECTION_OR_DECODE_MISMATCH")

    issues: list[AksharaIssue] = []
    expected_index = 0
    for op, expected, heard in aligned:
        if op == "extra":
            continue
        shloka, charan = owners[min(expected_index, len(owners) - 1)] if owners else (0, "")
        if op in {"substitute", "missing"}:
            issues.append(AksharaIssue(
                shloka=shloka,
                charan=charan,
                expected=expected,
                heard=heard,
                kind=op,
                # One decode is useful evidence, but not enough for a categorical verdict.
                confidence="Review",
            ))
        expected_index += 1
    return AksharaMatch(transcript, similarity, tuple(issues[:24]), True)
