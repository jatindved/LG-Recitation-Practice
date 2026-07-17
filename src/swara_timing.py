from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .geeta_data import Verse
from .sanskrit_rules import syllable_rules
from .trainer_audio import TrainerIndex


@dataclass(frozen=True)
class AksharaTiming:
    chapter: int
    shloka: int
    charan: str
    index: int
    akshara: str
    vowel: str
    length: str
    weight: str
    reason: str
    expected_sec: float
    start_sec: float
    end_sec: float


def _weight(length: str, weight: str) -> float:
    if length == "pluta":
        return 3.0
    if weight == "guru":
        return 2.0
    return 1.0


def akshara_timing_for_verse(index: TrainerIndex, verse: Verse, charan_texts: dict[int, str] | None = None) -> list[AksharaTiming]:
    frame = index.frame
    rows = frame[(frame.chapter == verse.chapter) & (frame.shloka == verse.shloka)].copy()
    if rows.empty or "charan" not in rows.columns:
        return []
    rows["charan_num"] = pd.to_numeric(rows["charan"], errors="coerce")
    rows = rows[rows["charan_num"].notna()].sort_values("charan_num")
    rows = rows[pd.to_numeric(rows["end_sec"], errors="coerce") > pd.to_numeric(rows["start_sec"], errors="coerce")]
    timings: list[AksharaTiming] = []
    for _, row in rows.iterrows():
        cnum = int(row.charan_num)
        charan_text = (charan_texts or {}).get(cnum)
        if not charan_text:
            if cnum < 1 or cnum > len(verse.charans):
                continue
            charan_text = verse.charans[cnum - 1]
        if not charan_text:
            continue
        rules = syllable_rules(charan_text)
        if not rules:
            continue
        weights = [_weight(rule.vowel_length, rule.weight) for rule in rules]
        total = sum(weights) or len(weights)
        start = float(row.start_sec)
        end = float(row.end_sec)
        duration = max(0.01, end - start)
        cursor = start
        label = "ABCD"[cnum - 1] if cnum <= 4 else str(cnum)
        for pos, (rule, wt) in enumerate(zip(rules, weights), start=1):
            expected = duration * wt / total
            timings.append(AksharaTiming(
                chapter=verse.chapter,
                shloka=verse.shloka,
                charan=label,
                index=pos,
                akshara=rule.text,
                vowel=rule.vowel,
                length=rule.vowel_length,
                weight=rule.weight,
                reason=rule.reason,
                expected_sec=round(expected, 3),
                start_sec=round(cursor, 3),
                end_sec=round(cursor + expected, 3),
            ))
            cursor += expected
    return timings


def akshara_timing_table(index: TrainerIndex, verses: list[Verse], recitation_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for verse in verses:
        charan_texts: dict[int, str] = {}
        if recitation_frame is not None and not recitation_frame.empty:
            subset = recitation_frame[
                (recitation_frame["Chapter"] == verse.chapter)
                & (recitation_frame["Shloka"] == verse.shloka)
                & (recitation_frame["Text_Type"].astype(str).str.upper() == "SHLOKA")
            ].copy()
            mapping = {"A": 1, "B": 2, "C": 3, "D": 4}
            for _, r in subset.iterrows():
                cnum = mapping.get(str(r.get("Charan", "")).strip().upper())
                text = str(r.get("Expected_Text_From_CSV", "") or r.get("Transcript_Devanagari", "")).strip()
                if cnum and text:
                    charan_texts[cnum] = text
        for item in akshara_timing_for_verse(index, verse, charan_texts):
            rows.append({
                "Chapter": item.chapter,
                "Shloka": item.shloka,
                "Charan": item.charan,
                "No.": item.index,
                "Akshara": item.akshara,
                "Vowel": item.vowel,
                "Length": item.length,
                "Weight": item.weight,
                "Expected_sec": item.expected_sec,
                "Start_sec": item.start_sec,
                "End_sec": item.end_sec,
                "Rule": item.reason,
            })
    return pd.DataFrame(rows)
