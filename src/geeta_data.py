from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SPEAKER_SUFFIX = "उवाच"


@dataclass(frozen=True)
class Verse:
    chapter: int
    shloka: int
    speaker: str
    charans: tuple[str, ...]
    canonical_text: str
    iast_text: str


def _int_key(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _clean_lines(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return [re.sub(r"\s+", " ", line).strip() for line in str(text).splitlines() if line.strip()]


def _strip_verse_number(text: str) -> str:
    return re.sub(r"॥\s*\d+(?:\.\d+)?\s*॥\s*$", "॥", text).strip()


def _parse_text(text: object) -> tuple[str, tuple[str, ...], str]:
    lines = _clean_lines(text)
    speaker = ""
    if lines and lines[0].endswith(SPEAKER_SUFFIX):
        speaker = lines.pop(0)
    cleaned = tuple(_strip_verse_number(line) for line in lines)
    canonical = "\n".join(([speaker] if speaker else []) + list(cleaned))
    return speaker, cleaned, canonical


class GeetaRepository:
    def __init__(self, hindi_csv: Path, english_csv: Path, aghata_csv: Path):
        self.hindi_csv = Path(hindi_csv)
        self.english_csv = Path(english_csv)
        self.aghata_csv = Path(aghata_csv)
        self._verses = self._load_verses()
        self._aghata = self._load_aghata()

    def _load_verses(self) -> dict[tuple[int, int], Verse]:
        hin = pd.read_csv(self.hindi_csv)
        eng = pd.read_csv(self.english_csv)
        eng_map: dict[tuple[int, int], str] = {}
        for _, row in eng.iterrows():
            ch, sh = _int_key(row.iloc[0]), _int_key(row.iloc[1])
            if ch is not None and sh is not None:
                eng_map[(ch, sh)] = str(row.iloc[2]) if not pd.isna(row.iloc[2]) else ""

        result: dict[tuple[int, int], Verse] = {}
        for _, row in hin.iterrows():
            ch, sh = _int_key(row.iloc[0]), _int_key(row.iloc[1])
            if ch is None or sh is None or sh == 0:
                continue
            speaker, charans, canonical = _parse_text(row.iloc[2])
            _, _, iast = _parse_text(eng_map.get((ch, sh), ""))
            result[(ch, sh)] = Verse(ch, sh, speaker, charans, canonical, iast)
        return result

    def _load_aghata(self) -> pd.DataFrame:
        frame = pd.read_csv(self.aghata_csv, low_memory=False)
        # Only mapped shloka marks are relevant to the learner report. Other official text
        # types remain in the source master and are never silently deleted.
        mask = frame["Text Type"].astype(str).str.upper().eq("SHLOKA")
        mapped = frame.loc[mask].copy()
        mapped["Chapter"] = pd.to_numeric(mapped["Chapter"], errors="coerce").astype("Int64")
        mapped["Shloka Number"] = pd.to_numeric(mapped["Shloka Number"], errors="coerce").astype("Int64")
        if "Review Status" in mapped.columns:
            bad_status = {"DECORATION_REVIEW", "REJECT", "REJECTED", "FALSE_POSITIVE"}
            mapped = mapped[~mapped["Review Status"].astype(str).str.upper().isin(bad_status)].copy()
        return mapped

    def chapters(self) -> list[int]:
        return sorted({ch for ch, _ in self._verses})

    def shlokas(self, chapter: int) -> list[int]:
        return sorted(sh for ch, sh in self._verses if ch == chapter)

    def verse(self, chapter: int, shloka: int) -> Verse | None:
        return self._verses.get((chapter, shloka))

    def verses(self, chapter: int, start: int, end: int) -> list[Verse]:
        return [v for n in range(start, end + 1) if (v := self.verse(chapter, n)) is not None]

    def aghata_for(self, chapter: int, shloka: int) -> pd.DataFrame:
        frame = self._aghata
        return frame[(frame["Chapter"] == chapter) & (frame["Shloka Number"] == shloka)].copy()

    def anusvara_for(self, chapter: int, shloka: int) -> pd.DataFrame:
        frame = self._aghata
        rows = frame[(frame["Chapter"] == chapter) & (frame["Shloka Number"] == shloka)].copy()
        if rows.empty:
            return rows
        present = pd.Series(False, index=rows.index)
        for column in ["Anusvara Present", "Anunasika Present"]:
            if column in rows.columns:
                present = present | rows[column].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "present"})
        for column in ["Anusvara Guide Output", "Anusvara Following Sound", "Anusvara Position"]:
            if column in rows.columns:
                present = present | rows[column].fillna("").astype(str).str.strip().ne("")
        return rows[present].copy()
