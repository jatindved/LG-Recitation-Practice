from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATASET_REVISION = "f65d41bcb717e6887bf2b069e40ee672e6cf0076"
BASE_URL = "https://huggingface.co/datasets/jatindved/geeta-master-audios/resolve"


def trainer_audio_url(chapter: int) -> str:
    return f"{BASE_URL}/{DATASET_REVISION}/master_audios/CH{chapter:02d}.mp3"


def find_local_trainer_audio(chapter: int, search_dirs: list[Path]) -> Path | None:
    names = [f"CH{chapter:02d}.mp3", f"CH{chapter:02d}.wav", f"CH{chapter:02d}.m4a"]
    for directory in search_dirs:
        directory = Path(directory)
        for name in names:
            candidate = directory / name
            if candidate.exists() and candidate.stat().st_size > 100_000:
                return candidate
    return None


def cached_trainer_audio(chapter: int, cache_dir: Path, local_dirs: list[Path] | None = None) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = find_local_trainer_audio(chapter, list(local_dirs or []) + [cache_dir])
    if local is not None:
        return local
    target = cache_dir / f"CH{chapter:02d}.mp3"
    if target.exists() and target.stat().st_size > 100_000:
        return target
    partial = target.with_suffix(".part")
    req = urllib.request.Request(trainer_audio_url(chapter), headers={"User-Agent": "LearnGeeta/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    os.replace(partial, target)
    return target


@dataclass(frozen=True)
class TrainerSegment:
    chapter: int
    audio_file: str | None
    shloka: int
    charan: int | None
    start_sec: float
    end_sec: float
    confidence: float
    verified: bool
    status: str


class TrainerIndex:
    REQUIRED = {"chapter", "shloka", "start_sec", "end_sec", "confidence", "verified"}

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.frame = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame(columns=sorted(self.REQUIRED | {"audio_file", "charan"}))
        frame = pd.read_csv(self.csv_path)
        missing = self.REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"Trainer index is missing columns: {sorted(missing)}")
        if "audio_file" not in frame.columns:
            frame["audio_file"] = None
        if len(frame):
            frame["verified"] = frame["verified"].map(
                lambda value: str(value).strip().lower() in {"true", "1", "yes", "verified", "auto_verified"}
            )
        return frame

    def has_verified_shloka(self, chapter: int, shloka: int) -> bool:
        rows = self.frame[(self.frame.chapter == chapter) & (self.frame.shloka == shloka)]
        return bool(len(rows) and rows.verified.all())

    def shloka_segment(self, chapter: int, shloka: int) -> TrainerSegment | None:
        rows = self.frame[(self.frame.chapter == chapter) & (self.frame.shloka == shloka)]
        if "audio_file" in rows.columns:
            audio_rows = rows[rows.audio_file.notna()]
            if len(audio_rows):
                rows = audio_rows
        if "charan" in rows.columns:
            total = rows[rows.charan.isna()]
            if len(total):
                rows = total
        if len(rows) != 1:
            return None
        r = rows.iloc[0]
        audio_file = None if pd.isna(r.audio_file) else str(r.audio_file)
        return TrainerSegment(
            chapter,
            audio_file,
            shloka,
            None,
            float(r.start_sec),
            float(r.end_sec),
            float(r.confidence),
            bool(r.verified),
            str(r.status) if "status" in rows.columns and not pd.isna(r.status) else "",
        )
