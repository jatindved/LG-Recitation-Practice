from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html

import numpy as np
import pandas as pd

from .audio_analysis import AudioData, TARGET_SR
from .geeta_data import Verse
from .sanskrit_rules import syllable_rules
from .trainer_audio import TrainerIndex


@dataclass(frozen=True)
class SwaraAudioIssue:
    shloka: int
    charan: str
    akshara: str
    length: str
    weight: str
    expected_sec: float
    observed_sec: float
    trainer_start_sec: float
    trainer_end_sec: float
    ratio: float
    issue: str
    confidence: str


def issue_key(shloka: int, charan: str, akshara: str) -> str:
    return f"{shloka}|{charan}|{akshara}"


def _load_trainer(path: Path, start: float, end: float) -> AudioData:
    import librosa
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True, offset=max(0.0, start), duration=max(0.05, end - start))
    y = np.asarray(y, dtype=np.float32)
    return AudioData(y, TARGET_SR, len(y) / TARGET_SR)


def _mfcc(audio: AudioData):
    import librosa
    hop = 256
    y = audio.samples
    if len(y) < 512:
        return np.empty((20, 0)), np.array([])
    mfcc = librosa.feature.mfcc(y=y, sr=audio.sample_rate, n_mfcc=20, hop_length=hop)
    mfcc = librosa.util.normalize(mfcc, axis=1)
    times = librosa.times_like(mfcc, sr=audio.sample_rate, hop_length=hop)
    return mfcc, times


def _dtw_time_map(trainer: AudioData, student: AudioData):
    import librosa
    x, tx = _mfcc(trainer)
    y, ty = _mfcc(student)
    if x.shape[1] < 3 or y.shape[1] < 3:
        return None
    _, wp = librosa.sequence.dtw(X=x, Y=y, metric="cosine")
    # wp rows are [trainer_frame, student_frame], usually reverse order.
    pairs = sorted((int(a), int(b)) for a, b in wp)
    mapping: dict[int, list[int]] = {}
    for a, b in pairs:
        mapping.setdefault(a, []).append(b)
    return mapping, tx, ty


def _charan_rows(index: TrainerIndex, verse: Verse) -> pd.DataFrame:
    frame = index.frame
    rows = frame[(frame.chapter == verse.chapter) & (frame.shloka == verse.shloka)].copy()
    if rows.empty or "charan" not in rows.columns:
        return pd.DataFrame()
    rows["charan_num"] = pd.to_numeric(rows["charan"], errors="coerce")
    rows = rows[rows["charan_num"].notna()].copy()
    rows = rows[pd.to_numeric(rows["end_sec"], errors="coerce") > pd.to_numeric(rows["start_sec"], errors="coerce")]
    return rows.sort_values("charan_num")


def _akshara_intervals(verse: Verse, charan_start: float, charan_end: float, charan_num: int):
    if charan_num < 1 or charan_num > len(verse.charans):
        return []
    rules = syllable_rules(verse.charans[charan_num - 1])
    if not rules:
        return []
    weights = []
    for rule in rules:
        if rule.vowel_length == "pluta":
            weights.append(3.0)
        elif rule.weight == "guru":
            weights.append(2.0)
        else:
            weights.append(1.0)
    total_weight = sum(weights) or len(weights)
    dur = max(0.01, charan_end - charan_start)
    out = []
    cursor = charan_start
    label = "ABCD"[charan_num - 1] if 1 <= charan_num <= 4 else str(charan_num)
    for rule, weight in zip(rules, weights):
        length = dur * weight / total_weight
        out.append((cursor, cursor + length, label, rule))
        cursor += length
    return out


def _charan_text_map(frame: pd.DataFrame | None, verse: Verse) -> dict[int, str]:
    if frame is None or frame.empty:
        return {}
    subset = frame[
        (frame["Chapter"] == verse.chapter)
        & (frame["Shloka"] == verse.shloka)
        & (frame["Text_Type"].astype(str).str.upper() == "SHLOKA")
    ].copy()
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4}
    out: dict[int, str] = {}
    for _, r in subset.iterrows():
        cnum = mapping.get(str(r.get("Charan", "")).strip().upper())
        text = str(r.get("Expected_Text_From_CSV", "") or r.get("Transcript_Devanagari", "")).strip()
        if cnum and text:
            out[cnum] = text
    return out


def _akshara_intervals_from_text(verse: Verse, charan_start: float, charan_end: float, charan_num: int, charan_text: str | None):
    if not charan_text:
        return _akshara_intervals(verse, charan_start, charan_end, charan_num)
    rules = syllable_rules(charan_text)
    if not rules:
        return []
    weights = []
    for rule in rules:
        if rule.vowel_length == "pluta":
            weights.append(3.0)
        elif rule.weight == "guru":
            weights.append(2.0)
        else:
            weights.append(1.0)
    total_weight = sum(weights) or len(weights)
    dur = max(0.01, charan_end - charan_start)
    out = []
    cursor = charan_start
    label = "ABCD"[charan_num - 1] if 1 <= charan_num <= 4 else str(charan_num)
    for rule, weight in zip(rules, weights):
        length = dur * weight / total_weight
        out.append((cursor, cursor + length, label, rule))
        cursor += length
    return out


def swara_audio_issues(
    student: AudioData,
    trainer_path: Path,
    index: TrainerIndex,
    verses: list[Verse],
    segment_start: float,
    segment_end: float,
    recitation_frame: pd.DataFrame | None = None,
) -> list[SwaraAudioIssue]:
    trainer = _load_trainer(trainer_path, segment_start, segment_end)
    mapped = _dtw_time_map(trainer, student)
    if mapped is None:
        return []
    mapping, trainer_times, student_times = mapped
    global_ratio = max(0.2, min(3.0, student.duration_sec / max(trainer.duration_sec, 0.01)))
    issues: list[SwaraAudioIssue] = []

    for verse in verses:
        charan_texts = _charan_text_map(recitation_frame, verse)
        for _, row in _charan_rows(index, verse).iterrows():
            cnum = int(row.charan_num)
            cstart = float(row.start_sec)
            cend = float(row.end_sec)
            intervals = _akshara_intervals_from_text(verse, cstart, cend, cnum, charan_texts.get(cnum))
            for pos, (abs_start, abs_end, label, rule) in enumerate(intervals):
                is_last_in_charan = pos == len(intervals) - 1
                rel_start = abs_start - segment_start
                rel_end = abs_end - segment_start
                trainer_frame_idx = np.where((trainer_times >= rel_start) & (trainer_times <= rel_end))[0]
                student_idx: list[int] = []
                for idx in trainer_frame_idx:
                    student_idx.extend(mapping.get(int(idx), []))
                if len(student_idx) < 2:
                    if is_last_in_charan or rel_end >= trainer.duration_sec - 0.75:
                        expected = max(0.01, float((abs_end - abs_start) * global_ratio))
                        issues.append(SwaraAudioIssue(
                            shloka=verse.shloka,
                            charan=label,
                            akshara=rule.text,
                            length=rule.vowel_length,
                            weight=rule.weight,
                            expected_sec=round(expected, 2),
                            observed_sec=0.0,
                            trainer_start_sec=round(abs_start, 3),
                            trainer_end_sec=round(abs_end, 3),
                            ratio=0.0,
                            issue="Akshara may be skipped or cut off",
                            confidence="Low",
                        ))
                    continue
                st = student_times[min(student_idx)]
                en = student_times[max(student_idx)]
                observed = max(0.01, float(en - st))
                expected = max(0.01, float((abs_end - abs_start) * global_ratio))
                ratio = observed / expected
                issue = ""
                if rule.vowel_length == "hrasva" and ratio > 1.55:
                    issue = "Hrasva likely elongated"
                elif rule.vowel_length in {"dirgha", "pluta"} and ratio < 0.62:
                    issue = "Long vowel likely shortened"
                elif rule.weight == "guru" and ratio < 0.62:
                    issue = "Guru akshara likely too short"
                elif rule.weight == "laghu" and ratio > 1.75:
                    issue = "Laghu akshara likely too long"
                elif is_last_in_charan and ratio < 0.48:
                    issue = "Final akshara may be swallowed"
                if issue:
                    confidence = "Medium" if len(set(student_idx)) >= 4 else "Low"
                    issues.append(SwaraAudioIssue(
                        shloka=verse.shloka,
                        charan=label,
                        akshara=rule.text,
                        length=rule.vowel_length,
                        weight=rule.weight,
                        expected_sec=round(expected, 2),
                        observed_sec=round(observed, 2),
                        trainer_start_sec=round(abs_start, 3),
                        trainer_end_sec=round(abs_end, 3),
                        ratio=round(ratio, 2),
                        issue=issue,
                        confidence=confidence,
                    ))
    return issues[:25]


def issues_to_keys(issues: list[SwaraAudioIssue]) -> set[str]:
    return {issue_key(i.shloka, i.charan, i.akshara) for i in issues}
