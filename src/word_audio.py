from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .audio_analysis import AudioData
from .geeta_data import Verse
from .swara_audio import _dtw_time_map, _load_trainer


@dataclass(frozen=True)
class WordAudioIssue:
    shloka: int
    charan: str
    word: str
    expected_sec: float
    observed_sec: float
    distance: float
    onset_distance: float
    issue: str
    confidence: str


def word_issue_key(shloka: int, charan: str, word: str) -> str:
    return f"{shloka}|{charan}|{word}"


def _clean_word(value: str) -> str:
    text = str(value or "")
    for ch in "()[]{}<>.,;:!?।॥'\"“”‘’ \t\r\n":
        text = text.replace(ch, "")
    return text


def _word_windows(recitation_frame: pd.DataFrame, verse: Verse) -> list[tuple[int, str, str, float, float]]:
    if recitation_frame.empty:
        return []
    rows = recitation_frame[
        (pd.to_numeric(recitation_frame["Chapter"], errors="coerce") == verse.chapter)
        & (pd.to_numeric(recitation_frame["Shloka"], errors="coerce") == verse.shloka)
        & (recitation_frame["Text_Type"].astype(str).str.upper() == "SHLOKA")
    ].copy()
    out: list[tuple[int, str, str, float, float]] = []
    labels = {"A": 1, "B": 2, "C": 3, "D": 4}
    for _, row in rows.sort_values("Charan").iterrows():
        label = str(row.get("Charan", "")).strip().upper()
        cnum = labels.get(label, 0)
        start = float(pd.to_numeric(row.get("Start_sec", None), errors="coerce"))
        end = float(pd.to_numeric(row.get("End_sec", None), errors="coerce"))
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        words = [w.strip() for w in str(row.get("Words", "")).split(",") if w.strip() and w.strip() != "-"]
        if not words:
            text = str(row.get("Expected_Text_From_CSV", "") or row.get("Transcript_Devanagari", ""))
            words = [w.strip() for w in text.split() if w.strip()]
        clean_words = [_clean_word(w) for w in words if _clean_word(w)]
        if not clean_words:
            continue
        weights = [max(1, len(w)) for w in clean_words]
        total = float(sum(weights)) or 1.0
        cursor = start
        for word, weight in zip(clean_words, weights):
            dur = (end - start) * weight / total
            out.append((cnum, label, word, cursor, cursor + dur))
            cursor += dur
    return out


def _mfcc_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float | None:
    import librosa

    if len(a) < sr * 0.08 or len(b) < sr * 0.08:
        return None
    hop = 160
    xa = librosa.feature.mfcc(y=a.astype(np.float32), sr=sr, n_mfcc=13, hop_length=hop)
    xb = librosa.feature.mfcc(y=b.astype(np.float32), sr=sr, n_mfcc=13, hop_length=hop)
    if xa.shape[1] < 2 or xb.shape[1] < 2:
        return None
    xa = librosa.util.normalize(xa, axis=1)
    xb = librosa.util.normalize(xb, axis=1)
    cost, _ = librosa.sequence.dtw(X=xa, Y=xb, metric="cosine")
    return float(cost[-1, -1] / max(1, xa.shape[1] + xb.shape[1]))


def _slice_rel(y: np.ndarray, sr: int, start_ratio: float, end_ratio: float) -> np.ndarray:
    if y.size == 0:
        return y
    s = max(0, min(len(y), int(len(y) * start_ratio)))
    e = max(s + int(sr * 0.06), min(len(y), int(len(y) * end_ratio)))
    return y[s:e]


def word_audio_issues(
    student: AudioData,
    trainer_path: Path,
    verses: list[Verse],
    segment_start: float,
    segment_end: float,
    recitation_frame: pd.DataFrame,
) -> list[WordAudioIssue]:
    trainer = _load_trainer(trainer_path, segment_start, segment_end)
    mapped = _dtw_time_map(trainer, student)
    if mapped is None:
        return []
    mapping, trainer_times, student_times = mapped
    issues: list[WordAudioIssue] = []

    for verse in verses:
        for _, label, word, abs_start, abs_end in _word_windows(recitation_frame, verse):
            rel_start = abs_start - segment_start
            rel_end = abs_end - segment_start
            trainer_idx = np.where((trainer_times >= rel_start) & (trainer_times <= rel_end))[0]
            student_idx: list[int] = []
            for idx in trainer_idx:
                student_idx.extend(mapping.get(int(idx), []))
            if len(student_idx) < 3:
                continue
            t0 = max(0, int(rel_start * trainer.sample_rate))
            t1 = min(len(trainer.samples), int(rel_end * trainer.sample_rate))
            s0 = max(0, int(student_times[min(student_idx)] * student.sample_rate))
            s1 = min(len(student.samples), int(student_times[max(student_idx)] * student.sample_rate))
            expected = max(0.01, (t1 - t0) / trainer.sample_rate)
            observed = max(0.01, (s1 - s0) / student.sample_rate)
            dist = _mfcc_distance(trainer.samples[t0:t1], student.samples[s0:s1], trainer.sample_rate)
            if dist is None:
                continue
            trainer_word = trainer.samples[t0:t1]
            student_word = student.samples[s0:s1]
            onset_dist = _mfcc_distance(
                _slice_rel(trainer_word, trainer.sample_rate, 0.0, 0.42),
                _slice_rel(student_word, student.sample_rate, 0.0, 0.42),
                trainer.sample_rate,
            )
            onset_value = float(onset_dist if onset_dist is not None else dist)
            # Give extra weight to the beginning of the word, where पाण्डव→काण्डव and
            # परन्तप→करन्तप type mistakes occur.
            combined = max(dist, onset_value * 1.18)
            if combined > 0.30:
                issues.append(WordAudioIssue(
                    shloka=verse.shloka,
                    charan=label,
                    word=word,
                    expected_sec=round(expected, 2),
                    observed_sec=round(observed, 2),
                    distance=round(dist, 3),
                    onset_distance=round(onset_value, 3),
                    issue="Expected word or starting consonant sounds different from trainer",
                    confidence="Medium" if combined > 0.38 else "Low",
                ))
    return issues[:20]
