from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .audio_analysis import AudioData, TARGET_SR
from .swara_audio import _dtw_time_map, _load_trainer
from .trainer_audio import TrainerIndex


@dataclass(frozen=True)
class EvidenceIssue:
    category: str
    shloka: int
    charan: str
    word: str
    target: str
    evidence: str
    issue: str
    confidence: str


def _energy(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y * y)) + 1e-12)


def _zcr(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(y)))))


def _student_window(mapping, trainer_times, student_times, rel_center: float, rel_width: float = 0.18):
    trainer_idx = np.where((trainer_times >= rel_center - rel_width) & (trainer_times <= rel_center + rel_width))[0]
    student_idx: list[int] = []
    for idx in trainer_idx:
        student_idx.extend(mapping.get(int(idx), []))
    if len(student_idx) < 2:
        return None
    return float(student_times[min(student_idx)]), float(student_times[max(student_idx)])


def _charan_label(row: pd.Series) -> str:
    labels = {1: "A", 2: "B", 3: "C", 4: "D"}
    value = pd.to_numeric(row.get("Charan Number", None), errors="coerce")
    if pd.isna(value):
        return ""
    return labels.get(int(value), "")


def _mapped_context(student: AudioData, trainer_path: Path, segment_start: float, segment_end: float):
    trainer = _load_trainer(trainer_path, segment_start, segment_end)
    mapped = _dtw_time_map(trainer, student)
    if mapped is None:
        return None
    mapping, trainer_times, student_times = mapped
    return trainer, mapping, trainer_times, student_times


def _clean_word(value: str) -> str:
    text = str(value or "")
    for ch in "()[]{}<>.,;:!?।॥'\"“”‘’ \t\r\n":
        text = text.replace(ch, "")
    return text


def _csv_word_window(
    row: pd.Series,
    recitation_frame: pd.DataFrame | None,
    segment_start: float,
    segment_end: float,
) -> tuple[float, float] | None:
    if recitation_frame is None or recitation_frame.empty:
        return None
    chapter_value = pd.to_numeric(row.get("Chapter", None), errors="coerce")
    shloka_value = pd.to_numeric(row.get("Shloka Number", None), errors="coerce")
    if pd.isna(chapter_value) or pd.isna(shloka_value):
        return None
    chapter = int(chapter_value)
    shloka = int(shloka_value)
    cnum = pd.to_numeric(row.get("Charan Number", None), errors="coerce")
    if pd.isna(cnum):
        return None
    label = {1: "A", 2: "B", 3: "C", 4: "D"}.get(int(cnum), "")
    if not label:
        return None
    rows = recitation_frame[
        (pd.to_numeric(recitation_frame["Chapter"], errors="coerce") == chapter)
        & (pd.to_numeric(recitation_frame["Shloka"], errors="coerce") == shloka)
        & (recitation_frame["Text_Type"].astype(str).str.upper() == "SHLOKA")
        & (recitation_frame["Charan"].astype(str).str.upper() == label)
    ]
    if rows.empty:
        return None
    rec = rows.iloc[0]
    start = float(pd.to_numeric(rec.get("Start_sec", None), errors="coerce"))
    end = float(pd.to_numeric(rec.get("End_sec", None), errors="coerce"))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None
    words = [w.strip() for w in str(rec.get("Words", "")).split(",") if w.strip() and w.strip() != "-"]
    if not words:
        text = str(rec.get("Expected_Text_From_CSV", "") or rec.get("Transcript_Devanagari", ""))
        words = [w.strip() for w in text.split() if w.strip()]
    if not words:
        return max(0.0, start - segment_start), max(0.05, end - segment_start)
    target_word = _clean_word(row.get("Associated Word", ""))
    clean_words = [_clean_word(w) for w in words]
    match_idx = 0
    for idx, clean in enumerate(clean_words):
        if target_word and (target_word in clean or clean in target_word):
            match_idx = idx
            break
    weights = [max(1, len(w)) for w in clean_words]
    total = float(sum(weights)) or float(len(weights))
    cursor = start
    for idx, weight in enumerate(weights):
        dur = (end - start) * weight / total
        if idx == match_idx:
            pad = min(0.08, dur * 0.2)
            return max(0.0, cursor - pad - segment_start), min(segment_end - segment_start, cursor + dur + pad - segment_start)
        cursor += dur
    return None


def nasal_evidence_issues(
    student: AudioData,
    trainer_path: Path,
    rule_frame: pd.DataFrame,
    chapter: int,
    shlokas: tuple[int, ...],
    segment_start: float,
    segment_end: float,
    recitation_frame: pd.DataFrame | None = None,
) -> list[EvidenceIssue]:
    ctx = _mapped_context(student, trainer_path, segment_start, segment_end)
    if ctx is None or rule_frame.empty:
        return []
    _, mapping, trainer_times, student_times = ctx
    rows = rule_frame[
        (rule_frame["Chapter"] == chapter)
        & (rule_frame["Shloka Number"].isin(shlokas))
        & (
            rule_frame.get("Anusvara Guide Output", pd.Series("", index=rule_frame.index)).fillna("").astype(str).str.strip().ne("")
            | rule_frame.get("Anunasika Present", pd.Series("", index=rule_frame.index)).astype(str).str.lower().isin({"yes", "true", "1"})
        )
    ].copy()
    issues: list[EvidenceIssue] = []
    for _, row in rows.iterrows():
        sh = int(row.get("Shloka Number"))
        csv_win = _csv_word_window(row, recitation_frame, segment_start, segment_end)
        if csv_win is not None:
            rel_center = (csv_win[0] + csv_win[1]) / 2
            rel_width = max(0.12, (csv_win[1] - csv_win[0]) / 2)
        else:
            group = rows[rows["Shloka Number"] == sh]
            order = list(group.index).index(row.name) if row.name in group.index else 0
            rel_center = (order + 1) / (len(group) + 1) * max(0.1, segment_end - segment_start)
            rel_width = 0.18
        win = _student_window(mapping, trainer_times, student_times, rel_center, rel_width=rel_width)
        if win is None:
            continue
        s0, s1 = win
        y = student.samples[int(s0 * student.sample_rate): int(s1 * student.sample_rate)]
        # Nasalized/sustained portions often have relatively stable voicing and low zero-crossing.
        zcr = _zcr(y)
        en = _energy(y)
        if en < 0.002:
            issues.append(EvidenceIssue("Anusvara/Anunasika", sh, _charan_label(row), str(row.get("Associated Word", "")), str(row.get("Anusvara Guide Output", "")), f"very low local energy {en:.4f}", "Nasal evidence weak or unclear", "Low"))
        elif zcr > 0.18:
            issues.append(EvidenceIssue("Anusvara/Anunasika", sh, _charan_label(row), str(row.get("Associated Word", "")), str(row.get("Anusvara Guide Output", "")), f"high local zero-crossing {zcr:.2f}", "Expected nasal sound may be unclear", "Low"))
    return issues[:20]


def aghata_evidence_issues(
    student: AudioData,
    trainer_path: Path,
    rule_frame: pd.DataFrame,
    chapter: int,
    shlokas: tuple[int, ...],
    segment_start: float,
    segment_end: float,
    recitation_frame: pd.DataFrame | None = None,
) -> list[EvidenceIssue]:
    ctx = _mapped_context(student, trainer_path, segment_start, segment_end)
    if ctx is None or rule_frame.empty:
        return []
    _, mapping, trainer_times, student_times = ctx
    rows = rule_frame[
        (rule_frame["Chapter"] == chapter)
        & (rule_frame["Shloka Number"].isin(shlokas))
        & (rule_frame["Text Type"].astype(str).str.upper() == "SHLOKA")
        & (rule_frame.get("Following Conjunct", pd.Series("", index=rule_frame.index)).fillna("").astype(str).str.strip().ne(""))
    ].copy()
    issues: list[EvidenceIssue] = []
    for sh, group in rows.groupby("Shloka Number"):
        group = group.reset_index(drop=True)
        for order, row in group.iterrows():
            csv_win = _csv_word_window(row, recitation_frame, segment_start, segment_end)
            if csv_win is not None:
                rel_center = (csv_win[0] + csv_win[1]) / 2
                rel_width = max(0.10, (csv_win[1] - csv_win[0]) / 2)
            else:
                rel_center = (order + 1) / (len(group) + 1) * max(0.1, segment_end - segment_start)
                rel_width = 0.14
            win = _student_window(mapping, trainer_times, student_times, rel_center, rel_width=rel_width)
            if win is None:
                continue
            s0, s1 = win
            y = student.samples[int(s0 * student.sample_rate): int(s1 * student.sample_rate)]
            if y.size < 32:
                continue
            env = np.abs(y)
            peak = float(np.max(env))
            rms = _energy(y)
            # Dvitva/aghata often creates a local consonant closure/release contrast.
            # If the local contrast is flat, evidence is weak.
            contrast = peak / max(rms, 1e-6)
            if contrast < 3.0:
                issues.append(EvidenceIssue(
                    "Aghata/Dvitva",
                    int(sh),
                    _charan_label(row),
                    str(row.get("Associated Word", "")),
                    str(row.get("Following Conjunct", "")),
                    f"local contrast {contrast:.2f}",
                    "Aghata/dvitva evidence may be weak",
                    "Low",
                ))
    return issues[:20]
