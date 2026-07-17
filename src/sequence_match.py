from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_analysis import AudioData, TARGET_SR, dtw_distance
from .trainer_audio import TrainerIndex


@dataclass(frozen=True)
class SequenceResult:
    selected: tuple[int, ...]
    detected: tuple[int, ...] | None
    distance: float | None
    margin: float | None
    reliable: bool
    message_code: str


def _load_segment(path: Path, start: float, end: float) -> AudioData:
    import librosa
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True, offset=max(0.0, start), duration=max(0.01, end - start))
    return AudioData(np.asarray(y, dtype=np.float32), sr, len(y) / sr)


def _sequence_audio(path: Path, index: TrainerIndex, chapter: int, sequence: tuple[int, ...]) -> AudioData | None:
    pieces: list[np.ndarray] = []
    for shloka in sequence:
        seg = index.shloka_segment(chapter, shloka)
        if seg is None or not seg.verified:
            return None
        audio_path = path
        if seg.audio_file:
            candidate = Path(seg.audio_file)
            if candidate.is_absolute() and candidate.exists():
                audio_path = candidate
            else:
                relative_candidate = path.parent / candidate
                if relative_candidate.exists():
                    audio_path = relative_candidate
        pieces.append(_load_segment(audio_path, seg.start_sec, seg.end_sec).samples)
    if not pieces:
        return None
    y = np.concatenate(pieces)
    return AudioData(y, TARGET_SR, len(y) / TARGET_SR)


def candidate_sequences(available: list[int], selected: tuple[int, ...]) -> list[tuple[int, ...]]:
    length = len(selected)
    candidates = {selected}
    if not available:
        return [selected]
    for n in range(max(1, length - 1), min(length + 1, 4) + 1):
        for i in range(0, len(available) - n + 1):
            seq = tuple(available[i:i + n])
            if abs(seq[0] - selected[0]) <= 2 or abs(seq[-1] - selected[-1]) <= 2:
                candidates.add(seq)
    return sorted(candidates)


def identify_sequence(student: AudioData, trainer_path: Path, index: TrainerIndex, chapter: int,
                      selected: tuple[int, ...], available: list[int]) -> SequenceResult:
    scored: list[tuple[float, tuple[int, ...]]] = []
    for seq in candidate_sequences(available, selected):
        ref = _sequence_audio(trainer_path, index, chapter, seq)
        if ref is not None:
            scored.append((dtw_distance(student, ref), seq))
    if not scored:
        return SequenceResult(selected, None, None, None, False, "INDEX_MISSING")
    scored.sort(key=lambda x: x[0])
    best_dist, best_seq = scored[0]
    second = scored[1][0] if len(scored) > 1 else float("inf")
    margin = second - best_dist
    # Conservative calibration defaults. They deliberately abstain on weak or ambiguous matches.
    reliable = bool(np.isfinite(best_dist) and best_dist < 0.42 and margin > 0.025)
    if not reliable:
        return SequenceResult(selected, None, best_dist, margin, False, "UNABLE_TO_IDENTIFY")
    if best_seq == selected:
        code = "MATCHED"
    elif set(selected) - set(best_seq) and set(best_seq) - set(selected):
        code = "MISSING_AND_EXTRA"
    elif set(selected) - set(best_seq):
        code = "MISSING_SELECTED_SHLOKA"
    elif set(best_seq) - set(selected):
        code = "EXTRA_UNSELECTED_SHLOKA"
    else:
        code = "OUT_OF_ORDER"
    return SequenceResult(selected, best_seq, best_dist, margin, True, code)
