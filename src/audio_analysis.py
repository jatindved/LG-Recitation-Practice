from __future__ import annotations

import io
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TARGET_SR = 16_000


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration_sec: float


@dataclass(frozen=True)
class QualityReport:
    duration_sec: float
    peak: float
    rms_dbfs: float
    clipping_fraction: float
    silence_fraction: float
    passed: bool
    reasons: tuple[str, ...]


def decode_audio(payload: bytes, suffix: str = ".wav") -> AudioData:
    import librosa
    import soundfile as sf
    # soundfile handles WAV/FLAC directly. librosa/audioread can use ffmpeg for MP3/M4A.
    try:
        y, sr = sf.read(io.BytesIO(payload), always_2d=False, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            path = tmp.name
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
        finally:
            Path(path).unlink(missing_ok=True)
    y = np.asarray(y, dtype=np.float32)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y = np.nan_to_num(y)
    return AudioData(y, int(sr), len(y) / float(sr) if sr else 0.0)


def quality_report(audio: AudioData) -> QualityReport:
    y = audio.samples
    if not len(y):
        return QualityReport(0, 0, -120, 0, 1, False, ("No decodable audio",))
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y * y)) + 1e-12)
    rms_dbfs = 20 * math.log10(rms)
    clipping = float(np.mean(np.abs(y) >= 0.995))
    frame_length, hop = 1024, 256
    if len(y) < frame_length:
        frame_rms = np.array([rms], dtype=float)
    else:
        starts = range(0, len(y) - frame_length + 1, hop)
        frame_rms = np.array([np.sqrt(np.mean(y[i:i + frame_length] ** 2)) for i in starts])
    silence_threshold = max(10 ** (-48 / 20), float(np.percentile(frame_rms, 20)) * 1.8)
    silence = float(np.mean(frame_rms < silence_threshold)) if len(frame_rms) else 1.0
    reasons: list[str] = []
    if audio.duration_sec < 1.5:
        reasons.append("Recording is too short")
    if rms_dbfs < -42:
        reasons.append("Speech level is too low")
    if clipping > 0.01:
        reasons.append("Microphone clipping detected")
    if silence > 0.75:
        reasons.append("Too much silence")
    return QualityReport(audio.duration_sec, peak, rms_dbfs, clipping, silence, not reasons, tuple(reasons))


def normalized_mfcc(audio: AudioData) -> np.ndarray:
    import librosa
    y, _ = librosa.effects.trim(audio.samples, top_db=35)
    mfcc = librosa.feature.mfcc(y=y, sr=audio.sample_rate, n_mfcc=20, hop_length=256)
    mfcc = librosa.util.normalize(mfcc, axis=1)
    return mfcc


def dtw_distance(left: AudioData, right: AudioData) -> float:
    import librosa
    a, b = normalized_mfcc(left), normalized_mfcc(right)
    if not a.shape[1] or not b.shape[1]:
        return float("inf")
    cost, wp = librosa.sequence.dtw(X=a, Y=b, metric="cosine")
    return float(cost[-1, -1] / max(len(wp), 1))


def normalized_pitch(audio: AudioData) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    f0, voiced, _ = librosa.pyin(audio.samples, fmin=65, fmax=500, sr=audio.sample_rate, frame_length=2048, hop_length=256)
    times = librosa.times_like(f0, sr=audio.sample_rate, hop_length=256)
    valid = voiced & np.isfinite(f0)
    if np.count_nonzero(valid) < 3:
        return np.array([]), np.array([])
    median = float(np.nanmedian(f0[valid]))
    semitones = np.full_like(f0, np.nan, dtype=float)
    semitones[valid] = 12.0 * np.log2(f0[valid] / median)
    return times, semitones
