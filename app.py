from __future__ import annotations

import html
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.audio_analysis import decode_audio, quality_report
from src.akshara_match import AksharaIssue, AksharaMatch, compare_aksharas
from src.data_sources import data_file
from src.geeta_data import GeetaRepository, Verse
from src.i18n import LANGUAGES, texts
from src.pronunciation_audio import EvidenceIssue, nasal_evidence_issues, aghata_evidence_issues
from src.swara_audio import swara_audio_issues, issues_to_keys, issue_key
from src.swara_timing import akshara_timing_table
from src.trainer_audio import TrainerIndex, cached_trainer_audio, trainer_audio_url
from src.word_audio import WordAudioIssue, word_audio_issues, word_issue_key


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache" / "trainer_audio"
LOCAL_AUDIO_DIRS = [
    ROOT / "master_audios",
    DATA / "master_audios",
    ROOT / "audio",
    Path(r"D:\Geeta Aghata"),
    Path(r"D:\Geeta Aghata\master_audios"),
]
EXTRA_AUDIO_DIRS = [Path(p) for p in os.environ.get("LEARNGEETA_AUDIO_DIRS", "").split(os.pathsep) if p.strip()]
LOCAL_AUDIO_DIRS = EXTRA_AUDIO_DIRS + LOCAL_AUDIO_DIRS

PARAYAN_CHAPTER_MINUTES = {
    1: 18, 2: 15, 3: 9, 4: 8, 5: 6, 6: 9, 7: 6, 8: 6, 9: 7,
    10: 9, 11: 13, 12: 4, 13: 7, 14: 6, 15: 4, 16: 5, 17: 6, 18: 16,
}

st.set_page_config(page_title="LearnGeeta Recitation Practice", page_icon="🕉️", layout="wide")


def fmt_seconds(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


@st.cache_resource
def repository() -> GeetaRepository:
    return GeetaRepository(
        data_file(DATA / "Geeta - Hin.csv"),
        data_file(DATA / "Geeta - Eng.csv"),
        data_file(DATA / "Gita_Aghata_Master_FINAL.csv"),
    )


@st.cache_resource
def trainer_index() -> TrainerIndex:
    return TrainerIndex(data_file(DATA / "trainer_index.csv"))


@st.cache_data(show_spinner=False)
def recitation_segments() -> pd.DataFrame:
    path = data_file(DATA / "Bhagavad Gita recitation.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in ["Chapter", "Shloka", "Start_sec", "End_sec"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def get_trainer_audio(chapter: int) -> str:
    return str(cached_trainer_audio(chapter, CACHE, LOCAL_AUDIO_DIRS))


def verse_html(verse: Verse) -> str:
    title = f"Chapter {verse.chapter} · Shloka {verse.shloka}"
    speaker = f'<div class="speaker">{html.escape(verse.speaker)}</div>' if verse.speaker else ""
    lines = "".join(f'<div class="charan">{html.escape(line)}</div>' for line in verse.charans)
    return f'<article class="verse"><div class="verse-id">{title}</div>{speaker}{lines}</article>'


def trainer_timing_table(index: TrainerIndex, verses: list[Verse]) -> pd.DataFrame:
    rows = []
    for verse in verses:
        segment = index.shloka_segment(verse.chapter, verse.shloka)
        rows.append({
            "Audio": segment.audio_file if segment and segment.audio_file else f"CH{verse.chapter:02d}.mp3",
            "Chapter": verse.chapter,
            "Shloka": verse.shloka,
            "Start_sec": fmt_seconds(segment.start_sec if segment else None),
            "End_sec": fmt_seconds(segment.end_sec if segment else None),
            "Confidence": f"{segment.confidence:.4f}" if segment else "-",
            "Status": ("Verified" if segment.verified else (segment.status or "Review required")) if segment else "Not available",
        })
    return pd.DataFrame(rows)


def trainer_charan_timing_table(index: TrainerIndex, verses: list[Verse]) -> pd.DataFrame:
    rows = []
    frame = index.frame
    for verse in verses:
        subset = frame[(frame.chapter == verse.chapter) & (frame.shloka == verse.shloka)].copy()
        if "charan" not in subset.columns:
            continue
        subset["charan_num"] = pd.to_numeric(subset["charan"], errors="coerce")
        subset = subset[subset["charan_num"].notna()].sort_values("charan_num")
        for _, segment in subset.iterrows():
            cnum = int(segment.charan_num)
            rows.append({
                "Shloka": verse.shloka,
                "Charan": "ABCD"[cnum - 1] if 1 <= cnum <= 4 else str(cnum),
                "Start_sec": fmt_seconds(segment.start_sec),
                "End_sec": fmt_seconds(segment.end_sec),
                "Duration": fmt_seconds(float(segment.end_sec) - float(segment.start_sec)),
                "Status": "Verified" if bool(segment.verified) else "Review",
            })
    return pd.DataFrame(rows)


def selected_reference_duration(index: TrainerIndex, verses: list[Verse]) -> tuple[float | None, list[int], bool]:
    total = 0.0
    missing: list[int] = []
    all_verified = True
    for verse in verses:
        segment = index.shloka_segment(verse.chapter, verse.shloka)
        if segment is None:
            missing.append(verse.shloka)
            continue
        all_verified = all_verified and segment.verified
        total += max(0.0, segment.end_sec - segment.start_sec)
    return (round(total, 2), missing, all_verified) if not missing else (None, missing, False)


def selected_segment_bounds(index: TrainerIndex, chapter: int, verses: list[Verse], include_opening: bool) -> tuple[float | None, float | None, list[int]]:
    found = []
    missing: list[int] = []
    for verse in verses:
        segment = index.shloka_segment(chapter, verse.shloka)
        if segment is None:
            missing.append(verse.shloka)
        else:
            found.append(segment)
    if not found:
        return None, None, missing
    start_sec = 0.0 if include_opening else min(float(s.start_sec) for s in found)
    end_sec = max(float(s.end_sec) for s in found)
    return start_sec, end_sec, missing


def opening_context(frame: pd.DataFrame, chapter: int, first_shloka: int) -> tuple[float, pd.DataFrame]:
    if frame.empty or first_shloka != 1:
        return 0.0, pd.DataFrame()
    types = {"PRAYER", "CHAPTER_TITLE", "CHAPTER_HEADING"}
    mask = (frame["Chapter"] == chapter) & (
        frame["Text_Type"].astype(str).str.upper().isin(types)
        | ((frame["Text_Type"].astype(str).str.upper() == "SPEAKER_LABEL") & (frame["Shloka"] == first_shloka))
    )
    rows = frame.loc[mask].sort_values("Start_sec").copy()
    if rows.empty:
        return 0.0, rows
    start = float(rows["Start_sec"].min())
    end = float(rows["End_sec"].max())
    return max(0.0, round(end - start, 2)), rows


def opening_context_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out = rows[["Text_Type", "Start_sec", "End_sec", "Transcript_Devanagari"]].copy()
    out.columns = ["Part", "Start_sec", "End_sec", "Text"]
    return out


def parayan_chapter_context(chapter: int, selected_count: int, total_count: int) -> str:
    minutes = PARAYAN_CHAPTER_MINUTES.get(chapter)
    if not minutes or not total_count:
        return ""
    approx_selected = minutes * 60 * selected_count / total_count
    return (
        f"Approximate Parayan context for Chapter {chapter}: full chapter around {minutes} minutes; "
        f"selected range roughly {fmt_seconds(approx_selected)}s by simple average. "
        "This is only a broad tradition/practice context, not a grading rule."
    )


def duration_candidate(index: TrainerIndex, chapter: int, selected: tuple[int, ...], student_duration: float) -> tuple[tuple[int, ...], float] | None:
    frame = index.frame
    if frame.empty:
        return None
    rows = frame[(frame.chapter == chapter) & (frame.verified == True)].copy()  # noqa: E712
    if rows.empty:
        return None
    if "charan" in rows.columns:
        rows = rows[rows.charan.isna()]
    rows = rows.sort_values("shloka")
    shlokas = [int(v) for v in rows.shloka.tolist()]
    durations = {int(r.shloka): max(0.0, float(r.end_sec) - float(r.start_sec)) for _, r in rows.iterrows()}
    selected_set = tuple(selected)
    best_seq: tuple[int, ...] | None = None
    best_diff = 10**9
    max_len = min(4, len(selected_set) + 2)
    for length in range(1, max_len + 1):
        for i in range(0, len(shlokas) - length + 1):
            seq = tuple(shlokas[i:i + length])
            if seq == selected_set:
                continue
            total = sum(durations.get(s, 0.0) for s in seq)
            if total <= 0:
                continue
            diff = abs(student_duration - total)
            if diff < best_diff:
                best_diff = diff
                best_seq = seq
    if best_seq is None:
        return None
    # Only show a suspicion when another sequence is reasonably close in duration.
    if best_diff <= max(3.0, student_duration * 0.18):
        return best_seq, round(best_diff, 2)
    return None


def global_duration_candidate(
    index: TrainerIndex,
    selected_chapter: int,
    selected: tuple[int, ...],
    selected_reference_duration: float,
    student_duration: float,
) -> tuple[int, tuple[int, ...], float, float] | None:
    """Find an obviously better chapter/shloka duration match.

    This is a conservative gate for cases like selecting Chapter 1 while
    uploading Chapter 15 audio. It does not try to prove pronunciation; it only
    prevents a misleading report when the selected text is very likely wrong.
    """
    frame = index.frame
    if frame.empty or not selected:
        return None
    rows = frame[(frame.verified == True)].copy()  # noqa: E712
    if "charan" in rows.columns:
        rows = rows[rows.charan.isna()]
    if rows.empty:
        return None
    rows = rows.sort_values(["chapter", "shloka"])
    selected_seq = tuple(selected)
    selected_diff = abs(student_duration - selected_reference_duration)
    best: tuple[int, tuple[int, ...], float] | None = None
    for chap, group in rows.groupby("chapter"):
        shlokas = [int(v) for v in group.shloka.tolist()]
        durations = {int(r.shloka): max(0.0, float(r.end_sec) - float(r.start_sec)) for _, r in group.iterrows()}
        for length in range(max(1, len(selected_seq) - 1), min(len(selected_seq) + 2, 5) + 1):
            for i in range(0, len(shlokas) - length + 1):
                seq = tuple(shlokas[i:i + length])
                if int(chap) == selected_chapter and seq == selected_seq:
                    continue
                total = sum(durations.get(s, 0.0) for s in seq)
                diff = abs(student_duration - total)
                if best is None or diff < best[2]:
                    best = (int(chap), seq, diff)
    if best is None:
        return None
    best_chapter, best_seq, best_diff = best
    # Only stop the report when another sequence is clearly closer.
    close_enough = best_diff <= max(3.0, student_duration * 0.16)
    meaningfully_better = (selected_diff - best_diff) >= max(2.0, student_duration * 0.14)
    different_text = best_chapter != selected_chapter or best_seq != selected_seq
    if close_enough and meaningfully_better and different_text:
        return best_chapter, best_seq, round(best_diff, 2), round(selected_diff, 2)
    return None


def instant_guidance(student_duration: float, reference_duration: float, lex: dict[str, str]) -> tuple[str, str, str]:
    diff = round(student_duration - reference_duration, 2)
    ratio = student_duration / reference_duration if reference_duration else 0.0
    if ratio < 0.72:
        return lex["status_much_faster"], lex["much_faster_meaning"].format(diff=f"{abs(diff):.2f}"), lex["much_faster_next"]
    if ratio > 1.45:
        return lex["status_much_slower"], lex["much_slower_meaning"].format(diff=f"{diff:.2f}"), lex["much_slower_next"]
    if ratio < 0.88:
        return lex["status_faster"], lex["faster_meaning"].format(diff=f"{abs(diff):.2f}"), lex["faster_next"]
    if ratio > 1.18:
        return lex["status_slower"], lex["slower_meaning"].format(diff=f"{diff:.2f}"), lex["slower_next"]
    return lex["status_close"], lex["close_meaning"].format(diff=f"{diff:+.2f}"), lex["close_next"]


def status_card(status: str, meaning: str, next_step: str, lex: dict[str, str]) -> None:
    tone = "ok" if status == lex["status_close"] else "warn"
    st.html(
        f"""
        <section class="result-card {tone}">
          <div class="result-kicker">{html.escape(lex["practice_result"])}</div>
          <h2>{html.escape(status)}</h2>
          <p><b>{html.escape(lex["meaning"])}:</b> {html.escape(meaning)}</p>
          <p><b>{html.escape(lex["next"])}:</b> {html.escape(next_step)}</p>
        </section>
        """
    )


def compact_time_card(comparison_label: str, expected_sec: float, actual_sec: float) -> None:
    diff = actual_sec - expected_sec
    pct = diff / max(expected_sec, 0.01) * 100
    st.html(
        f"""
        <section class="scope-card">
          <b>{html.escape(lex["time_card_title"])}</b>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-top:.65rem">
            <div><small>{html.escape(lex["time_card_text"])}</small><br><b>{html.escape(comparison_label)}</b></div>
            <div><small>{html.escape(lex["time_card_expected"])}</small><br><b>{fmt_seconds(expected_sec)}s</b></div>
            <div><small>{html.escape(lex["time_card_actual"])}</small><br><b>{fmt_seconds(actual_sec)}s</b></div>
            <div><small>{html.escape(lex["time_card_difference"])}</small><br><b>{diff:+.2f}s ({pct:+.1f}%)</b></div>
          </div>
        </section>
        """
    )


def parayan_rule_card() -> None:
    st.html(
        f"""
        <section class="scope-card">
          <b>{html.escape(lex["parayan_mode"])}</b>
        </section>
        """
    )


def render_understand_correction(repo: GeetaRepository, index: TrainerIndex, verses: list[Verse], segments: pd.DataFrame, lex: dict[str, str], chapter_context: str = "") -> None:
    with st.expander(lex["details_title"], expanded=False):
        st.caption(lex["details_caption"])
        if chapter_context:
            st.caption(chapter_context)
        if not segments.empty:
            shloka_rows = segments[segments["Text_Type"].astype(str).str.upper() == "SHLOKA"].copy()
            bad_timing = shloka_rows[pd.to_numeric(shloka_rows["End_sec"], errors="coerce") <= pd.to_numeric(shloka_rows["Start_sec"], errors="coerce")]
            st.caption(
                f"{lex['csv_check']}: {shloka_rows[['Chapter','Shloka']].drop_duplicates().shape[0]} shlokas, "
                f"{len(shloka_rows)} charan rows. Bad timing rows: {len(bad_timing)}."
            )
        st.subheader(lex["details_akshara"])
        render_expected_akshara_timing(index, verses, segments)
        st.subheader(lex["details_aghata"])
        render_aghata(repo, verses, lex)
        st.subheader(lex["details_anusvara"])
        render_anusvara(repo, verses, lex)


def render_aghata(repo: GeetaRepository, verses: list[Verse], lex: dict[str, str]) -> None:
    rows: list[dict[str, object]] = []
    for verse in verses:
        marks = repo.aghata_for(verse.chapter, verse.shloka)
        for _, mark in marks.iterrows():
            rows.append({
                "Shloka": verse.shloka,
                "Word": mark.get("Associated Word", ""),
                "Following conjunct": mark.get("Following Conjunct", ""),
                "Dvitva": mark.get("Dvitva Result", mark.get("Dvitva Candidate", "")),
                "Status": "Expected location from official data",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info(lex["unable"])
    st.caption(lex["not_acoustic"])


def render_anusvara(repo: GeetaRepository, verses: list[Verse], lex: dict[str, str]) -> None:
    rows: list[dict[str, object]] = []
    for verse in verses:
        marks = repo.anusvara_for(verse.chapter, verse.shloka)
        for _, mark in marks.iterrows():
            rows.append({
                "Shloka": verse.shloka,
                "Word": mark.get("Associated Word", ""),
                "Position": mark.get("Anusvara Position", ""),
                "Following sound": mark.get("Anusvara Following Sound", ""),
                "Expected guidance": mark.get("Anusvara Guide Output", ""),
                "Status": "Expected guidance from official data",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info(lex["no_anusvara"])
    st.caption(lex["anusvara_caption"])


def render_expected_akshara_timing(index: TrainerIndex, verses: list[Verse], segments: pd.DataFrame) -> None:
    st.subheader(lex["akshara_reference"])
    table = akshara_timing_table(index, verses, segments)
    if table.empty:
        st.info(lex["akshara_missing"])
        return
    total = table["Expected_sec"].sum()
    st.caption(
        f"{lex['akshara_reference_caption']} "
        f"Selected shloka expected akshara total: {fmt_seconds(total)}s."
    )
    st.dataframe(table, hide_index=True, use_container_width=True)


def render_akshara_timing_summary(index: TrainerIndex, verses: list[Verse], segments: pd.DataFrame) -> None:
    table = akshara_timing_table(index, verses, segments)
    if table.empty:
        return
    total = table["Expected_sec"].sum()
    count = len(table)
    avg = total / count if count else 0
    st.caption(f"{lex['csv_timing_ready']}: {count} aksharas · expected {fmt_seconds(total)}s.")


def render_swara_audio_check(student: AudioData, trainer_path: Path, index: TrainerIndex, verses: list[Verse], start_sec: float, end_sec: float, segments: pd.DataFrame):
    st.subheader(lex["swara_check_title"])
    with st.spinner(lex["swara_spinner"]):
        issues = swara_audio_issues(student, trainer_path, index, verses, start_sec, end_sec, segments)
    if not issues:
        st.success(lex["swara_no_issue"])
        st.caption(lex["swara_caption"])
        return issues
    frame = pd.DataFrame([issue.__dict__ for issue in issues])
    frame = frame.rename(columns={
        "shloka": "Shloka",
        "charan": "Charan",
        "akshara": "Akshara",
        "length": "Swara length",
        "weight": "Laghu/Guru",
        "expected_sec": "Expected sec",
        "observed_sec": "Your sec",
        "trainer_start_sec": "Trainer start",
        "trainer_end_sec": "Trainer end",
        "ratio": "Ratio",
        "issue": "Practice hint",
        "confidence": "Confidence",
    })
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption(
        lex["swara_caption"]
    )
    return issues


def render_word_audio_check(student: AudioData, trainer_path: Path, verses: list[Verse], start_sec: float, end_sec: float, segments: pd.DataFrame) -> list[WordAudioIssue]:
    st.subheader(lex["word_check_title"])
    with st.spinner(lex["word_spinner"]):
        issues = word_audio_issues(student, trainer_path, verses, start_sec, end_sec, segments)
    if not issues:
        st.success(lex["word_no_issue"])
        st.caption(lex["word_caption"])
        return issues
    frame = pd.DataFrame([issue.__dict__ for issue in issues]).rename(columns={
        "shloka": "Shloka",
        "charan": "Charan",
        "word": "Expected word",
        "expected_sec": "Trainer word sec",
        "observed_sec": "Your word sec",
        "distance": "Acoustic difference",
        "onset_distance": "Starting sound difference",
        "issue": "Practice hint",
        "confidence": "Confidence",
    })
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption(
        lex["word_caption"]
    )
    return issues


def collect_pronunciation_evidence(student: AudioData, trainer_path: Path, repo: GeetaRepository, chapter: int, selected_seq: tuple[int, ...], start_sec: float, end_sec: float, segments: pd.DataFrame) -> list[EvidenceIssue]:
    frame = repo._aghata  # official mapped rule data loaded by repository
    with st.spinner("Checking pronunciation at CSV-timed word locations..."):
        nasal = nasal_evidence_issues(student, trainer_path, frame, chapter, selected_seq, start_sec, end_sec, segments)
        aghata = aghata_evidence_issues(student, trainer_path, frame, chapter, selected_seq, start_sec, end_sec, segments)
    return [*nasal, *aghata]


@st.cache_data(show_spinner=False)
def trainer_segment_clip(path_text: str, start_sec: float, end_sec: float) -> tuple[bytes, str]:
    import soundfile as sf
    import librosa
    path = Path(path_text)
    duration = max(0.05, end_sec - start_sec)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            # Accurate seek: input first, then -ss/-t. AAC m4a is smaller/faster for the browser player.
            subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(path),
                    "-ss", f"{max(0.0, start_sec):.3f}",
                    "-t", f"{duration:.3f}",
                    "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
                    str(tmp_path),
                ],
                check=True,
            )
            return tmp_path.read_bytes(), "audio/mp4"
        except Exception:
            pass
        finally:
            tmp_path.unlink(missing_ok=True)

    # 12 kHz mono keeps the reference player much lighter than a full-quality WAV.
    # It is for listening only; analysis still loads the original trainer segment.
    y, sr = librosa.load(str(path), sr=12000, mono=True, offset=max(0.0, start_sec), duration=duration)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), y, sr, format="WAV")
        return tmp_path.read_bytes(), "audio/wav"
    finally:
        tmp_path.unlink(missing_ok=True)


def _clean_devanagari_token(value: str) -> str:
    remove = "()[]{}<>.,;:!?।॥'\"“”‘’ \t\r\n"
    text = str(value or "")
    for ch in remove:
        text = text.replace(ch, "")
    return text


def evidence_highlight_keys(issues: list[EvidenceIssue], segments: pd.DataFrame) -> dict[str, set[str]]:
    from src.sanskrit_rules import syllable_rules

    out: dict[str, set[str]] = {}
    if not issues or segments.empty:
        return out
    for issue in issues:
        if str(getattr(issue, "confidence", "")).strip().lower() == "low":
            continue
        charan = str(getattr(issue, "charan", "") or "").strip().upper()
        if not charan:
            continue
        rows = segments[
            (segments["Shloka"] == issue.shloka)
            & (segments["Text_Type"].astype(str).str.upper() == "SHLOKA")
            & (segments["Charan"].astype(str).str.upper() == charan)
        ]
        if rows.empty:
            continue
        text = str(rows.iloc[0].get("Expected_Text_From_CSV", "") or rows.iloc[0].get("Transcript_Devanagari", ""))
        target = _clean_devanagari_token(issue.target)
        word = _clean_devanagari_token(issue.word)
        category_class = "nasal-akshara" if "Anusvara" in issue.category or "Anunasika" in issue.category else "aghata-akshara"
        for rule in syllable_rules(text):
            clean = _clean_devanagari_token(rule.text)
            if not clean:
                continue
            matched = False
            if target and (target in clean or clean in target):
                matched = True
            elif word and clean in word:
                matched = True
            if matched:
                out.setdefault(issue_key(issue.shloka, charan, rule.text), set()).add(category_class)
    return out


def word_highlight_keys(issues: list[WordAudioIssue]) -> set[str]:
    return {word_issue_key(i.shloka, i.charan, _clean_devanagari_token(i.word)) for i in issues if i.confidence != "Low"}


def selected_charans(verses: list[Verse], segments: pd.DataFrame) -> list[tuple[int, str, str]]:
    selected: list[tuple[int, str, str]] = []
    for verse in verses:
        rows = segments[
            (segments["Chapter"] == verse.chapter)
            & (segments["Shloka"] == verse.shloka)
            & (segments["Text_Type"].astype(str).str.upper() == "SHLOKA")
        ].copy()
        for _, row in rows.sort_values("Charan").iterrows():
            label = str(row.get("Charan", "")).strip().upper()
            text = str(row.get("Expected_Text_From_CSV", "") or row.get("Transcript_Devanagari", "")).strip()
            if label and text:
                selected.append((verse.shloka, label, text))
    return selected


def akshara_highlight_keys(issues: tuple[AksharaIssue, ...]) -> set[str]:
    return {issue_key(i.shloka, i.charan, i.expected) for i in issues if i.expected}


def render_highlighted_text(verses: list[Verse], segments: pd.DataFrame, swara_keys: set[str], evidence_keys: dict[str, set[str]], word_keys: set[str] | None = None, asr_keys: set[str] | None = None) -> None:
    from src.sanskrit_rules import syllable_rules

    st.subheader(lex["highlight_title"])
    word_keys = word_keys or set()
    asr_keys = asr_keys or set()
    if swara_keys or evidence_keys or word_keys or asr_keys:
        st.caption(lex["highlight_caption"])
    else:
        st.caption(lex["no_highlight_caption"])
    charan_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    blocks = []
    for verse in verses:
        rows = segments[
            (segments["Chapter"] == verse.chapter)
            & (segments["Shloka"] == verse.shloka)
            & (segments["Text_Type"].astype(str).str.upper() == "SHLOKA")
        ].copy()
        lines = []
        for _, row in rows.sort_values("Charan").iterrows():
            label = str(row.get("Charan", "")).strip().upper()
            text = str(row.get("Expected_Text_From_CSV", "") or row.get("Transcript_Devanagari", "")).strip()
            parts = []
            for rule in syllable_rules(text):
                key = issue_key(verse.shloka, label, rule.text)
                classes = set()
                if key in swara_keys:
                    classes.add("bad-akshara")
                if key in asr_keys:
                    classes.add("asr-akshara")
                for word_key in word_keys:
                    parts_key = word_key.split("|", 2)
                    if len(parts_key) == 3 and parts_key[0] == str(verse.shloka) and parts_key[1] == label:
                        if _clean_devanagari_token(rule.text) and _clean_devanagari_token(rule.text) in parts_key[2]:
                            classes.add("word-akshara")
                classes.update(evidence_keys.get(key, set()))
                cls = " ".join(sorted(classes)) if classes else "ok-akshara"
                parts.append(f'<span class="{cls}">{html.escape(rule.text)}</span>')
            lines.append(f'<div class="highlight-line"><b>{html.escape(label)}</b> {"".join(parts) if parts else html.escape(text)}</div>')
        blocks.append(f'<article class="verse correction"><div class="verse-id">Chapter {verse.chapter} · Shloka {verse.shloka}</div>{"".join(lines)}</article>')
    st.html("".join(blocks) if blocks else '<div class="notice">No highlightable text available.</div>')


st.html("""
<style>
.block-container{max-width:980px;padding-top:1.4rem}.hero{padding:1rem 1.25rem;border-radius:18px;background:linear-gradient(135deg,#fff7ed,#fff);border:1px solid #fed7aa;margin-bottom:1rem}.hero h1{margin:0;color:#7c2d12;font-size:2rem}.verse{background:#fff;border:1px solid #e5e7eb;border-left:5px solid #dc2626;border-radius:14px;padding:1rem 1.25rem;margin:.75rem 0;box-shadow:0 3px 12px #0000000a}.verse-id{font-size:.82rem;color:#991b1b;font-weight:700}.speaker{text-align:center;font-weight:700;margin:.45rem}.charan{text-align:center;font-family:"Noto Serif Devanagari",serif;font-size:1.35rem;line-height:1.8}.notice{padding:.85rem 1rem;border-radius:12px;background:#eff6ff;border:1px solid #bfdbfe}.no-marks{padding:.8rem 1rem;border-radius:12px;background:#f9fafb;border:1px solid #d1d5db;font-weight:600}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin:.75rem 0 1.1rem}.step{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:.8rem .9rem}.step b{color:#991b1b}.result-card{border-radius:18px;padding:1rem 1.2rem;margin:.8rem 0 1rem;border:1px solid #fde68a;background:#fffbeb}.result-card.ok{border-color:#bbf7d0;background:#f0fdf4}.result-card h2{margin:.15rem 0 .45rem;font-size:1.8rem}.result-kicker{text-transform:uppercase;letter-spacing:.08em;font-size:.75rem;color:#6b7280;font-weight:800}.scope-card{background:#f8fafc;border:1px solid #cbd5e1;border-radius:16px;padding:1rem;margin:.6rem 0 1rem}.scope-card ul{margin:.35rem 0 .8rem 1.2rem}.correction{font-family:"Noto Serif Devanagari",serif}.highlight-line{text-align:center;font-size:1.45rem;line-height:2.1}.bad-akshara{background:#fee2e2;color:#991b1b;border-bottom:3px solid #dc2626;border-radius:6px;padding:0 .08em}.asr-akshara{background:#ffedd5;color:#9a3412;border-bottom:3px solid #f97316;border-radius:6px;padding:0 .08em}.word-akshara{background:#ffedd5;color:#9a3412;border-bottom:3px solid #f97316;border-radius:6px;padding:0 .08em}.aghata-akshara{background:#f3e8ff;color:#6b21a8;border-bottom:3px solid #9333ea;border-radius:6px;padding:0 .08em}.nasal-akshara{background:#dbeafe;color:#1d4ed8;border-bottom:3px solid #2563eb;border-radius:6px;padding:0 .08em}.ok-akshara{padding:0 .04em}@media(max-width:800px){.steps{grid-template-columns:1fr}}
</style>
""")

lang_name = st.selectbox("Language", [name for name, _ in LANGUAGES], index=0)
lang_code = dict(LANGUAGES)[lang_name]
lex = texts(lang_code)

st.html(f"""
<section class="steps">
  <div class="step"><b>1. {html.escape(lex["step_select"])}</b><br>{html.escape(lex["step_select_desc"])}</div>
  <div class="step"><b>2. {html.escape(lex["step_record"])}</b><br>{html.escape(lex["step_record_desc"])}</div>
  <div class="step"><b>3. {html.escape(lex["step_check"])}</b><br>{html.escape(lex["step_check_desc"])}</div>
</section>
""")

st.html(f'<section class="hero"><h1>{html.escape(lex["title"])}</h1><p>{html.escape(lex["subtitle"])}</p></section>')

repo = repository()
index = trainer_index()
segments = recitation_segments()
chapters = repo.chapters()
c1, c2, c3 = st.columns(3)
chapter = c1.selectbox(lex["chapter"], chapters, index=0)
numbers = repo.shlokas(chapter)
start = c2.selectbox(lex["from"], numbers, index=0)
valid_end = [n for n in numbers if n >= start]
end = c3.selectbox(lex["to"], valid_end, index=0)
selected_verses = repo.verses(chapter, start, end)
chapter_context = parayan_chapter_context(chapter, len(selected_verses), len(numbers))
opening_duration, opening_rows = opening_context(segments, chapter, start)
include_opening = False
if start == 1 and opening_duration > 0:
    include_opening = st.checkbox(
        lex["opening_checkbox"],
        value=False,
        help=lex["opening_help"],
    )

st.subheader(f"1. {lex['selected_text']}")
parayan_rule_card()
if include_opening and not opening_rows.empty:
    st.info(lex["opening_included"])
    with st.expander(lex["opening_expander"], expanded=True):
        st.dataframe(opening_context_table(opening_rows), hide_index=True, use_container_width=True)
st.html("".join(verse_html(v) for v in selected_verses))
render_akshara_timing_summary(index, selected_verses, segments)

st.divider()
st.subheader(f"2. {lex['record_section']}")
left, right = st.columns(2)
with left:
    live = st.audio_input(lex["record"])
with right:
    uploaded = st.file_uploader(lex["upload"], type=["wav", "mp3", "m4a", "aac", "flac"])
source = live if live is not None else uploaded
if source is not None:
    st.caption(lex["listen"])
    st.audio(source)

run = st.button(f"3. {lex['run_button']}", type="primary", use_container_width=True, disabled=source is None)

if run and source is not None:
    payload = source.getvalue()
    suffix = Path(getattr(source, "name", "recording.wav")).suffix or ".wav"
    try:
        student = decode_audio(payload, suffix=suffix)
    except Exception as exc:
        st.error(f"{lex['unable']}: {exc}")
        st.stop()
    quality = quality_report(student)
    if not quality.passed:
        st.error(lex["quality_fail"] + " " + "; ".join(quality.reasons))
        st.stop()

    with st.spinner(lex["akshara_asr_spinner"]):
        akshara_match: AksharaMatch = compare_aksharas(student, selected_charans(selected_verses, segments))
    if akshara_match.available and akshara_match.message == "SELECTION_OR_DECODE_MISMATCH":
        st.error(lex["wrong_selection_title"])
        st.warning(lex["akshara_selection_warning"])
        st.stop()

    st.subheader(f"3. {lex['result_section']}")
    selected_seq = tuple(v.shloka for v in selected_verses)
    reference_duration, missing_index, timing_verified = selected_reference_duration(index, selected_verses)
    if reference_duration is not None:
        comparison_duration = reference_duration + (opening_duration if include_opening else 0.0)
        comparison_label = lex["opening_plus_selected"] if include_opening else lex["selected_only"]
        status, meaning, next_step = instant_guidance(student.duration_sec, comparison_duration, lex)
        status_card(status, meaning, next_step, lex)
        compact_time_card(comparison_label, comparison_duration, student.duration_sec)
        if not include_opening:
            global_match = global_duration_candidate(index, chapter, selected_seq, comparison_duration, student.duration_sec)
            if global_match:
                best_chapter, best_seq, best_diff, selected_diff = global_match
                st.error(lex["wrong_selection_title"])
                st.warning(
                    lex["wrong_selection_message"].format(
                        chapter=best_chapter,
                        shlokas=", ".join(map(str, best_seq)),
                        selected_chapter=chapter,
                        selected_shlokas=", ".join(map(str, selected_seq)),
                    )
                )
                st.info(
                    f"{lex['wrong_selection_action']} "
                    f"(selected difference {selected_diff:.2f}s; better match difference {best_diff:.2f}s)"
                )
                st.stop()
        if student.duration_sec > comparison_duration + 0.9:
            st.warning(lex["continue_warning"])
        if include_opening:
            st.caption(
                f"{lex['opening_duration']}: {fmt_seconds(opening_duration)}s; "
                f"{lex['selected_duration']}: {fmt_seconds(reference_duration)}s."
            )
        if not timing_verified:
            st.info(lex["timing_review"])
        candidate = None if include_opening else duration_candidate(index, chapter, selected_seq, student.duration_sec)
        if candidate:
            seq, diff = candidate
            st.warning(
                f"{lex['selection_mismatch']}: by duration, this recording also resembles "
                f"shloka(s) {', '.join(map(str, seq))} within {diff:.2f}s. "
                "Please confirm whether the selected shloka range is correct."
            )
        try:
            trainer_path = Path(get_trainer_audio(chapter))
            segment_start, segment_end, segment_missing = selected_segment_bounds(index, chapter, selected_verses, include_opening)
            if segment_start is None or segment_end is None:
                st.warning(f"{lex['missing_trainer_segment']} Missing shlokas: {segment_missing}")
                st.stop()
            st.subheader(lex["trainer_segment"])
            st.caption(
                lex["trainer_segment_caption"].format(
                    start=fmt_seconds(segment_start),
                    end=fmt_seconds(segment_end),
                    duration=fmt_seconds(segment_end - segment_start),
                )
            )
            clip_bytes, clip_format = trainer_segment_clip(str(trainer_path), segment_start, segment_end)
            st.audio(clip_bytes, format=clip_format)
            swara_issues = render_swara_audio_check(student, trainer_path, index, selected_verses, segment_start, segment_end, segments)
            word_issues = render_word_audio_check(student, trainer_path, selected_verses, segment_start, segment_end, segments)
            render_highlighted_text(
                selected_verses,
                segments,
                issues_to_keys(swara_issues or []),
                {},
                word_highlight_keys(word_issues),
                akshara_highlight_keys(akshara_match.issues) if akshara_match.available else set(),
            )
            render_understand_correction(repo, index, selected_verses, segments, lex, chapter_context)
        except Exception as exc:
            st.info(f"{lex['unable']}: {exc}")
    else:
        st.warning(f"{lex['unable']}: trainer timestamps missing for shlokas {missing_index}.")
        st.info(lex["index_missing"])


