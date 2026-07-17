# LearnGeeta Recitation Practice

An evidence-first Streamlit app for known-text Bhagavad Gita recitation practice.

## Guarantees

- No marks, grades, percentage, ranking or pass/fail.
- No random timing, fabricated charts or predetermined error words.
- Canonical Sanskrit text is never rewritten.
- Official aghata locations come from `Gita_Aghata_Master_FINAL.csv`.
- Trainer audios are loaded from local files first; online download is only a fallback.
- The app gives fast practice guidance from verified trainer timestamps and abstains from unsupported pronunciation verdicts.

## Run

```bat
cd /d "C:\Users\jatin\Documents\Codex\2026-07-15\do-not-stop-after-writing-the"
RUN_APP.bat
```

For online deployment, the core app files are:

```text
app.py
src/
data/
requirements.txt
packages.txt
README_HUGGINGFACE_SPACE.md
.streamlit/config.toml
.hfignore
```

Windows helper files such as `.bat` scripts are only for local use and are not required on Hugging Face Spaces.

If CSV data is hosted in a Hugging Face Dataset, set:

```text
LEARNGEETA_DATA_BASE_URL=https://huggingface.co/datasets/<user>/<dataset>/resolve/main/data
```

The app will use local `data/` files first and download from that base URL only when a required file is missing.

All `.bat` files now use one shared Python environment:

```text
D:\GP\.venv
```

So dependencies are not installed again in every project folder. If you want a different shared location, set `GP_VENV` before running the batch file.

FFmpeg must be available on PATH for MP3/M4A decoding. Browser recording produces WAV and normally does not require FFmpeg.
If a dependency such as `pandas` is reported missing, run `REPAIR_DEPENDENCIES.bat` once and then run the app or indexer again.

If old duplicate project environments were created earlier, run:

```bat
CLEAN_OLD_DUPLICATE_ENVIRONMENTS.bat
```

It keeps `D:\GP\.venv` and removes only duplicate `.venv` folders after showing you the list and asking for confirmation.

## Trainer audio

The app uses local trainer audio first and downloads only as a fallback. Put files named `CH01.mp3` through `CH18.mp3` in one of these folders:

- `master_audios`
- `data\master_audios`
- `audio`
- `D:\Geeta Aghata`
- `D:\Geeta Aghata\master_audios`

The same local-first rule is used by `RUN_TRAINER_INDEX.bat`.

## Build the conservative trainer index

The repository intentionally ships with an empty `data/trainer_index.csv`: missing timestamps must not be guessed.

If `D:\Geeta Aghata\Bhagavad Gita recitation.csv` already contains chapter, shloka, charan and timestamp rows, use the fast import path:

```bat
IMPORT_TRAINER_INDEX_FROM_CSV.bat
```

This does not transcribe audio. It converts the prepared CSV into `data\trainer_index.csv`.

Only use the slower audio indexer when timestamps are not already available:

```bat
RUN_TRAINER_INDEX.bat
```

Rows imported from the prepared timing CSV are used as the trusted trainer timestamp index. The app refuses pronunciation verdicts for missing or unverified rows.
Timing is always per audio file. Each chapter/audio is treated independently and its timestamps restart from that audio's own zero point; timestamps are never carried across chapters.
The indexer shows progress, uses a faster CPU default (`small`, `beam-size 1`), and skips chapters already present in `data\trainer_index.csv`. Add `--rebuild` to process existing chapters again.

## Current acoustic scope

Implemented:

- learner-first three-step flow: select text, record/upload, instant check;
- full interface language selector with English, Hindi, Assamese, Bengali, Gujarati, Kannada, Malayalam, Manipuri, Marathi, Nepali, Odia, Devanagari Sindhi, Tamil and Telugu;
- Parayan-aware wording: one breath, one line; trainer audio is treated as a learning reference, not a strict speed limit;
- real audio decoding and resampling;
- duration, level, clipping and silence quality gates;
- normalized relative-pitch visualization;
- instant selected-shloka duration check from verified trainer timestamps;
- fast duration-based possible mismatch warning, without claiming a hard verdict;
- trainer charan timing display from the prepared CSV;
- expected anusvāra/anunāsika guidance from the official rule data when mapped;
- exact selected text preview and in-place record/upload/playback;
- official expected aghata locations without pretending to hear dvitva;
- no-score reporting.

Withheld until expert-calibrated phone alignment exists:

- hard letter-level pronunciation verdicts;
- missing/extra/shifted aghata verdicts;
- anusvara nasal-class verdicts;
- hrasva/dirgha and aspiration accusations.

This abstention is deliberate: the app must not trouble a learner with unsupported claims.

## Parayan timing principle

The app follows this interpretation:

- the uploaded trainer/master audio is primarily for learning;
- Parayan practice follows **one breath, one line**;
- with practice, recitation may become faster;
- faster or slower timing alone is not an error;
- timing guidance only asks the learner to check for missing text, extra text, repeated text or long pauses.

## What the learner sees

The main screen is intentionally simple:

1. Select chapter and shloka.
2. Read the selected Devanagari recitation text.
3. Record or upload the chanting.
4. Click **Check my practice**.
5. See a plain result card with:
   - what the timing suggests;
   - what to do next;
   - no marks, no score, no punishment language.

Technical details such as recording level, clipping, silence and pitch are kept inside collapsed advanced sections.
