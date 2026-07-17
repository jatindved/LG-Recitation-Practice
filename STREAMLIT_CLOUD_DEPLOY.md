# Deploy LearnGeeta App on GitHub + Streamlit Community Cloud

This is the recommended online path for the current app.

## What to upload to GitHub

Upload only these files/folders:

```text
app.py
src/
data/
requirements.txt
packages.txt
.streamlit/config.toml
README.md
STREAMLIT_CLOUD_DEPLOY.md
```

Do not upload:

```text
.cache/
.venv/
node_modules/
output/
outputs/
tmp/
work/
*.wav
*.m4a
*.mp3
user_result*.pdf
user_recording*
*.bat
```

The trainer chapter audio does not need to be stored in GitHub. The app uses the Hugging Face dataset:

```text
jatindved/geeta-master-audios
```

## GitHub steps

1. Create a new GitHub repository.
2. Name suggestion:

```text
learngeeta-recitation-practice
```

3. Upload the clean deployment package files.
4. Keep the repository private while testing.

## Streamlit Community Cloud steps

1. Open Streamlit Community Cloud.
2. Click **Create app** / **New app**.
3. Connect GitHub.
4. Select the repository.
5. Set:

```text
Branch: main
Main file path: app.py
```

6. Deploy.

## First test after deploy

Check:

- language dropdown opens;
- selected shloka text appears;
- recording/upload works;
- trainer audio segment plays;
- wrong chapter audio shows the mismatch warning;
- no marks/score/report-card language appears.

## Important notes

- Streamlit free apps can sleep when inactive, but this is still simpler than Hugging Face GPU/Docker for this app.
- The app does not run transcription online. It reads timing from CSV.
- `packages.txt` installs `ffmpeg` for audio clipping.
- If a data file is missing, the app will fail early instead of guessing.
