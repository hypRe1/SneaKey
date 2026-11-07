# SneaKey

SneaKey is a keystroke-acoustics research project: can individual keyboard keys be identified from the sound they make, recorded by a laptop's built-in microphone? It covers the whole chain: a browser tool that records synchronised keyboard audio and labelled key events, analysis that measures how well the two are actually aligned, extraction of per-keystroke audio clips, and classifiers evaluated on sessions they were not trained on.

All data so far comes from one person typing on one laptop (built-in microphone, 48 kHz) during a single evening, recorded with their own consent. Recordings are **not** in this repository (see [Data](#data-and-privacy)).

## Results at a glance

25 keys (Space, Backspace, Comma, Period and 21 letters; rarer keys have too few samples), 12 recorded sessions, about 12,000 key presses. Classifier: shrinkage LDA on spectral features of a 30 ms clip. Numbers are **leave-one-session-out** over 9 development sessions: every prediction comes from a model that never saw that session. Three further sessions are locked away as a final test set and **have not been evaluated yet**.

| What is measured | Accuracy | Top-3 | Notes |
|---|---|---|---|
| Clean clips: one isolated, confidently detected key sound | **79.2%** (balanced **65.0%**) | 91.0% | 38% of presses of the 25 keys qualify |
| Every press of one of the 25 known keys | **49.4%** | 64.1% | closest to real typing on this machine |
| Chance | 4% balanced | 12% | |

**The clean-clip number is not real-world typing accuracy.** It only covers presses where a single key sound was found cleanly, which excludes overlapping keystrokes and the half of all presses whose click was too quiet to detect. Across every press of a known key the model is right about half the time, and keys outside the 25 can't be predicted at all. Balanced accuracy averages per-key recall, so the very common Space key (98% recall) doesn't dominate it. Across the nine held-out sessions, clean balanced accuracy ranges from 45% to 70% and every-press accuracy from 33% to 64%.

## How it works

```
record_keystrokes.html ──> recordings/session_*/{audio.wav, keystrokes.csv, metadata.json}
        │
        ▼
analysis/sync_analysis.py ──> analysis_output/<session>/ (measured audio/keyboard offset, drift, matched onsets)
        │
        ▼
analysis/extract_keystroke_segments.py ──> datasets/<session>/ (labelled clips + manifest with clean/overlap flags)
        │
        ▼
analysis/experiments.py, analysis/error_analysis.py ──> analysis_output/experiments/, analysis_output/error_analysis/
```

**1. Recording** (`record_keystrokes.html`, `sneakey/`). A local, Monkeytype-style typing test in vanilla HTML/JS. It records mono 16-bit PCM from the microphone with echo cancellation, noise suppression and automatic gain disabled, and no processing applied. It logs every keydown and keyup with `event.key` (character), `event.code` (physical key, used as the label), modifiers, repeats, and what the app did with the key, including wrong characters and backspaces.

**2. Synchronisation.** The page stores raw `performance.now()` timestamps and maps them to audio sample indices from AudioWorklet frame counters and periodic clock pairs. That mapping cannot see microphone and driver input latency. So `analysis/sync_analysis.py` measures the real offset from the audio, without assuming the estimate is aligned:
- it detects acoustic onsets in a 1.5–10 kHz band;
- it searches for the lag between key events and onsets, across the whole recording and in sliding windows;
- it separates a constant offset (input latency) from drift.

Measured across all 12 sessions, sounds arrive **34–59 ms after the logged keydown**. The value is stable within a session but changes each time the microphone is opened. Drift stays within ±13 ppm, and the browser's clock-pair fit has about 3 ms residual. Key releases are matched using the press lag, because the press and release sounds share an input path.

**3. Segmentation** (`analysis/extract_keystroke_segments.py`). For each event, the clip is anchored on the detected onset. A clip is cut 5 ms before the onset and 25, 45 or 75 ms after it. Clips are bit-exact int16 slices of the original WAV, labelled by `event.code`, with key presses and releases kept as separate datasets.
- **Clean:** a clip counts as clean only if no other key's sound (detected, inseparable, or audible but undetected) and no unexplained transient falls inside it, and it is no louder than the session's own background.
- **Excluded clips are kept:** everything else is recorded with its reason in `manifest.csv`.
- **Calibrated rules:** the contamination checks have measured false-alarm rates.
- **Window length:** the 30 ms window carries almost all of a key's own sound energy; longer windows mainly pick up neighbouring keys.

**4. Classification** (`analysis/experiments.py`).
- **Features:** 64 log-spectral bands plus 20 MFCCs with deltas over the 30 ms clip.
- **Model:** a shrinkage LDA, trained on clean clips.
- **Protocol:** all model and feature choices were made by leave-one-session-out on 9 development sessions. Three later sessions are locked for a single final test.
- **Rejected alternatives:** LDA beat logistic regression, SVMs, gradient boosting, extra trees and nearest-neighbour models.
- **CNN status:** a small CNN with augmentation was started but stopped after 2 of 9 folds. On both of those folds it scored below LDA, so it remains inconclusive.

**5. Error analysis** (`analysis/error_analysis.py`, `analysis/error_report.py`). It decomposes where errors come from, using only the development sessions; see [Findings](#findings).

**Inspection tools.** These are local web pages that read generated outputs:
- `session_viewer.html` plays a recording with labels overlaid on a spectrogram.
- `prediction_dashboard.html` replays held-out predictions.
- `live_predict.html` runs the model on live microphone audio. It uses keyboard events to time and score each press, and the prediction comes from audio only.

## Repository layout

```
record_keystrokes.html      recorder / typing test
session_viewer.html         listen to a session with labels and spectrogram
prediction_dashboard.html   replay held-out predictions
live_predict.html           live microphone predictor (same machine only)
sneakey/                    browser modules: core logic, recorder + live AudioWorklets, passages, JS feature extraction
analysis/                   Python: sync analysis, segmentation, classifiers, experiments, error analysis, exports
tests/                      Node and Python tests, headless-Firefox end-to-end test
requirements.txt            Python dependencies
```

## Setup

All commands run from the repository root; scripts use paths relative to it.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# optional, only for the CNN experiment scripts (analysis/exp_cnn.py, analysis/exp_final.py):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m http.server 8000          # serve the HTML tools at http://localhost:8000/
```

Tested with Python 3.14.7, numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.2, scikit-learn 1.9.1, torch 2.15.0 (CPU), Node 24.13, and Firefox 154 on Linux/Wayland.

## Usage

1. **Record.** Open `http://localhost:8000/record_keystrokes.html`, press *Start session*, and type the paragraph. Download the session zip and unzip it into `recordings/`.
2. **Measure synchronisation**, per session: `.venv/bin/python analysis/sync_analysis.py recordings/<session> --out analysis_output/<session>`. This writes a report, statistics, matched onsets and plots.
3. **Extract clips**, per session: `.venv/bin/python analysis/extract_keystroke_segments.py --session recordings/<session> --analysis analysis_output/<session> --out datasets/<session>`.
4. **Experiments** (development sessions only; the locked test sessions are listed in `TEST_SESSIONS` in `analysis/experiments.py`):
   ```
   .venv/bin/python analysis/experiments.py cache                       # cache wide slices around every press
   .venv/bin/python analysis/experiments.py features                    # stage A: feature sets
   .venv/bin/python analysis/experiments.py models band64+mfcc20x10+d   # stage B: model families
   ```
5. **Error analysis:** `.venv/bin/python analysis/error_analysis.py && .venv/bin/python analysis/error_report.py` writes to `analysis_output/error_analysis/`.
6. **Tools:**
   - `session_viewer.html?dir=recordings/<session>` shows a session with labels.
   - `.venv/bin/python analysis/export_predictions.py` writes the data that `prediction_dashboard.html` replays.
   - `.venv/bin/python analysis/export_model.py` exports the model that `live_predict.html` runs.

   `classify_baseline.py`, `export_predictions.py` and `export_model.py` use **every** session in `datasets/`, including the locked test sessions. They are inspection tools, not part of the locked-test protocol.

## Tests

```
node --test tests/core.test.mjs              # recorder logic: WAV/CSV/zip/clock-sync/typing model (checked against Python's wave/csv/zipfile)
node --test tests/test_live_features.mjs     # browser feature extraction + classifier == Python (skipped until export_model.py has run)
.venv/bin/python -m unittest tests/test_sync_analysis.py tests/test_extract_segments.py
python3 tests/e2e_firefox.py                 # headless Firefox + fake microphone; needs the http.server on :8000
```

- **Python suites:** they generate synthetic sessions with known latency, drift, jitter, missing clicks and overlaps, and check that the analysis recovers them. One test checks that no clip marked clean contains another key's ground-truth click.
- **End-to-end test:** it drives the recorder with synthetic key events and Firefox's fake microphone. It tests the pipeline, not real keyboard acoustics.

## Findings

From the development-session error analysis (`analysis_output/error_analysis/report.md` when generated locally):

**Most errors come from finding the keystroke, not classifying it.** Of the 50.6% error across every press:
- 20.8 points are classifier mistakes on clean clips.
- **27.9 points come from presses with no detected onset.** These are half of all presses, and they score 23.0%.
- Overlapping presses with a detected onset cost under 1 point.

**The model is extremely sensitive to alignment.** Shifting clean clips by 2 ms drops their accuracy from 79% to 52–61%; a 5 ms shift drops it to 42–47%. Re-anchoring undetected presses on the nearest energy peak, using the same models, raises every-press accuracy from 49.4% to 54.8%.

**Confusions follow keyboard geometry.**
- Adjacent keys are confused about 2.8× as often as other pairs.
- Keys in the same row are confused about twice as often as keys at the same horizontal distance in different rows.
- D, E, L and O form one confusion cluster, accounting for 15% of clean-clip errors.
- Same-finger pairs are confused more, but that is mostly explained by those keys being adjacent.

**Sessions differ.** Typing speed did not explain accuracy. The worst session's keys sounded 5.5 dB quieter than the same keys in other sessions, and how far a session's sound differed from the others tracked its accuracy (Spearman −0.83). That is only 9 sessions and is correlational evidence.

**Class imbalance is minor.** The 14-point gap between accuracy and balanced accuracy is almost entirely the Space key. Correcting class priors gains about 2 balanced points.

## Limitations

- **One typist, one laptop, one microphone, one evening.** Nothing has been tested on another keyboard, room, microphone or person, and results are not expected to transfer.
- **The final test set has not been evaluated.** The 65.0% / 79.2% figures are from development sessions that were also used to choose the model, so they are mildly optimistic.
- **Only 25 keys are classified.** Digits, rare letters (q, z, x, j) and most punctuation have too few clean samples.
- **Every-press accuracy is about 49%.** Onset detection and alignment are the main bottleneck.
- **The live predictor has not been validated on real typing.** Its feature and model maths match Python to about 1e-6, but only fake-microphone tests have been run.
- **The baseline model was chosen on these same sessions.** The label set was also defined from the development sessions' counts. The leave-one-session-out procedure and its leakage checks are described in the error-analysis output.

## Next steps

These follow from the error analysis:
1. **Make classification robust to misaligned clips, and re-anchor every press.** For example, train on copies shifted by ±1–3 ms.
2. **Reduce differences between sessions.** For example, apply unsupervised per-session level alignment, and record a `conditions` field (position, surface, posture) with each session.
3. **Collect targeted data for weak and confusable keys** (G, Period, K, W, V, Comma, Y, M and the D/E/L/O group), and steer the recorder towards them.

Only then evaluate the locked test sessions once, and revisit neural networks.

## Data and privacy

Recordings are audio of real typing together with the text that was typed, so none of it is committed. `.gitignore` excludes:
- `recordings/`, `session/` and session zips;
- extracted datasets (`dataset/`, `datasets/`);
- all generated outputs (`analysis_output/`: reports, plots, cached features, predictions, exported models);
- logs, virtual environments and caches.

All processing is local; the HTML tools only load files from the local `http.server` and make no external network requests.
