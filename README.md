# SneaKey

A local research prototype that asks: **can individual keyboard keys be identified from the sound of typing,
recorded by a laptop microphone?** It records consented typing tests, shows how each keystroke looks and sounds
through the processing pipeline, and evaluates an audio-only prediction pipeline against what was actually typed.

Everything runs on one laptop. The server listens on `127.0.0.1` only, the page makes no external requests, and
recordings are stored under `data/` (git-ignored).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py            # then open http://127.0.0.1:8000
```

Tested with Python 3.14, numpy 2.5, scipy 1.18, scikit-learn 1.9, torch 2.15 (CPU), FastAPI 0.162 and Firefox. The microphone must run at 48 kHz (the models were trained at that rate).

## The four sections

**1. Record.** Enter a participant ID and setup, confirm consent, and type a generated test while the
microphone records raw audio (echo cancellation, noise suppression and gain all off). Every keydown and keyup is
logged with its estimated position in the audio. The session is saved to `data/recordings/session_<timestamp>/`.

**2. Analyse.** Waveform and two spectrograms you can switch between the raw audio and each stage's filtered
audio (detector 180 Hz, model input 700 Hz, click band 1.2 kHz), with optional noise-floor subtraction. Also
shown: the detector's novelty curve, its threshold and candidates, and the ground-truth keystrokes placed on
their sounds. Clicking a keystroke opens the inspector: the CNN's exact input (log-mel patch), its key
probabilities, the press classifier's output and features, and raw vs processed playback.

**3. Train.** Retrain the key classifier on your own recordings. Choose which recordings to train on and which
to hold out for validation, then pick a model: an **SVM** or an **LDA** (trains in seconds). CNN training is turned
off for now because it is slow on a laptop; the shipped CNN is still available for prediction. The page
shows training progress, then scores the new model *and* the active model on the same held-out presses, at the
true press times and optionally end to end. You decide whether to activate it. Detection is unchanged; only the
step that names each key is retrained.

**4. Predict.** Runs the audio-only pipeline, then scores it against the keystroke log. You get detection, key
and text metrics, a character diff of typed vs predicted text, and every word with its decoder confidence.
Clicking a word shows the per-key probability distribution of each press. There's also a timeline of typed vs
predicted keys, per-key recall, a confusion matrix, and pooled results across recordings.

## How prediction works

```
audio ─► detector (v6) ─► press classifier (v6) ─► CNN key classifier ─► raw text
         spectral flux       gradient boosting       log-mel, 49 keys     └─► dictionary word decoder ─► words
```

1. **Candidates.** Spectral flux of the 180 Hz high-passed audio, divided by its 251 ms running median, peaks
   above 1.5. Each peak is snapped to the loudest millisecond of the 1.2 kHz high-passed energy.
2. **Presses.** A gradient-boosting classifier on 13 loudness, shape, spectrum and spacing features keeps the
   candidates that sound like key presses (probability ≥ 0.35), not releases or noise.
3. **Keys.** A small CNN reads a 64-band log-mel spectrogram of the 700 Hz high-passed audio from 15 ms before
   to 45 ms after each press, averaged over ±2 ms shifts.
4. **Words.** A dictionary decoder aligns runs of letter presses to English words, allowing for missed and
   extra presses, and keeps raw letters when no word fits.

**Ground truth.** The recorder can't see the microphone's input delay (30–60 ms, different every session).
The delay is measured from the audio by lining up logged keydowns with click onsets, then each event is snapped
to its click. These positions are the labels for training and evaluation.

## Evaluation

The two metrics protocols from the original project are kept exactly:

| Metric | Definition |
|---|---|
| Detection recall / precision | Detected presses matched one-to-one to true presses within 20 ms |
| Key accuracy (detected) | Share of matched presses named correctly, audio only, end to end |
| Key accuracy (logged timing) | CNN at the true press positions (repeats excluded): the classifier alone. Also top-3 and balanced (mean per-key recall) |
| CER / WER | Levenshtein distance of the lowercase typed text vs predicted text, divided by the typed length |

**Every recording carries a role**, shown as a badge, so scores aren't misread:

| Role | Meaning |
|---|---|
| `unseen` | No model was trained or tuned on it: its scores measure generalisation |
| `trained` | The *active* model was trained on it (listed in its manifest): its scores are optimistic |
| `validation` | Used to choose the active model (its manifest), or listed in `data/splits.json`: mildly optimistic |
| `test`, `locked` | Assigned by you in `data/splits.json`. Training refuses both. **Locked** recordings can be analysed but Predict refuses them, keeping a final test set untouched |

```json
// data/splits.json (optional)
{ "validation": ["session_..."], "test": ["session_..."], "locked": ["session_..."] }
```

**Baseline.** The shipped models were trained on the original project's recordings, so every new recording
starts as `unseen`. On that project's held-out test session (one typist, one laptop), this code reproduces the
published results exactly:

- detection: 77.0% recall, 85.0% precision;
- key accuracy: 83.9% on detected presses, 77.4% with logged timing (top-3 90.0%);
- character error rate: 38.7% from keys alone, 29.5% after the dictionary stage.

These numbers won't transfer to another keyboard, microphone or room. Retrain on new recordings for that.

## Sentence generation

The original recorder reused 13 fixed sentences, so the same key sequences recurred across sessions.
`sneakey/sentences.py` composes new text from a small probabilistic grammar:

- **12 sentence frames:** statements, yes/no and wh-questions, commands, compound, subordinate-first and
  subordinate-last sentences, relative clauses, fronted time phrases, existentials and reported speech.
- **Variation within a frame:** optional modifiers switch on independently, and tense, modality and person
  vary with subject-verb agreement.
- **Anti-repetition:** each sentence is the best of 8 candidates. Candidates are penalised for word trigrams
  used in earlier recordings, for reusing content words from the same test, and for repeating frames.

Over 25 consecutive simulated tests (~10,000 characters), no within-clause word trigram was repeated, and
letter frequencies stayed close to English. "Favour under-recorded keys" mode mildly prefers words with letters that have little data. The seed,
mode and per-sentence frames are saved with each session, so the text can be regenerated.

## Project layout

```
run.py                 start the server (127.0.0.1:8000)
sneakey/               backend package
  config.py            paths, constants, splits
  audio.py             WAV input/output, filters
  detection.py         v6 detector and press classifier (ported, numerics unchanged)
  classifier.py        CNN key classifier
  features.py          SVM/LDA features (band energies + log-mel)
  models.py            classifier registry: shipped + trained models, the active one
  training.py          Train-page jobs: dataset, SVM/LDA/CNN training, comparison
  decoder.py           dictionary word decoder (+ press spans and candidate scores)
  pipeline.py          audio → presses, probabilities, text, words
  groundtruth.py       logged events → positions of their sounds
  evaluation.py        metrics, alignment, confusion
  analysis.py          intermediate signals for the Analyse view
  sentences.py         random test text
  sessions.py          session storage (original on-disk format)
  service.py           loaded pipeline, caching, roles
  api.py               HTTP routes
models/                trained models and settings, with manifests of what they were trained on
web/                   single page: index.html, css/, js/ (ES modules, no build step)
training/              build_dataset.py → train_cnn.py → tune_decoder.py
tests/                 unit, API and browser tests
data/                  recordings/, cache/, training/, splits.json (git-ignored, created on first save)
```

**Session format** (unchanged from the original recorder): `audio.wav` (mono 16-bit 48 kHz),
`keystrokes.csv` (one row per key event, including `code`, the physical key used as the label, and
`sample_index_est`) and `metadata.json`. New sessions add `participant`, `conditions` and `generator`
fields.

**API** (all under `/api`, documented in `sneakey/api.py`):

| Endpoint | Purpose |
|---|---|
| `sentences` | New test text |
| `sessions` | List and create sessions; save audio; delete |
| `sessions/{id}/audio.wav?filter=` | Audio as a given stage hears it |
| `sessions/{id}/signals` | Detector signals and ground truth |
| `sessions/{id}/keystroke?time_ms=` | What the models compute for one press |
| `sessions/{id}/predict` | Prediction and evaluation, cached per model version in `data/cache/` |
| `summary` | Pooled results across sessions |
| `models`, `models/{id}/activate` | List, activate and delete key classifiers |
| `training`, `training/cancel` | Start a training run, watch its progress and result, cancel it |

## Training models

The **Train page** is the normal way to retrain:

| Model | Input | Notes |
|---|---|---|
| SVM | 464 hand-made features per press: 64 band energies plus a 10-step × 40-band log-mel patch (the best SVM features from the original experiments) | RBF kernel, calibrated probabilities |
| LDA | Same features as the SVM | Shrinkage; trains in seconds |
| CNN | 64-band log-mel spectrogram | Same network and recipe as the shipped model. **Training disabled for now**: re-enable by adding `"cnn"` to `ENABLED_KINDS` in `sneakey/training.py` and restoring its button in `web/index.html` |

How each model did on the original project's recordings, scored on the same 2 held-out sessions with the
classifier given the true press times:

| Training data | Model | Key accuracy | Balanced | Top-3 | Train time |
|---|---|---|---|---|---|
| 4 sessions (4,184 presses) | LDA | 60.0% | 30.2% | 75.4% | 1 s |
| 4 sessions | SVM (C=10) | 55.9% | 26.3% | 73.8% | 24 s |
| 12 sessions (12,596 presses) | LDA | 66.9% | 37.4% | 79.5% | 3 s |
| 12 sessions | SVM (C=10) | 65.3% | 35.1% | 78.8% | 154 s |

In an earlier comparison on ~20k presses, scored on 4 different validation sessions, the CNN led with 60.3%
against 53.2% for the SVM and 53.1% for the LDA. Which model wins depends on how much data you have, so compare
them on your own recordings with the Train page.

Trained models are saved in `data/models/<id>/` with a manifest of their training and validation sessions. The
active model is recorded in `data/active_model.json`, and the shipped CNN stays available. Keys with fewer than
"min. presses" examples are left out of training, and the word decoder tolerates models that lack some keys.

The command-line scripts below reproduce the original CNN recipe exactly, using whole-recording filtering:

```bash
# 1. Assign roles in data/splits.json: a few validation sessions, a few test sessions (optionally locked ones).
.venv/bin/python training/build_dataset.py        # labelled presses → data/training/presses.npz
.venv/bin/python training/train_cnn.py validate   # train on train sessions, score on validation
.venv/bin/python training/train_cnn.py final      # train on train+validation → models/key_cnn.pt (+ .json manifest)
.venv/bin/python training/tune_decoder.py --save  # re-tune the word decoder for the new model
```

`final` keeps the previous model as `models/key_cnn.previous.pt`. The detector and press classifier are v6's
and are not retrained by these scripts.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v   # pipeline, ground truth, metrics, generator, API
.venv/bin/python tests/browser_check.py             # headless Firefox: record → analyse → predict
```

`tests/test_core.py` includes parity tests that compare against the original project (in `../SneaKey`, or set
`SNEAKEY_ORIGINAL`). They check detection times, keys, probabilities, text, ground truth and the published
metrics on its held-out test session; they're skipped if the original isn't present. The browser test uses
Firefox's fake microphone (a tone, not key sounds), so it tests the recording path, not acoustic accuracy.

## What changed from the original project

| Area | Change |
|---|---|
| Kept unchanged | Trained models, detector and press-classifier numerics, CNN, word decoder, labelling, metrics, session file format, and the recorder's capture and clock-sync logic |
| Consolidated | The used parts of the vendored v6 package (about 60 lines) were ported into `detection.py`/`groundtruth.py` instead of being imported via `sys.path`, and the temp-folder workaround for v6's labelling was removed. Verified identical by the parity tests |
| Dropped | v6's 19 MB SVM fallback (less accurate than the CNN), the live-microphone and WAV-upload pages, the standalone HTML tools and the earlier LDA experiment scripts. All remain in the original repository |
| Added | Train page with SVM/LDA/CNN and model registry, Analyse view, per-word probability view, evaluation roles and locked sets, prediction cache, sentence generator, participant/consent metadata, empty-recording safeguards |

## Experiment: LLM decoding (optional, off by default)

`sneakey/llm_decoder.py` tests replacing the dictionary decoder with DeepSeek. It **sends data off the laptop**:
each detected press's candidate letters (and, by default, the dictionary decoder's word guesses), never the
audio or the typed text. It reads its key from `data/deepseek.env` (git-ignored).

On the **Predict page** the "DeepSeek decoder" panel runs it. The panel is labelled as sending data, asks for
confirmation before every send, and shows the result next to the local decoders, with a diff and exactly what was
sent. Stored results are shown again without re-sending.

Results on held-out session 182727, with the same CNN probabilities given to every decoder:

| Decoder | CER | WER |
|---|---|---|
| Keys only | 38.7% | 92.9% |
| **Dictionary (the app's decoder, local)** | **29.5%** | **67.9%** |
| DeepSeek, press-by-press prompt | 129.7% | 204.2% |
| DeepSeek, word-slot prompt | 889% | 1303% |
| DeepSeek, top-3 candidate letters per press, grouped into words (`style=candidates`) | 55.2% | 82.7% |
| Dictionary decoder on the CNN + SVM + LDA probabilities averaged (local) | 32.2% | 73.2% |
| DeepSeek, averaged CNN + SVM + LDA candidates per press + dictionary word guesses (`style=ensemble`) | 36.9% | 72.0% |
| DeepSeek, the same candidates + the CNN dictionary decoder's word guesses | 35.3% | 73.2% |

The LLM either wrote fluent invented text, starting with pangrams such as "the quick brown fox…", or echoed the
noisy top letters. It never did the evidence alignment the dictionary decoder does, so the app keeps the
dictionary decoder. The candidate-letters format was the best LLM variant: 53 of 168 words exactly right against
the dictionary's 68. It recovered some phrases ("because the baby has fallen asleep") but rewrote others into
plausible, wrong sentences. The combined variant gave DeepSeek every model's candidate letters plus the best local
decoder's word guesses. It got 70 words exactly right against 68, but its other changes mostly invented words, so
both error rates were worse than the dictionary decoder's. The SVM and LDA were trained on 12 other sessions,
never on the test recording.

## Limitations

- The shipped models come from one typist on one laptop in one evening. Expect low accuracy on any other
  keyboard, microphone or room until you retrain.
- Text is lowercase: Shift and Caps Lock aren't reconstructed. Rare keys (digits, most punctuation) have little
  training data.
- The detector misses about 20% of presses and adds spurious ones. Most remaining errors come from detection and
  alignment, not classification.
- Generated sentences are grammatical but often nonsensical. That keeps a language model from guessing them
  easily, but it also differs from natural typing.
