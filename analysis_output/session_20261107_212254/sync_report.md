# Synchronisation report: session_20261107_212254

* WAV: `recordings/session_20261107_212254\audio.wav`  (115.76 s, 48000 Hz)
* keystrokes.csv: `recordings/session_20261107_212254\keystrokes.csv`
* metadata.json: `recordings/session_20261107_212254\metadata.json`

## Acoustic transient detection

Band-pass 1500-10000 Hz (zero-phase, so no delay is added), 0.5 ms RMS envelope, peaks > 4.0x the rolling 1 s median floor, onset = leading edge where the envelope first rises 20% of the way from floor to peak. **373 onsets** (3.22/s).

## Events

2040 events (1020 keydown, 1020 keyup). 0 auto-repeat keydowns are excluded from matching because a held key makes no new sound. Offsets below are `onset - sample_index_est/sample_rate`; positive = sound later than the estimate.

## 1. Does a timing relationship exist? (lag search, no alignment assumed)

All (onset - event) differences within the window were histogrammed. A peak is called significant when the number of pairs within +/-3 ms of the mode is implausible under the measured background rate (Poisson p < 1e-6) and the peak is at least 3x the background. This is done for the whole recording and again in sliding 20 s windows; the matching gate follows the *tracked* lag, so drift cannot push real onsets out of the window.

| class | whole-recording mode (ms) | p-value | significant | windows significant | tracked lag range (ms) | search window (ms) |
|---|---|---|---|---|---|---|
| down | 337.3 | 3.4e-03 | no | 0/11 (**NO**) | n/a | -150 .. 400 |
| up | 278.1 | 2.4e-03 | no | 0/11 (**NO**) | n/a | -150 .. 400 |

If the whole-recording peak is weak but most windows are significant with a moving mode, the offset is **drifting** (a sync error), not absent.

Note: keyups matched around the keydown lag (shared input path); keyup free-search mode 278.1 ms is diagnostic only (a negative value indicates rollover onto the next key's press).

![lag search](03_lag_search.png)

## 2. Matching, missed detections and ambiguity

Each eligible event gets a window of +/-40 ms around (estimate + its class's lag mode). *matched* = exactly one onset in the window and no other event claims it; *ambiguous* = several onsets in the window, or the onset is also in another event's window (typical in fast typing, including a release click landing in the next press window); *missed* = no onset in the window.

| class/category | eligible | matched | ambiguous | missed | miss rate |
|---|---|---|---|---|---|
| down/char | 962 | 28 | 120 | 814 | 84.6% |
| down/backspace | 35 | 1 | 4 | 30 | 85.7% |
| down/modifier | 23 | 1 | 4 | 18 | 78.3% |
| up/char | 962 | 23 | 126 | 813 | 84.5% |
| up/backspace | 35 | 2 | 5 | 28 | 80.0% |
| up/modifier | 23 | 0 | 5 | 18 | 78.3% |

Onsets not inside any event window: **62** (16.6% of all onsets): other sounds (speech, desk, mouse, touchpad), quiet keys detected late, or windows the gate excluded.

![match status](07_match_status_and_intervals.png)
![zoom](02_zoom_windows.png)
![overview](01_overview_waveform.png)

Gate check: nearest onset in the full lag window for each character keydown lies inside the gate for **27.2%**; a low value means the gate is cutting off a real tail and the statistics below under-state the spread.

## 3. Offset statistics (ms; onset - estimate)

| set | n | mean | median | std | robust sigma (MAD) | 2.5%..97.5% | min..max | IQR |
|---|---|---|---|---|---|---|---|---|
| down/char/matched | 28 | 331.88 | 331.90 | 20.30 | 26.70 | 304.16 .. 367.08 | 303.63 .. 369.19 | 34.78 |
| down/backspace/matched | 1 | 352.44 | 352.44 | 0.00 | 0.00 | 352.44 .. 352.44 | 352.44 .. 352.44 | 0.00 |
| down/modifier/matched | 1 | 319.71 | 319.71 | 0.00 | 0.00 | 319.71 .. 319.71 | 319.71 .. 319.71 | 0.00 |
| up/char/matched | 23 | 351.95 | 364.17 | 23.76 | 15.23 | 308.58 .. 376.95 | 307.90 .. 377.27 | 38.30 |
| up/backspace/matched | 2 | 338.18 | 338.18 | 14.92 | 15.64 | 328.15 .. 348.20 | 327.62 .. 348.73 | 10.55 |
| down/char/matched+ambiguous | 148 | 334.85 | 336.52 | 18.52 | 17.54 | 302.89 .. 373.00 | 300.85 .. 377.31 | 25.21 |
| up/char/matched+ambiguous | 149 | 341.24 | 341.58 | 20.43 | 25.11 | 303.57 .. 371.86 | 298.40 .. 377.27 | 33.96 |
| ungated nearest onset, keydown/char | 545 | 185.74 | 216.42 | 155.18 | 174.76 | -126.48 .. 393.98 | -148.67 .. 399.98 | 256.73 |

![offset distribution](04_offset_distribution.png)
![event-triggered](06_event_triggered_envelope.png)

### Press versus release

Median offset for keydown **331.9 ms**, for keyup **364.2 ms** (difference +32.3 ms; sd 20.3 / 23.8 ms). Both go through the same input chain and clock mapping, so a shared constant offset points to latency / mapping. A *difference* between them is physical (release click timing versus when the OS reports the release) and cannot be a clock error. Every key can click on both press and release, so release events are matched in their own class rather than counted as misses of the press.

## 4. Clock-synchronisation error versus input latency

* **Constant component**: median keydown offset **331.9 ms**, no significant drift. This is the sum of microphone/driver/browser input latency, the physical delay between finger contact and the OS key event, any constant bias in the clock mapping, and the detector's definition of 'onset'. These cannot be separated by this recording alone; they are not clock *error* because they do not change over time.
* **Drift (clock rate mismatch)**: Theil-Sen slope **-39 ppm** (95% CI -310 .. +201); accumulates **-4.2 ms** across the typed span. A CI excluding 0 means the sync model's slope is measurably wrong; real oscillators differ by <~100 ppm.
* Median offset in first / middle / last third of the typing: 336.1, 317.7, 324.4 ms.

Same matched onsets re-timed with each clock mapping saved in metadata.json (primary = `output_timestamp`):

| mapping | median offset (ms) | std (ms) | slope source | effective rate (Hz) | its own fit residual RMS (ms) |
|---|---|---|---|---|---|
| output_timestamp | 331.9 | 20.30 | fitted | 47999.9 | 3.57 |
| current_time | 331.9 | 20.30 | fitted | 47999.9 | 3.57 |
| worklet_arrival | 331.6 | 20.30 | fitted | 47999.9 | 4.66 |

The spread of medians across mappings is mapping uncertainty alone; the page's own sync-fit residual shows how well it tracked its clock pairs, which says nothing about microphone input latency.

![offset vs time](05_offset_vs_time.png)

## 5. Sensitivity to detector settings

The headline offset depends on how 'onset' is defined (a later threshold crossing gives a later onset). If the median moves by several ms across rows, treat the constant offset as uncertain by that much.

| threshold (x floor) | rise fraction | onsets | matched | ambiguous | missed | median (ms) | std (ms) |
|---|---|---|---|---|---|---|---|
| 3 | 0.1 | 462 | 26 | 146 | 790 | 327.9 | 24.87 |
| 3 | 0.2 | 462 | 24 | 147 | 791 | 326.2 | 23.36 |
| 3 | 0.5 | 462 | 25 | 145 | 792 | 324.6 | 23.59 |
| 4 | 0.1 | 373 | 29 | 120 | 813 | 330.8 | 22.16 |
| 4 | 0.2 | 373 | 28 | 120 | 814 | 331.9 | 20.30 |
| 4 | 0.5 | 373 | 26 | 119 | 817 | 332.1 | 19.51 |
| 6 | 0.1 | 236 | 17 | 84 | 861 | 332.1 | 23.35 |
| 6 | 0.2 | 236 | 18 | 83 | 861 | 337.4 | 23.23 |
| 6 | 0.5 | 236 | 18 | 84 | 860 | 323.8 | 24.02 |
| 10 | 0.1 | 120 | 9 | 39 | 914 | 332.1 | 22.15 |
| 10 | 0.2 | 120 | 11 | 36 | 915 | 342.4 | 22.58 |
| 10 | 0.5 | 120 | 10 | 39 | 913 | 339.3 | 24.51 |

Median offset ranges over **323.8 .. 342.4 ms** across detector settings.

## 6. Can we cut labelled per-keystroke clips?

| criterion | value | threshold | pass |
|---|---|---|---|
| significant lag peak in a majority of time windows (keydown) | 0/11 | >= 50% | **NO** |
| detection rate, char keydowns | 15.4% | >= 90% | **NO** |
| unambiguous one-to-one matches | 2.9% | >= 80% | **NO** |
| p95 abs residual, constant offset removed | 33.2 ms | <= 10.0 ms | **NO** |

With the slow component also removed the p95 residual would be 33.2 ms (this is what an offset that tracks drift could reach).

Fraction of character keydown onsets landing within +/-X ms of (estimate + constant offset): 5 ms: 14%, 10 ms: 29%, 20 ms: 57%, 30 ms: 89%, 50 ms: 100%.

Crowding (another key press or release within +/-W of a character keydown): 50ms: 74%, 100ms: 99%, 150ms: 100%.

* **Fixed-offset clips** (cut at estimate + one constant offset, no re-anchoring): **NOT supported** at this precision.
* **Offset + linear drift clips** (cut at estimate + fitted line): **NOT supported** (p95 residual 33.0 ms vs 10.0 ms).
* **Onset-snapped clips** (find the onset near each estimate, cut around it, label from the matching keystroke): **NOT supported** for the 3% of keystrokes with an unambiguous match; ambiguous and missed keystrokes should be dropped or reviewed rather than labelled by guess.

## 7. Suggested correction (NOT applied)

Evidence-backed constant offset: **+331.9 ms** (keydown median; detector-setting range 323.8 .. 342.4 ms; see sections 3-5). `keystrokes.csv` and `metadata.json` are untouched. To write a *copy* with an extra `sample_index_corrected` column using a value you have decided on: `python analysis/sync_analysis.py recordings/session_20261107_212254\audio.wav --write-corrected-csv <ms>`.

## Limitations

* Onset detection is a heuristic; a quiet key, overlapping keys or room noise change what is found. The sensitivity table bounds this but does not remove it.
* The keyboard event time is not the moment of finger contact: switch actuation and key travel add a physical delay of a few ms to tens of ms that varies key to key, and it is folded into 'latency' above.
* This recording can show that offsets are stable or drifting; it cannot split input latency from physical delay without an independent reference (e.g. a loopback or a known acoustic source).

## Files

* `01_overview_waveform.png`
* `02_zoom_windows.png`
* `03_lag_search.png`
* `04_offset_distribution.png`
* `05_offset_vs_time.png`
* `06_event_triggered_envelope.png`
* `07_match_status_and_intervals.png`
* `event_matches.csv`
* `sync_stats.json`
