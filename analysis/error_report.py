#!/usr/bin/env python3
"""Build problems_ranked.csv and report.md from analysis_output/error_analysis/ (no fitting, no test sessions)."""
import csv, json
import numpy as np
from scipy import stats

D = "analysis_output/error_analysis"
R = json.load(open(f"{D}/error_analysis.json"))
rows = list(csv.reader(open(f"{D}/confusion_raw.csv")))
keys = rows[0][1:]; C = np.array([[int(v) for v in r[1:]] for r in rows[1:]]); n = C.sum(1)
ix = {k: i for i, k in enumerate(keys)}; short = lambda k: k.replace("Key", "")
N = int(n.sum()); errors = int(N - np.trace(C))

# row effect (controlled for horizontal gap), recomputed from the saved confusion matrix
R0 = {0: ("qwertyuiop", 0.0), 1: ("asdfghjkl", 0.25), 2: ("zxcvbnm,.", 0.75)}
POS = {(f"Key{c.upper()}" if c.isalpha() else {",": "Comma", ".": "Period"}[c]): (x + i, r) for r, (s, x) in R0.items() for i, c in enumerate(s)}
L = [k for k in keys if k in POS]; prs = [(a, b) for i, a in enumerate(L) for b in L[i + 1:]]
rate = np.array([(C[ix[a], ix[b]] + C[ix[b], ix[a]]) / (n[ix[a]] + n[ix[b]]) for a, b in prs])
dx = np.array([abs(POS[a][0] - POS[b][0]) for a, b in prs]); same_row = np.array([POS[a][1] == POS[b][1] for a, b in prs])
row_bands = []
for lo, hi in ((0, 1.5), (1.5, 3.5), (3.5, 10)):
    m = (dx > lo) & (dx <= hi)
    row_bands.append((lo, hi, rate[m & same_row].mean(), rate[m & ~same_row].mean(), stats.mannwhitneyu(rate[m & same_row], rate[m & ~same_row], alternative="greater").pvalue))
cluster = ["KeyD", "KeyE", "KeyL", "KeyO"]
OTHER_PAIRS = [("KeyI", "KeyK"), ("KeyH", "KeyN"), ("KeyM", "KeyN"), ("KeyC", "Comma"), ("KeyT", "Backspace"), ("KeyI", "KeyR"), ("KeyI", "KeyM")]
clus_err = int(sum(C[ix[a], ix[b]] for a in cluster for b in cluster if a != b))

pk = R["per_key"]; seg = R["classifier_vs_segmentation"]; cat = seg["categories"]; imb = R["imbalance_options"]
sh = {int(k): v for k, v in R["anchor_shift_sensitivity_clean_acc"].items()}; ra = R["reanchor_unconfident_presses"]
sess = R["sessions"]; worst = min(sess, key=lambda r: r["clean_bal_acc"])
weak = sorted([k for k in keys if pk[k]["recall"] < 0.5], key=lambda k: pk[k]["recall"])
weak_deficit = sum((1 - pk[k]["recall"]) for k in weak) / len(keys)
cbal = R["reproduces_stage_B"]["clean_bal_acc"]; cacc = R["reproduces_stage_B"]["clean_acc"]
corr = R["session_correlations_vs_clean_bal_acc"]; corr_all = R["session_correlations_vs_all_acc"]
qb = R["accuracy_by_onset_strength_confident"]

problems = [
    ("Presses with no detected acoustic onset", "every-press accuracy",
     f"{cat['no_onset_detected(estimated)']['share']*100:.1f}% of all presses; accuracy {cat['no_onset_detected(estimated)']['acc']*100:.1f}% (top-3 {cat['no_onset_detected(estimated)']['top3']*100:.0f}%)",
     f"{seg['error_budget_all_presses']['extra_error_from_each_category']['no_onset_detected(estimated)']*100:.1f} of the {seg['error_budget_all_presses']['observed_error']*100:.1f} error points"),
    ("Extreme sensitivity to clip alignment", "all metrics",
     f"same models, clean clips shifted: -2 ms {sh[-2]*100:.0f}%, +2 ms {sh[2]*100:.0f}%, +/-5 ms {sh[-5]*100:.0f}/{sh[5]*100:.0f}% vs {sh[0]*100:.0f}% aligned",
     f"mechanism behind #1 and live use; re-anchoring undetected presses (inference only) {ra['all_press_acc_before']*100:.1f}% -> {ra['all_press_acc_after']*100:.1f}% every-press"),
    ("Session-to-session acoustic shift", "clean balanced accuracy",
     f"held-out balanced accuracy {min(r['clean_bal_acc'] for r in sess)*100:.0f}-{max(r['clean_bal_acc'] for r in sess)*100:.0f}%; feature shift rho {corr['feature_shift'][0]:+.2f} (p={corr['feature_shift'][1]:.3f}); worst session {worst['session'][-6:]} keys {worst['level_offset_db']:+.1f} dB vs other sessions",
     f"{(np.mean([r['clean_bal_acc'] for r in sess]) - worst['clean_bal_acc'])*100:.0f} points below the mean in the worst session"),
    ("Rare keys with recall below 50%", "clean balanced accuracy",
     ", ".join(f"{short(k)} {pk[k]['recall']:.2f} (n={pk[k]['support']})" for k in weak),
     f"{weak_deficit*100:.1f} of the {(1-cbal)*100:.1f} balanced-accuracy deficit points"),
    ("D / E / L / O confusion cluster", "clean accuracy",
     f"{clus_err} errors among these 4 keys ({clus_err/errors*100:.0f}% of all {errors} clean errors); same-finger pairs D-E, L-O plus same-row cross-hand pairs D-L, E-O",
     f"{clus_err/N*100:.1f} points of clean accuracy"),
    ("Quiet presses", "clean/confident accuracy",
     f"confident-press accuracy by onset-strength quintile: {qb[0]['acc']*100:.0f}% (weakest) -> {qb[-1]['acc']*100:.0f}% (strongest)",
     "partly overlaps #1 (undetected presses are the quietest)"),
    ("Other confusable pairs (row/adjacency)", "clean accuracy",
     "; ".join(f"{short(a)}<->{short(b)} {C[ix[a], ix[b]] + C[ix[b], ix[a]]}" for a, b in OTHER_PAIRS),
     f"{sum(C[ix[a], ix[b]] + C[ix[b], ix[a]] for a, b in OTHER_PAIRS)} errors = {sum(C[ix[a], ix[b]] + C[ix[b], ix[a]] for a, b in OTHER_PAIRS)/N*100:.1f} points of clean accuracy"),
    ("Overlapping but detected presses", "every-press accuracy",
     f"{cat['detected_but_overlapping']['share']*100:.1f}% of presses, accuracy {cat['detected_but_overlapping']['acc']*100:.1f}%",
     f"{seg['error_budget_all_presses']['extra_error_from_each_category']['detected_but_overlapping']*100:.1f} error points"),
    ("Class-prior bias", "clean balanced accuracy",
     f"post-hoc prior correction {imb['baseline_frequency_priors']['clean_bal_acc']*100:.1f}% -> {imb['posthoc_prior_correction_same_models']['clean_bal_acc']*100:.1f}% balanced (accuracy {imb['baseline_frequency_priors']['clean_acc']*100:.1f}% -> {imb['posthoc_prior_correction_same_models']['clean_acc']*100:.1f}%); uniform-prior refit {imb['refit_uniform_priors']['clean_bal_acc']*100:.1f}%; balanced subsampling {imb['balanced_subsampling_median_cap_3_seeds']['clean_bal_acc']*100:.1f}%",
     "about 2 balanced points; not the main problem"),
]
with open(f"{D}/problems_ranked.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rank", "problem", "metric_affected", "evidence", "impact"])
    for i, p in enumerate(problems, 1): w.writerow([i] + list(p))

T = R["technical_checks"]; P = R["proximity"]; dec = R["accuracy_minus_balanced_decomposition"]
md = [
    "# Error analysis: shrinkage LDA (band64 + MFCC20x10 + deltas), clean 30 ms press clips", "",
    f"Leave-one-session-out on the 9 development sessions, 25 keys. Refit reproduces stage B: clean balanced accuracy **{cbal*100:.1f}%**, accuracy **{cacc*100:.1f}%**. "
    "The 3 locked test sessions were removed before any feature was computed and were not evaluated.", "",
    "## Where the errors come from (every press of a known key)", "",
    "| category | share of presses | accuracy | error points added |", "|---|---|---|---|",
    f"| clean clip | {cat['clean']['share']*100:.1f}% | {cat['clean']['acc']*100:.1f}% | {seg['error_budget_all_presses']['classifier_error_at_clean_quality']*100:.1f} (classifier error) |"]
for c_ in ("detected_but_overlapping", "uncertain_onset", "no_onset_detected(estimated)"):
    md.append(f"| {c_.replace('_', ' ')} | {cat[c_]['share']*100:.1f}% | {cat[c_]['acc']*100:.1f}% | {seg['error_budget_all_presses']['extra_error_from_each_category'][c_]*100:.1f} |")
md += ["", f"Every-press accuracy is **{seg['all_press_acc']*100:.1f}%**: {seg['error_budget_all_presses']['classifier_error_at_clean_quality']*100:.1f} error points are the classifier itself, "
       f"**{seg['error_budget_all_presses']['extra_error_from_each_category']['no_onset_detected(estimated)']*100:.1f} come from presses whose onset was never detected** (half of all presses), and overlap costs almost nothing once an onset is found.", "",
       "Why undetected presses fail: (a) the model is extremely alignment-sensitive. Shifting clean clips with the same fitted models:", "",
       "| shift | -10 ms | -5 ms | -2 ms | 0 | +2 ms | +5 ms | +10 ms |", "|---|---|---|---|---|---|---|---|",
       "| clean accuracy | " + " | ".join(f"{sh[d]*100:.0f}%" for d in (-10, -5, -2, 0, 2, 5, 10)) + " |", "",
       f"Undetected presses are anchored at estimate + session lag, whose per-press error is ~5-11 ms (onset-lag sd per session), i.e. squarely in the collapse region. (b) They are also the quietest presses: even among detected presses accuracy rises from {qb[0]['acc']*100:.0f}% to {qb[-1]['acc']*100:.0f}% from the weakest to the strongest onset quintile.", "",
       f"Inference-only test: re-anchoring the {ra['n']} undetected/uncertain presses to the strongest high-frequency peak within +/-15 ms (same models) raises their accuracy {ra['acc_before']*100:.1f}% -> {ra['acc_after']*100:.1f}% and every-press accuracy **{ra['all_press_acc_before']*100:.1f}% -> {ra['all_press_acc_after']*100:.1f}%**.", "",
       "## Confusions", "",
       f"{errors} errors on {N} clean clips. Most confused pairs (both directions):", "",
       "| pair | errors | a->b / b->a | rate | physical distance | same finger |", "|---|---|---|---|---|---|"]
for t in R["top15_confused_pairs"]:
    md.append(f"| {t['pair']} | {t['errors']} | {t['a_to_b']}/{t['b_to_a']} | {t['rate']:.3f} | {'' if t['physical_distance'] is None else f'{t['physical_distance']:.1f}'} | {'yes' if t['same_finger'] else ''} |")
md += ["", f"* **Adjacency matters**: adjacent pairs are confused {P['confusion_rate_adjacent_vs_not'][0]/P['confusion_rate_adjacent_vs_not'][1]:.1f}x as often as others (p={P['mannwhitney_adjacent_gt_not_p']:.1e}); distance vs confusion Spearman {P['spearman_physical_distance_vs_confusion_rate'][0]:+.2f} (p={P['spearman_physical_distance_vs_confusion_rate'][1]:.4f}). This updates the 5-session result, which found no distance effect.",
       f"* **Same finger is mostly adjacency**: same-finger pairs are confused more overall (p={P['mannwhitney_same_finger_gt_not_p']:.3f}) but at matched distance the effect is small ({P['same_finger_by_distance_band'][0]['same_finger'][0]:.3f} vs {P['same_finger_by_distance_band'][0]['other'][0]:.3f} for adjacent pairs) and absent beyond. The earlier 'typist-specific finger signature' reading was overstated.",
       "* **Rows matter**: at matched horizontal gap, keys in the same row are confused about twice as often: " + "; ".join(f"gap {lo}-{hi}: {a:.4f} vs {b:.4f} (p={p:.3f})" for lo, hi, a, b, p in row_bands) + ". The D/E/L/O cluster is two same-finger pairs (D-E, L-O) joined by same-row cross-hand pairs (D-L home row, E-O top row); I-R and C-Comma are further same-row cross-hand pairs. The acoustics seem to encode vertical position better than horizontal.",
       f"* Same hand alone does not matter (p={P['mannwhitney_same_hand_gt_not_p']:.2f}). Acoustic distance between class means predicts confusion strongly (rho {P['spearman_acoustic_distance_vs_confusion_rate'][0]:+.2f}) but correlates only weakly with physical distance (rho {P['spearman_physical_vs_acoustic_distance'][0]:+.2f}).", "",
       "Per-key precision / recall / F1: `per_key_metrics.csv`. Weakest recall: " + ", ".join(f"{short(k)} {pk[k]['recall']:.2f}" for k in sorted(keys, key=lambda k: pk[k]['recall'])[:9]) + ".", "",
       "## Session generalisation", "",
       "| session | wpm | clean bal | clean acc | every-press acc | clean fraction | onset lag sd | feature shift | level vs others |", "|---|---|---|---|---|---|---|---|---|"]
for r in sess:
    md.append(f"| {r['session'][-6:]} | {r['wpm']:.0f} | {r['clean_bal_acc']*100:.1f}% | {r['clean_acc']*100:.1f}% | {r['all_acc']*100:.1f}% | {r['clean_fraction']:.2f} | {r['onset_lag_sd_ms']:.1f} ms | {r['feature_shift']:.3f} | {r['level_offset_db']:+.1f} dB |")
md += ["", "Spearman correlation with clean balanced accuracy across the 9 sessions (n=9, 12 covariates, p uncorrected; with Bonferroni only p < 0.004 would survive):", "",
       "| covariate | rho (clean bal) | p | rho (every-press) | p |", "|---|---|---|---|---|"]
for c_ in corr:
    md.append(f"| {c_} | {corr[c_][0]:+.2f} | {corr[c_][1]:.3f} | {corr_all[c_][0]:+.2f} | {corr_all[c_][1]:.3f} |")
wk = R["worst_session"]
md += ["", f"* **Typing speed does not explain it** (wpm rho {corr['wpm'][0]:+.2f}, p={corr['wpm'][1]:.2f}). The two slow sessions differ: 175531 (55 wpm) reaches {[r for r in sess if r['session'].endswith('175531')][0]['clean_acc']*100:.0f}% clean accuracy, 184222 (53 wpm) {worst['clean_acc']*100:.0f}%.",
       f"* **184222 is acoustically different**: largest feature shift ({worst['feature_shift']:.3f} vs {min(r['feature_shift'] for r in sess if r is not worst):.3f}-{max(r['feature_shift'] for r in sess if r is not worst):.3f} elsewhere), the same keys {worst['level_offset_db']:+.1f} dB quieter than in other sessions, the least precise onsets (lag sd {worst['onset_lag_sd_ms']:.1f} ms). Its biggest recall drops: " + ", ".join(f"{short(d['key'])} {d['recall_here']:.2f} vs {d['recall_elsewhere']:.2f}" for d in wk["key_recall_drop_vs_other_sessions"][:5]) + ". Softer, more deliberate strikes (or a changed position) are a more likely cause than speed itself; recordings carry no `conditions` metadata, so this cannot be confirmed.",
       f"* Across sessions, **feature shift** (rho {corr['feature_shift'][0]:+.2f}) and **backspace rate** (rho {corr['backspace_rate'][0]:+.2f}) track clean accuracy; **detection rate** (rho {corr_all['detection_rate'][0]:+.2f}) and **onset-lag sd** (rho {corr_all['onset_lag_sd_ms'][0]:+.2f}) track every-press accuracy, consistent with the segmentation finding.",
       "* Most inconsistent keys across sessions (recall sd): " + ", ".join(f"{short(d['key'])} {d['min']:.2f}-{d['max']:.2f}" for d in R["inconsistent_keys"][:6]) + " (per-key x session recall: `per_key_by_session_recall.csv`).", "",
       "## Class imbalance", "",
       f"Accuracy minus balanced accuracy = {dec['gap']*100:.1f} points, and it decomposes exactly as sum_k (n_k/N - 1/K) x recall_k: **Space alone contributes {dec['largest_positive_contributors'][0]['contribution']*100:.1f} points** ({dec['largest_positive_contributors'][0]['share']*100:.1f}% of clips, recall {dec['largest_positive_contributors'][0]['recall']:.2f}), then T, Backspace, E, A; rare low-recall keys (B, F, C, V, W, Y) pull the other way. Train/validation counts per fold: `fold_counts_train_over_validation.csv`.", "",
       "| option | clean acc | clean bal | every-press acc |", "|---|---|---|---|"]
for k_, lab in (("baseline_frequency_priors", "baseline (frequency priors)"), ("posthoc_prior_correction_same_models", "post-hoc prior correction, same models"), ("refit_uniform_priors", "refit with uniform priors"), ("balanced_subsampling_median_cap_3_seeds", "balanced subsampling (3 seeds)")):
    v = imb[k_]; md.append(f"| {lab} | {v['clean_acc']*100:.1f}% | {v['clean_bal_acc']*100:.1f}% | {v['all_acc']*100:.1f}% |")
md += ["", "Rebalancing trades about 1 point of accuracy for at most 2 points of balanced accuracy. The rare keys are weak because they have few samples and sit in confusable groups, not because of prior bias.", "",
       "## Technical checks", "",
       f"* Scaler statistics equal the training rows' statistics in every fold (max diff {T['scaler_mean_equals_training_rows_mean_max_abs_diff']:.1e}); refitting LDA on the training rows alone reproduces the fold model; features are computed per clip; predictions do not depend on batch composition.",
       f"* Every fold asserts the held-out session and the locked sessions are absent from training; {T['identical_audio_slices_shared_across_sessions']} identical audio slices are shared across sessions.",
       f"* Caveats: {T['label_set_note']} {T['selection_note']}. {T['segmentation_note']}.", "",
       "## Ranked problems", ""] + [f"{i}. **{p[0]}** ({p[1]}): {p[2]}. Impact: {p[3]}." for i, p in enumerate(problems, 1)] + [
       "", "## Recommended improvements (in order)", "",
       f"1. **Make classification robust to onset misalignment, and re-anchor every press.** Evidence: half of all presses have no detected onset and score {cat['no_onset_detected(estimated)']['acc']*100:.0f}%; a 2 ms shift alone costs 18-28 points; re-anchoring with the existing models already gives +{(ra['all_press_acc_after']-ra['all_press_acc_before'])*100:.1f} points every-press. Next: train with copies of each clean clip shifted by about +/-1-3 ms (or align features on the energy peak at both training and test), then re-anchor at inference. This is where the largest recoverable error sits ({seg['error_budget_all_presses']['extra_error_from_each_category']['no_onset_detected(estimated)']*100:.0f} points) and it directly affects the live predictor. Expect a ceiling: undetected presses are also the quietest.",
       f"2. **Reduce session/condition shift.** Evidence: held-out balanced accuracy ranges {min(r['clean_bal_acc'] for r in sess)*100:.0f}-{max(r['clean_bal_acc'] for r in sess)*100:.0f}% and tracks feature shift (rho {corr['feature_shift'][0]:+.2f}); the worst session's keys are {worst['level_offset_db']:+.1f} dB quieter. Next: test an unsupervised per-session level alignment (subtract each session's median band level, computed from its unlabelled audio) against level-free (shape-only) features, under the same LOSO; and record a `conditions` field (position, surface, posture) so shifts can be explained, not just measured. Evidence here is correlational (n=9).",
       f"3. **Targeted data for the weak, confusable keys, plus the free prior correction.** Evidence: {len(weak)} keys with recall under 50% account for {weak_deficit*100:.0f} of the {(1-cbal)*100:.0f} balanced-accuracy deficit points, and rebalancing cannot fix them. Next: steer the recorder towards G, Period, K, W, V, Comma, Y, M and the D/E/L/O and I/K/H/N/M groups (drill text, not just more natural English), and apply post-hoc prior correction (+{(imb['posthoc_prior_correction_same_models']['clean_bal_acc']-imb['baseline_frequency_priors']['clean_bal_acc'])*100:.1f} balanced points, no retraining) if balanced accuracy is the target metric.", "",
       "Not recommended yet: broader model searches (stage B showed LDA beats SVM/logistic/trees/kNN here) or CNNs before alignment robustness is fixed, since every model inherits the segmentation loss. Later, the 91% top-3 suggests a word-level language model could resolve many same-row confusions.", "",
       "![confusion](confusion_normalised.png)", "![raw](confusion_raw.png)", "![diagnostics](diagnostics.png)"]
open(f"{D}/report.md", "w").write("\n".join(md) + "\n")
print(f"wrote {D}/report.md and problems_ranked.csv; clean errors {errors}, DELO cluster {clus_err}, weak-key deficit {weak_deficit:.3f}")
