// SneaKey live feature extraction + linear/softmax classifier, reimplementing analysis/classify_baseline.py
// exactly so the browser predictor matches the Python model. Validated against analysis_output/live_model/validation.json.
(function (root) {
  "use strict";

  // ---- radix-2 iterative complex FFT (in place) ------------------------------------------------
  function makeFFT(N) {
    const lev = Math.log2(N), rev = new Uint32Array(N), cos = new Float64Array(N / 2), sin = new Float64Array(N / 2);
    for (let i = 0; i < N; i++) { let r = 0; for (let b = 0; b < lev; b++) r |= ((i >> b) & 1) << (lev - 1 - b); rev[i] = r; }
    for (let i = 0; i < N / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / N); sin[i] = -Math.sin(2 * Math.PI * i / N); }
    return (re, im) => {
      for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
      for (let size = 2; size <= N; size *= 2) {
        const half = size / 2, step = N / size;
        for (let i = 0; i < N; i += size) for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j, b = a + half, tr = re[b] * cos[k] - im[b] * sin[k], ti = re[b] * sin[k] + im[b] * cos[k];
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        }
      }
    };
  }

  // |rfft(signal, nfft)|^2 for bins 0..nfft/2 (signal zero-padded/truncated to nfft)
  function powerSpectrum(sig, nfft, fft, re, im) {
    re.fill(0); im.fill(0);
    const n = Math.min(sig.length, nfft);
    for (let i = 0; i < n; i++) re[i] = sig[i];
    fft(re, im);
    const B = nfft / 2 + 1, p = new Float64Array(B);
    for (let k = 0; k < B; k++) p[k] = re[k] * re[k] + im[k] * im[k];
    return p;
  }

  // scipy.signal.windows.tukey(n, alpha)
  function tukey(n, alpha) {
    const w = new Float64Array(n);
    if (alpha <= 0) { w.fill(1); return w; }
    const width = Math.floor(alpha * (n - 1) / 2);
    for (let i = 0; i < n; i++) {
      if (i <= width) w[i] = 0.5 * (1 + Math.cos(Math.PI * (2 * i / (alpha * (n - 1)) - 1)));
      else if (i >= n - 1 - width) w[i] = 0.5 * (1 + Math.cos(Math.PI * (2 * i / (alpha * (n - 1)) - 2 / alpha + 1)));
      else w[i] = 1;
    }
    return w;
  }
  // numpy.hanning(n)
  function hanning(n) {
    const w = new Float64Array(n);
    if (n === 1) { w[0] = 1; return w; }
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
    return w;
  }

  function geomspace(a, b, n) {
    const out = new Float64Array(n), la = Math.log(a), lb = Math.log(b);
    for (let i = 0; i < n; i++) out[i] = Math.exp(la + (lb - la) * i / (n - 1));
    return out;
  }

  function melFilterbank(sr, nFft, nMels, fmin, fmax) {
    const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
    const mel2hz = (m) => 700 * (Math.pow(10, m / 2595) - 1);
    const lo = hz2mel(fmin), hi = hz2mel(Math.min(fmax, sr / 2)), B = nFft / 2 + 1;
    const pts = new Float64Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) pts[i] = mel2hz(lo + (hi - lo) * i / (nMels + 1));
    const bins = new Float64Array(B);
    for (let k = 0; k < B; k++) bins[k] = k * sr / nFft;      // rfftfreq(nFft, 1/sr)
    const fb = [];
    for (let m = 0; m < nMels; m++) {
      const l = pts[m], c = pts[m + 1], h = pts[m + 2], row = new Float64Array(B);
      for (let k = 0; k < B; k++) row[k] = Math.max(0, Math.min((bins[k] - l) / (c - l), (h - bins[k]) / (h - c)));
      fb.push(row);
    }
    return fb;
  }

  // Build a feature extractor bound to a config (cfg = model.feature_config) and sample rate.
  function makeExtractor(cfg, sr, clipSamples) {
    const b = cfg.band, mel = cfg.mel;
    const fftBand = makeFFT(b.nfft), reB = new Float64Array(b.nfft), imB = new Float64Array(b.nfft);
    const fftMel = makeFFT(mel.n_fft), reM = new Float64Array(mel.n_fft), imM = new Float64Array(mel.n_fft);
    const tuk = tukey(clipSamples, b.tukey_alpha);
    const hann = hanning(mel.n_fft);
    const fBand = (k) => k * sr / b.nfft;
    const edges = geomspace(b.fmin, Math.min(b.fmax, sr / 2 * 0.98), b.bands + 1);
    const fb = melFilterbank(sr, mel.n_fft, mel.n_mels, mel.fmin, mel.fmax);

    return function extract(seg) {
      // --- band features: 10*log10(mean power in each geometric band) of the tukey-windowed clip ---
      const w = new Float64Array(seg.length);
      for (let i = 0; i < seg.length; i++) w[i] = seg[i] * (tuk[i] || 0);
      const p = powerSpectrum(w, b.nfft, fftBand, reB, imB);
      const out = new Float64Array(b.bands + mel.n_mels * mel.t_pool);
      for (let m = 0; m < b.bands; m++) {
        const lo = edges[m], hi = edges[m + 1]; let s = 0, c = 0;
        for (let k = 0; k < p.length; k++) { const f = fBand(k); if (f >= lo && f < hi) { s += p[k]; c++; } }
        out[m] = 10 * Math.log10((c ? s / c : 0) + 1e-20);
      }
      // --- log-mel: frames hop over the clip, mel power, log, pooled into t_pool time slices ---
      const starts = [];
      for (let i = 0; i + mel.n_fft <= seg.length; i += mel.hop) starts.push(i);
      const nF = starts.length, S = []; // S[mel][frame]
      for (let m = 0; m < mel.n_mels; m++) S.push(new Float64Array(nF));
      const fr = new Float64Array(mel.n_fft);
      for (let t = 0; t < nF; t++) {
        const s0 = starts[t];
        for (let i = 0; i < mel.n_fft; i++) fr[i] = seg[s0 + i] * hann[i];
        const pm = powerSpectrum(fr, mel.n_fft, fftMel, reM, imM);
        for (let m = 0; m < mel.n_mels; m++) { let acc = 0; const row = fb[m]; for (let k = 0; k < pm.length; k++) acc += row[k] * pm[k]; S[m][t] = Math.log(Math.max(acc, 1e-12)); }
      }
      // pool: edges = linspace(0, nF, t_pool+1).astype(int); slice means, slice-major order
      let o = b.bands;
      for (let sl = 0; sl < mel.t_pool; sl++) {
        const a = Math.trunc(nF * sl / mel.t_pool), e = Math.max(a + 1, Math.trunc(nF * (sl + 1) / mel.t_pool));
        for (let m = 0; m < mel.n_mels; m++) { let s = 0; for (let t = a; t < e && t < nF; t++) s += S[m][t]; out[o++] = s / (Math.min(e, nF) - a); }
      }
      return out;
    };
  }

  // linear + softmax == StandardScaler + shrinkage-LDA (verified identical in export_model.py)
  function makeClassifier(model) {
    const mean = model.scaler_mean, scale = model.scaler_scale, coef = model.coef, inter = model.intercept, C = model.classes;
    return function predict(f) {
      const z = new Float64Array(f.length);
      for (let i = 0; i < f.length; i++) z[i] = (f[i] - mean[i]) / scale[i];
      const d = new Float64Array(C.length); let mx = -Infinity;
      for (let c = 0; c < C.length; c++) { let s = inter[c], row = coef[c]; for (let i = 0; i < z.length; i++) s += row[i] * z[i]; d[c] = s; if (s > mx) mx = s; }
      let sum = 0; for (let c = 0; c < C.length; c++) { d[c] = Math.exp(d[c] - mx); sum += d[c]; }
      const probs = C.map((k, c) => [k, d[c] / sum]).sort((a, b) => b[1] - a[1]);
      return probs; // [[code, prob], ...] sorted desc
    };
  }

  root.SneaKeyFeatures = { makeExtractor, makeClassifier, makeFFT, tukey, hanning, melFilterbank, geomspace };
  if (typeof module !== "undefined" && module.exports) module.exports = root.SneaKeyFeatures;
})(typeof globalThis !== "undefined" ? globalThis : this);
