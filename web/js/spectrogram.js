// A spectrogram of a whole recording, for display in the session viewer.
// Levels are stored as bytes in 0.5 dB steps. Each frequency's median level over the recording is kept as its
// noise floor, so the display can subtract it ("whitening"): steady background hiss disappears and key clicks
// stand out.

const SIZE = 512;            // FFT size: 10.7 ms at 48 kHz
const HOP_SECONDS = 0.002;   // one column every 2 ms

export const CLICK_BAND = [1500, 10000];   // Hz; where key clicks are loudest relative to background

export async function computeSpectrogram(samples, sampleRate, onProgress) {
  const hop = Math.round(sampleRate * HOP_SECONDS);
  const bins = SIZE / 2 + 1;
  const frames = Math.max(0, Math.floor((samples.length - SIZE) / hop) + 1);
  const levels = new Uint8Array(frames * bins);
  const clickEnergy = new Float32Array(frames);       // dB in CLICK_BAND, per column
  const histogram = new Uint32Array(bins * 256);      // for each bin, how often each level occurs

  const fft = makeFFT(SIZE);
  const real = new Float64Array(SIZE);
  const imag = new Float64Array(SIZE);
  const window = Float64Array.from({ length: SIZE }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / SIZE));
  const windowSum = window.reduce((sum, w) => sum + w, 0);
  const normalise = 4 / (windowSum * windowSum);
  const binHz = sampleRate / SIZE;
  const [clickLow, clickHigh] = CLICK_BAND.map((hz) => Math.round(hz / binHz));

  for (let frame = 0; frame < frames; frame++) {
    const offset = frame * hop;
    for (let i = 0; i < SIZE; i++) {
      real[i] = samples[offset + i] * window[i];
      imag[i] = 0;
    }
    fft(real, imag);
    let energy = 0;
    for (let bin = 0; bin < bins; bin++) {
      const power = (real[bin] ** 2 + imag[bin] ** 2) * normalise;
      if (bin >= clickLow && bin <= clickHigh) energy += power;
      const level = Math.max(0, Math.min(255, Math.round((10 * Math.log10(power + 1e-14) + 130) * 2)));
      levels[frame * bins + bin] = level;
      histogram[bin * 256 + level]++;
    }
    clickEnergy[frame] = 10 * Math.log10(energy + 1e-14);
    if (frame % 4096 === 4095) {
      onProgress?.(frame / frames);
      await new Promise((resolve) => setTimeout(resolve, 0));   // let the page repaint
    }
  }

  // Per-bin median (noise floor), and overall percentiles for the unwhitened colour range.
  const noiseFloor = new Uint8Array(bins);
  const overall = new Float64Array(256);
  for (let bin = 0; bin < bins; bin++) {
    let seen = 0;
    for (let level = 0; level < 256; level++) {
      const count = histogram[bin * 256 + level];
      overall[level] += count;
      seen += count;
      if (!noiseFloor[bin] && seen >= frames / 2) noiseFloor[bin] = level;
    }
  }
  const sortedEnergy = Float32Array.from(clickEnergy).sort();

  return {
    sampleRate, hop, size: SIZE, bins, frames, levels, noiseFloor, clickEnergy,
    clickFloor: sortedEnergy[Math.floor(frames / 2)] ?? 0,
    binHz,
  };
}

// Colour map from black through purple and orange to pale yellow.
export const COLOURS = (() => {
  const stops = [[0, 0, 0, 4], [0.25, 60, 15, 110], [0.5, 180, 55, 100], [0.75, 250, 140, 40], [1, 252, 250, 190]];
  const table = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, ...from] = stops[k];
    const [t1, ...to] = stops[k + 1];
    const u = (t - t0) / (t1 - t0);
    for (let c = 0; c < 3; c++) table[i * 3 + c] = from[c] + (to[c] - from[c]) * u;
  }
  return table;
})();

// Iterative radix-2 FFT, in place.
function makeFFT(size) {
  const levels = Math.log2(size);
  const reversed = new Uint32Array(size);
  for (let i = 0; i < size; i++) {
    let r = 0;
    for (let b = 0; b < levels; b++) r |= ((i >> b) & 1) << (levels - 1 - b);
    reversed[i] = r;
  }
  const cos = Float64Array.from({ length: size / 2 }, (_, i) => Math.cos((2 * Math.PI * i) / size));
  const sin = Float64Array.from({ length: size / 2 }, (_, i) => -Math.sin((2 * Math.PI * i) / size));

  return (real, imag) => {
    for (let i = 0; i < size; i++) {
      const j = reversed[i];
      if (j > i) {
        [real[i], real[j]] = [real[j], real[i]];
        [imag[i], imag[j]] = [imag[j], imag[i]];
      }
    }
    for (let width = 2; width <= size; width *= 2) {
      const half = width / 2;
      const step = size / width;
      for (let start = 0; start < size; start += width) {
        for (let j = 0; j < half; j++) {
          const a = start + j;
          const b = a + half;
          const k = j * step;
          const tr = real[b] * cos[k] - imag[b] * sin[k];
          const ti = real[b] * sin[k] + imag[b] * cos[k];
          real[b] = real[a] - tr;
          imag[b] = imag[a] - ti;
          real[a] += tr;
          imag[a] += ti;
        }
      }
    }
  };
}
