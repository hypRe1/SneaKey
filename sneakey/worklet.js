// AudioWorkletProcessor: copies raw mono input into batches, with no filtering, gain or normalisation.
// Every render quantum (128 frames) is counted, so `startFrame` is the exact sample index of a batch's first
// sample, and `ctxFrame` (the AudioContext's currentFrame at that batch's first quantum) ties that index to the
// context clock. If the browser ever supplies no input channel we write zeros (and count them) so the sample
// timeline never silently shifts.
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.batchSize = 4096;
    this.buf = new Float32Array(this.batchSize);
    this.fill = 0;
    this.captured = 0;          // samples emitted + buffered so far
    this.batchStartFrame = 0;   // sample index of buf[0]
    this.batchCtxFrame = 0;     // currentFrame at buf[0]
    this.batchCtxTime = 0;
    this.missingFrames = 0;
    this.stopped = false;
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "stop") {
        this.flush();
        this.stopped = true;
        this.port.postMessage({ type: "stopped", totalFrames: this.captured, missingFrames: this.missingFrames });
      }
    };
  }

  flush() {
    if (this.fill === 0) return;
    const out = this.buf.slice(0, this.fill);
    this.port.postMessage(
      { type: "audio", samples: out, startFrame: this.batchStartFrame, ctxFrame: this.batchCtxFrame, ctxTime: this.batchCtxTime },
      [out.buffer]
    );
    this.fill = 0;
  }

  process(inputs) {
    if (this.stopped) return false;
    const ch = inputs[0] && inputs[0][0];
    const n = ch ? ch.length : 128;
    if (!ch) this.missingFrames += n;
    let i = 0;
    while (i < n) {
      if (this.fill === 0) {
        this.batchStartFrame = this.captured;
        this.batchCtxFrame = currentFrame + i;
        this.batchCtxTime = currentTime + i / sampleRate;
      }
      const k = Math.min(n - i, this.batchSize - this.fill);
      if (ch) this.buf.set(ch.subarray(i, i + k), this.fill);
      else this.buf.fill(0, this.fill, this.fill + k);
      this.fill += k; this.captured += k; i += k;
      if (this.fill === this.batchSize) this.flush();
    }
    return true;
  }
}
registerProcessor("sneakey-recorder", RecorderProcessor);
