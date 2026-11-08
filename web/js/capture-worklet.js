// Runs on the audio thread. Converts the microphone input to 16-bit samples and sends them to the page in
// batches of 4096 (about 85 ms). The conversion matches the original recorder, so live audio is identical in
// format to the training recordings. No filtering or gain is applied.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.batch = new Int16Array(4096);
    this.count = 0;
    this.batchStartFrame = 0;
    this.peak = 0;
    this.port.onmessage = (event) => {
      if (event.data === "flush") {
        this.sendBatch();
        this.port.postMessage({ flushed: true });
      }
    };
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    const length = channel ? channel.length : 128;   // with no input, record silence so the timeline stays intact
    for (let i = 0; i < length; i++) {
      if (this.count === 0) this.batchStartFrame = currentFrame + i;
      const sample = channel ? Math.max(-1, Math.min(1, channel[i])) : 0;
      this.peak = Math.max(this.peak, Math.abs(sample));
      this.batch[this.count++] = Math.round(sample < 0 ? sample * 32768 : sample * 32767);
      if (this.count === this.batch.length) this.sendBatch();
    }
    return true;
  }

  sendBatch() {
    if (this.count === 0) return;
    const samples = this.batch.slice(0, this.count);
    this.port.postMessage({ samples, startFrame: this.batchStartFrame, peak: this.peak }, [samples.buffer]);
    this.count = 0;
    this.peak = 0;
  }
}

registerProcessor("capture", CaptureProcessor);
