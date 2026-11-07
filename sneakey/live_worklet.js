// Forwards each render block of raw mono mic input to the main thread with its absolute start-frame index,
// so the main thread can keep a rolling ring buffer and map event times to sample positions. No processing.
class LiveProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) { const copy = ch.slice(0); this.port.postMessage({ frame: currentFrame, samples: copy }, [copy.buffer]); }
    return true;
  }
}
registerProcessor("sneakey-live", LiveProcessor);
