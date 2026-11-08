// Microphone capture at 48 kHz mono, 16-bit. Used by the recorder and the live page.

export const SAMPLE_RATE = 48000;

// The browser's own processing would change the sound the models were trained on, so turn it all off.
export const CONSTRAINTS = { channelCount: { ideal: 1 }, echoCancellation: false, noiseSuppression: false, autoGainControl: false };

// Starts capturing. onBatch receives {samples: Int16Array, startFrame, peak} for every ~85 ms of audio, where
// startFrame is the AudioContext frame of the batch's first sample.
// Call this directly from a click handler: browsers only allow audio to start in response to the user.
export async function openMicrophone(onBatch) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser can't use the microphone here. Open the page via http://localhost.");
  }
  const context = new AudioContext({ sampleRate: SAMPLE_RATE });
  context.resume().catch(() => {});   // may be rejected if the context closes first

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: CONSTRAINTS });
  } catch (error) {
    context.close();
    if (error.name === "NotAllowedError") throw new Error("Microphone permission was denied.");
    if (error.name === "NotFoundError") throw new Error("No microphone was found.");
    throw error;
  }

  let node;
  try {
    await context.audioWorklet.addModule(new URL("./capture-worklet.js", import.meta.url));
    node = new AudioWorkletNode(context, "capture", { numberOfOutputs: 0, channelCount: 1, channelCountMode: "explicit" });
    context.createMediaStreamSource(stream).connect(node);
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    context.close();
    throw new Error(`Couldn't start audio capture at 48 kHz (${error.message}). Set the microphone to 48 kHz in your system sound settings.`);
  }

  let onFlushed = null;
  node.port.onmessage = (event) => {
    if (event.data.flushed) onFlushed?.();
    else onBatch(event.data);
  };

  return {
    context,
    trackSettings: stream.getAudioTracks()[0]?.getSettings() ?? {},

    // Stops the microphone after delivering any samples still buffered in the worklet.
    async stop() {
      await new Promise((resolve) => {
        onFlushed = resolve;
        node.port.postMessage("flush");
      });
      stream.getTracks().forEach((track) => track.stop());
      await context.close();
    },
  };
}
