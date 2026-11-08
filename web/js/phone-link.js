// Laptop side of recording with a phone microphone (see sneakey/phone.py): pairing, the phone's live status,
// round-trip timing pings to the server, and turning a finished phone recording into a session.
import { $, api, el } from "./common.js";
import { drawQr } from "./qr.js";

let statusTimer = null;
let lastQrLink = null;
let status = { paired: false };
const pings = [];              // this page's clock vs the server clock, for mapping key events
let pinging = false;

export const phoneSelected = () => $("micSource").value === "phone";

export function init() {
  $("micSource").addEventListener("change", () => { localStorage.setItem("sneakey-mic-source", $("micSource").value); update(); });
  try { $("micSource").value = localStorage.getItem("sneakey-mic-source") || "laptop"; } catch { /* default */ }
  $("pairPhone").addEventListener("click", pair);
  update();
}

function update() {
  const on = phoneSelected();
  $("phonePanel").classList.toggle("hidden", !on);
  clearInterval(statusTimer);
  if (on) {
    refresh();
    statusTimer = setInterval(refresh, 700);
    startPinging();
  }
}

export function leave() { clearInterval(statusTimer); statusTimer = null; }
export function resume() { if (phoneSelected()) update(); }

async function pair() {
  try {
    await api("api/phone/pair", { method: "POST" });
  } catch (error) {
    $("phoneStatus").replaceChildren(el("span", { style: "color:var(--bad)" }, error.message));
    return;
  }
  refresh();
}

async function refresh() {
  try { status = await api("api/phone/status"); } catch { return; }
  if (!status.enabled) {
    $("phoneLinks").replaceChildren();
    $("phoneStatus").replaceChildren(el("span", { style: "color:var(--bad)" },
      "Phone recording is off. Restart the server with:  python run.py --phone"));
    $("pairPhone").disabled = true;
    return;
  }
  $("pairPhone").disabled = false;
  $("pairPhone").textContent = status.paired ? "New pairing code" : "Pair a phone";
  const links = status.links || [];
  $("phoneLinks").replaceChildren(...links.map((link) => el("div", { class: "phone-link mono" }, link)));
  // A QR of the first link, so the phone can scan instead of typing it.
  if (links.length && lastQrLink !== links[0]) {
    lastQrLink = links[0];
    try {
      drawQr($("phoneQr"), links[0], 220);
      $("phoneQrWrap").classList.remove("hidden");
    } catch { $("phoneQrWrap").classList.add("hidden"); }
  }
  if (!links.length) { $("phoneQrWrap").classList.add("hidden"); lastQrLink = null; }
  const phone = status.phone || {};
  $("phoneStatus").replaceChildren(
    !status.paired ? "Press “Pair a phone”, then open the link on the phone (same Wi-Fi)." :
      status.connected ? el("span", { style: "color:var(--ok)" }, `● Phone connected${phone.sample_rate ? ` · ${phone.sample_rate} Hz` : ""}`) :
        el("span", { class: "muted" }, "○ Waiting for the phone… open the link on it and tap “Start microphone”. Accept the certificate warning once."));
  $("phoneLevel").style.width = `${Math.min(100, Math.sqrt(status.level || 0) * 100)}%`;
  if (status.recording) $("level").style.width = `${Math.min(100, Math.sqrt(status.level || 0) * 100)}%`;
}

export const connected = () => status.connected;

// ---- timing: this page's clock vs the server clock ----

async function startPinging() {
  if (pinging) return;
  pinging = true;
  while (pinging) {
    const t0 = performance.now();
    try {
      const { server } = await api("api/clock");
      pings.push({ t0, t1: performance.now(), server });
      if (pings.length > 3000) pings.splice(0, 1000);
    } catch { /* missed: fine */ }
    await new Promise((resolve) => setTimeout(resolve, 500));
    if (!phoneSelected()) pinging = false;
  }
}

// ---- recording ----

export async function start() {
  if (!status.connected) throw new Error("The phone isn't connected: pair it and tap “Start microphone” on the phone.");
  const { recording_id } = await api("api/phone/start", { method: "POST" });
  // Wait for the phone's first audio before letting anyone type, so no keystroke falls before the recording.
  const started = Date.now();
  while (Date.now() - started < 6000) {
    const s = await api("api/phone/status");
    if (s.recording?.id === recording_id && s.recording.state === "recording") return { recordingId: recording_id, startedPerf: performance.now() };
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  await api("api/phone/stop", { method: "POST" }).catch(() => {});
  throw new Error("The phone didn't start recording. Check the phone page is open and its microphone is on.");
}

export async function stop(recordingId, onProgress) {
  await api("api/phone/stop", { method: "POST" });
  const started = Date.now();
  while (Date.now() - started < 60000) {
    const s = await api("api/phone/status");
    if (s.recording?.id === recordingId && s.recording.state === "done") return;
    onProgress?.(s.recording?.seconds);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("The phone didn't finish sending its audio.");
}

export async function save(recordingId, metadata, events, startedPerf) {
  const from = startedPerf - 30000;
  const body = { recording_id: recordingId, metadata, events, laptop_pings: pings.filter((p) => p.t0 >= from) };
  return api("api/sessions/phone", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}
