// Tab routing and the shared session list. URLs: #record, #analyse/<session>, #train, #predict/<session>.
import { api, roleBadge, setStickyError, showError } from "./common.js";
import * as record from "./record.js";
import * as analyse from "./inspect.js";   // not "analyse.js": ad/tracker blockers block that name
import * as predict from "./predict.js";
import * as train from "./train.js";
import * as player from "./player.js";

const VIEWS = { record, analyse, train, predict };
export const state = { sessions: [], view: null, sessionId: null };

export async function refreshSessions() {
  try {
    state.sessions = await api("api/sessions");
  } catch (error) {
    showError(`Couldn't load recordings: ${error.message}`);
    state.sessions = [];
  }
  for (const select of document.querySelectorAll("[data-session-select]")) fillSelect(select);
  record.showSessions?.(state.sessions);
  return state.sessions;
}

function fillSelect(select) {
  select.replaceChildren(...state.sessions.filter((s) => s.has_audio).map((s) => {
    const label = `${s.id.replace("session_", "")}  ·  ${s.participant || "—"}  ·  ${(s.duration_s ?? 0).toFixed(0)} s  ·  ${s.role}`;
    return new Option(label, s.id, false, s.id === state.sessionId);
  }));
}

function updateSessionChrome(name) {
  const usable = state.sessions.filter((s) => s.has_audio);
  document.querySelector(`[data-empty="${name}"]`)?.classList.toggle("hidden", usable.length > 0);
  document.querySelector(`[data-content="${name}"]`)?.classList.toggle("hidden", usable.length === 0);
  const session = usable.find((s) => s.id === state.sessionId);
  const badge = document.querySelector(`[data-role-badge="${name}"]`);
  if (badge) badge.replaceChildren(session ? roleBadge(session.role) : "");
}

async function route() {
  const [name = "record", sessionId] = location.hash.slice(1).split("/");
  const view = VIEWS[name] ? name : "record";
  showError("");
  for (const key of Object.keys(VIEWS)) {
    document.getElementById(`view-${key}`).classList.toggle("hidden", key !== view);
    document.querySelector(`.tabs a[data-view="${key}"]`).setAttribute("aria-current", key === view ? "page" : "false");
  }
  if (state.view && state.view !== view) VIEWS[state.view].leave?.();
  state.view = view;

  if (view === "record") return record.enter();
  if (view === "train") return train.enter();
  const usable = state.sessions.filter((s) => s.has_audio);
  state.sessionId = usable.some((s) => s.id === sessionId) ? sessionId : (usable.some((s) => s.id === state.sessionId) ? state.sessionId : usable[0]?.id);
  for (const select of document.querySelectorAll("[data-session-select]")) select.value = state.sessionId ?? "";
  // Keep the other tab links pointing at the same session.
  for (const key of ["analyse", "predict"]) document.querySelector(`.tabs a[data-view="${key}"]`).href = `#${key}${state.sessionId ? "/" + state.sessionId : ""}`;
  updateSessionChrome(view);
  if (state.sessionId) await VIEWS[view].enter(state.sessionId);
}

for (const select of document.querySelectorAll("[data-session-select]")) {
  select.addEventListener("change", () => { location.hash = `${select.dataset.sessionSelect}/${select.value}`; });
}

export function sessionInfo(id) {
  return state.sessions.find((s) => s.id === id);
}

// The page and server are updated together; a server left running from before an update lacks newer endpoints
// (e.g. training), which shows up as confusing 404/405 errors. Say so plainly instead.
const API_VERSION = 4;    // must match API_VERSION in sneakey/api.py
api("api/health").then((health) => {
  if (health.api_version !== API_VERSION) {
    setStickyError("The server is out of date: it was started before the latest update. Stop it (Ctrl+C) and start it again with `python run.py`, then reload this page.");
  }
}).catch(() => {});

// Playback volume, shared by every view.
const volume = document.getElementById("volume");
const showVolume = () => { document.getElementById("volumeValue").textContent = `${player.getVolumeDb() > 0 ? "+" : ""}${player.getVolumeDb()} dB`; };
volume.value = player.getVolumeDb();
volume.addEventListener("input", () => { player.setVolumeDb(Number(volume.value)); showVolume(); });
showVolume();

window.__sneakeyStarted = true;   // tells the fallback in index.html that the modules loaded
window.addEventListener("hashchange", route);
refreshSessions().then(route);
