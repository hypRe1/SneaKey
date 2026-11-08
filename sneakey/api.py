"""HTTP API and static file server. Bound to 127.0.0.1 by run.py: nothing leaves the machine.

  GET    /api/health                              model version and settings
  GET    /api/sentences?chars=350&mode=natural    a new random typing test (mode: natural | coverage)
  GET    /api/key-counts                          recorded presses per key
  GET    /api/sessions                            sessions with their evaluation role
  POST   /api/sessions                            create a session: {metadata, events} -> {id}
  PUT    /api/sessions/{id}/audio                 its audio: raw little-endian int16 mono at 48 kHz
  DELETE /api/sessions/{id}                       remove a session
  GET    /api/sessions/{id}                       metadata and key events
  GET    /api/sessions/{id}/audio.wav?filter=raw  audio as each pipeline stage hears it (raw|detector|model|click)
  GET    /api/sessions/{id}/signals               detector novelty, candidates and ground-truth positions
  GET    /api/sessions/{id}/keystroke?time_ms=    what the models compute for one press position
  POST   /api/sessions/{id}/predict?refresh=0     run the pipeline and evaluate it (cached per model version)
  GET    /api/summary                             metrics pooled over all evaluated sessions, by role
  GET    /api/llm                                 whether the optional DeepSeek decoder is configured
  POST   /api/sessions/{id}/llm-decode            OPT-IN, SENDS DATA OFF THE LAPTOP: decode key probabilities with DeepSeek
  GET    /api/sessions/{id}/llm-decode            a stored DeepSeek result, if any (never sends anything)
  GET    /api/models                              key classifiers (shipped + trained), with the active one
  POST   /api/models/{id}/activate                use this classifier for prediction
  DELETE /api/models/{id}                         remove a trained classifier
  GET    /api/training                            status of the current/last training run (progress, log, result)
  POST   /api/training                            start a run: {kind, params, train, validation, name, end_to_end}
  POST   /api/training/cancel                     stop the run at the next safe point
"""
import asyncio
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import analysis, config, llm_decoder, phone, service, sessions
from .audio import AudioFormatError, wav_bytes

WEB_DIR = config.ROOT / "web"
# Bump whenever the API changes; the page checks it so a server started before an update says "restart me"
# instead of failing with 404/405 errors. Must match API_VERSION in web/js/app.js.
API_VERSION = 5
MAX_AUDIO_BYTES = 400 * 1024 * 1024   # ~70 minutes at 48 kHz

app = FastAPI(title="SneaKey", docs_url=None, redoc_url=None)


# The phone port (run.py --phone) is reachable from the local network, so it serves only the phone page and the
# phone's own endpoints (which need the pairing code). Everything else stays on the localhost-only port.
PHONE_PORT_PATHS = ("/p/", "/api/phone/d/", "/img/")
PHONE_PORT_FILES = {"/phone.html", "/js/phone.js", "/js/capture-worklet.js", "/css/app.css"}


@app.middleware("http")
async def phone_port_only_serves_the_phone(request: Request, call_next):
    server = request.scope.get("server") or (None, None)
    if config.PHONE_PORT and server[1] == config.PHONE_PORT:
        path = request.url.path
        if not (path.startswith(PHONE_PORT_PATHS) or path in PHONE_PORT_FILES):
            return Response("Not found", status_code=404)
    return await call_next(request)


@app.middleware("http")
async def revalidate_pages(request: Request, call_next):
    """Make the browser revalidate the page and its scripts on every load. Without this, a browser that once
    visited another app on the same address (e.g. the original SneaKey web app on :8000, which also had
    js/common.js) can reuse those stale modules, and the page fails to start. Local, so the cost is negligible."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _session_errors(function):
    """Translate missing sessions and bad audio into HTTP errors."""
    try:
        return function()
    except sessions.NotFound:
        raise HTTPException(404, "No such session, or it has no audio.")
    except service.Locked as error:
        raise HTTPException(403, str(error))
    except llm_decoder.NotConfigured as error:
        raise HTTPException(400, str(error))
    except RuntimeError as error:                      # remote LLM failures
        raise HTTPException(502, str(error))
    except (AudioFormatError, ValueError) as error:
        raise HTTPException(400, str(error))


async def _run(function):
    return await asyncio.to_thread(_session_errors, function)


@app.get("/api/health")
async def health():
    p = await asyncio.to_thread(service.pipeline)
    return {"status": "ok", "api_version": API_VERSION, "model_version": p.version, "sample_rate": config.SAMPLE_RATE, "keys": p.key_names,
            "detector": p.detector.settings, "classifier_trained_sessions": len(p.classifier.trained_on),
            "decoder": vars(p.decoder.settings)}


@app.get("/api/sentences")
async def sentences(chars: int = 350, mode: str = "natural", seed: int | None = None):
    if mode not in ("natural", "coverage"):
        raise HTTPException(400, "mode must be natural or coverage")
    return await asyncio.to_thread(service.new_test, max(60, min(chars, 3000)), mode, seed)


@app.get("/api/key-counts")
async def key_counts(sessions: str | None = None):
    ids = [s for s in sessions.split(",") if s] if sessions is not None else None
    return await _run(lambda: service.key_counts(ids))


@app.get("/api/sessions")
async def list_sessions():
    return await asyncio.to_thread(service.list_sessions)


@app.post("/api/sessions")
async def create_session(request: Request):
    body = await request.json()
    if not isinstance(body.get("events"), list) or not isinstance(body.get("metadata"), dict):
        raise HTTPException(400, "Expected {metadata: {...}, events: [...]}.")
    return {"id": sessions.create_session(body["metadata"], body["events"])}


@app.put("/api/sessions/{session_id}/audio")
async def save_audio(session_id: str, request: Request):
    pcm = await request.body()
    if len(pcm) % 2 or len(pcm) > MAX_AUDIO_BYTES or len(pcm) < 2 * config.SAMPLE_RATE // 2:
        raise HTTPException(400, "Audio must be whole 16-bit samples, between 0.5 s and 400 MB.")
    _session_errors(lambda: sessions.save_audio(session_id, pcm, config.SAMPLE_RATE))
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    _session_errors(lambda: sessions.delete_session(session_id))
    return {"ok": True}


@app.get("/api/sessions/{session_id}")
async def session(session_id: str):
    return await _run(lambda: service.session_detail(session_id))


@lru_cache(maxsize=6)
def _filtered_wav(session_id, variant, modified):
    return wav_bytes(analysis.filtered(service.samples(session_id), variant))


@app.get("/api/sessions/{session_id}/audio.wav")
async def session_audio(session_id: str, filter: str = "raw"):
    if filter == "raw":
        return _session_errors(lambda: FileResponse(sessions.audio_path(session_id), media_type="audio/wav"))
    if filter not in analysis.FILTERS:
        raise HTTPException(400, f"filter must be one of {sorted(analysis.FILTERS)}")
    data = await _run(lambda: _filtered_wav(session_id, filter, sessions.audio_path(session_id).stat().st_mtime_ns))
    return Response(data, media_type="audio/wav")


@app.get("/api/sessions/{session_id}/signals")
async def signals(session_id: str):
    return await _run(lambda: service.signals(session_id))


@app.get("/api/sessions/{session_id}/keystroke")
async def keystroke(session_id: str, time_ms: float):
    return await _run(lambda: service.keystroke(session_id, time_ms))


@app.post("/api/sessions/{session_id}/predict")
async def predict(session_id: str, refresh: bool = False):
    return await _run(lambda: service.predict(session_id, refresh))


@app.get("/api/llm")
async def llm_status():
    return {"configured": llm_decoder.is_configured(), "endpoint": llm_decoder.API_URL, "model": llm_decoder.MODEL}


@app.post("/api/sessions/{session_id}/llm-decode")
async def llm_decode(session_id: str, refresh: bool = False, style: str = "ensemble", guesses: str = "active",
                     models: str | None = None):
    """Opt-in: SENDS this session's per-press candidate letters (and word guesses) to DeepSeek; see llm_decoder.py."""
    ids = [m for m in models.split(",") if m] if models else None
    return await _run(lambda: service.llm_decode(session_id, refresh, style, ids, guesses))


@app.get("/api/sessions/{session_id}/llm-decode")
async def llm_decode_cached(session_id: str, style: str = "ensemble", guesses: str = "active"):
    """A stored DeepSeek result, if any (404 otherwise). Never sends anything."""
    return await _run(lambda: service.llm_decode(session_id, False, style, None, guesses, cached_only=True))


@app.get("/api/summary")
async def summary():
    return await asyncio.to_thread(service.summary)


# ---- models and training ----

@app.get("/api/models")
async def list_models():
    return await asyncio.to_thread(service.list_models)


@app.post("/api/models/{model_id}/activate")
async def activate_model(model_id: str):
    await _run(lambda: service.activate_model(model_id))
    return {"ok": True}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    await _run(lambda: service.delete_model(model_id))
    return {"ok": True}


@app.get("/api/training")
async def training_status():
    return service.training_status()


@app.post("/api/training")
async def start_training(request: Request):
    body = await request.json()
    return await _run(lambda: service.start_training(body))


@app.post("/api/training/cancel")
async def cancel_training():
    service.training.cancel()
    return {"ok": True}


# ---- phone recording ----

def _phone_errors(function):
    try:
        return function()
    except phone.NotPaired as error:
        raise HTTPException(409, str(error))
    except ValueError as error:
        raise HTTPException(400, str(error))


@app.get("/api/clock")
def clock():
    """The server's clock, for the laptop page's round-trip timing pings."""
    return {"server": phone.server_ms()}


@app.post("/api/phone/pair")
def phone_pair():
    if not config.PHONE_PORT:
        raise HTTPException(409, "Phone recording is off. Restart the server with:  python run.py --phone")
    pairing = phone.new_pairing()
    return {"code": pairing.code, "links": service.phone_links(pairing.code)}


@app.get("/api/phone/status")
def phone_status():
    status = phone.status()
    if status.get("paired"):
        status["links"] = service.phone_links(status["code"])
    status["enabled"] = bool(config.PHONE_PORT)
    return status


@app.post("/api/phone/start")
def phone_start():
    return _phone_errors(phone.start)


@app.post("/api/phone/stop")
def phone_stop():
    return _phone_errors(phone.stop)


@app.post("/api/sessions/phone")
async def create_phone_session(request: Request):
    body = await request.json()
    if not all(k in body for k in ("recording_id", "metadata", "events", "laptop_pings")):
        raise HTTPException(400, "Expected {recording_id, metadata, events, laptop_pings}.")
    return await asyncio.to_thread(_phone_errors, lambda: service.create_phone_session(body))


# The phone's own endpoints: reachable on the phone port, all need the pairing code.

@app.get("/p/{code}")
def phone_page(code: str):
    return FileResponse(WEB_DIR / "phone.html", media_type="text/html")


@app.get("/api/phone/d/{code}/clock")
def phone_clock(code: str):
    _phone_errors(lambda: phone._pairing_for(code))
    return {"server": phone.server_ms()}


@app.post("/api/phone/d/{code}/hello")
async def phone_hello(code: str, request: Request):
    info = await request.json()
    return _phone_errors(lambda: phone.phone_hello(code, info))


@app.get("/api/phone/d/{code}/poll")
def phone_poll(code: str, level: float | None = None):
    return _phone_errors(lambda: phone.phone_poll(code, level))


@app.post("/api/phone/d/{code}/audio")
async def phone_audio(code: str, recording_id: str, start_frame: int, request: Request):
    pcm = await request.body()
    if len(pcm) % 2 or len(pcm) > 4 * 1024 * 1024:
        raise HTTPException(400, "Audio chunks must be whole 16-bit samples, at most 4 MB.")
    return _phone_errors(lambda: phone.phone_audio(code, recording_id, start_frame, pcm))


@app.post("/api/phone/d/{code}/finish")
async def phone_finish(code: str, recording_id: str, request: Request):
    sync = await request.json()
    return _phone_errors(lambda: phone.phone_finish(code, recording_id, sync))


# The single-page interface. Mounted last so the API routes take priority.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
