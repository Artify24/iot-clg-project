"""
StudyGuard AIoT - Local IoT Gateway Server
==========================================
Acts as the local hub for the StudyGuard edge-AIoT stack:

* Receives live video frames over WebSockets from the Phone IoT node or the
  Laptop webcam (`/ws/stream`).
* Runs inference through :class:`engine.focus_engine.FocusEngine`.
* Streams real-time focus analytics back to the node and broadcasts them to
  collaborative study rooms (`/ws/room/{room_id}`).
* Generates a QR code pointing the phone at the gateway pairing page.
* Produces an end-of-session diagnostic report via a rule engine, optionally
  upgraded with Google Gemini when ``GEMINI_API_KEY`` is set.

Run:
    python server.py                       # -> http://<laptop-ip>:8000
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import socket
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import qrcode
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qrcode.constants import ERROR_CORRECT_L
from starlette.websockets import WebSocketDisconnected
from uvicorn.protocols.utils import ClientDisconnected

from engine.focus_engine import FocusConfig, FocusEngine

# Disconnect family raised by starlette/uvicorn when a WebSocket dies
# mid-receive or mid-send. They are sibling Exception classes (not a shared
# base), so all three must be handled together.
_WS_DISCONNECTS = (WebSocketDisconnect, WebSocketDisconnected, ClientDisconnected)

# --------------------------------------------------------------------------- #
# Module-level configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"

HOST = "0.0.0.0"
PORT = int(os.getenv("STUDYGUARD_PORT", "8000"))

MAX_IMAGE_B64_CHARS = 25_000_000  # ~18 MB of raw JPEG bytes, plenty for 4K
JPEG_QUALITY = 70  # annotated preview only (detection uses the raw frame)

DEFAULT_ROOM = "ROOM-101"

# Optional Google Gemini enrichment (google-genai package + GEMINI_API_KEY).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
try:
    from google import genai as _genai  # type: ignore
    GEMINI_AVAILABLE = True
except Exception:  # pragma: no cover - package simply missing
    _genai = None
    GEMINI_AVAILABLE = False
GEMINI_ENABLED = GEMINI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("studygarde")


# --------------------------------------------------------------------------- #
# Local network discovery
# --------------------------------------------------------------------------- #

def detect_local_ip() -> str:
    """Detect the laptop's LAN IP so the phone can pair without hardcoding.

    Prefers the machine hostname's non-loopback addresses
    (socket.gethostbyname_ex) with WiFi/private subnet priority, and falls back
    to the classic UDP-socket trick when the name query is not conclusive.
    """
    candidates: List[str] = []

    def _rank(ip: str) -> int:
        if ip.startswith("192.168."):
            return 0
        if ip.startswith("10."):
            return 1
        if ip.startswith("172."):
            return 2
        return 3

    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
        candidates = [a for a in addrs if not a.startswith("127.")]
    except socket.error as exc:  # pragma: no cover - name resolution failure
        logger.warning("hostname IP discovery failed: %s", exc)

    if candidates:
        candidates.sort(key=_rank)
        return candidates[0]

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:  # provoke the OS routing table without sending traffic
        udp.connect(("8.8.8.8", 80))
        ip = udp.getsockname()[0]
        return ip if not ip.startswith("127.") else "127.0.0.1"
    except OSError:
        return "127.0.0.1"
    finally:
        udp.close()


APP_IP = detect_local_ip()
# Public pairing URL. Defaults to the demo's fixed ngrok tunnel; override with
# $env:STUDYGARD_PUBLIC_URL="https://your-tunnel.ngrok-free.dev" for other hosts.
_PUBLIC_URL = (
    os.getenv("STUDYGARD_PUBLIC_URL")
    or "https://fondue-froth-mowing.ngrok-free.dev"
).rstrip("/")
_LAN_BASE = f"http://{APP_IP}:{PORT}"
NODE_URL = f"{_PUBLIC_URL or _LAN_BASE}/node"
DASHBOARD_URL = f"{_PUBLIC_URL or _LAN_BASE}/"


# --------------------------------------------------------------------------- #
# Helpers: frame <-> base64 JPEG
# --------------------------------------------------------------------------- #

def decode_b64_jpeg(b64: str) -> np.ndarray:
    if not isinstance(b64, str) or not b64:
        raise ValueError("expected non-empty base64 string under 'image'")
    if len(b64) > MAX_IMAGE_B64_CHARS:
        raise ValueError("image payload too large")
    if b64.startswith("data:"):  # tolerate "data:image/jpeg;base64,...."
        b64 = b64.split(",", 1)[-1]
    raw = base64.b64decode(b64, validate=True)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode image bytes as JPEG")
    return frame


def encode_b64_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> Optional[str]:
    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _drop_frame_key(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the focus payload without the numpy frame (JSON-safe)."""
    return {k: v for k, v in payload.items() if k != "processed_frame"}


# --------------------------------------------------------------------------- #
# Session tracking (behind /api/end-session)
# --------------------------------------------------------------------------- #

@dataclass
class SessionTracker:
    started_at: Optional[float] = None
    total_frames: int = 0
    score_sum: float = 0.0
    phone_events: int = 0
    distraction_events: int = 0
    drowsy_events: int = 0
    phone_marks: List[float] = field(default_factory=list)  # minutes since start
    distraction_marks: List[float] = field(default_factory=list)
    drowsy_marks: List[float] = field(default_factory=list)
    last_state: Optional[Dict[str, Any]] = None
    _prev_phone: bool = False
    _prev_distracted: bool = False
    _prev_drowsy: bool = False

    def record(self, state: Dict[str, Any]) -> None:
        now = time.time()
        if self.started_at is None:
            self.started_at = now
        self.total_frames += 1
        self.score_sum += float(state.get("focus_score", 0))
        elapsed_min = (now - self.started_at) / 60.0

        phone = bool(state.get("phone_detected"))
        distracted = bool(state.get("is_distracted"))
        drowsy = bool(state.get("is_drowsy"))

        if phone and not self._prev_phone:
            self.phone_events += 1
            self.phone_marks.append(elapsed_min)
        if distracted and not self._prev_distracted:
            self.distraction_events += 1
            self.distraction_marks.append(elapsed_min)
        if drowsy and not self._prev_drowsy:
            self.drowsy_events += 1
            self.drowsy_marks.append(elapsed_min)

        self._prev_phone = phone
        self._prev_distracted = distracted
        self._prev_drowsy = drowsy
        self.last_state = state

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        duration = max(0.0, now - self.started_at) if self.started_at else 0.0
        avg = (self.score_sum / max(1, self.total_frames)) if self.total_frames else 0
        return {
            "duration_seconds": int(duration),
            "avg_focus": int(round(avg)),
            "distraction_count": self.distraction_events,
            "phone_count": self.phone_events,
            "drowsy_count": self.drowsy_events,
            "phone_marks": list(self.phone_marks),
            "drowsy_marks": list(self.drowsy_marks),
        }


# --------------------------------------------------------------------------- #
# Collaborative study rooms
# --------------------------------------------------------------------------- #

class RoomHub:
    """Multiplexer for collaborative study rooms.

    Holds the connected WebSocket subscribers per room and fans focus updates
    out to every member; dead sockets are garbage-collected on the failed
    broadcast itself, so a crashing tab can never wedge the room.
    """

    def __init__(self) -> None:
        self._rooms: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, room_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._rooms[room_id].add(ws)

    async def disconnect(self, room_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._rooms[room_id].discard(ws)
            if not self._rooms[room_id]:
                self._rooms.pop(room_id, None)

    def member_count(self, room_id: str) -> int:
        return len(self._rooms.get(room_id, ()))

    def active_rooms(self) -> List[str]:
        return list(self._rooms)

    async def broadcast(self, room_id: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            peers = list(self._rooms.get(room_id, ()))
        if not peers:
            return
        dead: List[WebSocket] = []
        for peer in peers:
            try:
                await peer.send_json(payload)
            except Exception:
                dead.append(peer)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._rooms[room_id].discard(ws)
                if not self._rooms.get(room_id):
                    self._rooms.pop(room_id, None)


# --------------------------------------------------------------------------- #
# End-of-session report (rule engine + optional Gemini)
# --------------------------------------------------------------------------- #

class SessionSummaryIn(BaseModel):
    duration_seconds: int = Field(default=0, ge=0)
    avg_focus: int = Field(default=0, ge=0, le=100)
    distraction_count: int = Field(default=0, ge=0)
    phone_count: int = Field(default=0, ge=0)
    drowsy_count: int = Field(default=0, ge=0)


def format_duration(seconds: int) -> str:
    m, s = divmod(max(0, int(seconds)), 60)
    minutes = "1 min" if m == 1 else f"{m} mins"
    secs = "1 sec" if s == 1 else f"{s} secs"
    if m == 0:
        return secs
    return f"{minutes} {secs}"


def productivity_grade(avg_focus: int) -> str:
    bands = (
        (93, "A+"), (88, "A"), (82, "A-"), (76, "B+"), (70, "B"),
        (64, "B-"), (58, "C"), (50, "D"),
    )
    for threshold, grade in bands:
        if avg_focus >= threshold:
            return grade
    return "F"


def build_summary(
    avg_focus: int,
    duration_s: int,
    phone_count: int,
    drowsy_count: int,
    distraction_count: int,
    phone_marks: List[float],
    drowsy_marks: List[float],
) -> str:
    minutes = duration_s / 60.0
    if avg_focus >= 85:
        flow = max(1, int(round(minutes * (avg_focus / 100.0))))
        text = (
            f"Great session! You stayed in deep flow for the first {flow} "
            f"minutes of a {format_duration(duration_s)} session."
        )
    elif avg_focus >= 70:
        text = (
            "Solid session with good consistency, though a couple of focus dips "
            "pulled the average down from the A range."
        )
    elif avg_focus >= 55:
        text = (
            "Moderate session - several distractions broke your study flow and "
            "dragged the average down."
        )
    else:
        text = (
            "Low-focus session - significant time was lost to distractions and "
            "phone usage; use today as the baseline to improve on."
        )
    if phone_count:
        text += f" Phone use was flagged {phone_count} time(s)."
    if drowsy_count:
        text += f" Drowsiness was picked up {drowsy_count} time(s) mid-session."
    if distraction_count >= 6:
        text += " Lateral gaze drift was unusually frequent."
    return text


def build_recommendations(
    duration_s: int,
    avg_focus: int,
    phone_count: int,
    drowsy_count: int,
    distraction_count: int,
    phone_marks: List[float],
    drowsy_marks: List[float],
) -> List[str]:
    recs: List[str] = []

    if duration_s >= 25 * 60:
        recs.append(
            "Take a 5-minute break now to reset ocular fatigue and protect "
            "screen focus (the Pomodoro 25/5 rhythm works best for most users)."
        )
    elif duration_s >= 45 * 60:
        recs.append(
            "You crossed 45 continuous minutes - step away from the screen for "
            "a short break so retention does not drop."
        )

    if phone_count:
        if phone_marks:
            marks = ", ".join(f"{m:.0f}" for m in sorted(phone_marks))
            recs.append(
                f"Phone usage was detected {phone_count} time(s) around the "
                f"{marks}-minute mark; consider keeping your phone in another room."
            )
        else:
            recs.append(
                "Phone usage broke your focus this session; try leaving your "
                "phone in another room next time."
            )

    if drowsy_count:
        where = f" around the {drowsy_marks[0]:.0f}-minute mark" if drowsy_marks else ""
        recs.append(
            f"Fatigue signs appeared{where}. Switch to active recall or hydrate "
            "before continuing."
        )

    if distraction_count >= 6:
        recs.append(
            "Lateral gaze drift was frequent - apply the 20-20-20 rule (every "
            "20 minutes look 20 feet away for 20 seconds)."
        )

    if avg_focus < 60:
        recs.append(
            "Before tomorrow's session, spend 2 minutes writing a single launch "
            "task; a clear starting point dramatically cuts warm-up time."
        )
    elif not phone_count and not drowsy_count and avg_focus >= 80:
        recs.append(
            "Excellent discipline today - keep the same start time and "
            "environment tomorrow to keep the streak going."
        )

    recs.append(
        "Review today's notes within 24 hours to move this material from "
        "working memory into long-term storage."
    )
    return recs[:5]


async def enhance_with_gemini(
    report: Dict[str, Any], base_recommendations: List[str]
) -> List[str]:
    """Ask Gemini to refine the recommendations (only when key present)."""
    try:
        client = _genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = (
            "You are a student focus coach for the StudyGuard app. Here is a "
            "session report as JSON: "
            f"{json.dumps(report)}. "
            "Generate exactly 4 concise, actionable study recommendations "
            "(max 20 words each) as a JSON array of strings. Return only valid JSON."
        )
        resp = await asyncio.to_thread(
            client.models.generate_content, model=GEMINI_MODEL, contents=prompt
        )
        text = (resp.text or "").strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("Gemini response did not contain a JSON array")
        ai_recs = json.loads(match.group(0))
        if not isinstance(ai_recs, list) or not all(
            isinstance(r, str) and r for r in ai_recs
        ):
            raise ValueError("Gemini returned a malformed recommendation list")
        logger.info("Gemini enriched the session recommendations")
        return (ai_recs + base_recommendations)[:6]
    except Exception as exc:  # never fail the report because Gemini hiccuped
        logger.warning("Gemini enrichment skipped: %s", exc)
        return base_recommendations


# --------------------------------------------------------------------------- #
# Mock study peers (viva-demo liveliness)
# --------------------------------------------------------------------------- #

_MOCK_PEERS: List[Dict[str, Any]] = [
    {"name": "Priya M.", "focus_score": 92, "status": "Focused",
     "sub_status": "Deep Flow", "emoji": "\U0001F7E2"},
    {"name": "Aman K.", "focus_score": 78, "status": "Slacking",
     "sub_status": "Slacking / Timepass", "emoji": "\U0001F7E1"},
]


async def _peer_simulator(app: FastAPI) -> None:
    """Background task: every 3s mutate the mock peers and fan to active rooms."""
    import random

    statuses = [("Focused", "Deep Flow", "\U0001F7E2"),
                ("Focused", "Deep Flow", "\U0001F7E2"),
                ("Slacking", "Phone Break", "\U0001F7E1"),
                ("Distracted", "Looking Away", "\U0001F7E0")]
    while True:
        await asyncio.sleep(3)
        for peer in _MOCK_PEERS:
            drift = random.randint(-5, 5)
            peer["focus_score"] = max(55, min(99, peer["focus_score"] + drift))
            st, sub, emoji = random.choice(statuses)
            peer["status"] = st
            peer["sub_status"] = sub
            peer["emoji"] = emoji
        update = [dict(p) for p in _MOCK_PEERS]
        for room_id in app.state.room_hub.active_rooms():
            await app.state.room_hub.broadcast(
                room_id, {"type": "peers_update", "room_id": room_id, "peers": update}
            )


def startup_banner(engine_ready: bool) -> None:
    bar = "=" * 62
    print(bar)
    print("        StudyGuard AIoT  -  Local IoT Gateway")
    print(bar)
    print(f"  Dashboard ............ {DASHBOARD_URL}")
    print(f"  PHONE PAIRING URL .... {NODE_URL}")
    print(f"  Scan the QR on that page from your phone to join.")
    if _PUBLIC_URL:
        print(f"  LAN fallback ......... {_LAN_BASE}   (same Wi-Fi only)")
    print(f"  Engine ready ......... {engine_ready}")
    print(f"  Gemini suggestions .... {'ON' if GEMINI_ENABLED else 'OFF (set GEMINI_API_KEY)'}")
    print(bar)


# --------------------------------------------------------------------------- #
# FastAPI app + lifespan
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine: Optional[FocusEngine] = None
    try:
        engine = FocusEngine(config=FocusConfig())
    except Exception as exc:  # pragma: no cover - environment problem
        logger.error("FocusEngine failed to initialise: %s", exc)
    app.state.engine = engine
    app.state.engine_lock = asyncio.Lock()
    app.state.session = SessionTracker()
    app.state.room_hub = RoomHub()

    peer_task = asyncio.create_task(_peer_simulator(app))
    startup_banner(engine_ready=engine is not None)
    yield
    peer_task.cancel()
    try:
        await peer_task
    except asyncio.CancelledError:
        pass
    if engine is not None:
        engine.close()


app = FastAPI(
    title="StudyGuard AIoT Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# --------------------------------------------------------------------------- #
# HTTP endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
def dashboard_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/node")
def node_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "node.html")


@app.get("/api/info")
def api_info() -> Dict[str, str]:
    """Pairing details for clients / fleet onboarding."""
    return {
        "ip": APP_IP,
        "port": PORT,
        "node_url": NODE_URL,
        "dashboard_url": DASHBOARD_URL,
        "qr_url": "/api/qr",
        "gemini_enabled": GEMINI_ENABLED,
    }


@app.get("/api/qr")
def api_qr() -> Response:
    """Dynamic PNG QR that points a phone at the pairing page."""
    qr = qrcode.make(NODE_URL, error_correction=ERROR_CORRECT_L)
    buffer = io.BytesIO()
    qr.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/end-session")
async def end_session(payload: SessionSummaryIn) -> Dict[str, Any]:
    """Produce the structured diagnostic end-of-session report.

    Client-supplied statistics win; the live-tracked session fills any gaps
    (which keeps the viva demo meaningful even with a 0-supplying client).
    """
    tracked = app.state.session.snapshot()

    duration_s = payload.duration_seconds or tracked["duration_seconds"]
    avg_focus = payload.avg_focus or tracked["avg_focus"]
    distraction_count = max(payload.distraction_count, tracked["distraction_count"])
    phone_count = max(payload.phone_count, tracked["phone_count"])
    drowsy_count = max(payload.drowsy_count, tracked["drowsy_count"])
    phone_marks = tracked["phone_marks"]
    drowsy_marks = tracked["drowsy_marks"]

    grade = productivity_grade(avg_focus)
    summary = build_summary(
        avg_focus, duration_s, phone_count, drowsy_count,
        distraction_count, phone_marks, drowsy_marks,
    )
    report = {
        "duration_formatted": format_duration(duration_s),
        "avg_focus": avg_focus,
        "productivity_grade": grade,
        "summary": summary,
        "ai_recommendations": [],
    }

    recommendations = build_recommendations(
        duration_s, avg_focus, phone_count, drowsy_count,
        distraction_count, phone_marks, drowsy_marks,
    )
    if GEMINI_ENABLED:
        recommendations = await enhance_with_gemini(report, recommendations)
    report["ai_recommendations"] = recommendations
    return report


# --------------------------------------------------------------------------- #
# WebSocket: live video stream (phone node / laptop webcam)
# --------------------------------------------------------------------------- #

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    engine: Optional[FocusEngine] = app.state.engine
    engine_lock: asyncio.Lock = app.state.engine_lock
    tracker: SessionTracker = app.state.session
    hub: RoomHub = app.state.room_hub

    if engine is None:
        await ws.send_json({"error": "FocusEngine not available", "status": "error"})
        await ws.close()
        return

    logger.info("stream connected from %s", ws.client.host)
    try:
        while True:
            raw = await ws.receive_json()
            try:
                frame = decode_b64_jpeg(raw.get("image", ""))
                room_id = str(raw.get("room_id") or "").strip().upper()

                async with engine_lock:
                    result = await asyncio.to_thread(engine.process_frame, frame)

                payload = _drop_frame_key(result)
                annotated = result["processed_frame"]
                b64 = encode_b64_jpeg(annotated)
                payload["annotated_frame"] = b64

                await ws.send_json(payload)
                tracker.record(payload)
                if room_id:
                    await hub.broadcast(
                        room_id,
                        {
                            "type": "self_update",
                            "room_id": room_id,
                            "data": _drop_frame_key(payload),
                        },
                    )
            except (ValueError, TypeError) as exc:
                await ws.send_json({"error": str(exc), "status": "error"})
                await asyncio.sleep(0.05)
            except _WS_DISCONNECTS:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("stream processing failure")
                await ws.send_json({"error": f"internal error: {exc}", "status": "error"})
                await asyncio.sleep(0.05)
    except _WS_DISCONNECTS:
        pass
    finally:
        logger.info("stream disconnected from %s", ws.client.host)


# --------------------------------------------------------------------------- #
# WebSocket: collaborative study room
# --------------------------------------------------------------------------- #

@app.websocket("/ws/room/{room_id}")
async def ws_room(ws: WebSocket, room_id: str) -> None:
    room_id = room_id.strip().upper() or DEFAULT_ROOM
    await ws.accept()
    hub: RoomHub = app.state.room_hub
    await hub.connect(room_id, ws)
    logger.info("room '%s' member joined (total %d)", room_id, hub.member_count(room_id))
    try:
        await ws.send_json(
            {
                "type": "hello",
                "room_id": room_id,
                "member_count": hub.member_count(room_id),
            }
        )
        last = app.state.session.last_state
        if last:
            await ws.send_json(
                {"type": "self_update", "room_id": room_id, "data": _drop_frame_key(last)}
            )
        # Keepalive drain: browsers may send pings; we just consume and ignore.
        while True:
            await ws.receive()
    except _WS_DISCONNECTS:
        pass
    finally:
        await hub.disconnect(room_id, ws)
        logger.info("room '%s' member left (total %d)", room_id, hub.member_count(room_id))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    logger.info("Starting StudyGuard gateway on %s:%d", HOST, PORT)
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        ws="websockets",
        access_log=True,
    )


if __name__ == "__main__":
    main()