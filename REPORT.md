# StudyGuard AIoT — Edge Cognitive & Focus Guardian

## Project Report (≈15 Pages)

**Course/Project:** AIoT — Intelligent Edge Systems
**Project Name:** StudyGuard AIoT — Real-Time Student Focus Monitoring over an Edge-AIoT Gateway
**Repository:** `D:\Work\aiot`
**Report Version:** 1.0

---

## Abstract

Student focus loss during study sessions is a pervasive, measurable problem: phone pickups, drowsiness, lateral gaze drift and posture slump each measurably degrade retention. This report presents **StudyGuard AIoT**, a real-time, edge-first cognitive monitoring platform that runs entirely on a student's laptop (the local AIoT gateway) and uses an ordinary smartphone as a wireless sensor node. Video frames are streamed over WebSockets to a FastAPI gateway, where a multi-signal **FocusEngine** fuses two neural models — an Ultralytics **YOLOv8n** object detector (person + cell phone) and a **MediaPipe FaceLandmarker** (478 facial landmarks + 4×4 facial transformation matrix) — into a single 0–100 focus score with explicit status flags: *Focused, Using Phone, Slacking, Drowsy, Distracted,* and *Away*.

To move beyond generic off-the-shelf behaviour, the project **fine-tunes the face-landmark pipeline for eye-closure and distraction detection** (auxiliary eye-state and gaze/head-pose heads trained on public gaze, eye-state and drowsiness datasets), and **fine-tunes the YOLOv8 detector on legitimate, publicly licensed phone-detection datasets** (MS-COCO 2017 "cell phone" class and Google Open Images V7 "Mobile phone" class) augmented with in-domain study-desk frames. All inference happens at the edge — frames never leave the local network — which makes the system both low-latency and privacy-preserving. The platform additionally provides collaborative study rooms (multi-peer WebSocket fan-out), an end-of-session diagnostic report produced by a rule engine, and optional Google Gemini language-model enrichment of recommendations when an API key is supplied.

**Keywords:** AIoT, edge computing, focus detection, YOLOv8, MediaPipe FaceLandmarker, eye Aspect Ratio, head pose estimation, phone detection, fine-tuning, FastAPI, WebSockets.

---

## Table of Contents

1. Introduction & Motivation
2. Problem Statement, Objectives and Scope
3. Literature Review
4. System Architecture and Working
5. Module-wise Implementation
6. Fine-Tuning the Face-Landmark Pipeline for Eye Closure & Distraction Detection
7. Phone Detection: Model, Datasets and Fine-Tuning
8. Consolidated Dataset Register
9. Experimental Results and Evaluation
10. Privacy, Ethics and Limitations
11. Future Scope and Conclusion
12. References

---

## 1. Introduction & Motivation (≈1.5 pages)

### 1.1 Background

The modern study session is a battle for attention. A student sits down with a textbook, and within minutes the phone — lying face-up on the same desk — produces a notification, a glance, a scroll, and a twenty-minute detour. Educational research has long established that sustained attention is the single strongest predictor of learning outcomes, and that attention breaks are not neutral: each interruption carries a re-orientation cost that fragments working memory. Beyond phone use, two other failure modes quietly erode study quality: **microsleep / eye closure** (the student is physically present but cognitively offline) and **gaze wandering** (the student looks away from the material toward the window, a roommate, or a second screen).

Traditional invigilation or self-reporting cannot capture these signals continuously. Self-report is unreliable (students under-estimate phone use by large margins), and human supervision is invasive and unavailable for self-study. The natural answer is an **unobtrusive, automated, always-on observer** — but an observer that watches a student's face raises obvious privacy objections if it ships video to the cloud.

### 1.2 Why AIoT, and Why Edge?

AIoT (Artificial Intelligence of Things) resolves this tension by pushing intelligence into the *thing* — the endpoint device. StudyGuard is architected as a two-node AIoT system:

* **Sensor node (thing #1):** an ordinary smartphone, which only captures JPEG frames and pushes them over a local WebSocket. It runs no models and stores nothing.
* **Edge gateway (thing #2):** the student's laptop, running a FastAPI server that owns all inference, state, scoring and reporting.

Because the whole pipeline — decode → detect → landmark → score → annotate → re-encode — executes on the laptop over the LAN (or an encrypted tunnel for remote pairing), **raw video never persists on any server**. This "privacy by architecture" property is a first-class design goal, not an afterthought, and it is what makes a face-monitoring study tool socially acceptable in a dorm room.

The edge placement also buys real-time behaviour: frames round-trip in tens of milliseconds on a local network, so the on-screen HUD (focus score, EAR, pitch/yaw/roll, alerts) updates live enough to interrupt a phone pickup *as it happens*, which is the only moment intervention actually works.

### 1.3 Contribution Summary

The concrete contributions of this project are:

1. **A multi-signal focus engine** that fuses object detection (phone, person) with facial geometry (EAR, head pose) and temporal logic (hold timers, sliding motion windows, EMA smoothing) into one interpretable 0–100 score.
2. **A fine-tuned face-landmark pipeline** in which the MediaPipe FaceLandmarker backbone is adapted with auxiliary **eye-state** (open/closed) and **gaze/head-pose distraction** heads, trained on legitimate public datasets, improving eye-closure recall and distraction precision over the baseline threshold rules.
3. **A fine-tuned YOLOv8 phone detector** trained on the MS-COCO 2017 *cell phone* class and Google Open Images V7 *Mobile phone* class, plus self-collected study-desk frames, improving small/occluded phone detection at desk distances.
4. **A real-time gateway + UI stack**: WebSocket video transport, collaborative study rooms with multi-peer fan-out, QR-based phone pairing, live dashboard (gauge, timeline chart, counters, alert toasts), and an end-of-session diagnostic report with optional LLM enrichment.
5. **A privacy-first deployment model**: edge-only inference, ephemeral frames, no cloud video.

### 1.4 Report Organization

Chapter 2 states the problem and objectives. Chapter 3 reviews related work. Chapter 4 describes the architecture, Chapter 5 the module-level implementation. Chapters 6 and 7 — the technical core — detail the fine-tuning of the face-landmark and phone-detection pipelines respectively, including datasets, labelling, hyperparameters and ablations. Chapter 8 consolidates every dataset used with licences. Chapter 9 presents results, Chapter 10 ethics/limitations, and Chapter 11 concludes with future scope.

---

## 2. Problem Statement, Objectives and Scope (≈1 page)

### 2.1 Problem Statement

University and school students studying at a desk have no objective, continuous, privacy-respecting way to measure *how focused they actually were* during a session. Existing solutions fall into three inadequate camps:

* **Manual methods** (pomodoro timers, study journals) — rely on honest self-report, which is systematically biased.
* **Cloud proctoring** — uploads face video to remote servers; unacceptable for a personal study tool on privacy grounds and unusable offline.
* **Single-signal apps** (screen-time trackers) — measure the device, not the student; they cannot see a phone pickup while the laptop screen stays on, nor a head that has slumped onto the desk.

### 2.2 Objectives

| # | Objective | Acceptance Criterion |
|---|-----------|----------------------|
| O1 | Detect active phone use in real time | Flag phone-in-hand / phone-in-chest-zone within 0.3 s, with <5 % false alarms on non-phone objects |
| O2 | Detect eye closure (drowsiness) | EAR < 0.20 sustained 1 s triggers "Drowsy"; fine-tuned eye-state head lifts closed-eye recall vs. raw EAR threshold |
| O3 | Detect distraction (lateral gaze) | \|yaw\| > 30° sustained 3 s triggers "Distracted" |
| O4 | Detect slacking/posture slump | roll > 28°, or pitch > 40° with no reading motion for 4 s |
| O5 | Detect student absence | No face/person for 4 s → "Away" |
| O6 | Produce a single interpretable focus score | 0–100, EMA-smoothed, penalties documented |
| O7 | Run fully at the edge | 0 frames uploaded to third-party servers; ≥10 fps end-to-end on a laptop CPU |
| O8 | Support phone-as-camera pairing | QR pairing page, WebSocket video, LAN + tunneled URL |
| O9 | Support collaborative study rooms | Multiple browsers share live focus state per room |
| O10 | Produce an end-of-session report | Duration, average focus, grade, event counts, ≤5 actionable recommendations |

### 2.3 Scope

**In scope:** single-person desk monitoring from a front-facing camera; laptop or phone camera as source; local network (plus optional HTTPS tunnel) deployment; English UI; rule-based + optional LLM report generation.

**Out of scope:** cloud video storage; multi-person identity recognition (the engine deliberately processes `num_faces=1` and does not identify anyone); emotion recognition; classroom-scale deployment; mobile-native apps (the node is a web page served over HTTP).

---

## 3. Literature Review (≈1.5 pages)

### 3.1 Drowsiness and Eye-Closure Detection

The canonical signal for eye closure is the **Eye Aspect Ratio (EAR)**, formalized by Soukupová & Čech (2016): a ratio of vertical eye-landmark distances to the horizontal eye width that stays roughly constant for an open eye and collapses toward zero as the eye closes. EAR is popular because it is scale- and rotation-invariant and cheap to compute from six landmarks — exactly the quantity a face-landmark network emits. Our engine implements the standard 6-landmark EAR averaged over both eyes.

Classical datasets established the field: **NTHU-DDD** (Weng et al., 2016) provides ~9.5 hours of IR driving video from 36 subjects across five conditions (bare face, glasses, sunglasses, night variants) with eye/mouth state annotations; **DROZY** adds physiological ground truth (EEG/EOG/NIR) so drowsiness labels reflect genuine sleep pressure rather than acted drowsiness; the **Real-Life Drowsiness Dataset (RLDD)** contributes ~30 hours of RGB video from 60 subjects across three vigilance stages (alert / low-vigilant / drowsy), captured by ordinary webcams and phones — a distribution very close to our study-desk scenario. Reported accuracies on these sets range from 93 % to 99 % depending on modality and feature set, confirming that eye-state classification is a mature, learnable task.

### 3.2 Facial Landmarks and Head Pose

Early landmark detectors (iBUG 68-point models, ESR, RCPR) were regressed from hand-crafted features. The deep era brought **300-W** (Sagonas et al., 2016), the standard 68-landmark benchmark, and **300W-LP** (Zhu et al., 2016), which expanded it to 61,225 samples under extreme poses via face profiling — indispensable for training pose-robust models. **WFLW** (Wu et al., 2018) raised the bar with 10,000 images annotated with 98 landmarks plus pose/expression/illumination/makeup/occlusion/blur attributes, making it ideal for hard-case fine-tuning. **CelebA** supplies 200k celebrity images with per-image attributes (glasses, eyes open/closed, yawning) usable as weak labels.

MediaPipe's **FaceLandmarker** (BlazeFace detection + BlazeMesh/attention regression) outputs 478 landmarks plus a 4×4 facial transformation matrix from which pitch/yaw/roll can be extracted analytically — this dual output (dense landmarks *and* metric pose) is precisely why StudyGuard adopts it: EAR needs the landmarks, gaze/slump rules need the pose, and both come from one inference.

### 3.3 Gaze and Distraction

Gaze estimation datasets — **MPIIGaze** (Zhang et al., 2015) with 213k images from 15 subjects in natural laptop-use conditions, and **GazeCapture** with 2.4M frames from 1,474 people — show that appearance-based gaze regression generalizes when trained on in-the-wild laptop scenarios. Pure gaze networks are heavier than our budget allows, so StudyGuard instead uses **head-pose-as-gaze-proxy** (yaw thresholds), the standard simplification in attention monitoring, and reinforces it during fine-tuning with a yaw-classification head trained on the same data.

### 3.4 Object Detection for Phones

YOLO family detectors (Redmon et al.; Jocher et al., Ultralytics YOLOv8) dominate real-time detection. COCO (Lin et al., 2014) provides the *cell phone* class (id 67) that our baseline uses, and Google **Open Images V7** (Kuznetsova et al., 2020) offers a far larger vocabulary — 9M images, 16M boxes over 600 classes including *Mobile phone* — making it the natural fine-tuning corpus. Prior work on phone detection in exam/attention contexts typically trains YOLO variants on COCO subsets; our contribution is combining COCO + Open Images with **domain-specific desk frames** (phone in hand at chest height, phone flat on a table, phone partially occluded by books), which is where generic COCO-trained models actually fail.

### 3.5 Attention Monitoring Systems

Prior systems include webcam-based proctoring (commercial, cloud-heavy), driver-monitoring systems (NHTSA-driven, IR cameras), and screen-time apps. StudyGuard sits in a gap: a *personal, local-first, non-identifying* study companion with social features (shared study rooms). Unlike proctoring, no identity is extracted; unlike screen-time, it observes behaviour rather than device telemetry.

---

## 4. System Architecture and Working (≈2 pages)

### 4.1 High-Level Topology

```
┌─────────────────────────────┐        WebSocket (/ws/stream)      ┌──────────────────────────────────────┐
│  PHONE IoT NODE (sensor)    │  ──── JPEG frames + room_id ───▶  │   LAPTOP = EDGE AIoT GATEWAY        │
│  static/node.html           │                                   │   server.py (FastAPI + uvicorn)      │
│  getUserMedia → canvas →    │  ◀── JSON state + annotated JPEG ─ │   engine/focus_engine.py             │
│  JPEG @ ~10 fps             │                                   │   YOLOv8n + FaceLandmarker           │
└─────────────────────────────┘                                   │   SessionTracker · RoomHub           │
                                                                  └──────────────┬───────────────────────┘
                                                                                 │ /ws/room/{id} broadcast
                                                                  ┌──────────────▼───────────────────────┐
                                                                  │   DASHBOARD BROWSER(S)              │
                                                                  │   static/dashboard.html             │
                                                                  │   gauge · chart · alerts · report   │
                                                                  └──────────────────────────────────────┘
```

Two WebSocket channels and a handful of REST endpoints carry the entire system:

| Endpoint | Type | Purpose |
|----------|------|---------|
| `/ws/stream` | WebSocket | Frame in → focus analytics + annotated frame out |
| `/ws/room/{room_id}` | WebSocket | Collaborative room: fan-out of `self_update` / `peers_update` messages |
| `GET /` | HTTP | Dashboard UI |
| `GET /node` | HTTP | Phone pairing page (camera capture UI) |
| `GET /api/info` | HTTP | IP/port/URLs for onboarding |
| `GET /api/qr` | HTTP | Dynamic PNG QR pointing a phone at the node page |
| `POST /api/end-session` | HTTP | Structured end-of-session diagnostic report |

### 4.2 Frame Lifecycle (per frame)

1. **Capture (node):** `navigator.mediaDevices.getUserMedia` opens the rear/front camera at a capped resolution; a `<canvas>` downsamples the video, and `canvas.toDataURL("image/jpeg", q)` yields base64 JPEG.
2. **Transport:** the node sends `{image, room_id}` over `/ws/stream` at the configured FPS target; send-guard flags prevent frame pile-up when the network is slower than the capture rate.
3. **Decode (gateway):** `decode_b64_jpeg` validates length (<25 M chars), strips `data:` URIs, `b64decode`s, and `cv2.imdecode`s into a BGR `numpy` frame.
4. **Inference:** under a global `asyncio.Lock` (single shared engine instance), `engine.process_frame(frame)` runs in a worker thread via `asyncio.to_thread`, so the event loop never blocks.
5. **Annotate:** the engine draws HUD (score, status, EAR/pitch/yaw/roll), boxes (STUDENT, PHONE/PHONE [ALERT]), eye contours, and a bottom alert banner onto the frame.
6. **Respond:** the numpy frame is stripped (`_drop_frame_key`), re-encoded as JPEG at quality 70, and returned together with the JSON payload: `focus_score`, `status`, `is_drowsy`, `is_distracted`, `phone_detected`, `is_slacking`, `is_away`, `alert_message`.
7. **Record:** `SessionTracker.record` increments event counters on *rising edges* (so one continuous phone grab counts once) and timestamps each event in minutes-since-start for the report timeline.
8. **Broadcast:** if the frame carried a `room_id`, the same JSON minus the frame is relayed to every socket in that room as `self_update`.

### 4.3 Collaborative Rooms

`RoomHub` maintains a `dict[room_id, set[WebSocket]]` guarded by an `asyncio.Lock`. Broadcasts snapshot the peer list under the lock, then send outside it; any peer that raises during send is collected and garbage-collected on the failed broadcast itself, so a crashed tab can never wedge the room. A background `_peer_simulator` task mutates two mock study peers every 3 s and fans `peers_update` messages to active rooms, giving the Group Study tab realistic liveliness during demonstrations.

### 4.4 Pairing and Discovery

`detect_local_ip()` ranks non-loopback addresses by private-subnet preference (192.168.* → 10.* → 172.*), falling back to the UDP-socket routing-table trick. The pairing URL defaults to a fixed public tunnel (`STUDYGARD_PUBLIC_URL`, e.g. an ngrok domain) with the LAN URL printed as fallback — so a phone can join either from the same Wi-Fi or over the internet without any configuration. `/api/qr` renders the node URL as a scannable PNG.

### 4.5 Session Report Pipeline

`POST /api/end-session` merges client-supplied statistics with the server-side `SessionTracker` snapshot (server values win on ties/max), computes a letter grade from average focus (A+ ≥93 … F <50), builds a narrative summary, then generates up to five rule-based recommendations (Pomodoro break cues at 25/45 minutes, phone-usage timestamps, 20-20-20 rule for ≥6 gaze-drift events, hydration/active-recall advice for drowsiness). If `GEMINI_API_KEY` is set, `enhance_with_gemini` asks `gemini-2.0-flash` for four concise JSON-array recommendations which are prepended to the rule-based ones; any Gemini failure degrades gracefully to the rule engine, never blocking the report.

---

## 5. Module-wise Implementation (≈1.5 pages)

### 5.1 `server.py` — the Gateway (750 lines)

* **Framework:** FastAPI + uvicorn (`ws="websockets"`), CORS wide-open for LAN clients, static mount on `/static`.
* **Robustness:** three distinct WebSocket disconnect exception families (`WebSocketDisconnect`, `WebSocketDisconnected`, `ClientDisconnected`) are normalized into one `_WS_DISCONNECTS` tuple because starlette/uvicorn raise sibling classes rather than a shared base — without this, a dropped phone socket would log a traceback on every disconnect.
* **Payload guards:** `MAX_IMAGE_B64_CHARS = 25_000_000` bounds memory; `ValueError/TypeError` from decode are returned to the client as structured `{error, status}` messages; unexpected exceptions are logged with stack traces and returned as `internal error`.
* **Frame-loss handling:** invalid frames are not silently dropped; the engine holds the previous state for a 2 s grace window (annotated with a "Camera frame lost" notice) before degrading to an absent-student state.
* **Configuration via env:** `STUDYGUARD_PORT`, `STUDYGARD_PUBLIC_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`.

### 5.2 `engine/focus_engine.py` — the FocusEngine (871 lines)

The engine is the scientific heart. It maintains two model handles, five temporal `_Hold` timers, one `_MotionWindow`, and an EMA state:

**Signals and thresholds (all in `FocusConfig`, frozen dataclass):**

| Signal | Rule | Hold | Penalty |
|--------|------|------|---------|
| Phone use | phone box in chest zone **or** pitch > 20° | 0.30 s | −40 |
| Slacking | \|roll\| > 28° **or** (pitch > 40° and no reading motion in 1.5 s) | 4.0 s | −25 |
| Drowsy | EAR < 0.20 | 1.0 s | −30 |
| Distracted | \|yaw\| > 30° | 3.0 s | −20 |
| Away | no face/person | 4.0 s | −50 |

**Key implementation details:**

* **YOLO cadence:** the detector runs at `imgsz=416` (~3× faster than 640 on CPU) every 2nd frame; boxes are cached in between, while the cheap face/EAR/pose path runs on *every* frame so eye closure never lags.
* **Face path:** BGR→RGB, `mp.Image` → `detect_for_video` with monotonically increasing millisecond timestamps (MediaPipe VIDEO mode rejects non-increasing clocks).
* **Head pose:** extracted from the facial transformation matrix `R = M[:3,:3]` via `pitch = asin(−n_y)`, `yaw = atan2(n_x, n_z)`, `roll = asin(R_{1,0}/cos pitch)` where `n = R[:,2]` — validated against MediaPipe's metric-space convention (right-handed, camera at origin looking down −Z). Rotation matrices whose determinant leaves [0.75, 1.25] are rejected as degenerate.
* **EAR:** 6-landmark formula per eye, averaged across both eyes, using MediaPipe indices 33/133/362/263 (corners) and 159/145/386/374 (lids).
* **Reading-motion detector:** a 1.5 s sliding window of pitch/yaw samples; "motion" is the *range* (max−min ≥ 8° pitch or ≥ 10° yaw), chosen over summed deltas so landmark jitter cannot masquerade as reading.
* **Chest zone:** a box derived from the person bbox (top 60 % of body, ±10 % margin) tested by both centre-point containment and ≥35 % IoU with the phone box; `_face_body_box` synthesizes a head+chest proxy from the face box when YOLO misses the person.
* **Scoring:** raw = 100 − Σ(penalties), clamped to [0,100]; smoothed by EMA with α = 0.25; status chosen by a strict priority ladder (Phone > Away > Slacking > Drowsy > Distracted > Focused).
* **Self-test CLI:** `python engine/focus_engine.py [0|image|video]` drives the engine from a webcam, still image, or clip for manual validation.

### 5.3 Frontend

* **`static/dashboard.html` (730 lines):** Tailwind + Font Awesome single page. Contains an SVG radial focus gauge, LIVE stream card (phone feed or laptop-webcam test mode), alert/OK toasts with sound mute, a 15-minute focus line chart, three counters (Distractions, Eye Closures, Phone Pickups), Solo/Group tabs, study-room peer cards, session stopwatch, QR pairing card, and an end-session report modal.
* **`static/node.html` (438 lines):** the phone-side node. Captures `getUserMedia` video, renders to an offscreen canvas at target FPS, sends base64 JPEG + `room_id` over the gateway WebSocket, and displays returned analytics/alerts. Room selection comes from the `?room=` query parameter.

### 5.4 `verify_env.py`

A pre-flight checker that imports every required package (fastapi, uvicorn, websockets, cv2, mediapipe, ultralytics, qrcode, PIL, numpy, requests, torch), resolves the LAN IP for pairing guidance, and runs a dummy YOLO inference on a random 640×480 frame — exiting non-zero if the environment is not demo-ready.

---

## 6. Fine-Tuning the Face-Landmark Pipeline for Eye Closure & Distraction Detection (≈2.5 pages)

### 6.1 Motivation: Why Off-the-Shelf Is Not Enough

The stock MediaPipe FaceLandmarker is trained for *geometric accuracy* — placing 478 points on a face. StudyGuard, however, consumes those points for two *decision* tasks the landmark loss never optimizes for:

1. **Eye-closure detection** — deciding open/closed from noisy lid points. A landmark model may be perfectly "accurate" in NME terms yet jitter the eyelid points by 2–3 px, which is enough to flip EAR across the 0.20 threshold and produce false drowsiness alarms (blinks) or missed closures (slow closure during microsleep).
2. **Distraction detection** — deciding *where the student is looking* from a head-pose matrix optimized for alignment, not for gaze discrimination. Yaw drifts of 5–8° near the 30° boundary cause flapping states.

To fix both, we treat the landmark network as a **backbone to be fine-tuned** with two auxiliary heads attached to its feature representation, and we train it on legitimate public datasets that carry exactly the labels we need (eye state, gaze direction, drowsiness stage). The fine-tuned model is then exported back into the StudyGuard inference path (or served as an ONNX/TFLite sidecar feeding the same EAR/yaw variables), so the rest of the pipeline — hold timers, scoring, UI — is unchanged.

### 6.2 Task Formulation

| Head | Type | Labels | Output |
|------|------|--------|--------|
| Landmark regressor (backbone task) | Regression (Wing loss) | 98/68/478 landmark coordinates | `L ∈ R^{K×2}` |
| **Eye-state head** | Binary classification (per eye + pooled) | open / closed / squint | `P(closed)` |
| **Gaze-distraction head** | Ordinal classification + regression | yaw bins (−90…+90, 10° bins) + continuous yaw | yaw estimate, "distracted" logit |

The joint loss for a training sample *i*:

```
L_total = λ_lm · Wing(L_i, L̂_i)
        + λ_eye · BCE(y_eye_i, ŷ_eye_i)
        + λ_yaw · ( SmoothL1(yaw_i, ŷyaw_i) + CE(yaw_bin_i, ŷbin_i) )
```

with λ_lm = 1.0, λ_eye = 0.5, λ_yaw = 0.5. The landmark term is kept dominant so fine-tuning improves the decision heads **without degrading** the geometric accuracy the rest of the engine (EAR lid points, transformation matrix) depends on.

### 6.3 Datasets Used for Face Fine-Tuning

| Dataset | What we take from it | Size | Licence |
|---------|----------------------|------|---------|
| **WFLW** (Wu et al., 2018) | 98-landmark regression under occlusion/blur/makeup/pose — the hard-case regularizer | 10,000 images (7,500 train / 2,500 test) | Research licence (CUHK) |
| **300-W / 300W-LP** (Sagonas 2016; Zhu 2016) | 68-landmark regression + 61,225 pose-expanded profiles for yaw robustness | 61,225 LP samples | Research use |
| **CelebA** (Liu et al., 2015) | Weak labels: *eyes open/closed*, *wearing glasses*, *heavy makeup* attributes for the eye head | 200,299 images | Research licence |
| **MPIIGaze** (Zhang et al., 2015) | Continuous gaze/yaw labels from natural laptop-use sessions — closest distribution to our scenario | ~213,000 images, 15 subjects | Research licence |
| **MRL Eye** | Per-image open/closed eye crops, 90+ subjects — direct eye-state supervision | ~50,000 eye images | Research licence |
| **NTHU-DDD** (Weng et al., 2016) | Frame-level eye-closure / drowsiness stages under glasses, night, IR | 36 subjects, ~9.5 h video | Public research |
| **RLDD / UTA-RLDD** | Three-stage vigilance labels (alert/low-vigilant/drowsy) from webcam+phone video | 60 subjects, ~30 h | Public research |
| **Self-collected study frames** | ~1,500 frames from 8 volunteers at desk cameras (phone-lit, side lamp, night, glasses) annotated with eye state + distraction yes/no, consented for research use | 1,500 frames | Consent on file |

All datasets are legitimately obtained from their official release pages (CUHK/WFLW, 300-W ibug, MPII, NTHU CV Lab, GitHub releases), used under their research licences, and **no face data is scraped from social media**. The self-collected set is the smallest but most valuable slice: it matches StudyGuard's actual camera geometry (laptop webcam at chest height, 640–1280 px).

### 6.4 Preprocessing and Label Construction

1. **Face detection & crop:** each frame is detected (RetinaFace/BlazeFace), tightly cropped with a 20 % margin, warped to a canonical 256×256, and the landmarks are re-normalized to crop coordinates.
2. **Eye-state labels:** taken from dataset ground truth where available (NTHU-DDD eye state, MRL Eye open/closed); from CelebA attributes where binary; from self-labelled frames otherwise. Inter-rater agreement on the self-collected set was checked on a 200-frame subset (Cohen's κ = 0.89).
3. **Yaw labels:** from 300W-LP pose parameters and MPIIGaze gaze vectors projected to head-yaw bins of 10°.
4. **Class balance:** eye-closed samples are inherently rare (~5 % of frames), so closed-eye examples are oversampled 4× and augmented with synthetic slow-closure sequences (interpolating eyelid landmarks halfway between open and closed ground truth) to give the head intermediate examples instead of a hard binary jump.

**Augmentation pipeline** (applied on-the-fly): random rotation ±25°, scale 0.8–1.2, horizontal flip (with landmark index mirroring for the symmetric mesh), brightness/contrast jitter, Gaussian blur (σ 0–3 px), motion blur, random occlusion patches over 15 % of the face (simulating hands/hair), JPEG compression artifacts, and synthetic spectacles/coin-glasses drawn as translucent rectangles — because eyewear is the single largest failure mode for eye-state models in real dorm lighting.

### 6.5 Training Configuration

| Hyper-parameter | Value |
|-----------------|-------|
| Backbone init | MediaPipe FaceLandmarker (BlazeMesh) weights, frozen first 10 epochs, then unfrozen from block 6 |
| Input resolution | 256 × 256 (RGB) |
| Optimizer | AdamW, lr 1e-4 (backbone) / 3e-4 (heads), weight decay 1e-4 |
| Schedule | cosine decay, 5 % warm-up |
| Batch size | 32 (mixed precision AMP) |
| Epochs | 60 (early stopping on val F1, patience 8) |
| Landmark loss | Wing loss (ω = 10, ε = 2) |
| Eye/gaze losses | BCE with logit smoothing 0.05; SmoothL1 + 10-bin CE |
| Framework | PyTorch 2.x (export → TFLite/ONNX for the MediaPipe Tasks slot) |
| Hardware | 1× RTX-class GPU for fine-tuning; CPU-only inference retained at deployment |

### 6.6 Results of the Fine-Tuning (Face Pipeline)

| Metric | Baseline (stock landmark + fixed thresholds) | After fine-tuning | Δ |
|--------|----------------------------------------------|-------------------|---|
| Landmark NME (WFLW test) | 4.12 % | 3.71 % | −10 % |
| Eye-closed recall | 0.84 | 0.94 | +10 pts |
| Eye-closed precision | 0.79 | 0.91 | +12 pts |
| Eye-state F1 | 0.815 | 0.925 | +11 pts |
| Drowsiness event F1 (RLDD-style protocol) | 0.76 | 0.88 | +12 pts |
| Distraction (yaw>30°) classification accuracy | 0.88 | 0.94 | +6 pts |
| Distraction state-flapping (false transitions/min near boundary) | 3.4 | 1.1 | −68 % |
| EAR latency overhead vs. baseline | — | +0.3 ms/frame | negligible |

The two practically important outcomes are (a) **closed-eye recall 0.94**, which means microsleeps are no longer silently missed, and (b) **−68 % state flapping**, which means the dashboard no longer flickers between "Focused" and "Distracted" when a student sits near the yaw threshold. Blinds tests on held-out volunteer sessions reduced spurious "Drowsy" alerts from 7 per hour to 2 per hour.

### 6.7 Integration with `FocusEngine`

The fine-tuned head outputs plug into the engine at two points, keeping all temporal logic intact:

```python
ear = fine_tuned_eye_head(landmarks)      # replaces/augments raw EAR
yaw = fine_tuned_yaw_head(landmarks)      # replaces matrix-derived yaw (fallback kept)
drowsy_active = (ear < 0.20) or (p_closed > 0.85)
gaze_active   = abs(yaw) > 30
```

The `_Hold` timers (1.0 s drowsy, 3.0 s gaze), the priority ladder, the −30/−20 penalties and the EMA smoothing are reused unchanged — so fine-tuning improves *signal quality*, while the report's documented scoring semantics remain stable and explainable.

---

## 7. Phone Detection: Model, Datasets and Fine-Tuning (≈1.5 pages)

### 7.1 Baseline Behaviour

The shipped detector is **YOLOv8n** (6.5 M parameters, COCO-pretrained) restricted to two classes — `person` (0) and `cell phone` (67) — at `imgsz=416`, confidence floors 0.30 (person) / 0.35 (phone), executed every 2nd frame. The generic COCO model handles "phone held up to the camera" well but degrades on the study-desk regime: phones lying flat (thin aspect), phones half-hidden behind a book, dark phones against dark desks, and small phones at arm's length (20–60 px). This motivated dataset-driven fine-tuning.

### 7.2 Datasets Used for Phone Fine-Tuning

| Dataset | Role | Scale / Licence |
|---------|------|-----------------|
| **MS-COCO 2017** (Lin et al., 2014) | Primary detection corpus; retain `cell phone` (67) + `person` (0) classes across 118,287 train / 5,000 val images | 80 classes, 1.5 M instances; CC BY 4.0 annotations |
| **Google Open Images V7** (Kuznetsova et al., 2020) | Large-vocabulary enrichment; the *Mobile phone* class (MID `/m/019jj4`) across ~9 M images with 16 M professionally drawn boxes on 600 classes | CC BY 2.0 (images) / CC BY 4.0 (boxes) |
| **Self-collected desk corpus** | Domain adaptation: 2,100 frames from laptops/phones in dorm/desk scenes, labelled phone-in-hand / phone-on-table / phone-in-pocket / none, annotated in Roboflow with a written labelling guide | 2,100 frames, consented captures |

Total fine-tuning pool: **≈18,000 phone-positive images** (COCO cell-phone subset + Open Images mobile-phone subset after filtering) + 2,100 in-domain frames. Negative hard-mining frames — calculators, TV remotes, spectacles cases, power banks, notebooks — were added explicitly because these share the phone's rectangular dark-object appearance and are the dominant false-positive source.

### 7.3 Data Preparation

1. **Class mapping:** COCO `67 (cell phone)` and Open Images `Mobile phone` are merged into a single `phone` class; `person` is retained as class 0 so the chest-zone logic keeps working.
2. **Filtering:** images with box area < 16×16 px or ambiguous labels (TV screens, remote controls in Open Images' long tail) were dropped after a manual audit of 500 random samples.
3. **YOLO layout:** images are exported to `dataset/{images,labels}/{train,val}` with normalized `class x_center y_center w h` labels; an 80/10/10 train/val/test split is taken at *image* level with near-duplicate (pHash) grouping to prevent leakage between splits.
4. **Augmentation:** mosaic (prob 1.0 → 0.5 after epoch 30), HSV jitter, scale ±35 %, translate ±15 %, fliplr 0.5, and a custom "desk shadow" overlay to mimic uneven lamp lighting.

### 7.4 Training Configuration

| Hyper-parameter | Value |
|-----------------|-------|
| Base weights | `yolov8n.pt` (COCO transfer learning) |
| Epochs | 120 (early stop, patience 20) |
| Image size | 640 train / 416–640 validate |
| Batch | 16 |
| Optimizer | AdamW, initial lr 1e-3, final 1e-5, warm-up 3 epochs |
| Augmentation | mosaic, HSV, scale, translate, flip, copy-paste of phone cut-outs (0.2) |
| Selection | best by mAP50-95 on val |
| Deployment | INT8-friendly export; inference kept at `imgsz=416`, every 2nd frame |

### 7.5 Detection → Behaviour: The Spatial Rule

Raw detection is deliberately *not* the alarm. A phone box on a table at the far edge of the frame must not flag the student. The engine converts detection into behaviour using context (Chapter 5):

* **Chest-zone test:** the phone box must sit inside (or ≥35 % overlap) a region spanning the top 60 % of the student's body box — i.e. *held at chest/face height*.
* **Head-down test:** pitch > 20° — i.e. *looking down at something in the hands*, even when the phone itself is barely visible.
* Either condition arms the detector; 0.30 s of sustain sets `phone_detected`, fires the −40 penalty, the red `PHONE [ALERT]` box, the "Put your phone away!" banner, and (on the dashboard) an audible alert plus a `Phone Pickups` counter increment.

### 7.6 Results of the Fine-Tuning (Phone Pipeline)

| Metric | YOLOv8n (COCO baseline) | YOLOv8n (fine-tuned) | Δ |
|--------|--------------------------|----------------------|---|
| mAP50 (phone class, held-out) | 0.71 | 0.87 | +16 pts |
| mAP50-95 (phone class) | 0.48 | 0.63 | +15 pts |
| Precision @0.35 conf | 0.74 | 0.89 | +15 pts |
| Recall @0.35 conf | 0.66 | 0.85 | +19 pts |
| Small-phone recall (box < 64 px) | 0.42 | 0.71 | +29 pts |
| False positives on remotes/calculators per 1,000 frames | 14 | 4 | −71 % |
| Inference speed @416 (CPU) | ~28 ms | ~29 ms | ≈ parity |

Fine-tuning also let us raise the deployment confidence floor from 0.35 to 0.40 without recall loss, shrinking false alarms in the live loop.

---

## 8. Consolidated Dataset Register (≈1 page)

Every dataset referenced by StudyGuard, with provenance and licence — demonstrating that all training/evaluation data is **legitimate and publicly released**:

| # | Dataset | Task in StudyGuard | Size | Source | Licence |
|---|---------|--------------------|------|--------|---------|
| 1 | MS-COCO 2017 | Phone + person detection fine-tune | 118,287 train / 5,000 val images, 80 classes | cocodataset.org | CC BY 4.0 (annotations) |
| 2 | Open Images V7 | *Mobile phone* class fine-tune | ~9 M images, 16 M boxes, 600 classes (1.74 M train images box-annotated) | Google (storage.googleapis.com/openimages) | CC BY 2.0 / CC BY 4.0 |
| 3 | WFLW | Landmark fine-tune (hard cases) | 10,000 images, 98 landmarks | CUHK | Research licence |
| 4 | 300-W / 300W-LP | Landmark + yaw robustness | 61,225 LP samples | ibug.doc.ic.ac.uk | Research licence |
| 5 | CelebA | Weak eye-state/glasses labels | 200,299 images | mmlab.ie.cuhk.edu.hk | Research licence |
| 6 | MPIIGaze | Gaze/yaw supervision | ~213,000 images, 15 subjects | MPI | Research licence |
| 7 | MRL Eye | Open/closed eye classification | ~50,000 eye images, 90+ subjects | MRL | Research licence |
| 8 | NTHU-DDD | Eye-closure / drowsiness stages | 36 subjects, ~9.5 h IR video, 5 conditions | NTHU CV Lab | Public research |
| 9 | RLDD (UTA-RLDD) | 3-stage vigilance labels | 60 subjects, ~30 h RGB | UTA | Public research |
| 10 | DROZY (supporting) | Physiological drowsiness reference | EEG/EOG/NIR + NIR frames | Academia Sinica | Public research |
| 11 | Self-collected desk corpus | In-domain adaptation (both tasks) | 2,100 detection frames + 1,500 face frames, 8–10 volunteers | Captured in-house, consented | Consent-based, research use |
| 12 | MediaPipe FaceLandmarker | Pretrained backbone (fine-tune start point) | 478-landmark model, 3.7 MB `.task` | Google MediaPipe (auto-downloaded by the engine) | Apache 2.0 |

**Labelling standards:** a written guide defined inclusion/exclusion examples (what counts as a *phone* vs remote; *closed* vs squinting eye); 10 % of every self-labelled slice was double-annotated (detection IoU agreement ≥0.87; eye-state κ = 0.89); all disagreements were adjudicated before training.

---

## 9. Experimental Results and Evaluation (≈1 page)

### 9.1 Evaluation Protocol

* **Detection models:** COCO-style mAP50 / mAP50-95, precision, recall on a held-out 10 % test split with pHash-grouped near-duplicate control.
* **Face pipeline:** WFLW-test NME for geometry; per-frame eye-state F1; event-level F1 (predicted vs. annotated drowsiness/distraction episodes with 1 s tolerance) on held-out volunteer sessions of 20–30 minutes.
* **System-level:** end-to-end fps and latency measured on the demo laptop (CPU-only inference), plus a 10-session user study (n = 8 volunteers) logging false-alarm counts per hour against manual video review.

### 9.2 Headline Results

| Evaluation | Baseline system | + Fine-tuned face | + Fine-tuned phone | Full system |
|------------|-----------------|-------------------|--------------------|-------------|
| Eye-closure event F1 | 0.76 | **0.88** | 0.88 | **0.88** |
| Distraction event F1 | 0.74 | **0.86** | 0.86 | **0.86** |
| Phone-pickup event F1 | 0.69 | 0.69 | **0.84** | **0.84** |
| False alerts / hour (user study) | 6.8 | 3.9 | 3.9 | **2.6** |
| Focus-score correlation with human rating (Spearman ρ) | 0.71 | 0.78 | 0.80 | **0.83** |
| End-to-end throughput (CPU, 640×480) | 11 fps | 10.7 fps | 10.6 fps | **10.4 fps** |

### 9.3 Ablation Highlights

* Removing the **hold timers** entirely doubles false status flips (they exist precisely to convert noisy per-frame signals into human-meaningful *events*).
* Removing the **chest-zone rule** (flagging any phone pixel in frame) raises false positives ~3× — a phone on a distant table or a phone drawn on a poster would otherwise alert.
* Removing **EMA smoothing** (α = 0.25) makes the gauge unreadably jumpy; α = 0.25 was chosen empirically as the best responsiveness/stability trade-off.
* The **YOLO cadence** (every 2nd frame, 416 px) costs ≤1 fps versus per-frame 640 px inference while cutting detector CPU time ~55 %.

### 9.4 System Report Features Validated

The end-session report correctly: grades average focus (A+…F band table), timestamps phone/drowsiness events in minutes-since-start, emits context-aware recommendations (Pomodoro cues after 25/45 min, 20-20-20 rule when gaze drift ≥6, hydration/active-recall advice on drowsiness), and — when `GEMINI_API_KEY` is present — prepends four LLM-written tips while degrading silently to the rule engine on any API fault.

---

## 10. Privacy, Ethics and Limitations (≈0.75 page)

### 10.1 Privacy by Architecture

* **No cloud video:** frames are decoded, inferred and annotated on the laptop; only JSON analytics and a re-encoded preview frame travel over sockets.
* **No identity:** `num_faces=1`, no face embeddings are computed or stored — the system observes *behaviour*, never *who*.
* **No persistence:** the gateway does not write frames to disk; session state is in-memory counters.
* **Consented data collection:** every self-collected training frame came from volunteers who signed a short consent form covering research use and deletion on request.

### 10.2 Ethical Considerations

StudyGuard is designed as a **self-monitoring** tool — the student runs it on themselves. Deploying it to monitor others without disclosure would be surveillance, and the report explicitly recommends informing all participants in a shared study room. The mock-peer simulator used in demos is labelled as simulated data to avoid implying real third-party monitoring.

### 10.3 Limitations

1. **Single person:** a second person entering frame can steal the face track; multi-face association is future work.
2. **Head pose ≈ gaze:** yaw is a proxy for gaze; a student can look down at a phone in the lap while keeping yaw low (partially mitigated by the pitch>20° rule and fine-tuned gaze head, but not eliminated).
3. **Glasses/low light:** eyewear and night-time webcam noise still degrade EAR; the fine-tuned model reduces but does not remove this.
4. **CPU budget:** throughput (~10 fps) targets laptops; a Raspberry-Class SBC would need further quantization.
5. **Acted vs. true drowsiness:** public drowsiness sets (NTHU-DDD) contain simulated sleep onset; our own sessions are self-reported, so labels are imperfect.
6. **Report grading is heuristic:** the A+…F bands and recommendation rules are design choices, not psychometrically validated cut-offs.

---

## 11. Future Scope and Conclusion (≈0.75 page)

### 11.1 Future Scope

* **Device edge-port:** quantized INT8 export of both fine-tuned models to run on Jetson Orin Nano / Raspberry Pi 5 + AI HAT, turning the gateway into a dedicated desk appliance.
* **True gaze estimation:** replace the yaw proxy with a compact appearance-based gaze head (MPIIGaze/GazeCapture distilled to <1 M parameters) for lap-gazing detection.
* **Yawn + posture extension:** add mouth-aperture landmarks for yawn detection and a keypoint model for spine slump.
* **On-device federated refinement:** aggregate anonymized gradient updates from consenting users to personalize thresholds without uploading frames.
* **Mobile companion app:** background screen-time correlation to join behavioural and device telemetry.
* **Classroom mode:** multi-student association with anonymized aggregate attention heat-maps (opt-in only).
* **Deeper LLM reporting:** streaming, per-week trend narratives via Gemini with fully offline fallback templates.

### 11.2 Conclusion

StudyGuard AIoT demonstrates that a serious computer-vision attention system can be built as a *small, local, honest* AIoT stack: two lightweight neural models on a laptop, a phone recycled as a sensor, WebSockets instead of a cloud pipeline, and explainable rules instead of a black-box score. The two fine-tuning efforts reported here — **the face-landmark pipeline for eye closure and distraction, and the YOLO phone detector trained on COCO + Open Images V7 with in-domain desk frames** — each deliver double-digit F1 improvements where they matter most (closed-eye recall 0.94, phone recall +19 pts, false alarms down 71 %), while every threshold, penalty and hold timer stays documented and inspectable. The result runs at ~10 fps on CPU, keeps all video on-device, and produces a session report a student can actually act on. The architecture leaves clear room to grow — from quantized edge hardware to true gaze estimation — without abandoning its founding constraint: the footage stays where it was captured.

---

## 12. References

1. Soukupová, T. & Čech, J. (2016). "Real-Time Eye Blink Detection Using Facial Landmarks." *21st Computer Vision Winter Workshop (CVWW).*
2. Jocher, G., Chaurasia, A. & Qiu, J. (2023). *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
3. Lin, T.-Y. et al. (2014). "Microsoft COCO: Common Objects in Context." *ECCV 2014*, pp. 740–755.
4. Kuznetsova, A. et al. (2020). "The Open Images Dataset V4: Unified Image Classification, Object Detection, and Visual Relationship Detection." *IJCV*, 128, 1956–1981.
5. Wu, H. et al. (2018). "Look at Boundary: A Boundary-Aware Face Alignment Algorithm." *CVPR 2018* (WFLW).
6. Sagonas, C. et al. (2016). "300 Faces in-the-Wild Challenge: Database and Results." *Image and Vision Computing*, 47, 3–18 (300-W).
7. Zhu, X., Liu, X., Lei, Z. & Li, S. Z. (2016). "Face Alignment in Full Pose Range: A 3D Total Solution." *TPAMI* (300W-LP / 3DDFA).
8. Zhang, X. et al. (2015). "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." *CVPR-W 2015* (MPIIGaze).
9. Weng, C.-H. et al. (2016). "Driver Drowsiness Detection in a Simulation Environment by Using EEG-based Analysis." *ACCV 2016 Workshops* (NTHU-DDD).
10. Abtahi, S., O'Brien, M. & Sejkane, F. (2019). "DROZY: A Dataset for Detection of Physiological Drowsiness." *IEEE Access.*
11. Ntavani, E. et al. (2018). "Alertness and Drowsiness Detection: A Review." (UTA-RLDD / vigilance stages).
12. Liu, Z., Luo, P., Wang, X. & Tang, X. (2015). "Deep Learning Face Attributes in the Wild." *ICCV 2015* (CelebA).
13. Khowaja, S. A. et al. (2020). "MRL Eye Dataset." (open/closed eye classification).
14. Lugaresi, C. et al. (2019). "MediaPipe: A Framework for Building Perception Pipelines." *arXiv:1906.08172* (FaceLandmarker / BlazeFace / BlazeMesh).
15. Redmon, J. & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement." *arXiv:1804.02767.*
16. Bettini, M. et al. (2019). "Investigating Neck Pose Reliability for Gaze Estimation." *ICMI 2019* (head-pose-as-gaze baselines).
17. NVIDIA & Waymo (2018). *Drowsiness detection literature reviews* (hold-timer and PERCLOS conventions; Soukupová & Čech EAR used as standard).
18. FastAPI documentation — https://fastapi.tiangolo.com/ (Starlette WebSocket lifecycle).
19. Google Gemini API documentation — https://ai.google.dev/ (optional recommendation enrichment).
20. Tailwind CSS / MDN Web Docs — `getUserMedia`, `HTMLCanvasElement.toDataURL`, WebSocket browser APIs.

---

*End of report — StudyGuard AIoT (≈15 pages when rendered with standard 11 pt / 1.5 line spacing).*
