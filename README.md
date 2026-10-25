# 🧠 StudyGuard AIoT
> **Zero-Hardware Edge-AIoT Cognitive & Distraction Monitoring System for Students**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/CV-MediaPipe%20FaceMesh-orange.svg)](https://developers.google.com/mediapipe)
[![YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8--nano-green.svg)](https://ultralytics.com/)
[![WebSockets](https://img.shields.io/badge/Protocol-WebSockets-yellow.svg)](https://websockets.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Project Overview

**StudyGuard AIoT** is a non-invasive, privacy-preserving cognitive monitoring system designed for students. Instead of requiring expensive microcontrollers (like ESP32 or Arduino), StudyGuard turns the student's ordinary **smartphone into a wireless Edge-AIoT Multi-Modal Sensor Node** propped on their desk.

The system continuously assesses student focus, posture degradation, drowsiness, and digital distractions in real time, synchronizing live telemetry with a collaborative study dashboard over low-latency WebSockets.

---

## 🏗️ 3-Tier AIoT System Architecture

```text
 ┌────────────────────────────────────────────────────────┐
 │            TIER 1: EDGE SENSING (PERCEPTION)           │
 │  📱 Smartphone on Desk Stand (IoT Edge Sensor Node)     │
 │  • Front Camera: Facial Landmark Geometry              │
 │  • Microphone: Acoustic Distraction Energy             │
 │  • HTML5 WebRTC / Sensor Stream via QR Code Pairing    │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼  WebSocket Telemetry (JSON / Frames)
 ┌────────────────────────────────────────────────────────┐
 │        TIER 2: TRANSPORT & PROCESSING (GATEWAY)        │
 │  💻 Local AI Gateway (FastAPI + Async Python)          │
 │  • Pre-trained Google MediaPipe (468-point FaceMesh)   │
 │  • Pre-trained Ultralytics YOLOv8-nano (Cell Phone)    │
 │  • Cognitive Sensor Fusion Engine                      │
 │  • Exponential Moving Average (EMA) Focus Scoring      │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼  Real-time Event Broadcasting
 ┌────────────────────────────────────────────────────────┐
 │          TIER 3: APPLICATION & COLLABORATION           │
 │  🖥️ Student Focus Hub (Tailwind CSS + Chart.js)        │
 │  • Live Circular Focus Score Gauge (0–100)             │
 │  • Real-Time Audio & Visual Distraction Alerts         │
 │  • Collaborative Study Rooms (Solo vs Group Mode)      │
 │  • End-of-Session AI Diagnostic Report                 │
 └────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

* **📱 Zero External Hardware Required:** Any Android or iOS smartphone acts as the wireless IoT sensor node by simply scanning a QR code on the laptop screen.
* **😴 Drowsiness & Fatigue Detection:** Calculates **Eye Aspect Ratio (EAR)** using eyelid landmark coordinates. Triggers wake-up alerts when prolonged eyelid closure ($>2\text{ s}$) is detected.
* **📱 Active Phone Usage Detection:** Leverages **YOLOv8-nano** to detect mobile phones in the student's hands. Correlates head downward pitch with phone proximity to flag active social media scrolling.
* **⚠️ Timepass & Postural Slump Detection:** Recognizes head-resting-on-palm, extreme lateral tilts (Roll $> 28^\circ$), or continuous gazing away (Yaw $> 30^\circ$).
* **👥 Collaborative Study Rooms:** Supports solo study as well as multi-student rooms (`STUDY-101`) where group members can see each other's live focus status for social accountability.
* **🔒 Privacy-Preserving Architecture:** Video frames are processed in volatile RAM and immediately discarded. Zero raw biometric video is stored or transmitted over public networks.
* **📊 Post-Session AI Diagnostic Report:** Summarizes total study duration, average focus score, distraction counts, and provides personalized study recommendations.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend & IoT Hub** | Python 3.10+, FastAPI, Uvicorn, WebSockets |
| **Edge Computer Vision** | Google MediaPipe FaceMesh / Face Landmarker, Ultralytics YOLOv8-nano (`yolov8n.pt`), OpenCV |
| **Frontend Dashboard** | HTML5, Tailwind CSS, FontAwesome, Chart.js, HTML5 Web Audio API |
| **IoT Protocol** | Bidirectional WebSockets over Local Wi-Fi |

---

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Artify24/iot-clg-project.git
cd iot-clg-project
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Environment
```bash
python verify_env.py
```

### 5. Launch the System
```bash
python server.py
```
* **Laptop Dashboard:** Open `http://localhost:8000` in your browser.
* **Phone IoT Node:** Scan the on-screen QR code or navigate to `http://<laptop-ip>:8000/node` on your phone connected to the same Wi-Fi.

---

## 🎯 Viva / Examiner Presentation Highlights

* **Why is this an AIoT project?**  
  The smartphone functions as an autonomous, multi-modal edge sensing node communicating over IoT transport protocols (WebSockets) to a centralized IoT Gateway.
* **Privacy Standard:**  
  Frames exist in volatile memory only for inference ($\sim 15\text{ ms}$) and are purged; only scalar telemetry (focus score, status) is stored.
* **Model Justification:**  
  Utilizes industry-standard pre-trained models (MediaPipe and YOLOv8-nano) to achieve 30+ FPS real-time processing on standard commodity CPUs without requiring an external GPU.

---

## 📄 License
This project is open-source and licensed under the MIT License.

