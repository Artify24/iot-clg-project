import importlib
import socket
import sys
import traceback

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("websockets", "websockets"),
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("ultralytics", "ultralytics"),
    ("qrcode", "qrcode[pil]"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("torch", "torch (via ultralytics)"),
]


def ok(msg):
    print(f"{GREEN}[OK]{RESET}   {msg}")


def fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")


def info(msg):
    print(f"{CYAN}[..]{RESET}   {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def check_imports():
    passed, failed = [], []
    for module, package in PACKAGES:
        try:
            importlib.import_module(module)
            ok(f"import {module:<14} ({package})")
            passed.append(package)
        except Exception as exc:
            fail(f"import {module:<14} ({package}) -> {exc}")
            failed.append(package)
    return failed


def check_local_ip():
    info("Resolving laptop local IP for phone pairing over Wi-Fi...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(1)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
        finally:
            sock.close()

        if ip.startswith("127."):
            warn(f"Resolved {ip} (loopback) - connect to Wi-Fi first")
            return False

        hostname = socket.gethostname()
        try:
            host_ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            host_ip = "n/a"

        ok(f"Local IP {ip} (hostname={hostname}, host-resolve={host_ip})")
        info(f"Pair phone with: http://{ip}:8000")
        return True
    except Exception as exc:
        fail(f"Could not resolve local IP -> {exc}")
        return False


def check_yolo():
    info("Loading yolov8n.pt (downloads on first run)...")
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        ok("yolov8n.pt loaded")

        frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        results = model.predict(frame, imgsz=320, verbose=False)
        count = len(results[0].boxes) if results and results[0].boxes is not None else 0

        ok(f"Dummy 1-frame inference passed (cv2 {cv2.__version__}, {count} detections)")
        return True
    except Exception as exc:
        fail(f"YOLO/PyTorch check failed -> {exc}")
        traceback.print_exc()
        return False


def main():
    print(f"\n{BOLD}StudyGuard Environment Verification{RESET}\n")

    print(f"{BOLD}1. Dependencies{RESET}")
    failed = check_imports()

    print(f"\n{BOLD}2. Network (phone pairing){RESET}")
    net_ok = check_local_ip()

    print(f"\n{BOLD}3. Model inference{RESET}")
    model_ok = check_yolo()

    print()
    if not failed and model_ok:
        print(f"{GREEN}{BOLD}[OK] All dependencies ready for StudyGuard AIoT{RESET}")
        if not net_ok:
            warn("Network check incomplete - verify Wi-Fi before pairing devices")
        sys.exit(0)

    print(f"{RED}{BOLD}[FAIL] Environment incomplete. Install missing packages:{RESET}")
    for pkg in failed:
        print(f"       pip install {pkg}")
    if not model_ok:
        print("       check GPU/CPU torch build and ultralytics install")
    sys.exit(1)


if __name__ == "__main__":
    main()
