#!/usr/bin/env python3
"""One-time ONLINE preparation. Resulting inference artifact works offline.

python scripts/prepare_models.py
python scripts/prepare_models.py --vocabulary path/to/classes.json
python scripts/prepare_models.py --verify-only  # blocks outbound sockets
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

PLATFORM = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("CENTREWATCH_MODELS_DIR", PLATFORM / "models"))
CLASSES = ["person", "chair", "workbench", "table", "lathe machine", "drill press", "sewing machine", "computer monitor", "laptop", "fire extinguisher", "whiteboard"]
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt"


def offline_verify(model_path: Path) -> dict:
    # A separate process provides evidence that neither inference nor loading
    # silently contacts a model CDN or downloads the CLIP text encoder.
    script = r'''
import os, socket, json, sys, time
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
attempts = []
def denied(*args, **kwargs):
    attempts.append("network_attempt")
    raise RuntimeError("Offline smoke test blocked an attempted network connection")
socket.getaddrinfo = denied
socket.create_connection = denied
socket.socket.connect = denied
socket.socket.connect_ex = denied
import numpy as np
from ultralytics import YOLOWorld
model = YOLOWorld(sys.argv[1])
model.predict(np.zeros((480,640,3), dtype=np.uint8), device="cpu", verbose=False)
time.sleep(.1)  # Allow any async usage callback to expose a blocked attempt.
assert not attempts, "An inference callback attempted network access"
print(json.dumps({"offlineLoadAndInference": True, "networkAttempts": len(attempts), "classes": list(model.names.values())}))
'''
    result = subprocess.run([sys.executable, "-c", script, str(model_path)], capture_output=True, text=True, timeout=180, check=True)
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
            if value.get("offlineLoadAndInference"):
                return value
        except ValueError:
            pass
    raise RuntimeError("Offline validation did not produce its confirmation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, help="JSON list of visual object names; must include person")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(MODEL_DIR / ".ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    output = MODEL_DIR / "centrewatch-world.pt"
    if args.verify_only:
        print(json.dumps(offline_verify(output), indent=2))
        return
    classes = json.loads(args.vocabulary.read_text()) if args.vocabulary else CLASSES
    if not isinstance(classes, list) or not classes or any(not isinstance(value, str) or not value.strip() for value in classes):
        raise ValueError("Vocabulary must be a non-empty JSON list of descriptive visual names")
    classes = list(dict.fromkeys(value.strip().lower() for value in classes))
    if "person" not in classes:
        classes.insert(0, "person")
    os.environ["YOLO_OFFLINE"] = "false"
    source = MODEL_DIR / "yolov8s-worldv2.pt"
    if not source.exists():
        print("Downloading official YOLO-World detector weights (one-time setup)...", flush=True)
        temporary = source.with_suffix(".download")
        request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "CentreWatch-Model-Setup/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        temporary.replace(source)
    from ultralytics import YOLOWorld
    # Keep the one-time text encoder cache inside the ignored model directory.
    # This module reads WEIGHTS_DIR at import; setting it avoids global caches.
    import ultralytics.nn.text_model as text_model
    text_model.WEIGHTS_DIR = MODEL_DIR
    print("Embedding approved visual vocabulary; CLIP encoder may download once...", flush=True)
    model = YOLOWorld(str(source))
    model.set_classes(classes)
    # txt_feats stores only object vocabulary embeddings, never person identities.
    model.model.clip_model = None
    model.save(str(output))
    verification = offline_verify(output)
    manifest = {"baseModel": "yolov8s-worldv2", "classes": classes, "modelFile": output.name,
                "createdAt": datetime.now(timezone.utc).isoformat(), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "sourceWeightsSha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source": WEIGHTS_URL,
                "ultralyticsVersion": importlib.metadata.version("ultralytics"), "torchVersion": importlib.metadata.version("torch"),
                "licenses": {"ultralytics": "AGPL-3.0 or commercial license", "CLIP": "MIT", "YOLO-World original repository": "GPL-3.0"},
                "licenseSources": {"ultralytics": "https://github.com/ultralytics/ultralytics/blob/main/LICENSE", "CLIP": "https://github.com/openai/CLIP/blob/main/LICENSE", "YOLO-World": "https://github.com/AILab-CVC/YOLO-World/blob/master/LICENSE"},
                "documentation": "https://docs.ultralytics.com/models/yolo-world/", "offlineVerification": verification,
                "limitation": "Open-vocabulary detections need measured accuracy on centre footage; object presence does not establish operability."}
    (MODEL_DIR / "vocabulary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"ready": True, "model": str(output), "classes": classes, "offlineVerified": True}, indent=2))


if __name__ == "__main__":
    main()
