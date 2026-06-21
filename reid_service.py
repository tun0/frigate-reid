#!/usr/bin/env python3
"""
Frigate Re-ID filter.

Subscribes to frigate/events, deduplicates zone-entry detections across
all cameras using appearance embeddings, and republishes survivors to
frigate/events/filtered. The HA automation listens to the filtered topic.

Deduplication logic:
  - On each zone-entry event, fetch the bounding-box crop from the Frigate
    snapshot API and compute a 576-dim MobileNetV3 feature vector.
  - Compare cosine similarity against all embeddings stored in the last
    EMBEDDING_TTL seconds (default 5 min).
  - If similarity >= SIMILARITY_THRESHOLD for any stored embedding → same
    person/object already notified → drop.
  - Otherwise → new detection → store embedding + forward event.

Upgrade path: replace the MobileNetV3 extractor with an OSNet model from
torchreid for better cross-camera accuracy once the simple model is validated.
"""

import json
import logging
import os
import time
from collections import defaultdict
from io import BytesIO
from threading import Lock

import numpy as np
import paho.mqtt.client as mqtt
import requests
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ["MQTT_PORT"])
FRIGATE_API = os.environ["FRIGATE_API"]
SIMILARITY_THRESHOLD = float(os.environ["SIMILARITY_THRESHOLD"])
EMBEDDING_TTL = int(os.environ["EMBEDDING_TTL"])
TARGET_ZONES = set(os.environ["TARGET_ZONES"].split(","))
TARGET_LABELS = set(os.environ["TARGET_LABELS"].split(","))

INPUT_TOPIC = "frigate/events"
OUTPUT_TOPIC = "frigate/events/filtered"

TRANSFORM = T.Compose([
    T.Resize((256, 128)),  # standard re-ID input size: height × width
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model() -> torch.nn.Module:
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier = torch.nn.Identity()  # 576-dim feature vector instead of class logits
    m.eval()
    return m


def compute_embedding(model: torch.nn.Module, img: Image.Image) -> np.ndarray:
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor).squeeze().numpy()
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


class EmbeddingStore:
    """Per-label TTL store. Embeddings expire after `ttl` seconds."""

    def __init__(self, ttl: int) -> None:
        self._ttl = ttl
        # label → list of (timestamp, embedding)
        self._store: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
        self._lock = Lock()

    def is_duplicate(self, label: str, emb: np.ndarray, threshold: float) -> bool:
        now = time.monotonic()
        with self._lock:
            # Evict expired entries
            self._store[label] = [
                (ts, e) for ts, e in self._store[label] if now - ts < self._ttl
            ]
            for _, stored in self._store[label]:
                if float(np.dot(emb, stored)) >= threshold:
                    return True
            self._store[label].append((now, emb))
        return False


def fetch_crop(event_id: str) -> Image.Image | None:
    """Fetch the bounding-box-cropped snapshot for a Frigate event."""
    url = f"{FRIGATE_API}/api/events/{event_id}/snapshot.jpg"
    for attempt in range(2):
        try:
            r = requests.get(url, params={"crop": 1, "quality": 70}, timeout=3)
            r.raise_for_status()
            return Image.open(BytesIO(r.content))
        except Exception as exc:
            if attempt == 0:
                time.sleep(0.5)  # snapshot may not be written yet on the first try
            else:
                log.warning("Snapshot fetch failed for %s: %s", event_id, exc)
    return None


def is_zone_entry(payload: dict) -> bool:
    if payload.get("type") != "update":
        return False
    after = payload.get("after", {})
    if after.get("label") not in TARGET_LABELS:
        return False
    before_zones = set((payload.get("before") or {}).get("entered_zones") or [])
    after_zones = set(after.get("entered_zones") or [])
    return bool((after_zones - before_zones) & TARGET_ZONES)


def make_handler(model: torch.nn.Module, store: EmbeddingStore, client: mqtt.Client):
    def on_message(_client, _userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        if not is_zone_entry(payload):
            return

        after = payload["after"]
        event_id: str = after.get("id", "?")
        label: str = after.get("label", "?")
        camera: str = after.get("camera", "?")

        img = fetch_crop(event_id)
        if img is None:
            # Can't verify — pass through rather than risk missing a real event
            log.info("PASS-THROUGH (no snapshot) %s/%s/%s", camera, label, event_id)
            client.publish(OUTPUT_TOPIC, msg.payload, qos=0, retain=False)
            return

        emb = compute_embedding(model, img)
        if store.is_duplicate(label, emb, SIMILARITY_THRESHOLD):
            log.info("DUPLICATE              %s/%s/%s", camera, label, event_id)
        else:
            log.info("NEW → forwarded        %s/%s/%s", camera, label, event_id)
            client.publish(OUTPUT_TOPIC, msg.payload, qos=0, retain=False)

    return on_message


def main() -> None:
    log.info("Loading re-ID model…")
    model = load_model()
    log.info("Model ready (MobileNetV3-Small, 576-dim features).")

    store = EmbeddingStore(ttl=EMBEDDING_TTL)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(c, _userdata, _flags, reason_code, _props):
        if reason_code == 0:
            c.subscribe(INPUT_TOPIC, qos=0)
            log.info("Connected to MQTT, subscribed to %s", INPUT_TOPIC)
        else:
            log.error("MQTT connect failed: %s", reason_code)

    client.on_connect = on_connect
    client.on_message = make_handler(model, store, client)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    log.info("Subscribing to %s → publishing new detections to %s", INPUT_TOPIC, OUTPUT_TOPIC)
    client.loop_forever()


if __name__ == "__main__":
    main()
