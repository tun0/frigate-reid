#!/usr/bin/env python3
"""
Frigate Re-ID filter.

Subscribes to frigate/events, deduplicates zone-entry detections across
all cameras using appearance embeddings, and republishes survivors to
frigate/events/filtered. The HA automation listens to the filtered topic.

Deduplication logic:
  - On each zone-entry event, fetch the bounding-box crop from the Frigate
    snapshot API and compute a 512-dim OSNet x0.25 feature vector.
  - Compare cosine similarity against all embeddings stored in the last
    EMBEDDING_TTL seconds.
  - If best similarity >= SIMILARITY_THRESHOLD → same person/object already
    notified → drop.
  - Otherwise → new detection → store embedding + forward event.

If SNAPSHOT_DIR is set, every evaluated crop is saved there for later
training use alongside a JSON sidecar with full metadata:
  {SNAPSHOT_DIR}/{YYYYMMDD}/{label}/{camera}_{HHMMSS_ffffff}_{id[:8]}_{new|dup}.jpg
  {SNAPSHOT_DIR}/{YYYYMMDD}/{label}/{camera}_{HHMMSS_ffffff}_{id[:8]}_{new|dup}.json

Three training scenarios, and which snapshots support each:
  - Detection FP  (shirt detected as person by Frigate model): ALL snapshots —
    a dup-classified image is still a detection FP if the underlying detection
    is wrong; status only tells you what the re-ID service decided.
  - Re-ID FP dup  (different people suppressed as duplicate): DUP snapshots —
    compare with matched_event_id image to confirm they are different people.
  - Re-ID FN dup  (same person fires twice, both as "new"): NEW snapshots —
    look for pairs of new images in the same window that depict the same person.

Dup sidecars include matched_event_id linking back to the new image they
were compared against — enables side-by-side review for re-ID FP labeling.

Upgrade path: fine-tune OSNet on domain-specific data collected via Label
Studio (same-person / different-person pair labels from the snapshot archive).
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from threading import Lock

import torch
import torchvision.transforms as T
import numpy as np
import paho.mqtt.client as mqtt
import requests
from PIL import Image
from osnet import osnet_x0_25, init_pretrained_weights

torch.backends.nnpack.enabled = False

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
SNAPSHOT_DIR: str | None = os.environ.get("SNAPSHOT_DIR") or None
GALLERY_PATH: str | None = os.environ.get("GALLERY_PATH") or None

INPUT_TOPIC = "frigate/events"
OUTPUT_TOPIC = "frigate/events/filtered"

TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_gallery(path: str) -> dict[str, np.ndarray]:
    """Load named identity gallery from JSON. Returns {name: mean_embedding}."""
    try:
        with open(path) as f:
            raw = json.load(f)
        gallery = {name: np.mean(np.array(embs), axis=0) for name, embs in raw.items()}
        # Re-normalise averaged embeddings
        for name, emb in gallery.items():
            norm = np.linalg.norm(emb)
            if norm > 0:
                gallery[name] = emb / norm
        log.info("Gallery loaded: %s", ", ".join(f"{n}({len(raw[n])})" for n in gallery))
        return gallery
    except Exception as exc:
        log.warning("Could not load gallery from %s: %s", path, exc)
        return {}


def identify(emb: np.ndarray, gallery: dict[str, np.ndarray], threshold: float) -> str | None:
    """Return the best-matching gallery identity if above threshold, else None."""
    best_name, best_sim = None, threshold - 0.01
    for name, gallery_emb in gallery.items():
        sim = float(np.dot(emb, gallery_emb))
        if sim > best_sim:
            best_sim, best_name = sim, name
    return best_name


def load_model() -> torch.nn.Module:
    # OSNet x0.25: purpose-built for person re-ID, 512-dim embeddings.
    # In eval mode forward() returns the 512-dim feature vector directly.
    m = osnet_x0_25(num_classes=1000, pretrained=False)
    init_pretrained_weights(m, 'osnet_x0_25')
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
        # label → list of (monotonic_timestamp, embedding, event_id)
        self._store: dict[str, list[tuple[float, np.ndarray, str]]] = defaultdict(list)
        self._lock = Lock()

    def check(self, label: str, emb: np.ndarray, event_id: str, threshold: float) -> tuple[bool, float, str | None]:
        """
        Return (is_duplicate, best_similarity, matched_event_id).

        Finds the stored embedding with the highest cosine similarity.
        If best_similarity >= threshold: duplicate — do NOT store, return matched_event_id.
        Otherwise: new — store embedding, return (False, best_similarity, None).
        """
        now = time.monotonic()
        with self._lock:
            self._store[label] = [
                (ts, e, eid) for ts, e, eid in self._store[label] if now - ts < self._ttl
            ]
            best_sim, best_id = 0.0, None
            for _, stored, stored_eid in self._store[label]:
                sim = float(np.dot(emb, stored))
                if sim > best_sim:
                    best_sim, best_id = sim, stored_eid
            if best_sim >= threshold:
                return True, best_sim, best_id
            self._store[label].append((now, emb, event_id))
        return False, best_sim, None


def save_snapshot(
    img: Image.Image,
    camera: str,
    label: str,
    event_id: str,
    zones: list[str],
    status: str,
    similarity: float,
    matched_event_id: str | None,
) -> None:
    if not SNAPSHOT_DIR:
        return
    now = datetime.now()
    dest = os.path.join(SNAPSHOT_DIR, now.strftime("%Y%m%d"), label)
    os.makedirs(dest, exist_ok=True)

    stem = f"{camera}_{now.strftime('%H%M%S_%f')}_{event_id[:8]}_{status}"

    meta: dict = {
        "event_id": event_id,
        "camera": camera,
        "label": label,
        "timestamp": now.isoformat(),
        "zones": zones,
        "status": status,
        "similarity": round(similarity, 4),
        "image": f"{stem}.jpg",
    }
    if status == "dup" and matched_event_id:
        meta["matched_event_id"] = matched_event_id

    try:
        img.save(os.path.join(dest, f"{stem}.jpg"), "JPEG", quality=90)
        with open(os.path.join(dest, f"{stem}.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:
        log.warning("Failed to save snapshot %s: %s", stem, exc)


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


def is_zone_entry(payload: dict) -> tuple[bool, list[str]]:
    """Return (is_zone_entry, new_target_zones)."""
    if payload.get("type") != "update":
        return False, []
    after = payload.get("after", {})
    if after.get("label") not in TARGET_LABELS:
        return False, []
    before_zones = set((payload.get("before") or {}).get("entered_zones") or [])
    after_zones = set(after.get("entered_zones") or [])
    new_zones = list((after_zones - before_zones) & TARGET_ZONES)
    return bool(new_zones), new_zones


def make_handler(model: torch.nn.Module, store: EmbeddingStore, gallery: dict[str, np.ndarray], client: mqtt.Client):
    def on_message(_client, _userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        entered, new_zones = is_zone_entry(payload)
        if not entered:
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
        is_dup, similarity, matched_id = store.check(label, emb, event_id, SIMILARITY_THRESHOLD)

        if is_dup:
            log.info("DUPLICATE (sim=%.2f) %s/%s/%s → matched %s", similarity, camera, label, event_id, matched_id)
            save_snapshot(img, camera, label, event_id, new_zones, "dup", similarity, matched_id)
        else:
            identity = identify(emb, gallery, SIMILARITY_THRESHOLD) if gallery else None
            if identity:
                log.info("NEW → forwarded      %s/%s/%s [%s]", camera, label, event_id, identity)
                payload["after"]["identity"] = identity
            else:
                log.info("NEW → forwarded      %s/%s/%s", camera, label, event_id)
            save_snapshot(img, camera, label, event_id, new_zones, "new", similarity, None)
            client.publish(OUTPUT_TOPIC, json.dumps(payload).encode(), qos=0, retain=False)

    return on_message


def main() -> None:
    log.info("Loading re-ID model…")
    model = load_model()
    log.info("Model ready (OSNet x0.25, 512-dim re-ID features).")
    if SNAPSHOT_DIR:
        log.info("Snapshots → %s", SNAPSHOT_DIR)

    gallery = load_gallery(GALLERY_PATH) if GALLERY_PATH else {}

    store = EmbeddingStore(ttl=EMBEDDING_TTL)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(c, _userdata, _flags, reason_code, _props):
        if reason_code == 0:
            c.subscribe(INPUT_TOPIC, qos=0)
            log.info("Connected to MQTT, subscribed to %s", INPUT_TOPIC)
        else:
            log.error("MQTT connect failed: %s", reason_code)

    client.on_connect = on_connect
    client.on_message = make_handler(model, store, gallery, client)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    log.info("Subscribing to %s → publishing new detections to %s", INPUT_TOPIC, OUTPUT_TOPIC)
    client.loop_forever()


if __name__ == "__main__":
    main()
