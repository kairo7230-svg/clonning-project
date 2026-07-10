"""
specificgesture.py  -  Custom Gesture Training + Clone Arc Trigger
===================================================================
QUICK START:
  1. Press 1  -> toggle recording gesture label "clone"
     (hold your gesture for ~3 sec, press 1 again to stop)
  2. Press 2  -> toggle recording label "idle"
     (show neutral hand for ~2 sec, press 2 again to stop)
  3. Press F  -> train
  4. Press V  -> save model
  5. Perform your "clone" gesture -> arc formation appears!

Keys: 1/2 = Record toggle  F = Train  V = Save  L = Load  C = Clear  Q = Quit

NOTE: matplotlib is stubbed because App Control blocks ft2font.dll.
"""

import cv2
import numpy as np
import time
import threading
import pickle
import os
import sys
import types

# ── matplotlib stub (must be before any mediapipe import) ───────────────────
for _m in ["matplotlib", "matplotlib.pyplot", "matplotlib.colors",
           "matplotlib.cm", "matplotlib.patches", "matplotlib.figure"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
# ───────────────────────────────────────────────────────────────────────────

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
TRIGGER_LABEL       = "clone"
MODEL_PATH          = "gesture_model.pkl"
CAM_W, CAM_H        = 640, 480
PREDICT_CONFIDENCE  = 0.45     # lowered – KNN vote fraction to trigger
KNN_K               = 5
GESTURE_HOLD_FRAMES = 3        # consecutive frames before activation
GESTURE_RESET_FRAMES= 30       # frames without gesture before counter resets

# Gesture labels mapped to number keys (add more if you like)
LABEL_KEYS = {ord('1'): "clone", ord('2'): "idle"}

_PERSON_CX = 0.50
_PERSON_FY = 0.93


# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════
latest_landmarks  = None
latest_seg_mask   = None
clone_activated   = False
clone_configs     = []

_data_lock        = threading.Lock()   # protects _train_X / _train_y
_state_lock       = threading.Lock()   # protects hysteresis counters

_recording        = False
_current_label    = None
_train_X          = []
_train_y          = []
_sample_counts    = {}     # {label: count} shown in HUD

_classifier       = None
_label_names      = []

_gesture_hold     = 0
_gesture_reset    = 0
_last_label       = None
_last_conf        = 0.0

# Smoke / clone spawn state
_smoke_active     = False   # True while smoke effect is playing
_smoke_countdown  = 0       # frames remaining before clones appear
SMOKE_DELAY_FRAMES = 45     # ~1.5 s at 30 fps

# ═══════════════════════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ═══════════════════════════════════════════════════════════════════════════
WRIST      = 0
MIDDLE_MCP = 9


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
def extract_features(hand_landmarks) -> np.ndarray:
    """
    63-d normalised landmark vector:
      - Wrist-relative (translation invariant)
      - Scale-normalised by wrist-to-middle-MCP distance
    """
    pts  = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    pts -= pts[WRIST]
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFIER  (KNN)
# ═══════════════════════════════════════════════════════════════════════════
def train_classifier():
    global _classifier, _label_names
    with _data_lock:
        if len(_train_X) < KNN_K:
            print(f"[TRAIN] Need at least {KNN_K} samples. Have {len(_train_X)}.")
            return False
        X = np.array(_train_X, dtype=np.float32)
        y = np.array(_train_y)

    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        print("[TRAIN] scikit-learn not installed! Run: pip install scikit-learn")
        return False

    unique_labels = sorted(set(y))
    if len(unique_labels) < 2:
        print(f"[TRAIN] Need at least 2 labels for classification. "
              f"Only found: {unique_labels}. "
              f"Record an 'idle' gesture too (press 2).")
        return False

    _label_names = unique_labels
    n_neigh = min(KNN_K, len(X))
    knn = KNeighborsClassifier(n_neighbors=n_neigh, metric="euclidean",
                               weights="distance")
    knn.fit(X, y)
    _classifier = knn
    counts = {lbl: int((y == lbl).sum()) for lbl in _label_names}
    print(f"[TRAIN] Done! Labels: {counts}")
    return True


def save_model(path):
    if _classifier is None:
        print("[SAVE] No model – press F first.")
        return
    with _data_lock:
        data = {"clf": _classifier, "labels": _label_names,
                "X": list(_train_X), "y": list(_train_y)}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[SAVE] Saved -> {path}")


def load_model(path):
    global _classifier, _label_names, _train_X, _train_y, _sample_counts
    if not os.path.exists(path):
        print(f"[LOAD] Not found: {path}")
        return False
    with open(path, "rb") as f:
        d = pickle.load(f)
    with _data_lock:
        _classifier  = d["clf"]
        _label_names = d.get("labels", [])
        _train_X     = list(d.get("X", []))
        _train_y     = list(d.get("y", []))
        y_arr = np.array(_train_y)
        _sample_counts = {lbl: int((y_arr == lbl).sum()) for lbl in _label_names}
    print(f"[LOAD] Loaded <- {path}  labels={_label_names}, samples={len(_train_X)}")
    return True


def predict_gesture(hand_landmarks):
    """Returns (label, confidence) or (None, 0.0). Thread-safe read."""
    clf = _classifier
    if clf is None:
        return None, 0.0
    try:
        feat  = extract_features(hand_landmarks).reshape(1, -1)
        proba = clf.predict_proba(feat)[0]
        top   = int(np.argmax(proba))
        return clf.classes_[top], float(proba[top])
    except Exception:
        return None, 0.0


def get_all_proba(hand_landmarks):
    """Returns dict {label: confidence} for all known labels."""
    clf = _classifier
    if clf is None:
        return {}
    try:
        feat  = extract_features(hand_landmarks).reshape(1, -1)
        proba = clf.predict_proba(feat)[0]
        return {clf.classes_[i]: float(proba[i]) for i in range(len(clf.classes_))}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  SMOKE PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
class SmokeParticle:
    """Single smoke puff with position, radius, opacity and velocity."""
    def __init__(self, x, y):
        angle  = np.random.uniform(0, 2 * np.pi)
        speed  = np.random.uniform(0.8, 3.5)
        self.x  = float(x)
        self.y  = float(y)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed - np.random.uniform(0.5, 2.0)  # drift upward
        self.r  = np.random.randint(8, 28)       # starting radius
        self.dr = np.random.uniform(1.2, 3.0)    # radius growth per frame
        self.alpha = np.random.uniform(180, 255)  # starting opacity
        self.da    = np.random.uniform(6, 14)     # fade per frame
        # Smoke colour: dark grey → white-ish with slight blue tint
        v = np.random.randint(60, 200)
        self.color = (v + 20, v + 10, v)          # BGR

    def update(self):
        self.x    += self.vx
        self.y    += self.vy
        self.vy   -= 0.08    # slight upward acceleration
        self.r    += self.dr
        self.alpha -= self.da

    @property
    def alive(self):
        return self.alpha > 0 and self.r < 120


class SmokeSystem:
    """Manages all active smoke particles and renders them via alpha blending."""
    def __init__(self):
        self._particles = []

    def burst(self, positions, count_per_pos=18):
        """Spawn `count_per_pos` particles at each (x, y) in positions."""
        for (x, y) in positions:
            for _ in range(count_per_pos):
                self._particles.append(SmokeParticle(x, y))

    def update_and_draw(self, display):
        """Advance physics, draw, and prune dead particles."""
        live = []
        # Use a temporary overlay for additive blending
        overlay = display.copy()
        for p in self._particles:
            p.update()
            if p.alive:
                live.append(p)
                a  = int(np.clip(p.alpha, 0, 255))
                cx = int(np.clip(p.x, 0, display.shape[1] - 1))
                cy = int(np.clip(p.y, 0, display.shape[0] - 1))
                r  = max(1, int(p.r))
                cv2.circle(overlay, (cx, cy), r, p.color, -1)
        self._particles = live
        # Alpha blend: smoke is semi-transparent
        cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)

    @property
    def active(self):
        return len(self._particles) > 0


_smoke_system = SmokeSystem()


def _clone_spawn_positions():
    """Return (x, y) screen positions for the smoke burst origin points."""
    W, H = CAM_W, CAM_H
    # Arc of positions roughly matching where clones will appear
    return [
        (int(0.22 * W), int(0.52 * H)),   # back-left
        (int(0.78 * W), int(0.52 * H)),   # back-right
        (int(0.10 * W), int(0.95 * H)),   # far-left
        (int(0.24 * W), int(0.90 * H)),
        (int(0.37 * W), int(0.85 * H)),
        (int(0.63 * W), int(0.85 * H)),
        (int(0.76 * W), int(0.90 * H)),
        (int(0.90 * W), int(0.95 * H)),   # far-right
        (int(0.50 * W), int(0.70 * H)),   # center
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  ARC CLONE FORMATION
# ═══════════════════════════════════════════════════════════════════════════
def make_arc_configs():
    W, H   = CAM_W, CAM_H
    FEET_Y = int(H * 0.98)

    def place(scale, cx_frac, fy_frac=None):
        fy = FEET_Y if fy_frac is None else int(H * fy_frac)
        sw = int(W * scale); sh = int(H * scale)
        px = int(cx_frac * W) - int(_PERSON_CX * sw)
        py = fy - int(_PERSON_FY * sh)
        return px, py

    cfg = []
    cfg.append((*place(0.27, 0.22, 0.52), 0.27))
    cfg.append((*place(0.27, 0.78, 0.52), 0.27))
    cfg.append((*place(0.38, 0.10),        0.38))
    cfg.append((*place(0.56, 0.24),        0.56))
    cfg.append((*place(0.74, 0.37),        0.74))
    cfg.append((*place(0.74, 0.63),        0.74))
    cfg.append((*place(0.56, 0.76),        0.56))
    cfg.append((*place(0.38, 0.90),        0.38))
    cfg.append((0, 0, 1.0))
    return cfg


def stamp_clone(display, person_bgra, px, py, scale):
    dh, dw = display.shape[:2]
    sh, sw = person_bgra.shape[:2]
    cw = int(sw * scale); ch = int(sh * scale)
    if cw < 4 or ch < 4:
        return
    small    = cv2.resize(person_bgra, (cw, ch), interpolation=cv2.INTER_LINEAR)
    alpha_ch = small[:, :, 3]
    sy0 = max(0, -py); sx0 = max(0, -px)
    dy0 = max(0,  py); dx0 = max(0,  px)
    dy1 = min(dh, py + ch); dx1 = min(dw, px + cw)
    sy1 = sy0 + (dy1 - dy0); sx1 = sx0 + (dx1 - dx0)
    if dy1 <= dy0 or dx1 <= dx0:
        return
    roi = display[dy0:dy1, dx0:dx1].astype(np.float32)
    cr  = small[sy0:sy1, sx0:sx1, :3].astype(np.float32)
    a   = (alpha_ch[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[:, :, np.newaxis]
    display[dy0:dy1, dx0:dx1] = (cr * a + roi * (1.0 - a)).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════
def receive_landmarks(result: vision.HandLandmarkerResult,
                      output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks, clone_activated, clone_configs
    global _gesture_hold, _gesture_reset
    global _last_label, _last_conf
    global _smoke_active, _smoke_countdown

    latest_landmarks = result

    if not result.hand_landmarks:
        with _state_lock:
            _gesture_reset += 1
            if _gesture_reset >= GESTURE_RESET_FRAMES:
                _gesture_hold  = 0
                _gesture_reset = 0
        _last_label, _last_conf = None, 0.0
        return

    hand = result.hand_landmarks[0]

    # ── Collect training samples ─────────────────────────────────────────
    if _recording and _current_label is not None:
        feat = extract_features(hand)
        with _data_lock:
            _train_X.append(feat)
            _train_y.append(_current_label)

    # ── Predict ──────────────────────────────────────────────────────────
    label, conf = predict_gesture(hand)
    _last_label = label
    _last_conf  = conf
    gesture_ok  = (label == TRIGGER_LABEL and conf >= PREDICT_CONFIDENCE)

    with _state_lock:
        if gesture_ok:
            _gesture_reset  = 0
            _gesture_hold  += 1
            # Trigger smoke + delayed clone spawn
            if _gesture_hold >= GESTURE_HOLD_FRAMES and not clone_activated and not _smoke_active:
                _smoke_active    = True
                _smoke_countdown = SMOKE_DELAY_FRAMES
                print(f"[GESTURE] '{TRIGGER_LABEL}' confirmed (conf={conf:.0%}) "
                      f"-> smoke burst! clones in ~{SMOKE_DELAY_FRAMES} frames.")
        else:
            _gesture_reset += 1
            if _gesture_reset >= GESTURE_RESET_FRAMES:
                _gesture_hold  = 0
                _gesture_reset = 0


def receive_segmentation(result, output_image: mp.Image, timestamp_ms: int):
    global latest_seg_mask
    if hasattr(result, "confidence_masks") and len(result.confidence_masks) > 15:
        mask = result.confidence_masks[15].numpy_view()
        latest_seg_mask = cv2.flip(mask, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  RICH HUD
# ═══════════════════════════════════════════════════════════════════════════
def draw_hud(display, hand_landmarks_for_proba=None):
    dh, dw = display.shape[:2]

    # ── Left panel background ─────────────────────────────────────────────
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (320, dh), (8, 8, 12), -1)
    cv2.addWeighted(overlay, 0.60, display, 0.40, 0, display)

    y = 0  # running y cursor

    def section(title, col=(80, 180, 255)):
        nonlocal y
        y += 18
        cv2.putText(display, title, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
        y += 4
        cv2.line(display, (8, y), (312, y), (40, 40, 55), 1)
        y += 2

    def row(msg, col=(180, 180, 180), bold=False):
        nonlocal y
        y += 16
        cv2.putText(display, msg, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 2 if bold else 1, cv2.LINE_AA)

    def bar(label, value, bar_col, max_w=295):
        """Draw a labelled progress bar."""
        nonlocal y
        y += 18
        bh = 12
        filled = int(max_w * min(value, 1.0))
        cv2.rectangle(display, (10, y), (10 + max_w, y + bh), (40, 40, 50), -1)
        if filled > 0:
            cv2.rectangle(display, (10, y), (10 + filled, y + bh), bar_col, -1)
        cv2.rectangle(display, (10, y), (10 + max_w, y + bh), (70, 70, 80), 1)
        cv2.putText(display, f"{label}: {value*100:.0f}%",
                    (12, y + bh - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Controls ──────────────────────────────────────────────────────────
    section("-- CONTROLS --", (80, 180, 255))
    row("[1] Record 'clone'  [2] Record 'idle'")
    row("[F] Train  [V] Save  [L] Load  [C] Clones  [X] Clear Data  [Q] Quit")

    # ── Recording status ──────────────────────────────────────────────────
    section("-- RECORDING --", (80, 180, 255))
    blink_on = int(time.time() * 2) % 2 == 0
    if _recording:
        dot_col = (0, 50, 200) if blink_on else (0, 80, 255)
        cv2.circle(display, (15, y + 9), 6, dot_col, -1)
        with _data_lock:
            sc = len([v for v in _train_y if v == _current_label])
        row(f"  REC: '{_current_label}'  ({sc} new samples)", (0, 160, 255), bold=True)
    else:
        cv2.circle(display, (15, y + 9), 6, (50, 50, 60), -1)
        row("  Idle - press 1 or 2 to record", (120, 120, 130))

    # Sample counts per label
    with _data_lock:
        y_arr = list(_train_y)
    labels_seen = sorted(set(y_arr)) if y_arr else []
    for lbl in labels_seen:
        cnt = y_arr.count(lbl)
        col = (0, 220, 120) if lbl == TRIGGER_LABEL else (160, 160, 160)
        row(f"  '{lbl}': {cnt} samples", col)
    if not labels_seen:
        row("  No samples yet", (80, 80, 90))

    # ── Model status ──────────────────────────────────────────────────────
    section("-- MODEL --", (80, 180, 255))
    if _classifier is not None:
        row(f"  READY  labels={_label_names}", (0, 220, 100), bold=True)
    else:
        row("  No model - press F to train", (100, 100, 110))

    # ── Live prediction ───────────────────────────────────────────────────
    section("-- LIVE PREDICTION --", (80, 180, 255))

    if hand_landmarks_for_proba is not None and _classifier is not None:
        all_p = get_all_proba(hand_landmarks_for_proba)
        for lbl in sorted(all_p, key=lambda l: -all_p[l]):
            conf = all_p[lbl]
            is_trigger = (lbl == TRIGGER_LABEL and conf >= PREDICT_CONFIDENCE)
            b_col = (0, 200, 80) if is_trigger else (60, 120, 200)
            bar(lbl, conf, b_col)
        y += 4
        # Show verdict
        top_lbl = max(all_p, key=all_p.get) if all_p else None
        top_conf = all_p.get(top_lbl, 0) if top_lbl else 0
        is_trig = (top_lbl == TRIGGER_LABEL and top_conf >= PREDICT_CONFIDENCE)
        if is_trig:
            pulse = (0, 255, 100) if blink_on else (0, 200, 70)
            row(f">> TRIGGER: '{TRIGGER_LABEL}' ({top_conf:.0%})", pulse, bold=True)
        else:
            row(f"  Top: '{top_lbl}' ({top_conf:.0%}) - below threshold",
                (140, 140, 140))
    elif hand_landmarks_for_proba is not None and _classifier is None:
        row("  Hand detected - train model first", (120, 120, 80))
    else:
        row("  No hand detected", (80, 80, 90))

    # ── Hysteresis progress bar ───────────────────────────────────────────
    section("-- HOLD PROGRESS --", (80, 180, 255))
    with _state_lock:
        hold = _gesture_hold
    progress = min(hold / max(GESTURE_HOLD_FRAMES, 1), 1.0)
    p_col = (0, 220, 80) if progress >= 1.0 else (60, 160, 255)
    bar(f"Gesture hold ({hold}/{GESTURE_HOLD_FRAMES})", progress, p_col)

    # ── Clone status ──────────────────────────────────────────────────────
    section("-- CLONE STATUS --", (80, 180, 255))
    if clone_activated:
        n = len(clone_configs)
        pulse = (0, 255, 140) if blink_on else (0, 200, 100)
        row(f"  ACTIVE: {n} clones  [C] to clear", pulse, bold=True)
        seg_ok = latest_seg_mask is not None
        row(f"  Segmentation: {'ready' if seg_ok else 'warming up...'}")
    else:
        row("  No clones - perform trigger gesture", (80, 80, 90))

    # ── Trigger label reminder ────────────────────────────────────────────
    y = dh - 20
    cv2.putText(display, f"Trigger='{TRIGGER_LABEL}'  threshold={PREDICT_CONFIDENCE:.0%}  hold={GESTURE_HOLD_FRAMES}f",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 130), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    global _recording, _current_label, clone_activated, clone_configs
    global _smoke_active, _smoke_countdown

    # Auto-load saved model
    if os.path.exists(MODEL_PATH):
        load_model(MODEL_PATH)

    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.4,
        result_callback=receive_landmarks
    )
    seg_opts = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path="deeplab_v3.tflite"),
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_category_mask=False,
        output_confidence_masks=True,
        result_callback=receive_segmentation
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    print("\n===========================================")
    print("   SPECIFIC GESTURE CLONE SYSTEM")
    print(f"   Trigger label: '{TRIGGER_LABEL}'")
    print("   Keys: 1=clone  2=idle  F=Train  V=Save")
    print("   L=Load  C=ClearClones  X=ClearData  Q=Quit")
    print("===========================================\n")

    with vision.HandLandmarker.create_from_options(hand_opts) as landmarker, \
         vision.ImageSegmenter.create_from_options(seg_opts) as segmenter:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts  = int(time.time() * 1000)

            landmarker.detect_async(img, ts)
            segmenter.segment_async(img, ts)

            display = cv2.flip(frame, 1)

            # ── Smoke effect (pre-clone) ──────────────────────────────────
            if _smoke_active:
                if _smoke_countdown == SMOKE_DELAY_FRAMES:
                    # First frame of smoke: fire a big burst at all positions
                    _smoke_system.burst(_clone_spawn_positions(), count_per_pos=22)
                elif _smoke_countdown == SMOKE_DELAY_FRAMES // 2:
                    # Mid-way: second smaller burst for density
                    _smoke_system.burst(_clone_spawn_positions(), count_per_pos=10)

                _smoke_system.update_and_draw(display)
                _smoke_countdown -= 1

                # Flash text while smoke is building
                blink = int(time.time() * 4) % 2 == 0
                if blink:
                    cv2.putText(display, "CLONING...",
                                (int(CAM_W * 0.35), int(CAM_H * 0.50)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                (180, 230, 255), 3, cv2.LINE_AA)

                if _smoke_countdown <= 0:
                    # Smoke done -> spawn clones
                    clone_configs   = make_arc_configs()
                    clone_activated = True
                    _smoke_active   = False
                    print(f"[CLONE] Arc formation spawned: {len(clone_configs)} clones.")

            # ── Clone compositing ─────────────────────────────────────────
            if clone_activated and clone_configs:
                dh, dw = display.shape[:2]
                if latest_seg_mask is not None:
                    # Full segmented clone
                    seg = latest_seg_mask
                    if seg.shape[:2] != (dh, dw):
                        seg = cv2.resize(seg, (dw, dh))
                    bm   = (seg > 0.35).astype(np.uint8) * 255
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    bm   = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, kern)
                    bm   = cv2.morphologyEx(bm, cv2.MORPH_OPEN,  kern)
                    sm   = cv2.GaussianBlur(bm, (13, 13), 0)
                    pbgra          = cv2.cvtColor(display, cv2.COLOR_BGR2BGRA)
                    pbgra[:, :, 3] = sm
                else:
                    # Segmentation not ready yet: use full frame as clone
                    pbgra = cv2.cvtColor(display, cv2.COLOR_BGR2BGRA)

                for (px, py, scale) in clone_configs:
                    stamp_clone(display, pbgra, px, py, scale)

                n = len(clone_configs)
                cv2.putText(display, f"CLONES ACTIVE: {n}",
                            (330, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 140), 2, cv2.LINE_AA)

            # ── Hand landmark dots ─────────────────────────────────────────
            hand_for_hud = None
            if latest_landmarks and latest_landmarks.hand_landmarks:
                fh, fw = display.shape[:2]
                for hand in latest_landmarks.hand_landmarks:
                    for lm in hand:
                        cx = fw - int(lm.x * fw)
                        cy = int(lm.y * fh)
                        cv2.circle(display, (cx, cy), 4, (0, 220, 255), -1)
                hand_for_hud = latest_landmarks.hand_landmarks[0]

            # ── HUD ───────────────────────────────────────────────────────
            draw_hud(display, hand_landmarks_for_proba=hand_for_hud)

            cv2.imshow("Gesture Trainer | Clone System", display)

            # ── Key handling (no input() blocking!) ───────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                if _recording and _current_label == label:
                    # Stop recording this label
                    _recording     = False
                    _current_label = None
                    with _data_lock:
                        cnt        = _train_y.count(label)
                        total      = len(_train_X)
                        counts_str = {lbl: _train_y.count(lbl) for lbl in set(_train_y)}
                    print(f"[REC] Stopped '{label}'. Samples per label: {counts_str}  (total={total})")
                    # Auto-train if we have enough samples (lock-safe check)
                    if total >= KNN_K:
                        print("[TRAIN] Auto-training...")
                        train_classifier()
                    else:
                        print(f"[TRAIN] Not enough samples yet ({total}/{KNN_K}). Record more data.")
                else:
                    # Start recording this label
                    _recording     = True
                    _current_label = label
                    print(f"[REC] Recording '{label}' ... press {chr(key)} again to stop.")

            elif key == ord('f'):
                print("[TRAIN] Training classifier...")
                ok = train_classifier()
                if ok:
                    print("[TRAIN] Press V to save.")

            elif key == ord('v'):
                save_model(MODEL_PATH)

            elif key == ord('l'):
                load_model(MODEL_PATH)

            elif key == ord('c'):
                clone_activated  = False
                clone_configs    = []
                _smoke_active    = False
                _smoke_countdown = 0
                _smoke_system._particles.clear()
                with _state_lock:
                    _gesture_hold  = 0
                    _gesture_reset = 0
                print("[CLEAR] Clones and smoke cleared.")

            elif key == ord('x'):
                # ── Wipe ALL training data and model (fresh start) ─────────
                _recording = False
                _current_label = None
                with _data_lock:
                    _train_X.clear()
                    _train_y.clear()
                    _sample_counts.clear()
                global _classifier, _label_names
                _classifier  = None
                _label_names = []
                # Also delete saved model file if present
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                    print("[X] Deleted saved model file.")
                print("[X] All training data and model cleared. Ready to record fresh samples!")

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()