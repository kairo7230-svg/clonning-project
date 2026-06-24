import cv2
import mediapipe as mp
import time
import threading
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── GLOBAL STATE ────────────────────────────────────────────────────────────
latest_landmarks  = None   # from HandLandmarker (LIVE_STREAM)
latest_seg_mask   = None   # from ImageSegmenter  (LIVE_STREAM)
clone_activated   = False
clone_configs     = []     # list of (px, py, scale) — set on each palm

# Palm state-machine with hysteresis
_palm_was_open    = False
_palm_gone_count  = 0
PALM_GONE_THRESHOLD = 20   # ~0.65 s at 30 fps
_capture_lock     = threading.Lock()

CAM_W, CAM_H = 640, 480   # must match cap.set() below

# ─── PALM-OPEN DETECTION ─────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4;  THUMB_MCP  = 2
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

def is_open_palm(landmarks) -> bool:
    lm = landmarks
    def tip_above_pip(tip_i, pip_i):
        return lm[tip_i].y < lm[pip_i].y
    thumb_open   = lm[THUMB_TIP].x < lm[THUMB_MCP].x
    fingers_open = (
        tip_above_pip(INDEX_TIP,  INDEX_PIP)  and
        tip_above_pip(MIDDLE_TIP, MIDDLE_PIP) and
        tip_above_pip(RING_TIP,   RING_PIP)   and
        tip_above_pip(PINKY_TIP,  PINKY_PIP)
    )
    return thumb_open and fingers_open

# Person appears at ~50% width and ~93% height in the full frame
_PERSON_CX = 0.50   # horizontal centre (normalised)
_PERSON_FY = 0.93   # feet y-position   (normalised)

def make_arc_configs():
    """
    Fixed V-arc formation (render back → front so center is always on top):

        [BK_L]              [BK_R]     ← small, chest-level (not above head)
      [0.38]                  [0.38]   ← left/right flanks
        [0.56]            [0.56]       ← approaching center
          [0.74]        [0.74]         ← near center
                [CENTER=1.0]           ← full-size, on top
    """
    W, H = CAM_W, CAM_H
    FEET_Y = int(H * 0.98)           # ground line in display pixels

    def place(scale, target_cx_frac, target_feet_frac=None):
        """Return (px, py) so the person inside the scaled clone lands at
        (target_cx_frac*W, target_feet_frac*H) in the display."""
        if target_feet_frac is None:
            feet_y = FEET_Y
        else:
            feet_y = int(H * target_feet_frac)
        s_w = int(W * scale)
        s_h = int(H * scale)
        px  = int(target_cx_frac * W) - int(_PERSON_CX * s_w)
        py  = feet_y - int(_PERSON_FY * s_h)
        return px, py

    configs = []

    # ─ BACK CLONES: small, placed at chest level (head stays below main head) ─
    # Main head ≈ y=0.17*H ≈ 82px. Back clone scale=0.27 → height=130px.
    # Placing at feet_y=0.52*H puts their top at 0.52*H - 0.93*130 ≈ 129px > 82px ✔
    configs.append((*place(0.27, 0.22, 0.52), 0.27))   # back-left
    configs.append((*place(0.27, 0.78, 0.52), 0.27))   # back-right

    # ─ LEFT ARC: far-left (small) → near-center (large) ─────────────────────
    configs.append((*place(0.38, 0.10), 0.38))
    configs.append((*place(0.56, 0.24), 0.56))
    configs.append((*place(0.74, 0.37), 0.74))

    # ─ RIGHT ARC: near-center (large) → far-right (small) ─────────────────
    configs.append((*place(0.74, 0.63), 0.74))
    configs.append((*place(0.56, 0.76), 0.56))
    configs.append((*place(0.38, 0.90), 0.38))

    # ─ CENTER: full-size clone on top of everything ────────────────────────
    configs.append((0, 0, 1.0))

    return configs

# ─── HAND LANDMARKER CALLBACK (LIVE_STREAM) ──────────────────────────────────
def receive_landmarks(result: vision.HandLandmarkerResult,
                      output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks, clone_activated, clone_configs
    global latest_seg_mask, _palm_was_open, _palm_gone_count

    latest_landmarks = result

    palm_open_now = (
        any(is_open_palm(h) for h in result.hand_landmarks)
        if result.hand_landmarks else False
    )

    should_spawn = False
    with _capture_lock:
        if palm_open_now:
            _palm_gone_count = 0
            if not _palm_was_open and latest_seg_mask is not None:
                _palm_was_open = True
                should_spawn   = True
        else:
            _palm_gone_count += 1
            if _palm_gone_count >= PALM_GONE_THRESHOLD:
                _palm_was_open   = False
                _palm_gone_count = 0

    if should_spawn:
        clone_configs   = make_arc_configs()
        n = len(clone_configs)
        print(f"[PALM] Arc formation activated ({n} clones).")
        clone_activated = True

# ─── SEGMENTATION CALLBACK (LIVE_STREAM) ─────────────────────────────────────
def receive_segmentation(result, output_image: mp.Image, timestamp_ms: int):
    global latest_seg_mask
    if hasattr(result, 'confidence_masks') and len(result.confidence_masks) > 15:
        mask = result.confidence_masks[15].numpy_view()
        latest_seg_mask = cv2.flip(mask, 1)

# ─── HELPER: stamp one clone onto display ────────────────────────────────────
def stamp_clone(display, person_bgra, px, py, scale):
    """Alpha-composite a scaled clone cutout at pixel position (px, py)."""
    dh, dw = display.shape[:2]
    src_h, src_w = person_bgra.shape[:2]

    clone_w = int(src_w * scale)
    clone_h = int(src_h * scale)
    if clone_w < 4 or clone_h < 4:
        return

    # Scale the BGRA cutout
    small = cv2.resize(person_bgra, (clone_w, clone_h),
                       interpolation=cv2.INTER_LINEAR)

    alpha_ch  = small[:, :, 3]    # soft-mask alpha channel

    # Clip region to display bounds
    src_y0 = max(0, -py);     src_x0 = max(0, -px)
    dst_y0 = max(0,  py);     dst_x0 = max(0,  px)
    dst_y1 = min(dh, py + clone_h);  dst_x1 = min(dw, px + clone_w)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
        return  # fully off-screen

    roi       = display[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    clone_roi = small[src_y0:src_y1, src_x0:src_x1, :3].astype(np.float32)
    a         = (alpha_ch[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
                 / 255.0)[:, :, np.newaxis]   # 1.0 = fully opaque

    display[dst_y0:dst_y1, dst_x0:dst_x1] = (
        clone_roi * a + roi * (1.0 - a)
    ).astype(np.uint8)

# ─── BUILD MEDIAPIPE TASKS ───────────────────────────────────────────────────
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=receive_landmarks
)

seg_options = vision.ImageSegmenterOptions(
    base_options=python.BaseOptions(model_asset_path="deeplab_v3.tflite"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_category_mask=False,
    output_confidence_masks=True,
    result_callback=receive_segmentation
)

# ─── CAMERA SETUP ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
print("[ON] Clone system ready! Show an OPEN PALM to deploy the arc formation.")
print("     Press 'q' to quit | 'c' to clear clones.")

with vision.HandLandmarker.create_from_options(hand_options) as landmarker, \
     vision.ImageSegmenter.create_from_options(seg_options) as segmenter:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)

        landmarker.detect_async(mp_image, timestamp)
        segmenter.segment_async(mp_image, timestamp)

        display = cv2.flip(frame, 1)

        # ── Multi-clone compositing ───────────────────────────────────────────
        if clone_activated and clone_configs and latest_seg_mask is not None:
            seg = latest_seg_mask
            h, w = display.shape[:2]

            if seg.shape[:2] != (h, w):
                seg = cv2.resize(seg, (w, h))

            # Build soft-edged person mask
            binary_mask = (seg > 0.35).astype(np.uint8) * 255
            kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
            soft_mask   = cv2.GaussianBlur(binary_mask, (13, 13), 0)

            # Build BGRA cutout of the segmented person from the live frame
            person_bgra            = cv2.cvtColor(display, cv2.COLOR_BGR2BGRA)
            person_bgra[:, :, 3]   = soft_mask

            # Stamp every clone
            for (px, py, scale) in clone_configs:
                stamp_clone(display, person_bgra, px, py, scale)

            n = len(clone_configs)
            cv2.putText(display, f"CLONES ACTIVE: {n}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 150), 2, cv2.LINE_AA)
            cv2.putText(display, "Open palm = new random spawn  |  'c' = clear",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(display, "Show OPEN PALM to spawn clones",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (100, 200, 255), 2, cv2.LINE_AA)

        # ── Hand skeleton (x mirrored to match flipped display) ──────────────
        if latest_landmarks and latest_landmarks.hand_landmarks:
            fh, fw = display.shape[:2]
            for hand in latest_landmarks.hand_landmarks:
                for lm in hand:
                    cx = fw - int(lm.x * fw)
                    cy = int(lm.y * fh)
                    cv2.circle(display, (cx, cy), 4, (0, 220, 255), -1)

        cv2.imshow("Palm Clone", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            clone_activated = False
            clone_configs   = []
            print("[CLEAR] All clones cleared.")

cap.release()
cv2.destroyAllWindows()
print("Closed cleanly.")