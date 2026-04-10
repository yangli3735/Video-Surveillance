"""
Smart Surveillance Demo - PC Prototype
Virtual fence intrusion detection + simple fall suspicion detection
"""

import cv2
import numpy as np
import os
import time
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"            # YOLO model (auto-downloads if missing)
VIDEO_SOURCE = 0                      # 0 = webcam, or path like "test.mp4"
CONFIDENCE_THRESHOLD = 0.45           # YOLO confidence threshold

# Virtual fence region (x1, y1, x2, y2) — adjust to your camera view
FENCE_REGION = (200, 150, 500, 400)

# Intrusion detection
INTRUSION_FRAME_THRESHOLD = 5         # consecutive frames to confirm intrusion

# Fall suspicion detection
FALL_RATIO_THRESHOLD = 1.2            # aspect ratio (h/w) below this = possible fall
FALL_FRAME_THRESHOLD = 5              # consecutive frames to confirm fall suspicion
MIN_BOX_AREA = 4000                   # ignore very small person boxes
FALL_DOWNWARD_THRESHOLD = 30          # pixels of downward center movement to count as falling
FALL_LOW_POSITION_RATIO = 0.65        # person bottom below this fraction of frame height = "low"
FALL_LOW_POSITION_FRAMES = 3          # consecutive low-position frames needed
PERSON_HISTORY_LENGTH = 15            # frames of position history to keep per person

# Edge-aware visibility detection
EDGE_MARGIN = 10                      # pixels from frame border to count as "touching"
TOO_CLOSE_AREA_RATIO = 0.25           # box area / frame area above this = TOO_CLOSE

# State-dependent intrusion confirmation thresholds
INTRUSION_THRESH_FULL = 5             # FULL_BODY: normal
INTRUSION_THRESH_BOTTOM = 7           # BOTTOM_TRUNCATED: slightly stricter
INTRUSION_THRESH_EDGE = 10            # EDGE_TRUNCATED: cautious
INTRUSION_THRESH_CLOSE = 12           # TOO_CLOSE: conservative

# State-dependent fall confirmation thresholds (0 = suppressed)
FALL_THRESH_FULL = 5                  # FULL_BODY: normal
FALL_THRESH_BOTTOM = 8                # BOTTOM_TRUNCATED: stricter
FALL_THRESH_EDGE = 0                  # EDGE_TRUNCATED: suppressed
FALL_THRESH_CLOSE = 0                 # TOO_CLOSE: suppressed

# Centroid matching tolerance for simple tracking (pixels)
MATCH_DISTANCE = 80

# Output
OUTPUT_DIR = "outputs"
SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")
LOG_FILE = os.path.join(OUTPUT_DIR, "events.log")

# ============================================================
# COLORS
# ============================================================

COLOR_FENCE = (0, 200, 255)       # orange
COLOR_NORMAL = (0, 255, 0)        # green
COLOR_INTRUSION = (0, 0, 255)     # red
COLOR_FALL = (255, 0, 255)        # magenta
COLOR_CRITICAL = (0, 0, 200)      # dark red
COLOR_TEXT_BG = (30, 30, 30)

# ============================================================
# STATE DEFINITIONS
# ============================================================

STATE_NORMAL = "NORMAL"
STATE_INTRUSION = "INTRUSION"
STATE_FALL = "FALL_SUSPECTED"
STATE_CRITICAL = "CRITICAL"

STATE_COLORS = {
    STATE_NORMAL: COLOR_NORMAL,
    STATE_INTRUSION: COLOR_INTRUSION,
    STATE_FALL: COLOR_FALL,
    STATE_CRITICAL: COLOR_CRITICAL,
}

# ============================================================
# VISIBILITY STATE DEFINITIONS
# ============================================================

VIS_FULL_BODY = "FULL_BODY"
VIS_BOTTOM_TRUNCATED = "BOTTOM_TRUNC"
VIS_EDGE_TRUNCATED = "EDGE_TRUNC"
VIS_TOO_CLOSE = "TOO_CLOSE"

# ============================================================
# HELPER: OUTPUT DIRECTORIES
# ============================================================

def init_output_dirs():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def log_event(state, prev_state):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] State changed: {prev_state} -> {state}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())


def save_screenshot(frame, state):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SCREENSHOT_DIR, f"{ts}_{state}.jpg")
    cv2.imwrite(path, frame)

# ============================================================
# HELPER: VIRTUAL FENCE DRAWING
# ============================================================

def draw_fence(frame, fence_region):
    x1, y1, x2, y2 = fence_region
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_FENCE, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FENCE, 2)
    cv2.putText(frame, "Restricted Area", (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FENCE, 2)

# ============================================================
# HELPER: CHECK IF POINT IS INSIDE FENCE
# ============================================================

def point_in_fence(cx, cy, fence_region):
    x1, y1, x2, y2 = fence_region
    return x1 <= cx <= x2 and y1 <= cy <= y2

# ============================================================
# SIMPLE CENTROID TRACKER
# ============================================================

class SimpleCentroidTracker:
    """
    Minimal centroid-based tracker that assigns stable IDs to detections
    across frames using nearest-neighbor matching.
    """

    def __init__(self, max_distance=MATCH_DISTANCE):
        self.next_id = 0
        self.objects = {}          # id -> (cx, cy)
        self.max_distance = max_distance

    def update(self, centroids):
        """
        centroids: list of (cx, cy)
        Returns: list of assigned IDs (same order as input centroids)
        """
        if len(centroids) == 0:
            self.objects = {}
            return []

        if len(self.objects) == 0:
            ids = []
            for c in centroids:
                self.objects[self.next_id] = c
                ids.append(self.next_id)
                self.next_id += 1
            return ids

        old_ids = list(self.objects.keys())
        old_cents = list(self.objects.values())

        # Distance matrix
        dists = np.zeros((len(old_cents), len(centroids)), dtype=np.float32)
        for i, oc in enumerate(old_cents):
            for j, nc in enumerate(centroids):
                dists[i, j] = np.hypot(oc[0] - nc[0], oc[1] - nc[1])

        assigned_ids = [None] * len(centroids)
        used_old = set()
        used_new = set()

        # Greedy closest-first matching
        flat = list(np.ndindex(dists.shape))
        flat.sort(key=lambda idx: dists[idx])
        for i, j in flat:
            if i in used_old or j in used_new:
                continue
            if dists[i, j] > self.max_distance:
                continue
            assigned_ids[j] = old_ids[i]
            used_old.add(i)
            used_new.add(j)

        # Assign new IDs for unmatched detections
        for j in range(len(centroids)):
            if assigned_ids[j] is None:
                assigned_ids[j] = self.next_id
                self.next_id += 1

        # Update stored objects
        new_objects = {}
        for j, cid in enumerate(assigned_ids):
            new_objects[cid] = centroids[j]
        self.objects = new_objects

        return assigned_ids

# ============================================================
# EDGE-AWARE VISIBILITY CLASSIFICATION
# ============================================================

def classify_visibility(bx1, by1, bx2, by2, frame_w, frame_h):
    """
    Classify a person box into a visibility state based on proximity
    to frame borders and box size relative to frame.
    """
    touch_bottom = (by2 >= frame_h - EDGE_MARGIN)
    touch_left = (bx1 <= EDGE_MARGIN)
    touch_right = (bx2 >= frame_w - EDGE_MARGIN)

    box_area = (bx2 - bx1) * (by2 - by1)
    frame_area = frame_w * frame_h
    area_ratio = box_area / frame_area if frame_area > 0 else 0

    # TOO_CLOSE: touches bottom and is very large
    if touch_bottom and area_ratio >= TOO_CLOSE_AREA_RATIO:
        return VIS_TOO_CLOSE
    # BOTTOM_TRUNCATED: touches bottom edge (feet likely out of frame)
    if touch_bottom:
        return VIS_BOTTOM_TRUNCATED
    # EDGE_TRUNCATED: touches left or right edge (body partially cropped)
    if touch_left or touch_right:
        return VIS_EDGE_TRUNCATED
    # FULL_BODY: no significant border contact
    return VIS_FULL_BODY


def compute_anchor_point(f):
    """
    Select a dynamic anchor point for intrusion checking
    based on the person's visibility state.

    FULL_BODY       -> bottom-center (standard)
    BOTTOM_TRUNCATED -> 75% height point (compensate for missing feet)
    EDGE_TRUNCATED   -> box center (conservative)
    TOO_CLOSE        -> box center (conservative)
    """
    bx1, by1, bx2, by2 = f['bx1'], f['by1'], f['bx2'], f['by2']
    vis = f['visibility']
    mid_x = (bx1 + bx2) / 2

    if vis == VIS_FULL_BODY:
        return mid_x, float(by2)
    elif vis == VIS_BOTTOM_TRUNCATED:
        return mid_x, by1 + 0.75 * f['h']
    else:  # EDGE_TRUNCATED or TOO_CLOSE
        return mid_x, (by1 + by2) / 2


# ============================================================
# SHARED FEATURE EXTRACTION
# ============================================================

def extract_person_features(person_dets, ids, centroids, person_history,
                            frame_w, frame_h):
    """
    Build a shared feature dict for each detected person.
    Includes visibility state and dynamic anchor point.
    Updates person_history in-place as a side effect.
    """
    features = []
    for idx, (bx1, by1, bx2, by2, conf) in enumerate(person_dets):
        pid = ids[idx]
        cx, cy = centroids[idx]
        w = bx2 - bx1
        h = by2 - by1
        area = w * h
        aspect_ratio = h / w if w > 0 else 999
        center_y = (by1 + by2) / 2
        bottom_y = float(by2)

        visibility = classify_visibility(bx1, by1, bx2, by2, frame_w, frame_h)

        # Update per-person position history
        if pid not in person_history:
            person_history[pid] = deque(maxlen=PERSON_HISTORY_LENGTH)
        person_history[pid].append((center_y, bottom_y))

        f = {
            'pid': pid,
            'bx1': bx1, 'by1': by1, 'bx2': bx2, 'by2': by2,
            'conf': conf,
            'cx': cx, 'cy': cy,
            'w': w, 'h': h, 'area': area,
            'aspect_ratio': aspect_ratio,
            'center_y': center_y, 'bottom_y': bottom_y,
            'visibility': visibility,
        }
        # Dynamic anchor point for intrusion checking
        ax, ay = compute_anchor_point(f)
        f['anchor_x'] = ax
        f['anchor_y'] = ay
        features.append(f)
    return features

# ============================================================
# BRANCH 1: INTRUSION DETECTION
# ============================================================

def detect_intrusion(features, intrusion_counters, fence_region):
    """
    For each person, determine whether they are intruding into the fence.
    Uses dynamic anchor point + state-dependent confirmation threshold.
    Returns list of bool (same order as features).
    """
    # Map visibility state to confirmation threshold
    thresh_map = {
        VIS_FULL_BODY: INTRUSION_THRESH_FULL,
        VIS_BOTTOM_TRUNCATED: INTRUSION_THRESH_BOTTOM,
        VIS_EDGE_TRUNCATED: INTRUSION_THRESH_EDGE,
        VIS_TOO_CLOSE: INTRUSION_THRESH_CLOSE,
    }

    results = []
    for f in features:
        pid = f['pid']
        # Use dynamic anchor point instead of always bottom-center
        inside = point_in_fence(f['anchor_x'], f['anchor_y'], fence_region)
        if inside:
            intrusion_counters[pid] = intrusion_counters.get(pid, 0) + 1
        else:
            intrusion_counters[pid] = 0
        threshold = thresh_map.get(f['visibility'], INTRUSION_THRESH_FULL)
        is_intruding = intrusion_counters[pid] >= threshold
        results.append(is_intruding)
    return results

# ============================================================
# BRANCH 2: FALL SUSPICION DETECTION
# ============================================================

def detect_fall(features, fall_counters, person_history, frame_height):
    """
    For each person, determine whether they are a possible fall case.
    Multi-condition: flat_box AND (downward_motion OR low_position),
    confirmed over state-dependent consecutive frames.
    Suppressed (threshold=0) for unreliable visibility states.
    Returns list of bool (same order as features).
    """
    # Map visibility state to fall confirmation threshold (0 = suppressed)
    thresh_map = {
        VIS_FULL_BODY: FALL_THRESH_FULL,
        VIS_BOTTOM_TRUNCATED: FALL_THRESH_BOTTOM,
        VIS_EDGE_TRUNCATED: FALL_THRESH_EDGE,
        VIS_TOO_CLOSE: FALL_THRESH_CLOSE,
    }

    results = []
    for f in features:
        pid = f['pid']
        vis = f['visibility']
        fall_threshold = thresh_map.get(vis, FALL_THRESH_FULL)

        # If threshold is 0, this visibility state suppresses fall detection
        if fall_threshold == 0:
            fall_counters[pid] = 0
            results.append(False)
            continue

        hist = person_history[pid]

        # Condition: box is large enough and flat enough
        flat_box = (f['area'] >= MIN_BOX_AREA
                    and f['aspect_ratio'] < FALL_RATIO_THRESHOLD)

        # Condition A: downward movement over recent history
        downward_motion = False
        if len(hist) >= 2:
            oldest_center_y = hist[0][0]
            if f['center_y'] - oldest_center_y > FALL_DOWNWARD_THRESHOLD:
                downward_motion = True

        # Condition B: person bottom stays in the low region of the frame
        low_position = False
        if len(hist) >= FALL_LOW_POSITION_FRAMES:
            recent = list(hist)[-FALL_LOW_POSITION_FRAMES:]
            low_threshold_y = frame_height * FALL_LOW_POSITION_RATIO
            if all(b_y > low_threshold_y for _, b_y in recent):
                low_position = True

        # Combined: flat box AND at least one motion/position signal
        fall_condition = flat_box and (downward_motion or low_position)
        if fall_condition:
            fall_counters[pid] = fall_counters.get(pid, 0) + 1
        else:
            fall_counters[pid] = 0

        is_fall_suspected = fall_counters[pid] >= fall_threshold
        results.append(is_fall_suspected)
    return results

# ============================================================
# FUSION LAYER
# ============================================================

def fuse_system_state(intrusion_results, fall_results):
    """
    Aggregate per-person branch outputs into a single system state.
    """
    any_intrusion = any(intrusion_results) if intrusion_results else False
    any_fall = any(fall_results) if fall_results else False

    if any_intrusion and any_fall:
        return STATE_CRITICAL
    elif any_fall:
        return STATE_FALL
    elif any_intrusion:
        return STATE_INTRUSION
    else:
        return STATE_NORMAL

# ============================================================
# DISPLAY: DRAW PERSON ANNOTATIONS
# ============================================================

def draw_person_annotations(display, features, intrusion_results, fall_results):
    """Draw boxes and labels for each person based on branch outputs."""
    for idx, f in enumerate(features):
        is_intruding = intrusion_results[idx]
        is_fall_suspected = fall_results[idx]

        box_color = COLOR_NORMAL
        labels = [f"person {f['conf']:.2f}"]

        # Show visibility state for debugging
        labels.append(f"[{f['visibility']}]")

        if is_intruding and is_fall_suspected:
            box_color = COLOR_CRITICAL
            labels.append("INTRUSION")
            labels.append("Possible Fall")
        elif is_fall_suspected:
            box_color = COLOR_FALL
            labels.append("Possible Fall")
        elif is_intruding:
            box_color = COLOR_INTRUSION
            labels.append("INTRUSION")

        bx1, by1, bx2, by2 = f['bx1'], f['by1'], f['bx2'], f['by2']
        cv2.rectangle(display, (bx1, by1), (bx2, by2), box_color, 2)

        # Draw dynamic anchor point (shows which point is used for intrusion)
        cv2.circle(display, (int(f['anchor_x']), int(f['anchor_y'])), 5,
                   box_color, -1)

        for li, label in enumerate(labels):
            ty = by1 - 10 - li * 22
            if ty < 12:
                ty = by1 + 20 + li * 22
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (bx1, ty - th - 4), (bx1 + tw + 4, ty + 4),
                          COLOR_TEXT_BG, -1)
            cv2.putText(display, label, (bx1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

# ============================================================
# MAIN
# ============================================================

def main():
    init_output_dirs()

    # --- Video input ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {VIDEO_SOURCE}")
        return

    # --- YOLO model ---
    model = YOLO(MODEL_PATH)

    # --- Tracker and per-ID state ---
    tracker = SimpleCentroidTracker(max_distance=MATCH_DISTANCE)
    intrusion_counters = {}   # person_id -> consecutive intrusion frame count
    fall_counters = {}        # person_id -> consecutive fall-condition frame count
    person_history = {}       # person_id -> deque of (center_y, bottom_y)
    intrusion_snapshot_taken = {}  # person_id -> bool, reset when intrusion clears
    fall_snapshot_taken = {}       # person_id -> bool, reset when fall clears

    prev_system_state = STATE_NORMAL

    print("Surveillance demo running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # ---- YOLO inference ----
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        # ---- Filter: keep only person detections (class 0) ----
        person_dets = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                person_dets.append((bx1, by1, bx2, by2, conf))

        # ---- Tracking ----
        centroids = [((bx1 + bx2) / 2, float(by2))
                     for (bx1, by1, bx2, by2, _) in person_dets]
        ids = tracker.update(centroids)

        # Clean up stale per-ID state
        active_ids = set(ids)
        intrusion_counters = {k: v for k, v in intrusion_counters.items() if k in active_ids}
        fall_counters = {k: v for k, v in fall_counters.items() if k in active_ids}
        person_history = {k: v for k, v in person_history.items() if k in active_ids}
        intrusion_snapshot_taken = {k: v for k, v in intrusion_snapshot_taken.items() if k in active_ids}
        fall_snapshot_taken = {k: v for k, v in fall_snapshot_taken.items() if k in active_ids}

        # ============================================
        # SHARED FEATURES
        # ============================================
        features = extract_person_features(
            person_dets, ids, centroids, person_history,
            frame_width, frame_height)

        # ============================================
        # BRANCH 1: INTRUSION
        # ============================================
        intrusion_results = detect_intrusion(
            features, intrusion_counters, FENCE_REGION)

        # ============================================
        # BRANCH 2: FALL SUSPICION
        # ============================================
        fall_results = detect_fall(
            features, fall_counters, person_history, frame_height)

        # ============================================
        # FUSION
        # ============================================
        system_state = fuse_system_state(intrusion_results, fall_results)

        # ---- Drawing ----
        draw_fence(display, FENCE_REGION)
        draw_person_annotations(display, features, intrusion_results, fall_results)

        # ---- Branch-level screenshots (per-person, after annotations drawn) ----
        for idx, f in enumerate(features):
            pid = f['pid']
            if intrusion_results[idx]:
                if not intrusion_snapshot_taken.get(pid, False):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = os.path.join(SCREENSHOT_DIR, f"{ts}_intrusion_pid{pid}.jpg")
                    cv2.imwrite(path, display)
                    intrusion_snapshot_taken[pid] = True
            else:
                intrusion_snapshot_taken[pid] = False

            if fall_results[idx]:
                if not fall_snapshot_taken.get(pid, False):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = os.path.join(SCREENSHOT_DIR, f"{ts}_fall_pid{pid}.jpg")
                    cv2.imwrite(path, display)
                    fall_snapshot_taken[pid] = True
            else:
                fall_snapshot_taken[pid] = False

        # System state banner
        state_color = STATE_COLORS[system_state]
        cv2.rectangle(display, (0, 0), (350, 45), COLOR_TEXT_BG, -1)
        cv2.putText(display, f"State: {system_state}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)

        # ---- Logging / screenshot on state change ----
        if system_state != prev_system_state:
            log_event(system_state, prev_system_state)
            save_screenshot(display, system_state)
            prev_system_state = system_state

        # ---- Show ----
        cv2.imshow("Surveillance Demo", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo stopped.")


if __name__ == "__main__":
    main()

