"""Model-free rules: scene loading, line counting, traffic light state, plate voting, pixelation.

Every function here is pure (data in, data out) and covered by tests/test_plateproof.py.
"""
import gzip
import json
import tomllib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

COCO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_ORDER = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
VEHICLES = {"car", "truck", "bus", "motorcycle"}
LIGHT_STATES = ["red", "yellow", "green"]


def load_scene(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_raw(out_dir):
    """The per-frame file written by `plateproof.analyze` (SENSITIVE: holds plate text)."""
    path = Path(out_dir) / "raw.json.gz"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run step 1 first: python -m plateproof.analyze <video> <scene> {out_dir}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ── counting ───────────────────────────────────────────────────────────────────

def foot_point(box):
    """The point that stands for a vehicle on the ground: middle of the box bottom edge."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def side(p, a, b):
    """> 0 when p is on the right-hand side of someone walking from a to b on screen (y grows downwards)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing(points, frames, line):
    """Frame where the trajectory crosses the line, or None.

    direction "right_to_left": goes from side > 0 to side <= 0, as seen by someone walking from `from` to `to`.
    direction "any": changes side either way.
    The point right after the crossing must fall within the length of the segment.
    """
    a, b = line["from"], line["to"]
    length2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    if length2 == 0:  # a typo in the scene file would otherwise count 0 forever, in silence
        raise ValueError(f"counting line {line.get('name', '?')!r} has zero length: 'from' and 'to' are the same point")
    any_way = line.get("direction", "right_to_left") == "any"
    for p0, p1, frame in zip(points, points[1:], frames[1:]):
        s0, s1 = side(p0, a, b), side(p1, a, b)
        crossed = (s0 * s1 <= 0 and s0 != s1) if any_way else (s0 > 0 >= s1)
        if not crossed:
            continue
        t = ((p1[0] - a[0]) * (b[0] - a[0]) + (p1[1] - a[1]) * (b[1] - a[1])) / length2
        if 0 < t <= 1:
            return frame
    return None


def assign_line(boxes, frames, lines):
    """(line name, crossing frame) for the first line in scene order the trajectory crosses.

    A line may also require a total displacement (end minus start, median of up to 5 points on each side)
    inside open intervals: dx = [min, max], dy = [min, max]. That is what separates a vehicle going straight
    from one that only clips the line while turning.
    """
    points = [foot_point(b) for b in boxes]
    k = max(1, min(5, len(points) // 2))  # the two windows must never overlap, or short tracks lose their travel
    total_dx, total_dy = np.median(points[-k:], axis=0) - np.median(points[:k], axis=0)
    for line in lines:
        dx = line.get("dx", [-np.inf, np.inf])
        dy = line.get("dy", [-np.inf, np.inf])
        if not (dx[0] < total_dx < dx[1] and dy[0] < total_dy < dy[1]):
            continue
        frame = crossing(points, frames, line)
        if frame is not None:
            return line["name"], frame
    return None, -1  # -1 rather than None: the frame travels into tracks.json, where the viewer wants a number


# ── traffic light ──────────────────────────────────────────────────────────────

def lens_brightness(frame, box):
    """Brightness (90th percentile of the V channel) of each third of the box: red, yellow, green."""
    x, y, w, h = box
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    if crop.shape[0] < 3 or crop.shape[1] < 1:
        raise ValueError(f"traffic light box {box} falls outside the {frame.shape[1]}x{frame.shape[0]} frame or is "
                         f"under 3 px tall: fix [[light]].box in the scene file")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    third = crop.shape[0] // 3
    return [round(float(np.percentile(hsv[k * third:(k + 1) * third, :, 2], 90)), 1) for k in range(3)]


def lit_lens(brightness, lit=230, margin=50):
    """State of one traffic light in one frame, or None when no lens stands out."""
    i = int(np.argmax(brightness))
    ranked = sorted(brightness)
    # ranked[-2] is the second brightest lens; on a tie at the top the margin is 0 and the frame reads None
    return LIGHT_STATES[i] if brightness[i] >= lit and brightness[i] - ranked[-2] >= margin else None


def light_phases(readings, fps, lit=230, margin=50):
    """readings[frame] = brightness list of every traffic light that drives the counted flow.

    Frame state = agreement between the lights that read something; then a majority vote over a ~0.5 s
    window, which absorbs occlusion (a truck passing in front) and camera flicker.
    Returns [[first_frame, last_frame, state]].
    """
    per_frame = []
    for frame in readings:
        states = {s for s in (lit_lens(b, lit, margin) for b in frame) if s}
        per_frame.append(states.pop() if len(states) == 1 else None)
    half_window = int(fps * 0.25)
    current = next((s for s in per_frame if s), None)
    phases = []
    for i in range(len(per_frame)):
        window = [s for s in per_frame[max(0, i - half_window):i + half_window + 1] if s]
        if window:
            current = Counter(window).most_common(1)[0][0]
        if not phases or phases[-1][2] != current:
            phases.append([i, i, current])
        phases[-1][1] = i
    return phases


# ── plates ─────────────────────────────────────────────────────────────────────

def plate_owner(plate, boxes, ids):
    """The vehicle a plate belongs to: the smallest box containing the plate centre."""
    cx, cy = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    best, best_area = None, float("inf")
    for (x1, y1, x2, y2), track_id in zip(boxes, ids):
        area = (x2 - x1) * (y2 - y1)
        if x1 <= cx <= x2 and y1 <= cy <= y2 and area < best_area:
            best, best_area = int(track_id), area
    return best


def vote_plate(reads, min_conf=0.6, length=(5, 8)):
    """Per-character vote across every OCR read of the same vehicle.

    reads = [(frame, text, confidence, width_px)]. Returns None with fewer than 2 usable reads of the same
    length. `agreement` = mean, per position, of the winning character's weight: 1.0 means unanimous.
    Ties, both on length and on a character, go to the read that came first.
    """
    good = [r for r in reads if r[2] >= min_conf and length[0] <= len(r[1]) <= length[1]]
    if len(good) < 2:
        return None
    by_length = Counter()
    for _, text, conf, _ in good:
        by_length[len(text)] += conf
    n = by_length.most_common(1)[0][0]
    good = [r for r in good if len(r[1]) == n]
    if len(good) < 2:  # two reads that disagree on character count are not a vote
        return None
    text, agreement = "", []
    for i in range(n):
        votes = Counter()
        for _, candidate, conf, _ in good:
            votes[candidate[i]] += conf
        char, weight = votes.most_common(1)[0]
        text += char
        agreement.append(weight / sum(votes.values()))
    same = [frame for frame, candidate, _, _ in good if candidate == text]
    return {"text": text, "reads": len(good), "agreement": float(np.mean(agreement)),
            "confidence": float(np.mean([c for _, _, c, _ in good])),
            "confirmed_frame": same[1] if len(same) >= 2 else good[1][0],
            "width": int(max(r[3] for r in good))}


def _strings(data):
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        for key, value in data.items():
            yield str(key)
            yield from _strings(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            yield from _strings(value)


def ensure_no_plate_text(data, texts):
    """Raise if any plate text (4+ chars) shows up in the data about to be published. Last safety net.

    Only strings and keys are checked: numbers in the file (coordinates, frame indices) collide by chance
    with digit-only OCR reads, and a loose number identifies nobody.
    """
    haystack = "\n".join(_strings(data)).upper()
    leaked = sorted(t for t in texts if len(t) >= 4 and t.upper() in haystack)
    if leaked:
        raise ValueError(f"{len(leaked)} plate text(s) found in data meant for publication; nothing was written")


# ── privacy ────────────────────────────────────────────────────────────────────

def pixelate(img, x1, y1, x2, y2, block=24):
    """Mosaic with `block` px cells, in place (img is modified). True when something was actually covered."""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return False
    small = cv2.resize(img[y1:y2, x1:x2], (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return True


def inside(point, regions):
    return any(x1 <= point[0] <= x2 and y1 <= point[1] <= y2 for x1, y1, x2, y2 in regions)
