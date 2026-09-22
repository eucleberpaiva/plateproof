"""Step 1 · one pass over the video, all on CPU, timing every stage.

Per frame: detect vehicles (YOLOX-s), track them (ByteTrack), detect plates (YOLOv9), read the plates that
are big enough (OCR) and measure the brightness of every traffic light lens. Nothing is drawn here: the
shareable video comes out of step 3.

Output in <out>/ (SENSITIVE, never publish):
  raw.json.gz   everything frame by frame, plate TEXT included
  crops/        best plate crop per vehicle, for the manual review

With --no-text the OCR never runs: plates are still detected (step 3 needs to know where to pixelate), but
no plate text and no crop is written. Use it when all you need is the count.

    python -m plateproof.analyze video.mp4 scene.toml out/ [--max 300] [--no-text]
"""
import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from plateproof import models
from plateproof.core import inside, lens_brightness, load_scene, plate_owner


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--max", type=int, default=0, help="process only the first N frames (quick test)")
    parser.add_argument("--no-text", action="store_true", help="skip the OCR: no plate text is written")
    args = parser.parse_args()

    scene = load_scene(args.scene)
    detection = scene["detection"]
    out_dir = Path(args.out)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    vehicle_model, plate_model, ocr = models.load(scene)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise SystemExit(f"{args.video}: the container reports fps={fps}, {width}x{height}. Re-mux it first: "
                         f"ffmpeg -i {args.video} -c copy fixed.mp4")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # streams and some containers report no frame count: read until the end instead of doing nothing
        total = args.max or 10 ** 9
    elif args.max:
        total = min(total, args.max)

    # ByteTrack was dropped from supervision in 0.31, which is why the version is pinned in requirements.txt.
    tracker = sv.ByteTrack(track_activation_threshold=detection["vehicle_conf"], lost_track_buffer=int(fps * 2),
                           minimum_matching_threshold=0.8, frame_rate=int(round(fps)), minimum_consecutive_frames=3)

    frames, times, best_crop = [], defaultdict(list), {}
    started = time.perf_counter()
    for index in range(total):
        t0 = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        t1 = time.perf_counter()

        xyxy, conf, classes = vehicle_model(frame, conf=detection["vehicle_conf"])
        t2 = time.perf_counter()

        detections = sv.Detections(xyxy=xyxy.astype(np.float32), confidence=conf.astype(np.float32),
                                   class_id=classes.astype(int))
        detections = tracker.update_with_detections(detections)
        t3 = time.perf_counter()

        plates = plate_model.predict(frame)
        t4 = time.perf_counter()

        plate_rows = []
        for plate in plates:
            box = plate.bounding_box
            x1, y1 = max(box.x1, 0), max(box.y1, 0)
            x2, y2 = min(box.x2, width), min(box.y2, height)
            text, text_conf = None, None
            if inside(((x1 + x2) / 2, (y1 + y2) / 2), detection["ignore"]):
                # a logo or a clock often reads as a plate: never read or count it, but keep the box so that
                # step 3 still pixelates it, because a real plate can drive through that area too
                plate_rows.append([None, x1, y1, x2, y2, round(float(plate.confidence), 3), None, None])
                continue
            track_id = plate_owner((x1, y1, x2, y2), detections.xyxy, detections.tracker_id) if len(detections) else None
            if (not args.no_text and plate.confidence >= detection["ocr_plate_conf"]
                    and x2 - x1 >= detection["ocr_min_width"] and y2 > y1):
                read = ocr.predict(frame[y1:y2, x1:x2])
                if read and read.text:
                    text = read.text.strip().upper().replace("_", "")
                    text_conf = float(np.mean(read.confidence)) if read.confidence else 0.0
                    score = text_conf * (x2 - x1)
                    if track_id is not None and text and score > best_crop.get(track_id, (0,))[0]:
                        best_crop[track_id] = (score, frame[max(0, y1 - 8):y2 + 8, max(0, x1 - 12):x2 + 12].copy())
            plate_rows.append([track_id, x1, y1, x2, y2, round(float(plate.confidence), 3), text,
                               None if text_conf is None else round(text_conf, 3)])
        t5 = time.perf_counter()

        lights = [lens_brightness(frame, light["box"]) for light in scene.get("light", [])]
        t6 = time.perf_counter()

        for name, elapsed in (("decode", t1 - t0), ("vehicles", t2 - t1), ("tracking", t3 - t2),
                              ("plate_detection", t4 - t3), ("plate_ocr", t5 - t4), ("traffic_light", t6 - t5)):
            times[name].append(round(elapsed * 1000, 2))
        frames.append({
            "vehicles": [[int(t), int(c), *map(int, b), round(float(s), 3)]
                         for b, s, c, t in zip(detections.xyxy, detections.confidence, detections.class_id,
                                               detections.tracker_id)] if len(detections) else [],
            "plates": plate_rows,
            "lights": lights,
        })
        if (index + 1) % 300 == 0:
            rate = (index + 1) / (time.perf_counter() - started)
            left = f"  ~{(total - index - 1) / rate / 60:.1f} min left" if total < 10 ** 9 else ""  # unknown length
            print(f"{index + 1}/{total if total < 10 ** 9 else '?'}  {rate:.1f} frames/s{left}", flush=True)

    for track_id, (_, crop) in best_crop.items():  # no plate text in the file name: names get indexed and synced
        cv2.imwrite(str(out_dir / "crops" / f"{track_id:05d}.png"), crop)
    elapsed = time.perf_counter() - started
    with gzip.open(out_dir / "raw.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"fps": fps, "width": width, "height": height, "frames": frames, "times": times,
                   "light_names": [light["name"] for light in scene.get("light", [])],
                   "cpu_seconds": round(elapsed, 1)}, f)
    print(f"ok: {len(frames)} frames in {elapsed / 60:.1f} min ({len(frames) / elapsed:.1f} frames/s) -> {out_dir}")


if __name__ == "__main__":
    main()
