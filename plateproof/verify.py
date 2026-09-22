"""Step 4a · attacks your own shareable video: the same plate reader, at a lower threshold, tries to read it.

The video is upscaled back to the original resolution first, which is what somebody would do to try to read
it. Every plate-shaped read is compared against the confident reads from the original (confidence >= 0.8 and
4 characters or more): an edit distance of 2 or less counts as a leaked plate. A weak OCR guess is left out
of the comparison, otherwise any street sign in the scene raises a false alarm.

Nothing is skipped, not even the `ignore` regions of the scene: a plate can drive behind a logo.
The texts come from out/texts.json (written by step 2). With none of them, the command stops with an error
instead of claiming it passed.

The command exits with an error whenever it flags anything, by design. The verdict is yours, looking at
sheets/verification.jpg crop by crop.

Blind spots: the attack uses the same plate detector as the analysis, so a plate it never found in the
original is not found here either; and a plate read below 0.8 confidence in the original is not in the
comparison list, so a leak of that plate is not flagged. Step 4b (python -m plateproof.sample) exists for that.

    python -m plateproof.verify out/public.mp4 scene.toml out/ [--step 1] [--max 300]
    ->  out/verification.json  and  out/sheets/verification.jpg (crops of the flagged reads)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from plateproof import models
from plateproof.core import load_scene


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        previous, row[0] = row[:], i
        for j, cb in enumerate(b, 1):
            row[j] = min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (ca != cb))
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("public_video")
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--step", type=int, default=1, help="analyse 1 frame out of every N (1 = all of them)")
    parser.add_argument("--max", type=int, default=0, help="analyse only the first N frames (quick test)")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be 1 or more")

    scene = load_scene(args.scene)
    scene["detection"]["plate_conf"] = 0.2  # more sensitive than the analysis: here over-detecting is the safe side
    out_dir = Path(args.out)
    for needed in ("summary.json", "texts.json"):
        if not (out_dir / needed).exists():
            raise SystemExit(f"{out_dir / needed} not found. Run step 2 first: "
                             f"python -m plateproof.summarize {args.scene} {out_dir}")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    width, height = summary["video"]["width"], summary["video"]["height"]
    known = {t for t in json.loads((out_dir / "texts.json").read_text(encoding="utf-8"))["confident"] if len(t) >= 4}
    if not known:
        raise SystemExit("No plate read in the original to compare against: this check would prove nothing here. "
                         "Use the visual review only (python -m plateproof.sample).")

    _, plate_model, ocr = models.load(scene, vehicles=False)
    capture = cv2.VideoCapture(args.public_video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.public_video}")
    low, high = scene["plates"]["length"]  # the same shape the analysis calls a plate, straight from the scene
    found, analysed, index = [], 0, 0
    while not args.max or index < args.max:
        if index % args.step:  # skipped frame: decode the packet without paying for the full retrieve
            if not capture.grab():
                break
            index += 1
            continue
        ok, frame = capture.read()
        if not ok:
            break
        analysed += 1
        large = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        for plate in plate_model.predict(large):
            box = plate.bounding_box
            x1, y1 = max(box.x1, 0), max(box.y1, 0)
            x2, y2 = min(box.x2, width), min(box.y2, height)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            read = ocr.predict(large[y1:y2, x1:x2])
            if not read or not read.text:
                continue
            text = read.text.strip().upper().replace("_", "")
            confidence = float(np.mean(read.confidence)) if read.confidence else 0.0
            if confidence >= 0.7 and low <= len(text) <= high:
                distance = min((edit_distance(text, t) for t in known), default=99)
                # keep a crop of everything flagged, plus the first few, so the rest can be sanity-checked
                # cap the crops kept: on a video that leaks badly this list would grow without bound
                keep = len([f for f in found if f[3] is not None]) < 96 and (distance <= 2 or len(found) < 24)
                crop = large[max(0, y1 - 10):y2 + 10, max(0, x1 - 20):x2 + 20].copy() if keep else None
                found.append((index, text, distance, crop))
        index += 1

    leaked = [f for f in found if f[2] <= 2]
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    kept = [f for f in sorted(found, key=lambda f: f[2]) if f[3] is not None][:48]
    if kept:
        sheet = np.full((-(-len(kept) // 6) * 110, 6 * 260, 3), 20, np.uint8)
        for i, (frame_index, _, distance, crop) in enumerate(kept):
            row, col = divmod(i, 6)
            scale = min(250 / crop.shape[1], 80 / crop.shape[0])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
            sheet[row * 110:row * 110 + crop.shape[0], col * 260:col * 260 + crop.shape[1]] = crop
            cv2.putText(sheet, f"frame {frame_index} distance {distance}", (col * 260 + 4, row * 110 + 100),
                        0, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / "sheets" / "verification.jpg"), sheet)
    result = {"frames_analysed": analysed, "plate_shaped_reads": len(found), "close_to_a_real_plate": len(leaked),
              "min_distance": min((f[2] for f in found), default=None),
              "flagged_frames": sorted({f[0] for f in leaked})}  # frame numbers only: no text is ever written
    (out_dir / "verification.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    if leaked:
        raise SystemExit(f"FLAGGED: {len(leaked)} reads close to a real plate. Look at sheets/verification.jpg "
                         f"crop by crop before publishing anything.")
    print("clear. Still missing the visual review: python -m plateproof.sample")


if __name__ == "__main__":
    main()
