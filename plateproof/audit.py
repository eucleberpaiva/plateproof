"""Manual review of the numbers: contact sheets to look at and mark up (SENSITIVE: plates are visible).

  plates.jpg        each accepted plate crop next to the voted text: check it character by character
  class_<name>.jpg  every counted vehicle of one class: see what it actually is
  line_<name>.jpg   every crossing of one line, in order: look for the same vehicle counted twice

    python -m plateproof.audit video.mp4 out/ [--class truck] [--line straight]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def sheet(items, target, columns, cell_w, cell_h, per_sheet):
    for start in range(0, len(items), per_sheet):
        page = items[start:start + per_sheet]
        canvas = np.full((-(-len(page) // columns) * (cell_h + 24), columns * cell_w, 3), 20, np.uint8)
        for i, (crop, label) in enumerate(page):
            row, col = divmod(i, columns)
            scale = min(cell_w / crop.shape[1], cell_h / crop.shape[0])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                              interpolation=cv2.INTER_CUBIC)
            y, x = row * (cell_h + 24), col * cell_w
            canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
            cv2.putText(canvas, label, (x + 4, y + cell_h + 17), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(target.with_name(f"{target.stem}_{start // per_sheet + 1}.jpg")), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("out")
    parser.add_argument("--class", dest="classes", action="append", default=[])
    parser.add_argument("--line", dest="lines", action="append", default=[])
    args = parser.parse_args()
    out_dir = Path(args.out)
    folder = out_dir / "sheets"
    folder.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "audit.json").exists():
        raise SystemExit(f"{out_dir / 'audit.json'} not found. Run step 2 first: "
                         f"python -m plateproof.summarize <scene> {out_dir}")
    audit = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))

    crops = {int(p.stem): p for p in (out_dir / "crops").glob("*.png")}  # analyze.py writes <track id>.png
    plates = [(cv2.imread(str(crops[int(track)])), f"#{int(track)} {read['text']}")
              for track, read in sorted(audit["read"].items(), key=lambda kv: kv[1]["confirmed_frame"])
              if int(track) in crops]
    if plates:
        sheet(plates, folder / "plates.jpg", 4, 300, 110, 24)

    chosen = [c for c in audit["counted"] if c["class"] in args.classes or c["line"] in args.lines]
    if (args.classes or args.lines) and not chosen:
        raise SystemExit(f"no counted vehicle matches --class {args.classes} --line {args.lines}; "
                         f"the names come from audit.json")
    wanted = {}
    for item in chosen:
        wanted.setdefault(item["crossing"], []).append(item)
    capture, images, index = cv2.VideoCapture(args.video), {}, 0
    for target in sorted(wanted):
        while index < target:
            capture.grab()
            index += 1
        ok, frame = capture.read()
        index += 1
        if not ok:
            break
        for item in wanted[target]:
            x1, y1, x2, y2 = item["box"]
            images[item["track"]] = frame[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10].copy()
    for klass in args.classes:
        items = [(images[c["track"]], f"#{c['track']} {c['line']} frame {c['crossing']}")
                 for c in chosen if c["class"] == klass and c["track"] in images]
        sheet(items, folder / f"class_{klass}.jpg", 6, 220, 160, 36)
    for line in args.lines:
        items = [(images[c["track"]], f"#{c['track']} frame {c['crossing']} x{(c['box'][0] + c['box'][2]) // 2}")
                 for c in sorted(chosen, key=lambda c: c["crossing"]) if c["line"] == line and c["track"] in images]
        sheet(items, folder / f"line_{line}.jpg", 8, 170, 120, 48)
    print(f"ok: {len(plates)} plates, {len(chosen)} vehicles -> {folder}")


if __name__ == "__main__":
    main()
