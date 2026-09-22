"""Step 4b · pulls WHOLE frames out of a video for the visual review.

Whole frames, not crops of the pixelated area: a crop centred on the mosaic only proves the mosaic is
blurred, never that the plate is inside it. Open every image and look for any readable plate.
Handy for calibration too: open a frame in an editor that shows the cursor position to read coordinates off.

    python -m plateproof.sample video.mp4 out/ [--n 24] [--frame 4500 --frame 9000] [--seed 16]
    ->  out/sheets/<video name>/frame_NNNNN.jpg
"""
import argparse
import random
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("out")
    parser.add_argument("--n", type=int, default=24, help="how many frames to draw")
    parser.add_argument("--frame", type=int, action="append", default=[], help="frame that is always included")
    parser.add_argument("--seed", type=int, default=16, help="same seed, same frames")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0 and not args.frame:
        raise SystemExit(f"{args.video} reports no frame count; pick the frames explicitly with --frame N")
    random.seed(args.seed)
    wanted = sorted(set(random.sample(range(max(0, total)), min(args.n, max(0, total)))) | set(args.frame))
    folder = Path(args.out) / "sheets" / Path(args.video).stem  # original and shareable never in the same folder
    folder.mkdir(parents=True, exist_ok=True)
    index = 0
    for target in wanted:
        while index < target:  # reading in sequence is more reliable than seeking in H.264
            capture.grab()
            index += 1
        ok, frame = capture.read()
        index += 1
        if not ok:
            break
        cv2.imwrite(str(folder / f"frame_{target:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{len(wanted)} frames -> {folder}")


if __name__ == "__main__":
    main()
