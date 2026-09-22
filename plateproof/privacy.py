"""Step 3 · builds the video you can show, with the plates hidden. No model runs: it reuses step 1.

Three layers, because no detector is right every frame:
  1. every detected plate, with margin
  2. every vehicle whose box is tall enough: the lower band of the box, pixelated in EVERY frame, plus a few
     frames before and after the track exists. This covers most of the plates the plate detector missed.
  3. fixed masks from the scene (for example the date burned into the camera clock)

One case is left over: vehicle AND plate missed in the same frame. That is what step 4 is for.
The story of what failed before this method is in docs/privacy.md.

    python -m plateproof.privacy video.mp4 scene.toml out/ [--max 300]   ->  out/public.mp4
"""
import argparse
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

from plateproof.core import load_scene, pixelate, read_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--max", type=int, default=0, help="process only the first N frames (quick test)")
    args = parser.parse_args()

    scene = load_scene(args.scene)
    privacy = scene["privacy"]
    raw = read_raw(args.out)
    frames, fps, width, height = raw["frames"], raw["fps"], raw["width"], raw["height"]
    if args.max:
        frames = frames[:args.max]
    out_width = privacy["output_width"]
    out_height = round(height * out_width / width / 2) * 2

    boxes, first, last = defaultdict(list), {}, {}
    for index, frame in enumerate(frames):
        for vehicle in frame["vehicles"]:
            boxes[index].append(vehicle[2:6])
            first.setdefault(vehicle[0], (index, vehicle[2:6]))
            last[vehicle[0]] = (index, vehicle[2:6])
    for ends, direction in ((first, -1), (last, 1)):
        for index, box in ends.values():
            for step in range(1, privacy["frame_slack"] + 1):
                boxes[index + direction * step].append(box)

    ffmpeg = shutil.which("ffmpeg")  # never bare "ffmpeg": on Windows CreateProcess searches . before PATH
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on your PATH (only this step needs it)")
    target = Path(args.out).resolve() / "public.mp4"  # resolve: an --out starting with - would read as a flag
    encoder = subprocess.Popen(
        [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_width}x{out_height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "32", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        stdin=subprocess.PIPE)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    # the boxes come from raw.json.gz, the pixels from this file: if they are not the same video, the
    # mosaics land in the wrong places and step 4 compares against the wrong plate list, so both checks
    # would pass over a video where every plate is readable
    got = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if got[:2] != (width, height) or (got[2] > 0 and not args.max and got[2] != len(raw["frames"])):
        raise SystemExit(f"{args.video} is {got[0]}x{got[1]} with {got[2]} frames, but raw.json.gz came from "
                         f"{width}x{height} with {len(raw['frames'])}. Run step 1 on this exact file.")
    counts = {"plates": 0, "bands": 0}
    bx1, by1, bx2, by2 = privacy["band"]
    for index, frame in enumerate(frames):
        ok, image = capture.read()
        if not ok:
            break
        for _, x1, y1, x2, y2, *_ in frame["plates"]:  # no exception for `ignore`: pixelating a logo costs nothing
            mx, my = (x2 - x1) * 0.3, (y2 - y1) * 0.5
            counts["plates"] += pixelate(image, x1 - mx, y1 - my, x2 + mx, y2 + my, privacy["block"])
        for vx1, vy1, vx2, vy2 in boxes.get(index, ()):  # .get: reading a defaultdict would insert a row per frame
            w, h = vx2 - vx1, vy2 - vy1
            if h >= privacy["min_height"]:
                counts["bands"] += pixelate(image, vx1 + bx1 * w, vy1 + by1 * h,
                                            vx1 + bx2 * w, vy1 + by2 * h, privacy["block"])
        for mask in privacy.get("masks", []):
            pixelate(image, *mask, privacy["block"])
        encoder.stdin.write(cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed while encoding the shareable video")
    print(f"ok: {counts['plates']} plates and {counts['bands']} bands pixelated -> {target}")
    print("before showing this video to anyone: python -m plateproof.verify and python -m plateproof.sample")


if __name__ == "__main__":
    main()
