"""Draws the README charts from the run's own data — no frame of the video, only numbers.

    python docs/charts/make_charts.py out/tracks.json out/summary.json

SVG rather than PNG for three reasons: a frame of your footage is still your footage and this repository
does not take images; SVG stays sharp at any width; and it is text, so a diff shows what changed.

Two things the GitHub renderer forces, and both break silently if you forget them: it loads an SVG inside
an <img>, which pulls no external font, so the font family has to be generic; and it puts no background
behind it, so each file carries its own, otherwise the drawing disappears in dark mode.
"""
import argparse
import json
from pathlib import Path

BG, INK, INK2, INK3, LINE, BLUE = "#F4F3EF", "#14161A", "#55534D", "#8A877F", "#DCD9D1", "#2B4CFF"
MONO = "ui-monospace, 'Cascadia Mono', 'DejaVu Sans Mono', monospace"
SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def text(x, y, s, size=13, fill=INK2, family=MONO, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}">{s}</text>')


def trails(tracks, out):
    """Every path a tracked vehicle took: the middle of the bottom edge of its box, frame by frame."""
    w, h = tracks["width"], tracks["height"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + 120}" width="1600" '
             f'height="{round((h + 120) * 1600 / w)}" role="img" '
             f'aria-label="Every vehicle trajectory drawn as a technical drawing, with the counting lines">',
             f'<rect width="{w}" height="{h + 120}" fill="{BG}"/>',
             f'<g fill="none" stroke="{INK}" stroke-opacity="0.30" stroke-width="2.4" stroke-linecap="round">']
    for track in tracks["tracks"]:
        boxes, points = track[5], []
        for i in range(len(boxes) // 4):
            x, y, bw, bh = boxes[i * 4:i * 4 + 4]
            points.append(f"{x + bw / 2:.0f},{y + bh:.0f}")
        if len(points) > 1:
            parts.append(f'<path d="M{" L".join(points)}"/>')
    parts.append("</g>")
    parts.append(f'<g fill="none" stroke="{BLUE}" stroke-width="5" stroke-dasharray="24 12">')
    for line in tracks["lines"]:
        (x1, y1), (x2, y2) = line["from"], line["to"]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
    parts.append("</g>")
    crossings = sum(len(v) for v in tracks["crossings_by_line"].values())
    parts.append(text(0, h + 60, f"{len(tracks['tracks'])} TRACKS · {crossings} LINE CROSSINGS · "
                                 f"{tracks['frames']:,} FRAMES".replace(",", ","), 34, INK3))
    parts.append(text(w, h + 60, "THE COUNTING LINES, IN BLUE", 34, INK3, anchor="end"))
    parts.append("</svg>")
    (out / "trails.svg").write_text("\n".join(parts), encoding="utf-8")


def read_rate(summary, out):
    """Read rate by plate width. A pale bar means a small sample: the caveat is drawn, not footnoted."""
    bands = summary["plates"]["by_width"]
    w, h, left, base, top = 1200, 560, 90, 440, 70
    step = (w - left - 40) / len(bands)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="1200" height="{h}" '
         f'role="img" aria-label="Plate read rate by plate width in pixels">',
         f'<rect width="{w}" height="{h}" fill="{BG}"/>']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        y = base - v * (base - top)
        p.append(f'<line x1="{left}" x2="{w - 40}" y1="{y}" y2="{y}" stroke="{LINE}" stroke-width="1"/>')
        p.append(text(left - 14, y + 5, f"{round(v * 100)}%", 20, INK3, anchor="end"))
    worst = min((b for b in bands if b["vehicles"] and b["read"]), key=lambda b: b["read"] / b["vehicles"],
                default=None)
    for i, band in enumerate(bands):
        n, read = band["vehicles"], band["read"]
        rate = read / n if n else 0
        x, bw = left + i * step + step * 0.18, step * 0.64
        y = base - rate * (base - top)
        colour = BLUE if band is worst else INK
        p.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{base - y:.0f}" fill="{colour}" '
                 f'fill-opacity="{"0.30" if n < 30 else "1"}"/>')
        p.append(text(x + bw / 2, y - 16, f"{round(rate * 100)}%" if read else "0", 30, INK, SANS, "middle"))
        p.append(text(x + bw / 2, base + 34, band["band"], 20, INK2, anchor="middle"))
        p.append(text(x + bw / 2, base + 60, f"n = {n}", 18, INK3, anchor="middle"))
    p.append(text(left, 36, "PLATE READ RATE, BY HOW WIDE THE PLATE GOT IN THE IMAGE", 22, INK3))
    p.append(text(left, h - 16, "PALE BAR = FEWER THAN 30 VEHICLES IN THE BAND, DO NOT READ MUCH INTO IT",
                  18, INK3))
    p.append("</svg>")
    (out / "read-rate.svg").write_text("\n".join(p), encoding="utf-8")


def per_stage(summary, out):
    """Where each frame goes. Bar is the median, tick is the 95th percentile."""
    names = {"decode": "decode", "vehicles": "detect vehicles", "tracking": "track",
             "plate_detection": "detect plates", "plate_ocr": "read plates (OCR)"}
    stages = [(label, summary["stage_ms"][key]["p50"], summary["stage_ms"][key]["p95"])
              for key, label in names.items() if key in summary["stage_ms"]]
    w, row, left, right = 1200, 66, 330, 150
    h = len(stages) * row + 90
    top = max(p95 for _, _, p95 in stages)
    slowest = max(stages, key=lambda s: s[1])[0]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="1200" height="{h}" '
         f'role="img" aria-label="Milliseconds per frame for each stage of the pipeline">',
         f'<rect width="{w}" height="{h}" fill="{BG}"/>',
         text(left, 36, "MILLISECONDS PER FRAME · BAR = MEDIAN · TICK = 95TH PERCENTILE", 22, INK3)]
    for i, (name, p50, p95) in enumerate(stages):
        y = 70 + i * row
        bar = (p50 / top) * (w - left - right)
        x95 = left + (p95 / top) * (w - left - right)
        p.append(f'<rect x="{left}" y="{y}" width="{max(3, bar):.0f}" height="30" '
                 f'fill="{BLUE if name == slowest else INK}"/>')
        p.append(f'<line x1="{x95:.0f}" x2="{x95:.0f}" y1="{y - 6}" y2="{y + 36}" stroke="{INK3}" stroke-width="3"/>')
        p.append(text(left - 20, y + 22, name, 22, INK2, SANS, "end"))
        p.append(text(x95 + 14, y + 22, f"{p50} / {p95} ms", 20, INK3))
    detectors = sum(p50 for name, p50, _ in stages if "detect" in name)
    soma = f"{detectors:.1f}".rstrip("0").rstrip(".")   # 162.5, não 162: a soma de duas medianas é exata
    p.append(text(left, h - 20, f"THE TWO DETECTORS ARE {soma} MS OF EVERY FRAME. "
                                f"EVERYTHING ELSE IS NOISE.", 18, INK3))
    p.append("</svg>")
    (out / "per-stage.svg").write_text("\n".join(p), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tracks", help="out/tracks.json from step 2")
    parser.add_argument("summary", help="out/summary.json from step 2")
    parser.add_argument("--out", default=str(Path(__file__).parent), help="where to write the SVG files")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    trails(tracks, out)
    read_rate(summary, out)
    per_stage(summary, out)
    for f in sorted(out.glob("*.svg")):
        print(f"{f.name:<16} {f.stat().st_size // 1024:>4} KB")


if __name__ == "__main__":
    main()
