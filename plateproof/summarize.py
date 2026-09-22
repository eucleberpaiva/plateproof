"""Step 2 · turns the per-frame file into numbers and into the viewer file.

Output in <out>/:
  summary.json  counts per line and class, traffic light phases, plate read rate, time per stage  PUBLISHABLE
  tracks.json   boxes per vehicle, crossings and phases, for the viewer                           PUBLISHABLE
  audit.json    plates with text and counted vehicles, for the manual review                      SENSITIVE
  texts.json    every text the OCR read, so step 4 can compare without loading the raw file       SENSITIVE

Both publishable files go through ensure_no_plate_text before being written: if any plate text shows up in
them, nothing is written.

    python -m plateproof.summarize scene.toml out/
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from plateproof.core import (CLASS_ORDER, COCO_CLASSES, LIGHT_STATES, VEHICLES, assign_line, ensure_no_plate_text,
                        light_phases, load_scene, read_raw, vote_plate)

VIEWER_STEP = 4          # the viewer gets one box every 4 frames and interpolates in between
FLOW_BUCKET_S = 10
WIDTH_BANDS = [(1, 40, "up to 40 px"), (40, 60, "40 to 60"), (60, 80, "60 to 80"), (80, 100, "80 to 100"),
               (100, 99999, "100 or more")]


def summarize(scene, raw):
    frames, fps = raw["frames"], raw["fps"]
    lines = scene.get("line", [])

    tracks = defaultdict(lambda: {"frames": [], "boxes": [], "votes": Counter()})
    for index, frame in enumerate(frames):
        for track_id, class_id, x1, y1, x2, y2, score in frame["vehicles"]:
            track = tracks[track_id]
            track["frames"].append(index)
            track["boxes"].append((x1, y1, x2, y2))
            track["votes"][class_id] += score
    tracks = {k: v for k, v in tracks.items() if len(v["frames"]) >= scene["tracking"]["min_frames"]}
    for track in tracks.values():
        track["class"] = COCO_CLASSES[track["votes"].most_common(1)[0][0]]
        track["line"], track["crossing"] = assign_line(track["boxes"], track["frames"], lines)
    counted = {k: v for k, v in tracks.items() if v["line"] and v["class"] in VEHICLES}

    # plates
    plate_cfg = scene["plates"]
    reads, plate_width = defaultdict(list), defaultdict(int)
    for index, frame in enumerate(frames):
        for track_id, x1, y1, x2, y2, _, text, conf in frame["plates"]:
            if track_id is None:
                continue
            plate_width[track_id] = max(plate_width[track_id], x2 - x1)
            if text and conf is not None:
                reads[track_id].append((index, text, conf, x2 - x1))
    votes = {}
    for track_id in tracks:
        voted = vote_plate(reads.get(track_id, []), plate_cfg["vote_min_conf"], plate_cfg["length"])
        if voted:
            votes[track_id] = voted
    plate_lines = {line["name"] for line in lines if line.get("plates")}
    candidates = [k for k, v in counted.items() if v["line"] in plate_lines]
    read_plates = {k: votes[k] for k in candidates
                   if k in votes and votes[k]["reads"] >= plate_cfg["min_reads"]
                   and votes[k]["agreement"] >= plate_cfg["min_agreement"]}

    # traffic light
    flow_lights = [i for i, light in enumerate(scene.get("light", [])) if light.get("flow")]
    phases = []
    if flow_lights:
        reading = scene.get("light_reading", {})
        phases = light_phases([[f["lights"][i] for i in flow_lights] for f in frames], fps,
                              reading.get("lit", 230), reading.get("margin", 50))
    full = phases[1:-1]  # the first and the last phase were cut by the start and the end of the video
    cycles = sum(1 for a, b, c in zip(full, full[1:], full[2:])
                 if (a[2], b[2], c[2]) == ("green", "yellow", "red"))
    crossings = {line["name"]: sorted(v["crossing"] for v in counted.values() if v["line"] == line["name"])
                 for line in lines}

    times = {k: np.array(v) for k, v in raw["times"].items()}
    summary = {
        "video": {"fps": fps, "frames": len(frames), "width": raw["width"], "height": raw["height"]},
        "tracks": len(tracks),
        "vehicles_counted": len(counted),
        # over `counted`, the same set as vehicles_counted: the breakdown has to add up to the headline
        "counts": {line["name"]: {"label": line.get("label", line["name"]),
                                  **{c: sum(t["line"] == line["name"] and t["class"] == c for t in counted.values())
                                     for c in CLASS_ORDER}}
                   for line in lines},
        "flow": {"bucket_seconds": FLOW_BUCKET_S,
                 "by_line": {name: np.bincount([int(f / fps) // FLOW_BUCKET_S for f in frame_list],
                                               minlength=int(len(frames) / fps) // FLOW_BUCKET_S + 1).tolist()
                             for name, frame_list in crossings.items()}},
        "traffic_light": {
            "full_cycles": cycles,
            "phases": [{"start_s": round(a / fps, 1), "end_s": round((b + 1) / fps, 1), "state": state,
                        "crossings": {name: sum(a <= f <= b for f in fl) for name, fl in crossings.items()}}
                       for a, b, state in phases],
            "duration_s": {state: [round((b - a + 1) / fps, 1) for a, b, s in full if s == state]
                           for state in LIGHT_STATES},
        },
        "plates": {
            "candidates": len(candidates),
            "with_plate_detected": sum(plate_width.get(k, 0) > 0 for k in candidates),
            "with_vote": sum(k in votes for k in candidates),
            "read": len(read_plates),
            "by_width": [{"band": label,
                          "vehicles": len(group := [k for k in candidates if low <= plate_width.get(k, 0) < high]),
                          "read": sum(k in read_plates for k in group)} for low, high, label in WIDTH_BANDS],
        },
        "stage_ms": {k: {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
                     for k, v in times.items()},
        "frames_per_second": round(len(frames) / raw["cpu_seconds"], 2),
    }

    viewer_plates = sorted([read_plates[k]["confirmed_frame"], round(read_plates[k]["confidence"], 2),
                            read_plates[k]["reads"], k] for k in read_plates)
    plate_index = {p[3]: i for i, p in enumerate(viewer_plates)}
    viewer_tracks = []
    for track_id in sorted(tracks, key=lambda k: tracks[k]["frames"][0]):
        track = tracks[track_id]
        by_frame, last, flat = dict(zip(track["frames"], track["boxes"])), track["boxes"][0], []
        for frame in range(track["frames"][0], track["frames"][-1] + 1, VIEWER_STEP):
            last = by_frame.get(frame, last)  # gap in the track: repeat the last known box
            x1, y1, x2, y2 = last
            flat += [x1, y1, x2 - x1, y2 - y1]
        viewer_tracks.append([track_id, CLASS_ORDER.index(track["class"]), track["frames"][0], track["crossing"],
                              plate_index.get(track_id, -1), flat])
    clock = scene.get("clock", {}).get("start")
    viewer = {
        "fps": fps, "step": VIEWER_STEP, "frames": len(frames), "width": raw["width"], "height": raw["height"],
        "classes": CLASS_ORDER, "light_states": LIGHT_STATES,
        "clock_start_s": sum(int(x) * m for x, m in zip(clock.split(":"), (3600, 60, 1))) if clock else None,
        "lines": [{"name": l["name"], "label": l.get("label", l["name"]), "from": l["from"], "to": l["to"],
                   "direction": l.get("direction", "right_to_left"), "plates": bool(l.get("plates"))} for l in lines],
        "tracks": viewer_tracks,
        "plates": viewer_plates,
        "crossings_by_class": [sorted(v["crossing"] for v in counted.values() if v["class"] == c) for c in CLASS_ORDER],
        "crossings_by_line": crossings,
        "phases": [[a, LIGHT_STATES.index(state)] for a, _, state in phases if state],
    }
    audit = {
        "read": {str(k): v for k, v in read_plates.items()},
        "counted": [{"track": k, "class": v["class"], "line": v["line"], "crossing": v["crossing"],
                     "box": v["boxes"][v["frames"].index(v["crossing"])]} for k, v in counted.items()],
    }
    texts = {v["text"] for v in votes.values()} | {r[1] for rs in reads.values() for r in rs}
    texts |= {p[6] for f in frames for p in f["plates"] if p[6]}  # including reads with no owner vehicle
    confident = sorted({p[6] for f in frames for p in f["plates"] if p[6] and p[7] and p[7] >= 0.8})
    return summary, viewer, audit, texts, confident


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene")
    parser.add_argument("out")
    args = parser.parse_args()
    out_dir = Path(args.out)
    summary, viewer, audit, texts, confident = summarize(load_scene(args.scene), read_raw(out_dir))
    ensure_no_plate_text(summary, texts)
    ensure_no_plate_text(viewer, texts)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "tracks.json").write_text(json.dumps(viewer, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    (out_dir / "texts.json").write_text(json.dumps({"all": sorted(texts), "confident": confident}, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"{summary['vehicles_counted']} vehicles counted across {summary['tracks']} tracks")
    for name, line in summary["counts"].items():
        print(f"  {name:<20} " + "  ".join(f"{k} {v}" for k, v in line.items() if k != "label" and v))
    plates = summary["plates"]
    print(f"plates: {plates['read']} read out of {plates['candidates']} candidates "
          f"({plates['with_plate_detected']} had a plate detected)")
    if summary["traffic_light"]["phases"]:
        print(f"traffic light: {summary['traffic_light']['full_cycles']} full cycles")
    print(f"ok -> {out_dir}/summary.json, tracks.json (publishable); audit.json and texts.json (SENSITIVE)")


if __name__ == "__main__":
    main()
