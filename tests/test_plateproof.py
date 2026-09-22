"""Tests that need no video and no model:  python -m unittest -v"""
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

from plateproof.core import (assign_line, crossing, ensure_no_plate_text, light_phases, lit_lens, load_scene, pixelate,
                        plate_owner, vote_plate)
from plateproof.summarize import summarize

ROOT = Path(__file__).resolve().parent.parent
UPWARD_LINE = {"from": [0, 100], "to": [200, 100], "direction": "right_to_left"}


class Counting(unittest.TestCase):
    def test_counts_the_right_direction(self):
        self.assertEqual(crossing([(50, 150), (50, 90)], [0, 1], UPWARD_LINE), 1)

    def test_ignores_the_other_direction(self):
        self.assertIsNone(crossing([(50, 90), (50, 150)], [0, 1], UPWARD_LINE))

    def test_any_direction(self):
        self.assertEqual(crossing([(50, 90), (50, 150)], [0, 1], {**UPWARD_LINE, "direction": "any"}), 1)

    def test_outside_the_segment_does_not_count(self):
        self.assertIsNone(crossing([(300, 150), (300, 90)], [0, 1], UPWARD_LINE))

    def test_short_track_keeps_its_displacement(self):
        boxes = [(40, 140 - 20 * i, 60, 160 - 20 * i) for i in range(6)]  # foot point rises 100 px in 6 frames
        picky = {**UPWARD_LINE, "name": "up", "dy": [-np.inf, -50]}
        self.assertEqual(assign_line(boxes, list(range(6)), [picky])[0], "up")

    def test_zero_length_line_is_rejected(self):
        with self.assertRaises(ValueError):
            crossing([(50, 150), (50, 90)], [0, 1], {"from": [100, 100], "to": [100, 100]})

    def test_first_matching_line_wins_and_displacement_filters(self):
        boxes = [(40, 140 - 10 * i, 60, 160 - 10 * i) for i in range(10)]  # foot point rises from y=160 to y=70
        other = {**UPWARD_LINE, "name": "other"}
        picky = {**UPWARD_LINE, "name": "picky", "dy": [-np.inf, -500]}
        self.assertEqual(assign_line(boxes, list(range(10)), [picky, other])[0], "other")


class TrafficLight(unittest.TestCase):
    def test_a_lit_lens_has_to_stand_out(self):
        self.assertEqual(lit_lens([255, 130, 120]), "red")
        self.assertIsNone(lit_lens([255, 240, 120]))
        self.assertIsNone(lit_lens([200, 130, 120]))

    def test_window_absorbs_flicker_and_disagreement_yields_nothing(self):
        green, red = [120, 120, 255], [255, 120, 120]
        readings = [[green, green]] * 30 + [[red, green]] * 2 + [[green, green]] * 30
        self.assertEqual([p[2] for p in light_phases(readings, fps=30)], ["green"])


class Plates(unittest.TestCase):
    def test_per_character_vote(self):
        reads = [(0, "ABC1234", 0.9, 60), (1, "ABC1Z34", 0.7, 60), (2, "ABC1234", 0.9, 62)]
        voted = vote_plate(reads)
        self.assertEqual((voted["text"], voted["reads"]), ("ABC1234", 3))
        self.assertLess(voted["agreement"], 1.0)

    def test_reads_of_different_lengths_do_not_vote(self):
        self.assertIsNone(vote_plate([(0, "ABC12", 0.9, 60), (1, "ABCD123", 0.95, 60)]))

    def test_weak_read_does_not_vote(self):
        self.assertIsNone(vote_plate([(0, "ABC1234", 0.9, 60), (1, "ABC1234", 0.3, 60)]))

    def test_owner_is_the_smallest_box_holding_the_plate(self):
        self.assertEqual(plate_owner((45, 45, 55, 50), [(0, 0, 100, 100), (40, 40, 60, 60)], [1, 2]), 2)

    def test_safety_net_blocks_plate_text(self):
        with self.assertRaises(ValueError):
            ensure_no_plate_text({"labels": ["car ABC1234"]}, {"ABC1234"})
        ensure_no_plate_text({"coordinates": [1234, 5678]}, {"1234"})  # a loose number is not a published plate


class Privacy(unittest.TestCase):
    def test_mosaic_destroys_detail(self):
        img = np.zeros((48, 48, 3), np.uint8)
        img[::2, ::2] = 255  # fine texture, like characters
        pixelate(img, 0, 0, 48, 48, block=24)
        self.assertEqual(len(np.unique(img[:24, :24].reshape(-1, 3), axis=0)), 1)


class PublishableOutput(unittest.TestCase):
    def test_publishable_files_carry_no_plate_text(self):
        scene = load_scene(ROOT / "scenes" / "example-intersection.toml")
        frames = []
        for i in range(40):  # one car drives up across the "straight" line with its plate read in every frame
            y2 = 900 - i * 10
            frames.append({"vehicles": [[7, 2, 900, y2 - 150, 1100, y2, 0.9]],
                           "plates": [[7, 980, y2 - 40, 1060, y2 - 20, 0.9, "XYZ9876", 0.95]],
                           "lights": [[120, 120, 255]] * 3})
        raw = {"fps": 30, "width": 1920, "height": 1080, "frames": frames, "cpu_seconds": 1,
               "times": {"vehicles": [1.0] * 40}}
        summary, viewer, audit, texts, confident = summarize(scene, raw)
        self.assertEqual(summary["vehicles_counted"], 1)
        self.assertEqual(summary["plates"]["read"], 1)
        for publishable in (summary, viewer):
            self.assertNotIn("XYZ9876", json.dumps(publishable))
        self.assertIn("XYZ9876", json.dumps(audit))  # the audit file needs the text, which is why it is sensitive
        self.assertIn("XYZ9876", confident)  # confident read: this is the list step 4 compares against


class Repository(unittest.TestCase):
    def test_no_video_model_or_result_is_versioned(self):
        try:
            files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                                   check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not inside a git repository")
        blocked = (".mp4", ".mov", ".avi", ".mkv", ".onnx", ".json.gz")
        results = {"raw.json", "audit.json", "texts.json", "verification.json", "summary.json", "tracks.json"}
        self.assertEqual([f for f in files if f.lower().endswith(blocked)], [])
        self.assertEqual([f for f in files if f.split("/")[0] in ("out", "output", "crops", "sheets", "models")
                          or f.split("/")[-1] in results], [])
        # not one image: a frame of the footage is still the footage, and a contact sheet from plateproof.audit
        # has the plate text drawn onto the pixels, where ensure_no_plate_text cannot see it
        self.assertEqual([f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))], [])
        # nobody else's scene calibration: it carries coordinates of a real camera
        self.assertEqual([f for f in files if f.startswith("scenes/")
                          and f != "scenes/example-intersection.toml"], [])
        # nothing versioned here is heavier than 1 MB: this repository is text
        big = [f for f in files if (ROOT / f).exists() and (ROOT / f).stat().st_size > 1_000_000]
        self.assertEqual(big, [])

if __name__ == "__main__":
    unittest.main()
