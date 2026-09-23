# Plateproof · vehicle counting, traffic-light phases and license-plate reading on CPU

[![tests](https://github.com/eucleberpaiva/plateproof/actions/workflows/tests.yml/badge.svg)](https://github.com/eucleberpaiva/plateproof/actions/workflows/tests.yml)

Point it at video from a fixed camera and get numbers out: vehicles counted by class and by direction,
traffic-light phases read from the lamps themselves, license plates read and then pixelated. Everything runs
on CPU through ONNX Runtime, with no GPU and no PyTorch. And before the video goes anywhere, two checks
attack it with the same plate reader to prove the plates are gone.

**This is a lab, not a product.** It ships no footage, no screenshot, no model weights and no results — you
bring your own video. Everything below was measured on one real scene: 7 minutes of a fixed camera at an
intersection, 12,750 frames, on a laptop with no GPU. The numbers include what it got wrong. Of the 53
vehicles the model called a truck, 8 were trucks. Four to six out of 168 were counted twice. How many it
missed entirely was never measured.

What step 2 prints at the end of that run:

```
316 vehicles counted across 606 tracks
  straight             car 137  truck 30  motorcycle 1
  oncoming             car 86  truck 12
  cross_westbound      car 39  truck 11
plates: 47 read out of 266 candidates (170 had a plate detected)
traffic light: 3 full cycles
ok -> out/summary.json, tracks.json (publishable); audit.json and texts.json (SENSITIVE)
```

[How it works, stage by stage](docs/how-it-works.md) · [The privacy method, and what it cannot
prove](docs/privacy.md)

## What it does

| Step | Command | What comes out |
|---|---|---|
| 0. Models | `python -m plateproof.models` | 3 ONNX models from their original source, SHA-256 verified |
| 1. Analyze | `python -m plateproof.analyze` | vehicles, tracks, plates and traffic lights, frame by frame, with the time each stage took |
| 2. Summarize | `python -m plateproof.summarize` | counts per line and class, light phases, plate read rate |
| 3. Hide plates | `python -m plateproof.privacy` | `public.mp4`, with the plates pixelated |
| 4. Check | `python -m plateproof.verify` and `python -m plateproof.sample` | an automatic attack on that video, plus whole frames to review by eye |
| View | `viewer/index.html` | video, boxes, trails, counting lines and live counters, in the browser |

## Measured on a real scene

Seven minutes of a fixed traffic camera at an intersection (12,750 frames, 1920x1080) on a laptop with an
Intel Core i7-13700H and no GPU. The calibration for that scene is in
[`scenes/example-intersection.toml`](scenes/example-intersection.toml).

- **316 vehicles counted** across 3 movements: 168 going straight, 98 coming the other way, 50 crossing.
- **3 full traffic light cycles** read from lens brightness, with no connection to the controller.
- **47 plates read** out of 266 passes where the plate faced the camera.
- **5.5 frames per second** with everything switched on. The two detectors take almost all of it: 75 ms
  (vehicles) and 88 ms (plates) per frame, median.

And what the manual review found in those numbers:

- **Plates:** 32 of the 47 were exactly right. 11 had one character wrong, nearly always an 8 read as a B,
  and 4 were unreadable even by eye.
- **Trucks:** of the 53 vehicles the model called a truck, 8 were trucks. The rest were 24 pickups, 13 vans,
  3 trailers and 5 SUVs. The model learned on COCO, where pickups and vans usually land in that class.
- **Double counts:** 4 to 6 of the 168 going straight were counted twice. Trailers come in separately from
  the vehicle towing them, and a pole in front of the camera makes the tracker switch IDs.
- **Not measured:** how many vehicles went by without being counted.

What drove the plate read rate was plate size in the image. This camera was installed to show the
intersection, and the plate rarely gets past 80 px wide; below 40 px the OCR does not even run, by
configuration. For a gate camera built to read plates, Axis asks for 130 px. Details in
[`docs/how-it-works.md`](docs/how-it-works.md).

## Stack

Computer vision on CPU, with no training and no fine-tuning: object detection (YOLOX-s), multi-object
tracking (ByteTrack), plate detection (YOLOv9-s) and OCR (a CCT model) — pre-trained deep-learning models
running inference on ONNX Runtime, wired together with OpenCV and NumPy. Python 3.12+, FFmpeg for the video
step, `unittest` on GitHub Actions for CI.

The accuracy numbers above are not a benchmark score. They are a hand-labelled evaluation against ground
truth, crop by crop, and the privacy check runs with a positive control on the original video. What was
never measured — how many vehicles went by uncounted — is stated as missing rather than rounded away.

## Install

You need Python 3.12 or newer (developed on 3.14) and [FFmpeg](https://ffmpeg.org) on your PATH, which only
step 3 uses.

```bash
git clone https://github.com/eucleberpaiva/plateproof.git
cd plateproof
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/macOS: source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m plateproof.models
python -m unittest -v
```

`--require-hashes` makes pip refuse any package whose hash is not the one pinned in `requirements.txt`.
`plateproof.models` does the same for the models: if the source swaps a file, the download fails instead of
running a different model. The tests need no video and no model.

## Run it on your own video

**1. Calibrate the scene.** Copy the example and adjust it to your camera:

```bash
cp scenes/example-intersection.toml scenes/my-scene.toml
python -m plateproof.sample my-video.mp4 out --n 3
```

Open the frames in `out/sheets/` in any image editor that shows the cursor position and write down the
coordinates: the counting lines, the box around each traffic light and the areas that are not plates (logo,
clock). Every field is explained inside the file. `scenes/*.toml` is git-ignored, so your calibration will
not end up in a commit by accident.

**2. Run it.** Try a few frames before feeding it the whole video:

```bash
python -m plateproof.analyze my-video.mp4 scenes/my-scene.toml out --max 300
python -m plateproof.summarize scenes/my-scene.toml out
```

**3. Watch it.** Open `viewer/index.html` in your browser and pick the video and `out/tracks.json`. Nothing
leaves your machine: the page makes no network request at all (`default-src 'none'`).

**4. Before you show the video to anybody:**

```bash
python -m plateproof.privacy my-video.mp4 scenes/my-scene.toml out
python -m plateproof.verify out/public.mp4 scenes/my-scene.toml out
python -m plateproof.sample out/public.mp4 out --n 24
```

`verify` exits with an error whenever it flags anything, by design: the verdict is yours, looking at
`out/sheets/verification.jpg` crop by crop.

Run it against your **original** video first, as a control. Zero flags means nothing until you have seen the
same command light up on unpixelated footage. On the example scene the control found 224 plate-shaped reads
in 400 frames, 142 of them within two characters of a real plate. The publishable video, all 12,750 frames of
it, returned zero. Then open every frame `sample` drew and look for a readable plate.

Why there are two checks, and what failed before them, is in [`docs/privacy.md`](docs/privacy.md).

## What is sensitive

| File | Contains | Publishable? |
|---|---|---|
| your source video | people and plates | no |
| `out/raw.json.gz` | text of every plate read, frame by frame | no |
| `out/crops/`, `out/audit.json`, `out/texts.json`, `out/sheets/` | plate images and text | no |
| `out/summary.json`, `out/tracks.json` | numbers, boxes and crossings, no plate text | yes, with the note below |
| `out/public.mp4` | video with plates pixelated; **faces are not handled** | only after step 4, and only if no face is recognizable |

`tracks.json` also carries what you typed into the scene file: the name and label of every counting line and
the wall clock of frame 0. Name a line after a street and you publish the location; keep `[clock]` and you
publish the time of day of the recording.

`summary.json` and `tracks.json` are checked before they are written: if any plate text shows up in them,
nothing is written. `out/`, videos, images and models are git-ignored, and a test fails if any of them ever
gets committed.

## Responsible use

A license plate can identify a person. Brazil has the LGPD, and other places have specific rules for
automatic plate reading. Before you point this at a camera:

- **Right to use the footage:** your own camera, with a notice to the people going by, or material under a
  license that allows it.
- **The minimum you need:** if all you want is the count, run `plateproof.analyze` with `--no-text`: plates are
  still detected so they can be pixelated, but no text and no crop is written. If you did read plates,
  delete `raw.json.gz`, `crops/`, `audit.json` and `texts.json` when the review is over.
- **Never publish the original.** Publish only what came through step 4.

It is here to learn from and to measure traffic with. The license does not allow anyone to break the law.

## Known limits

- **Brazilian plates:** the reader (`cct-xs-v2-global`) has not been tested against Mercosur plates, only
  against the US plates in this scene.
- **Speed:** 5.5 frames per second is not real time for 30 fps video. Every timing here was measured on
  onnxruntime 1.24.4; a different runtime version is a different measurement, so that one is pinned until
  it is re-measured.
- **Tracker:** ByteTrack was dropped from `supervision` in 0.31, which is why the version is pinned. The
  pinned version prints a `FutureWarning` about it on every run; nothing is broken.
- **Privacy:** the layer that covers undetected plates depends on the vehicle detector, so a vehicle and a
  plate missed in the same frame is not covered automatically, which is what the visual review is for. Faces
  are not handled, and neither is anything else that identifies somebody: the company name painted on a
  van, a bumper sticker, a school badge. Only the plate is treated as personal data here. The mosaic has
  not been tested against reconstruction attacks that combine many frames.

## How to contribute

It is a lab: issues are welcome, with no promise of a response time. A few things worth measuring:

- **Missed vehicles:** count a stretch by hand and compare it with the automatic count, the number this
  README is missing.
- **Tracker:** swap ByteTrack for a maintained one and compare the double counts.
- **Privacy:** an attack that combines several frames to try to reconstruct the mosaic, and a way to handle
  faces.

Run `python -m unittest` before sending a PR, and never attach video or images with a readable plate or
face, not even in an issue.

## Credits

- **Vehicle detection:** [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), Megvii, Apache-2.0. Trained
  on [COCO](https://cocodataset.org) (Lin et al., 2014), annotations under CC BY 4.0.
- **Plate detection:** [open-image-models](https://github.com/ankandrew/open-image-models), ankandrew, MIT.
  [YOLOv9](https://arxiv.org/abs/2402.13616) architecture (Wang, Yeh and Liao, 2024).
- **Plate reading:** [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) and
  [fast-alpr](https://github.com/ankandrew/fast-alpr), ankandrew, MIT.
- **Tracking:** ByteTrack (Zhang et al., 2022) through
  [supervision](https://github.com/roboflow/supervision), Roboflow, MIT.
- **Runtime:** [ONNX Runtime](https://onnxruntime.ai) (MIT), [OpenCV](https://opencv.org) (Apache-2.0),
  [FFmpeg](https://ffmpeg.org) (LGPL/GPL).

No model weights are redistributed here. Each one is downloaded from its original source, and it is worth
checking each model's license for your own use.

## License

Code under [MIT](LICENSE). © 2026 Cleber Paiva · [cleberpaiva.com.br](https://cleberpaiva.com.br)
