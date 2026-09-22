# Privacy: hiding plates, and proving they are hidden

Pixelating the plate the detector found is the easy part. The hard part is the plate it missed, and knowing
whether any got through. This is the method as it stands, the attempts that failed before it, and what the
checks cannot prove.

## The method · `plateproof/privacy.py`

**Three layers:**

1. **Detected plate:** pixelated with margin, at any confidence.
2. **Lower band of every nearby vehicle:** applies to a vehicle box 100 px tall or more. The band runs from
   30% of the box height down to the bottom, full width, and is pixelated in every frame, including 5 frames
   before and 5 after the track exists, because the tracker takes a few frames to confirm a vehicle. In the
   example scene, 99% of detected plates start below 38% of the vehicle box height, and 99% of the vehicles
   with a detected plate had a box 133 px tall or taller.
3. **Fixed masks:** for example the date burned into the camera clock.

A plate detected inside an `ignore` region of the scene (logo, clock) is neither read nor counted, but it is
pixelated all the same: a real plate can drive through that area too.

The mosaic uses 24 px cells in the original (16 px in the 1280-wide output). In the example scene a whole
plate is 22 px tall at the median and 39 px at p99, so it becomes one or two cells: unreadable, and much
cleaner than giant blocks.

**What is left over:**

- **Vehicle and plate missed in the same frame:** no layer covers it, which is what the visual review is for.
- **Reconstruction across frames:** the mosaic travels with the vehicle, and no attack that combines frames
  to rebuild what is underneath has been tried here.
- **Faces:** not handled.
- **Everything else that identifies somebody:** the company name painted on a van, a bumper sticker, a
  badge on a window. Only the plate is treated as personal data here. In the example scene a commercial
  van goes by with the business name readable on its side, plate pixelated.

## Attempts that failed

| Attempt | How it failed | Found by |
|---|---|---|
| Pixelate only the detected plate | The detector misses frames and whole vehicles. The cover frame showed a clean plate. | review of the cover image |
| Remember where the plate sat inside the box and keep pixelating there | On a half-hidden vehicle the box changes shape and the remembered spot lands in the wrong place. | by eye, on one frame |
| Skip the band when the vehicle already had a plate detected | A box truck had two plate-shaped areas. The detector found the wrong one and the real plate got through. | the automatic attack |
| Estimated zone across the box width (10% to 90%) | The box caught a car and a trailer together, and the zone landed on the trailer. | review of the cover image |

## Check 1 · attack your own video · `plateproof/verify.py`

The same plate reader, more sensitive than in the analysis (threshold 0.2), runs over the shareable video,
skipping nothing. The video is upscaled back to the original resolution first, which is what somebody would
do to try to read it. Every plate-shaped read is compared against the confident reads from the original
(confidence 0.8 or above, 4 characters or more), and an edit distance of 2 or less counts as a leak. If the
original has no read to compare against, the command stops with an error instead of claiming it passed.

The command exits with an error whenever it flags anything, by design. The verdict is yours, looking at
`sheets/verification.jpg` crop by crop.

**Result on the example scene:** across all 12,750 frames the reader returned 8,911 plate-shaped texts,
nearly all of them a street name sign burned into the corner of the image. Five got within two characters of a
confident read from the original, and the five crops show that same sign, half covered. No vehicle plate
came out readable.

**Blind spots:** the attack uses the same plate detector as the analysis, so a plate it never found in the
original is not found here either. And a plate whose original read fell below 0.8 confidence is not in the
comparison list, so a leak of that plate is not flagged. Passing this check proves nothing about those two
cases.

## Check 2b · faces, by eye

Faces are not pixelated, so the least you can do is look. The detector tracked 51 people and bicycles in the
example scene. Sorted by box height, the twelve biggest were reviewed crop by crop: eight of them are the
pedestrian signal head being read as a person, over and over. The other four show a truck mirror, a backlit
silhouette in dark clothing, and somebody from behind. No identifiable face. The tallest person box in the
whole video is 310 px, which puts a face at roughly 30 px across.

That is a review, not a guarantee, and it says nothing about a scene shot closer to the pavement.

## Check 2 · whole frames, by eye · `plateproof/sample.py`

A draw with a fixed seed, plus any frame you always want in, such as the cover image. You look at the
**whole frame**, hunting for any readable plate.

An earlier version drew crops of the pixelated area itself. It passed with 72 crops and no readable plate,
and still proved nothing: a crop centred on the mosaic only shows the mosaic is blurred, never that the plate
is inside it. A zone in the wrong place never made it into the sample.

**Result on the example scene:** 25 whole frames, cover image included, no readable plate.

## What never leaves your machine

Plate text never reaches the publishable files. `summarize.py` only writes `summary.json` and `tracks.json`
after checking that no text the OCR read (4 characters or more) shows up in them. The viewer shows "plate read" with the number of
reads and the confidence, and nothing else.
