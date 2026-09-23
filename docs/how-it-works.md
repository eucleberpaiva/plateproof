# How it works

Every stage, why it is built that way and what was measured. The numbers come from the example scene:
7 minutes, 12,750 frames at 1920x1080, on a laptop CPU with no GPU. The absolute times below are worth
less than the ratios between them: the two detectors are 97% of every frame, and that holds on any CPU.

## 1. Detect vehicles · `plateproof/yolox.py`

YOLOX-s finds cars, trucks, buses, motorcycles, bicycles and people in every frame. It runs straight on ONNX
Runtime, without PyTorch, which keeps the install small and identical across systems.

The exported model hands back the raw grid. The code does the rest: resize the frame keeping its aspect
ratio, pad with grey (value 114, as in training), decode the grid into boxes and apply NMS across classes.
That NMS is there so the same car does not come out as "car" and "truck" at the same time.

**Measured:** 75 ms per frame, median.

## 2. Track · ByteTrack

Detection alone cannot count. In the example scene a vehicle stays on screen for 136 frames, median (4.5 s).
The tracker links boxes from one frame to the next so it carries a single ID and is counted once. A track
shorter than 20 frames (0.67 s) is treated as detector noise.

Each track's class is decided by a vote weighted by per-frame confidence. A car the detector called a truck
for two frames is still a car.

## 3. Count · lines on the scene

A vehicle is represented by the middle of the bottom edge of its box, the point where it touches the ground.
Each line in the scene is a segment with a direction, and a vehicle counts once, on the first line in the
list that it crosses.

"Crossing" has an exact definition (`core.crossing`). Picture someone walking from `from` to `to`. The line
counts whoever passes from that person's right-hand side to the left, and the point right after has to fall
within the length of the segment.

Crossing the line is not always enough. A car making a turn can clip the line meant for through traffic, so
each line may require a total displacement, measured from the start to the end of the track. In the example
scene, "going straight" requires rising at least 60 px on screen.

**Where it goes wrong, from the manual review:** 4 to 6 extra counts out of 168. A trailer comes in as a
separate vehicle, and a pole in front of the right-hand lane hides the car and makes the tracker switch IDs.

## 4. Read the traffic light · from the lamps

No integration with the controller: a box covers each lamp housing and the code measures the brightness of
each third of it, red on top, yellow in the middle, green at the bottom. A lit lens goes past 230 and sits at
least 50 above the second brightest. In the example scene, lit reads around 255 and unlit sits between 110
and 160.

Two details keep the reading stable:

- **Agreement:** when two lights control the same flow, the state only counts if both agree.
- **Half-second window:** the state of a frame is the majority over a window around it. A truck passing in
  front of the light, or a camera flicker, does not open a fake phase.

**Measured:** 3 full cycles. Green from 53.2 to 62.4 s, yellow 5 s, red from 44.3 to 59.5 s. The durations
line up between cycles, which is a good sign that the reading is not inventing phases.

## 5. Read plates · detector + OCR + vote

**Plate detector (YOLOv9-s-608).** Compared across the same 5 frames:

| Model | Time | Plates found per frame |
|---|---|---|
| t-384 | 18 ms | 1 |
| t-640 | 57 ms | 1.6 |
| s-608 | 64 ms | 3.2 |

The s-608 finds more than three times as many plates for 46 ms more.

**OCR (cct-xs-v2).** By configuration it only runs on plates at least 40 px wide. On the test plate it read
all 7 characters correctly; the bigger model, cct-s-v2, swapped a 9 for an H.

**Vote.** Every read of the same vehicle votes, character by character, weighted by the confidence of each
read. A plate only counts as read with 3 reads or more and 75% mean agreement.

**Measured:** 47 plates read out of 266 passes. Of those 47, 32 were exactly right in the visual review.

**What matters most is how big the plate is in the image:**

| Widest the plate got | Vehicles | Read |
|---|---|---|
| up to 40 px | 29 | OCR disabled |
| 40 to 60 | 54 | 8 |
| 60 to 80 | 72 | 33 |
| 80 to 100 | 11 | 3 |
| 100 or more | 4 | 3 |

In another 96 passes the plate was never detected at all, nearly always a car coming from far away. To read
plates for real, the camera has to be installed for it. Axis asks for a plate 130 px wide, an angle within
30° and a distance of 2 to 7 m at a gate
([guide](https://help.axis.com/en-us/axis-license-plate-verifier)). Hikvision asks for characters 20 to 30 px
tall and an exposure time of 1/250 to 1/500 s for vehicles between 30 and 60 km/h
([guide](https://www.hikvision.com/content/dam/hikvision/au/firmware/ids-2cd7a46g0-p-izhs(y)/7xxx-ANPR-Installation-&-Configuration-GuidanceNew.pdf)).

## Performance

In its final shape the whole video ran at 5.5 frames per second. It did not start there: the first 300-frame
test ran at 1.3 frames per second, and the processor was not busy with inference. Three ONNX Runtime
sessions were keeping threads spinning idle, waiting for work, and fighting for the same cores as the video
encoder. Two changes together took that test to 4.8 frames per second, still writing a video: switching off
that spin wait (`session.intra_op.allow_spinning = 0`) and limiting the encoder to 2 threads in the old
single-step pipeline. Today the analysis writes no video at all; that happens in the privacy step, so
there is no encoder left in this file to limit.

**Time per frame over the whole video, median:**

| Stage | ms |
|---|---|
| Decode | 4.3 |
| Detect vehicles | 75.0 |
| Track | 2.2 |
| Detect plates | 87.5 |
| Read plates (OCR) | 0.1 |

The OCR sits near zero at the median because it only runs when there is a plate big enough. Its p95 is 7.9 ms.
