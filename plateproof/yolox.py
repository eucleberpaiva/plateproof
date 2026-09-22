"""YOLOX vehicle detector (Megvii, Apache-2.0) on ONNX Runtime / CPU, without PyTorch.

The exported model returns the raw grid; the pre-processing (resize keeping the aspect ratio and pad with
grey 114, as in training), the grid decoding and the NMS live here.
"""
import cv2
import numpy as np
import onnxruntime as ort

from plateproof.core import COCO_CLASSES as CLASSES  # only the classes that show up at an intersection


class YOLOX:
    def __init__(self, path, size=640, sess_options=None):
        options = sess_options or ort.SessionOptions()
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = size
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = size // stride
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def __call__(self, img, conf=0.35, nms=0.45):
        h, w = img.shape[:2]
        ratio = min(self.size / h, self.size / w)
        padded = np.full((self.size, self.size, 3), 114, np.uint8)
        padded[: int(h * ratio), : int(w * ratio)] = cv2.resize(
            img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        batch = padded.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(None, {self.input_name: batch})[0][0]
        out[:, :2] = (out[:, :2] + self.grid) * self.stride
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.stride
        scores = out[:, 5:] * out[:, 4:5]
        wanted = np.array(list(CLASSES))
        subset = scores[:, wanted]
        best = subset.argmax(1)
        score = subset[np.arange(len(subset)), best]
        keep = score >= conf
        if not keep.any():
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        boxes, score, classes = out[keep, :4] / ratio, score[keep], wanted[best[keep]]
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
        # class-agnostic NMS: one car must not come out as car and truck in the same box
        xywh = np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]
        # NMSBoxes filters with a strict >, so a detection scoring exactly `conf` comes back as an empty
        # float array: without the dtype it would index the boxes with floats and raise
        idx = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), conf, nms), dtype=int).reshape(-1)
        return xyxy[idx], score[idx], classes[idx]
