"""Detector de veículos YOLOX (Megvii, Apache-2.0) em ONNX Runtime/CPU, sem PyTorch.

O modelo exportado entrega a grade crua; aqui ficam o pré-processamento (redimensiona mantendo proporção e
completa com cinza 114, como no treino), a decodificação da grade e o NMS.
"""
import cv2
import numpy as np
import onnxruntime as ort

from vigia.nucleo import CLASSES_COCO as CLASSES  # só as classes que passam num cruzamento


class YOLOX:
    def __init__(self, path, size=640, sess_options=None):
        so = sess_options or ort.SessionOptions()
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size = size
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def __call__(self, img, conf=0.35, nms=0.45):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        x = pad.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0][0]
        out[:, :2] = (out[:, :2] + self.grid) * self.stride
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.stride
        cls_scores = out[:, 5:] * out[:, 4:5]
        keep_cls = np.array(list(CLASSES))
        sub = cls_scores[:, keep_cls]
        ci = sub.argmax(1)
        sc = sub[np.arange(len(sub)), ci]
        m = sc >= conf
        if not m.any():
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        b, sc, cls = out[m, :4] / r, sc[m], keep_cls[ci[m]]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        # NMS agnóstico de classe: um carro não vira carro+caminhão na mesma caixa
        xywh = np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), conf, nms)
        idx = np.array(idx).reshape(-1)
        return xyxy[idx], sc[idx], cls[idx]
