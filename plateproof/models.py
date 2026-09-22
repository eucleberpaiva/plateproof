"""Downloads the models into ./models and checks the SHA-256 of every file before use.

Fixed URL and fixed hash: if the source ever swaps a file, the download fails instead of silently running a
different model from the one these numbers were measured with. No model weights live in this repository;
each one comes from its original source.

    python -m plateproof.models
"""
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent / "models"

MODELS = {
    # vehicle and person detector · Megvii YOLOX-s · Apache-2.0
    "yolox_s.onnx": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    ),
    # plate detector · open-image-models (ankandrew) · MIT
    "yolo-v9-s-608-license-plates-end2end.onnx": (
        "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx",
        "2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a",
    ),
    # plate reader (OCR) · fast-plate-ocr (ankandrew) · MIT
    "cct_xs_v2_global.onnx": (
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.onnx",
        "8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44",
    ),
    "cct_xs_v2_global_plate_config.yaml": (
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global_plate_config.yaml",
        "0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_of(name):
    """Path of a model already downloaded and checked. Exits with instructions when missing or altered."""
    path = FOLDER / name
    if not path.exists():
        sys.exit(f"Model {name} not found. Run first: python -m plateproof.models")
    if sha256(path) != MODELS[name][1]:
        sys.exit(f"SHA-256 mismatch for {path}. Delete the file and run: python -m plateproof.models")
    return path


def download():
    FOLDER.mkdir(exist_ok=True)
    for name, (url, expected) in MODELS.items():
        target = FOLDER / name
        if target.exists() and sha256(target) == expected:
            print(f"ok         {name}")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        print(f"downloading {name}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as f:
                shutil.copyfileobj(response, f)
        except OSError as error:  # a stalled mirror must not hang step 0 forever
            partial.unlink(missing_ok=True)
            sys.exit(f"Download of {name} failed ({error}). Re-running this command is safe.")
        got = sha256(partial)
        if got != expected:
            partial.unlink()
            sys.exit(f"SHA-256 mismatch for {name}\n  expected {expected}\n  got      {got}")
        partial.replace(target)
        print(f"ok         {name}")


def load(scene, vehicles=True):
    """The three models ready for CPU inference, with the same settings the measurements used.

    vehicles=False skips YOLOX (step 4 only needs the plate reader, and this saves memory).
    """
    import onnxruntime as ort
    from fast_alpr.default_ocr import DefaultOCR
    from open_image_models.detection.core.yolo_v9.inference import YoloV9Detector

    from plateproof.yolox import YOLOX

    # Without this the three ONNX sessions spin idle threads fighting for the same cores (see docs/how-it-works.md).
    options = ort.SessionOptions()
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.enable_cpu_mem_arena = False  # the arena grows to the high-water mark and never gives it back
    detection = scene["detection"]
    vehicle_model = YOLOX(path_of("yolox_s.onnx"), sess_options=options) if vehicles else None
    plate_model = YoloV9Detector(path_of("yolo-v9-s-608-license-plates-end2end.onnx"),
                                 class_labels=("License Plate",), conf_thresh=detection["plate_conf"],
                                 providers=["CPUExecutionProvider"], sess_options=options)
    ocr = DefaultOCR(model_path=path_of("cct_xs_v2_global.onnx"),
                     config_path=path_of("cct_xs_v2_global_plate_config.yaml"),
                     device="cpu", sess_options=options)
    return vehicle_model, plate_model, ocr


if __name__ == "__main__":
    download()
