"""Baixa os modelos para ./modelos e confere o SHA-256 de cada arquivo antes de usar.

Endereço e hash fixos: se a fonte trocar o arquivo, o download falha em vez de rodar um modelo diferente
do que foi medido. Nenhum peso de modelo fica dentro do repositório; cada um vem da fonte original.

    python -m vigia.modelos
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

PASTA = Path(__file__).resolve().parent.parent / "modelos"

MODELOS = {
    # detector de veículos e pessoas · Megvii YOLOX-s · Apache-2.0
    "yolox_s.onnx": (
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    ),
    # detector de placa · open-image-models (ankandrew) · MIT
    "yolo-v9-s-608-license-plates-end2end.onnx": (
        "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx",
        "2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a",
    ),
    # leitor de placa (OCR) · fast-plate-ocr (ankandrew) · MIT
    "cct_xs_v2_global.onnx": (
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.onnx",
        "8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44",
    ),
    "cct_xs_v2_global_plate_config.yaml": (
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global_plate_config.yaml",
        "0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6",
    ),
}


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def caminho(nome):
    """Caminho de um modelo já baixado e conferido. Falha com instrução se faltar ou não bater."""
    p = PASTA / nome
    if not p.exists():
        sys.exit(f"Modelo {nome} não encontrado. Rode antes: python -m vigia.modelos")
    if sha256(p) != MODELOS[nome][1]:
        sys.exit(f"SHA-256 de {p} não confere. Apague o arquivo e rode: python -m vigia.modelos")
    return p


def baixar():
    PASTA.mkdir(exist_ok=True)
    for nome, (url, esperado) in MODELOS.items():
        destino = PASTA / nome
        if destino.exists() and sha256(destino) == esperado:
            print(f"ok      {nome}")
            continue
        temp = destino.with_suffix(destino.suffix + ".parcial")
        print(f"baixando {nome}")
        urllib.request.urlretrieve(url, temp)
        obtido = sha256(temp)
        if obtido != esperado:
            temp.unlink()
            sys.exit(f"SHA-256 não confere para {nome}\n  esperado {esperado}\n  obtido   {obtido}")
        temp.replace(destino)
        print(f"ok      {nome}")


def carregar(cena, veiculos=True):
    """Os três modelos prontos para inferência em CPU, com a mesma configuração usada nas medições.

    veiculos=False pula o YOLOX (a etapa 4 só precisa do leitor de placa e economiza memória).
    """
    import onnxruntime as ort
    from fast_alpr.default_ocr import DefaultOCR
    from open_image_models.detection.core.yolo_v9.inference import YoloV9Detector

    from vigia.yolox import YOLOX

    # Sem isto as três sessões ONNX giram threads ociosas disputando os mesmos núcleos (ver docs/como-funciona.md).
    so = ort.SessionOptions()
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    d = cena["deteccao"]
    veiculos = YOLOX(caminho("yolox_s.onnx"), sess_options=so) if veiculos else None
    placa = YoloV9Detector(caminho("yolo-v9-s-608-license-plates-end2end.onnx"), class_labels=("License Plate",),
                           conf_thresh=d["placa_conf"], providers=["CPUExecutionProvider"], sess_options=so)
    ocr = DefaultOCR(model_path=caminho("cct_xs_v2_global.onnx"), config_path=caminho("cct_xs_v2_global_plate_config.yaml"),
                     device="cpu", sess_options=so)
    return veiculos, placa, ocr


if __name__ == "__main__":
    baixar()
