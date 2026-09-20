"""Etapa 1 · um passe pelo vídeo, tudo em CPU, medindo o tempo de cada etapa.

Por quadro: detecta veículos (YOLOX-s), rastreia (ByteTrack), detecta placas (YOLOv9), lê as placas grandes o
bastante (OCR) e mede o brilho das lentes dos semáforos. Nada é desenhado aqui: o vídeo público sai da etapa 3.

Saída em <saida>/ (SENSÍVEL, nunca publicar):
  bruto.json.gz   tudo quadro a quadro, inclusive o TEXTO das placas
  recortes/       melhor recorte de placa por veículo, para a conferência manual

Com --sem-texto o OCR não roda: a placa ainda é detectada (a etapa 3 precisa saber onde pixelar), mas nenhum
texto de placa nem recorte é gravado. Use quando você só precisa contar.

    python -m vigia.analisar video.mp4 cena.toml saida/ [--max 300] [--sem-texto]
"""
import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from vigia import modelos
from vigia.nucleo import brilho_lentes, carregar_cena, dentro, dono


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("cena")
    ap.add_argument("saida")
    ap.add_argument("--max", type=int, default=0, help="processa só os N primeiros quadros (teste rápido)")
    ap.add_argument("--sem-texto", action="store_true", help="não lê placas: nenhum texto de placa é gravado")
    args = ap.parse_args()

    cena = carregar_cena(args.cena)
    d = cena["deteccao"]
    saida = Path(args.saida)
    (saida / "recortes").mkdir(parents=True, exist_ok=True)
    det_veiculos, det_placa, ocr = modelos.carregar(cena)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max:
        total = min(total, args.max)

    # ByteTrack saiu do supervision a partir da 0.31: por isso a versão fica travada em requirements.txt.
    tracker = sv.ByteTrack(track_activation_threshold=d["veiculos_conf"], lost_track_buffer=int(fps * 2),
                           minimum_matching_threshold=0.8, frame_rate=int(round(fps)), minimum_consecutive_frames=3)

    quadros, tempos, melhor_recorte = [], defaultdict(list), {}
    inicio = time.perf_counter()
    for fi in range(total):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()

        xyxy, conf, cls = det_veiculos(frame, conf=d["veiculos_conf"])
        t2 = time.perf_counter()

        dets = sv.Detections(xyxy=xyxy.astype(np.float32), confidence=conf.astype(np.float32), class_id=cls.astype(int))
        dets = tracker.update_with_detections(dets)
        t3 = time.perf_counter()

        placas = det_placa.predict(frame)
        t4 = time.perf_counter()

        registro = []
        for p in placas:
            b = p.bounding_box
            x1, y1, x2, y2 = max(b.x1, 0), max(b.y1, 0), min(b.x2, W), min(b.y2, H)
            texto, conf_ocr = None, None
            if dentro(((x1 + x2) / 2, (y1 + y2) / 2), d["ignorar"]):
                # logotipo ou relógio costuma virar "placa": não lê nem conta, mas grava a caixa para a etapa 3
                # pixelar, porque uma placa de verdade também pode passar ali
                registro.append([None, x1, y1, x2, y2, round(float(p.confidence), 3), None, None])
                continue
            tid = dono((x1, y1, x2, y2), dets.xyxy, dets.tracker_id) if len(dets) else None
            if not args.sem_texto and p.confidence >= d["ocr_placa_conf"] and x2 - x1 >= d["ocr_largura_min"]:
                r = ocr.predict(frame[y1:y2, x1:x2])
                if r and r.text:
                    texto = r.text.strip().upper().replace("_", "")
                    conf_ocr = float(np.mean(r.confidence)) if r.confidence else 0.0
                    nota = conf_ocr * (x2 - x1)
                    if tid is not None and texto and nota > melhor_recorte.get(tid, (0,))[0]:
                        melhor_recorte[tid] = (nota, frame[max(0, y1 - 8):y2 + 8, max(0, x1 - 12):x2 + 12].copy())
            registro.append([tid, x1, y1, x2, y2, round(float(p.confidence), 3), texto,
                             None if conf_ocr is None else round(conf_ocr, 3)])
        t5 = time.perf_counter()

        semaforos = [brilho_lentes(frame, s["caixa"]) for s in cena.get("semaforo", [])]
        t6 = time.perf_counter()

        for nome, dt in (("decodificar", t1 - t0), ("veiculos", t2 - t1), ("rastreio", t3 - t2),
                         ("placas", t4 - t3), ("ocr", t5 - t4), ("semaforo", t6 - t5)):
            tempos[nome].append(round(dt * 1000, 2))
        quadros.append({
            "v": [[int(t), int(c), *map(int, b), round(float(s), 3)]
                  for b, s, c, t in zip(dets.xyxy, dets.confidence, dets.class_id, dets.tracker_id)] if len(dets) else [],
            "p": registro,
            "s": semaforos,
        })
        if (fi + 1) % 300 == 0:
            ritmo = (fi + 1) / (time.perf_counter() - inicio)
            print(f"{fi + 1}/{total}  {ritmo:.1f} quadros/s  faltam ~{(total - fi - 1) / ritmo / 60:.1f} min", flush=True)

    for tid, (_, recorte) in melhor_recorte.items():  # sem a placa no nome: nome de arquivo é indexado e sincronizado
        cv2.imwrite(str(saida / "recortes" / f"{tid:05d}.png"), recorte)
    duracao = time.perf_counter() - inicio
    with gzip.open(saida / "bruto.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"fps": fps, "w": W, "h": H, "quadros": quadros, "tempos": tempos,
                   "semaforos": [s["nome"] for s in cena.get("semaforo", [])], "cpu_total_s": round(duracao, 1)}, f)
    print(f"ok: {len(quadros)} quadros em {duracao / 60:.1f} min ({len(quadros) / duracao:.1f} quadros/s) -> {saida}")


if __name__ == "__main__":
    main()
