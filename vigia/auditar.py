"""Conferência manual dos números: folhas de imagem para olhar e anotar (SENSÍVEL: mostra placas).

  placas.jpg          recorte de cada placa aceita + texto votado: confira caractere por caractere
  classe_<nome>.jpg   todo veículo contado de uma classe: veja o que ele é de fato
  linha_<nome>.jpg    todo cruzamento de uma linha, em ordem: procure o mesmo veículo contado duas vezes

    python -m vigia.auditar video.mp4 saida/ [--classe caminhao] [--linha frente]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def folha(itens, destino, colunas, larg, alt, por_folha):
    for k in range(0, len(itens), por_folha):
        parte = itens[k:k + por_folha]
        img = np.full((-(-len(parte) // colunas) * (alt + 24), colunas * larg, 3), 20, np.uint8)
        for i, (rec, rotulo) in enumerate(parte):
            r, c = divmod(i, colunas)
            s = min(larg / rec.shape[1], alt / rec.shape[0])
            rec = cv2.resize(rec, (max(1, int(rec.shape[1] * s)), max(1, int(rec.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
            y, x = r * (alt + 24), c * larg
            img[y:y + rec.shape[0], x:x + rec.shape[1]] = rec
            cv2.putText(img, rotulo, (x + 4, y + alt + 17), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(destino.with_name(f"{destino.stem}_{k // por_folha + 1}.jpg")), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("saida")
    ap.add_argument("--classe", action="append", default=[])
    ap.add_argument("--linha", action="append", default=[])
    args = ap.parse_args()
    saida = Path(args.saida)
    pasta = saida / "folhas"
    pasta.mkdir(exist_ok=True)
    a = json.loads((saida / "auditoria.json").read_text(encoding="utf-8"))

    recortes = {int(p.stem.split("_")[0]): p for p in (saida / "recortes").glob("*.png")}
    placas = [(cv2.imread(str(recortes[int(tid)])), f"#{int(tid)} {v['texto']}")
              for tid, v in sorted(a["lidas"].items(), key=lambda kv: kv[1]["quadro_confirmado"]) if int(tid) in recortes]
    if placas:
        folha(placas, pasta / "placas.jpg", 4, 300, 110, 24)

    escolhidos = [c for c in a["contados"] if c["classe"] in args.classe or c["linha"] in args.linha]
    precisa = {}
    for c in escolhidos:
        precisa.setdefault(c["cruza"], []).append(c)
    cap, recs, fi = cv2.VideoCapture(args.video), {}, 0
    for alvo in sorted(precisa):
        while fi < alvo:
            cap.grab()
            fi += 1
        ok, fr = cap.read()
        fi += 1
        if not ok:
            break
        for c in precisa[alvo]:
            x1, y1, x2, y2 = c["caixa"]
            recs[c["tid"]] = fr[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10].copy()
    for classe in args.classe:
        itens = [(recs[c["tid"]], f"#{c['tid']} {c['linha']} q{c['cruza']}") for c in escolhidos if c["classe"] == classe and c["tid"] in recs]
        folha(itens, pasta / f"classe_{classe}.jpg", 6, 220, 160, 36)
    for linha in args.linha:
        itens = [(recs[c["tid"]], f"#{c['tid']} q{c['cruza']} x{(c['caixa'][0] + c['caixa'][2]) // 2}")
                 for c in sorted(escolhidos, key=lambda c: c["cruza"]) if c["linha"] == linha and c["tid"] in recs]
        folha(itens, pasta / f"linha_{linha}.jpg", 8, 170, 120, 48)
    print(f"ok: {len(placas)} placas, {len(escolhidos)} veículos -> {pasta}")


if __name__ == "__main__":
    main()
