"""Etapa 4b · sorteia quadros INTEIROS de um vídeo para conferência a olho.

Quadro inteiro, e não recorte da área pixelada: recorte centrado no pixelado só prova que o pixelado está
borrado, nunca que a placa está dentro dele. Abra cada imagem e procure qualquer placa legível.
Também serve para achar coordenadas ao calibrar uma cena nova (abra num editor que mostre a posição do cursor).

    python -m vigia.amostrar video.mp4 saida/ [--n 24] [--quadro 4500 --quadro 9000] [--semente 16]
    ->  saida/folhas/<nome do vídeo>/quadro_NNNNN.jpg
"""
import argparse
import random
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("saida")
    ap.add_argument("--n", type=int, default=24, help="quantos quadros sortear")
    ap.add_argument("--quadro", type=int, action="append", default=[], help="quadro que entra sempre (ex.: o da capa)")
    ap.add_argument("--semente", type=int, default=16, help="mesma semente, mesmos quadros")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random.seed(args.semente)
    alvo = sorted(set(random.sample(range(total), min(args.n, total))) | set(args.quadro))
    pasta = Path(args.saida) / "folhas" / Path(args.video).stem  # original e público nunca na mesma pasta
    pasta.mkdir(parents=True, exist_ok=True)
    fi = 0
    for q in alvo:
        while fi < q:  # ler em sequência é mais confiável que pular para o quadro em H.264
            cap.grab()
            fi += 1
        ok, fr = cap.read()
        fi += 1
        if not ok:
            break
        cv2.imwrite(str(pasta / f"quadro_{q:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{len(alvo)} quadros -> {pasta}")


if __name__ == "__main__":
    main()
