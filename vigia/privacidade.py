"""Etapa 3 · gera o vídeo que pode ser mostrado, com placas escondidas. Não roda modelo: usa o bruto da etapa 1.

Três camadas, porque nenhum detector acerta sempre:
  1. toda placa detectada, com margem
  2. todo veículo com a caixa alta o bastante: a faixa inferior da caixa pixelada em TODO quadro, inclusive
     alguns quadros antes e depois do rastro existir. Cobre a maior parte das placas que o detector de placa perdeu.
  3. tarjas fixas da cena (ex.: data queimada no relógio da câmera)

Sobra um caso que nenhuma camada cobre: veículo E placa perdidos no mesmo quadro. Por isso a etapa 4 confere.
O histórico do que falhou antes de chegar aqui está em docs/privacidade.md.

    python -m vigia.privacidade video.mp4 cena.toml saida/ [--max 300]   ->  saida/publico.mp4
"""
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

from vigia.nucleo import carregar_cena, ler_bruto, pixelar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("cena")
    ap.add_argument("saida")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    cena = carregar_cena(args.cena)
    pv = cena["privacidade"]
    bruto = ler_bruto(args.saida)
    Q, fps, W, H = bruto["quadros"], bruto["fps"], bruto["w"], bruto["h"]
    if args.max:
        Q = Q[:args.max]
    ow = pv["largura_saida"]
    oh = round(H * ow / W / 2) * 2

    caixas, primeira, ultima = defaultdict(list), {}, {}
    for fi, q in enumerate(Q):
        for v in q["v"]:
            caixas[fi].append(v[2:6])
            primeira.setdefault(v[0], (fi, v[2:6]))
            ultima[v[0]] = (fi, v[2:6])
    for extremos, sinal in ((primeira, -1), (ultima, 1)):
        for fi, caixa in extremos.values():
            for k in range(1, pv["folga_quadros"] + 1):
                caixas[fi + sinal * k].append(caixa)

    destino = Path(args.saida) / "publico.mp4"
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "32", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(destino)],
        stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(args.video)
    n = {"placas": 0, "faixas": 0}
    fx1, fy1, fx2, fy2 = pv["faixa"]
    for fi, q in enumerate(Q):
        ok, fr = cap.read()
        if not ok:
            break
        for _, x1, y1, x2, y2, *_ in q["p"]:  # sem exceção para `ignorar`: pixelar um logotipo não custa nada
            mx, my = (x2 - x1) * 0.3, (y2 - y1) * 0.5
            pixelar(fr, x1 - mx, y1 - my, x2 + mx, y2 + my, pv["bloco"])
            n["placas"] += 1
        for vx1, vy1, vx2, vy2 in caixas[fi]:
            w, h = vx2 - vx1, vy2 - vy1
            if h >= pv["altura_min"]:
                pixelar(fr, vx1 + fx1 * w, vy1 + fy1 * h, vx1 + fx2 * w, vy1 + fy2 * h, pv["bloco"])
                n["faixas"] += 1
        for tarja in pv.get("tarjas", []):
            pixelar(fr, *tarja, pv["bloco"])
        enc.stdin.write(cv2.resize(fr, (ow, oh), interpolation=cv2.INTER_AREA).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg falhou (ele precisa estar no PATH)")
    print(f"ok: {n['placas']} placas e {n['faixas']} faixas pixeladas -> {destino}")
    print("próximo passo, antes de mostrar o vídeo a alguém: python -m vigia.verificar e python -m vigia.amostrar")


if __name__ == "__main__":
    main()
