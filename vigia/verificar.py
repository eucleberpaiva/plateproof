"""Etapa 4a · ataca o próprio vídeo público: o mesmo leitor de placa, com limiar baixo, tenta ler placas nele.

O vídeo é ampliado para a resolução original antes (o que alguém faria para tentar ler). Cada leitura com cara
de placa é comparada com as leituras confiáveis do original (confiança >= 0,8): distância de edição até 2 conta
como placa vazada. Palpite fraco do OCR não entra na comparação, senão qualquer letreiro da cena acusa falso.
Nada é pulado, nem as regiões `ignorar` da cena: uma placa pode passar atrás de um logotipo.
Os textos vêm de saida/textos.json (gerado pela etapa 2). Se não houver nenhum, o comando para com erro em vez de
dizer que passou.

Limite deste teste: ele tem o mesmo ponto cego do detector de placa. Uma placa que o detector nunca achou no
original também não é achada aqui. Por isso existe a etapa 4b (python -m vigia.amostrar), a olho.

    python -m vigia.verificar saida/publico.mp4 cena.toml saida/ [--passo 1]
    ->  saida/verificacao.json  e  saida/folhas/verificacao.jpg (recortes das leituras; SENSÍVEL se vazou algo)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vigia import modelos
from vigia.nucleo import carregar_cena


def distancia(a, b):
    linha = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        anterior, linha[0] = linha[:], i
        for j, cb in enumerate(b, 1):
            linha[j] = min(anterior[j] + 1, linha[j - 1] + 1, anterior[j - 1] + (ca != cb))
    return linha[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_publico")
    ap.add_argument("cena")
    ap.add_argument("saida")
    ap.add_argument("--passo", type=int, default=1, help="analisa 1 quadro a cada N (1 = todos)")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    cena = carregar_cena(args.cena)
    cena["deteccao"]["placa_conf"] = 0.2  # mais sensível que na análise: aqui achar demais é melhor que de menos
    saida = Path(args.saida)
    resumo = json.loads((saida / "resumo.json").read_text(encoding="utf-8"))
    W, H = resumo["video"]["largura"], resumo["video"]["altura"]
    reais = {t for t in json.loads((saida / "textos.json").read_text(encoding="utf-8"))["confiaveis"] if len(t) >= 4}
    if not reais:
        raise SystemExit("Nenhuma leitura de placa no original para comparar: esta verificação não prova nada aqui. "
                         "Use só a conferência a olho (python -m vigia.amostrar).")

    _, det_placa, ocr = modelos.carregar(cena, veiculos=False)
    cap = cv2.VideoCapture(args.video_publico)
    achados, analisados, fi = [], 0, 0
    while not args.max or fi < args.max:
        ok, fr = cap.read()
        if not ok:
            break
        if fi % args.passo == 0:
            analisados += 1
            grande = cv2.resize(fr, (W, H), interpolation=cv2.INTER_CUBIC)
            for p in det_placa.predict(grande):
                b = p.bounding_box
                x1, y1, x2, y2 = max(b.x1, 0), max(b.y1, 0), min(b.x2, W), min(b.y2, H)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                r = ocr.predict(grande[y1:y2, x1:x2])
                if not r or not r.text:
                    continue
                texto = r.text.strip().upper().replace("_", "")
                conf = float(np.mean(r.confidence)) if r.confidence else 0.0
                if conf >= 0.7 and 5 <= len(texto) <= 8:
                    perto = min((distancia(texto, t) for t in reais), default=99)
                    # guarda o recorte de tudo que foi marcado, e dos primeiros achados para conferir o resto
                    guardar = perto <= 2 or len(achados) < 24
                    recorte = grande[max(0, y1 - 10):y2 + 10, max(0, x1 - 20):x2 + 20].copy() if guardar else None
                    achados.append((fi, texto, perto, recorte))
        fi += 1

    vazadas = [a for a in achados if a[2] <= 2]
    (saida / "folhas").mkdir(exist_ok=True)
    guardados = [a for a in sorted(achados, key=lambda a: a[2]) if a[3] is not None][:48]
    if guardados:
        folha = np.full((-(-len(guardados) // 6) * 110, 6 * 260, 3), 20, np.uint8)
        for i, (f, texto, perto, rec) in enumerate(guardados):
            r, c = divmod(i, 6)
            s = min(250 / rec.shape[1], 80 / rec.shape[0])
            rec = cv2.resize(rec, (max(1, int(rec.shape[1] * s)), max(1, int(rec.shape[0] * s))))
            folha[r * 110:r * 110 + rec.shape[0], c * 260:c * 260 + rec.shape[1]] = rec
            cv2.putText(folha, f"q{f} distancia {perto}", (c * 260 + 4, r * 110 + 100), 0, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(saida / "folhas" / "verificacao.jpg"), folha)
    resultado = {"quadros_analisados": analisados, "leituras_com_cara_de_placa": len(achados),
                 "parecidas_com_placa_real": len(vazadas), "distancia_minima": min((a[2] for a in achados), default=None),
                 "quadros_marcados": sorted({a[0] for a in vazadas})}  # só o número do quadro: o texto não é gravado
    (saida / "verificacao.json").write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    print(json.dumps(resultado, indent=1))
    if vazadas:
        raise SystemExit(f"FALHOU: {len(vazadas)} leituras parecidas com placa real. Não publique; veja folhas/verificacao.jpg")
    print("passou. Falta a conferência a olho: python -m vigia.amostrar")


if __name__ == "__main__":
    main()
