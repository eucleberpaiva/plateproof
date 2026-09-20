"""Regras do Vigia que não dependem de modelo: cena, contagem por linha, semáforo, voto de placa, pixelado.

Tudo aqui é função pura (entra dado, sai dado) e tem teste em tests/test_vigia.py.
"""
import json
import tomllib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES_COCO = {0: "pessoa", 1: "bicicleta", 2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}
ORDEM = ["carro", "caminhao", "onibus", "moto", "bicicleta", "pessoa"]
VEICULOS = {"carro", "caminhao", "onibus", "moto"}
ESTADOS = ["vermelho", "amarelo", "verde"]


def carregar_cena(caminho):
    with open(caminho, "rb") as f:
        return tomllib.load(f)


# ── contagem ───────────────────────────────────────────────────────────────────

def pe(caixa):
    """Ponto que representa o veículo no chão: meio da base da caixa."""
    x1, y1, x2, y2 = caixa
    return ((x1 + x2) / 2, y2)


def lado(p, a, b):
    """> 0: p está à direita de quem anda de a para b, olhando a tela (y cresce para baixo)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def cruzamento(pontos, quadros, linha):
    """Quadro em que a trajetória cruza a linha, ou None.

    sentido "direita_para_esquerda": passa de lado > 0 para lado <= 0 (de quem anda de `de` para `ate`).
    sentido "qualquer": troca de lado em qualquer direção.
    O ponto depois do cruzamento precisa cair dentro do comprimento do segmento.
    """
    a, b = linha["de"], linha["ate"]
    comp2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    qualquer = linha.get("sentido", "direita_para_esquerda") == "qualquer"
    for p0, p1, f in zip(pontos, pontos[1:], quadros[1:]):
        s0, s1 = lado(p0, a, b), lado(p1, a, b)
        cruzou = (s0 * s1 <= 0 and s0 != s1) if qualquer else (s0 > 0 >= s1)
        if not cruzou:
            continue
        t = ((p1[0] - a[0]) * (b[0] - a[0]) + (p1[1] - a[1]) * (b[1] - a[1])) / comp2
        if 0 < t <= 1:
            return f
    return None


def movimento(caixas, quadros, linhas):
    """(nome da linha, quadro do cruzamento) da primeira linha que a trajetória cruza, na ordem da cena.

    Cada linha pode exigir um deslocamento total (fim menos início, mediana de 5 pontos em cada ponta),
    em intervalos abertos: desloc_x = [min, max], desloc_y = [min, max]. Isso separa, por exemplo,
    quem segue em frente de quem só atravessa a linha fazendo a curva.
    """
    pontos = [pe(c) for c in caixas]
    dx, dy = np.median(pontos[-5:], axis=0) - np.median(pontos[:5], axis=0)
    for linha in linhas:
        lx, ly = linha.get("desloc_x", [-np.inf, np.inf]), linha.get("desloc_y", [-np.inf, np.inf])
        if not (lx[0] < dx < lx[1] and ly[0] < dy < ly[1]):
            continue
        f = cruzamento(pontos, quadros, linha)
        if f is not None:
            return linha["nome"], f
    return None, -1


# ── semáforo ───────────────────────────────────────────────────────────────────

def brilho_lentes(frame, caixa):
    """Brilho (percentil 90 do canal V) de cada terço da caixa: vermelho, amarelo, verde."""
    x, y, w, h = caixa
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    terco = h // 3
    return [round(float(np.percentile(hsv[k * terco:(k + 1) * terco, :, 2], 90)), 1) for k in range(3)]


def lente_acesa(brilhos, acesa=230, margem=50):
    """Estado de um semáforo num quadro, ou None se nenhuma lente se destaca."""
    i = int(np.argmax(brilhos))
    ordenado = sorted(brilhos)
    return ESTADOS[i] if brilhos[i] >= acesa and brilhos[i] - ordenado[-2] >= margem else None


def fases_semaforo(leituras, fps, acesa=230, margem=50):
    """leituras[quadro] = lista de brilhos dos semáforos que mandam no fluxo.

    Estado do quadro = consenso entre os semáforos que leram algo; depois, maioria numa janela de ~0,5 s,
    que ignora oclusão (um caminhão passando na frente) e piscada. Devolve [[quadro_ini, quadro_fim, estado]].
    """
    bruto = []
    for q in leituras:
        estados = {e for e in (lente_acesa(b, acesa, margem) for b in q) if e}
        bruto.append(estados.pop() if len(estados) == 1 else None)
    meia = int(fps * 0.25)
    atual = next((e for e in bruto if e), None)
    fases = []
    for i in range(len(bruto)):
        janela = [e for e in bruto[max(0, i - meia):i + meia + 1] if e]
        if janela:
            atual = Counter(janela).most_common(1)[0][0]
        if not fases or fases[-1][2] != atual:
            fases.append([i, i, atual])
        fases[-1][1] = i
    return fases


# ── placas ─────────────────────────────────────────────────────────────────────

def dono(placa, caixas, ids):
    """Veículo dono da placa: a menor caixa que contém o centro da placa."""
    cx, cy = (placa[0] + placa[2]) / 2, (placa[1] + placa[3]) / 2
    melhor, area = None, float("inf")
    for (x1, y1, x2, y2), tid in zip(caixas, ids):
        a = (x2 - x1) * (y2 - y1)
        if x1 <= cx <= x2 and y1 <= cy <= y2 and a < area:
            melhor, area = int(tid), a
    return melhor


def votar(leituras, conf_min=0.6, tamanho=(5, 8)):
    """Voto por posição entre as leituras de OCR do mesmo veículo.

    leituras = [(quadro, texto, confianca, largura_px)]. Devolve None com menos de 2 leituras boas.
    `acordo` = média, por posição, do peso do caractere vencedor: 1,0 é unanimidade.
    """
    boas = [l for l in leituras if l[2] >= conf_min and tamanho[0] <= len(l[1]) <= tamanho[1]]
    if len(boas) < 2:
        return None
    por_tamanho = Counter()
    for _, t, c, _ in boas:
        por_tamanho[len(t)] += c
    n = por_tamanho.most_common(1)[0][0]
    boas = [l for l in boas if len(l[1]) == n]
    texto, acordo = "", []
    for i in range(n):
        votos = Counter()
        for _, t, c, _ in boas:
            votos[t[i]] += c
        ch, peso = votos.most_common(1)[0]
        texto += ch
        acordo.append(peso / sum(votos.values()))
    iguais = [f for f, t, _, _ in boas if t == texto]
    return {"texto": texto, "n": len(boas), "acordo": float(np.mean(acordo)),
            "conf": float(np.mean([c for _, _, c, _ in boas])),
            "quadro_confirmado": iguais[1] if len(iguais) >= 2 else boas[1][0],
            "largura": int(max(l[3] for l in boas))}


def _strings(dado):
    if isinstance(dado, str):
        yield dado
    elif isinstance(dado, dict):
        for k, v in dado.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(dado, (list, tuple)):
        for v in dado:
            yield from _strings(v)


def garantir_sem_placa(dado, textos):
    """Falha se algum texto de placa (4+ caracteres) aparece em qualquer texto do dado. Rede de segurança antes de publicar.

    Olha só strings e chaves: números do arquivo (coordenadas, quadros) coincidem por acaso com leituras de OCR
    só de dígitos, e um número solto não identifica ninguém.
    """
    s = "\n".join(_strings(dado)).upper()
    vazou = sorted(t for t in textos if len(t) >= 4 and t.upper() in s)
    if vazou:
        raise ValueError(f"{len(vazou)} texto(s) de placa no dado que seria publicado; nada foi gravado")


# ── privacidade ────────────────────────────────────────────────────────────────

def pixelar(img, x1, y1, x2, y2, bloco=24):
    """Mosaico de blocos de `bloco` px, no lugar (img é alterada)."""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return
    peq = cv2.resize(img[y1:y2, x1:x2], (max(1, w // bloco), max(1, h // bloco)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(peq, (w, h), interpolation=cv2.INTER_NEAREST)


def dentro(ponto, regioes):
    return any(x1 <= ponto[0] <= x2 and y1 <= ponto[1] <= y2 for x1, y1, x2, y2 in regioes)


def ler_bruto(saida):
    import gzip
    with gzip.open(Path(saida) / "bruto.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)
