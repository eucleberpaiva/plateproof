"""Etapa 2 · transforma o bruto (quadro a quadro) em números e no arquivo do visor.

Saída em <saida>/:
  resumo.json    contagem por linha e classe, fases do semáforo, leitura de placa, tempo por etapa   PUBLICÁVEL
  trilhas.json   caixas por veículo, cruzamentos e fases, para o visor                             PUBLICÁVEL
  auditoria.json placas lidas com texto e veículos contados, para a conferência manual            SENSÍVEL
  textos.json    todo texto que o OCR leu, para a etapa 4 comparar sem carregar o bruto            SENSÍVEL

Os dois arquivos publicáveis passam por garantir_sem_placa antes de serem gravados: se algum texto de placa
aparecer neles, nada é gravado.

    python -m vigia.resumo cena.toml saida/
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vigia.nucleo import (CLASSES_COCO, ESTADOS, ORDEM, VEICULOS, carregar_cena, fases_semaforo, garantir_sem_placa,
                          ler_bruto, movimento, votar)

PASSO_VISOR = 4          # o visor recebe 1 caixa a cada 4 quadros e interpola entre elas
PASSO_FLUXO_S = 10
FAIXAS_LARGURA = [(1, 40, "até 40 px"), (40, 60, "40 a 60"), (60, 80, "60 a 80"), (80, 100, "80 a 100"), (100, 99999, "100 ou mais")]


def resumir(cena, bruto):
    Q, fps = bruto["quadros"], bruto["fps"]
    linhas = cena.get("linha", [])

    trilhas = defaultdict(lambda: {"f": [], "b": [], "votos": Counter()})
    for fi, q in enumerate(Q):
        for tid, c, x1, y1, x2, y2, s in q["v"]:
            t = trilhas[tid]
            t["f"].append(fi)
            t["b"].append((x1, y1, x2, y2))
            t["votos"][c] += s
    trilhas = {tid: t for tid, t in trilhas.items() if len(t["f"]) >= cena["rastreio"]["min_quadros"]}
    for t in trilhas.values():
        t["classe"] = CLASSES_COCO[t["votos"].most_common(1)[0][0]]
        t["linha"], t["cruza"] = movimento(t["b"], t["f"], linhas)
    contados = {tid: t for tid, t in trilhas.items() if t["linha"] and t["classe"] in VEICULOS}

    # placas
    cp = cena["placas"]
    leituras, largura = defaultdict(list), defaultdict(int)
    for fi, q in enumerate(Q):
        for tid, x1, y1, x2, y2, _, texto, conf in q["p"]:
            if tid is None:
                continue
            largura[tid] = max(largura[tid], x2 - x1)
            if texto and conf is not None:
                leituras[tid].append((fi, texto, conf, x2 - x1))
    votos = {tid: v for tid in trilhas if (v := votar(leituras.get(tid, []), cp["voto_conf_min"], cp["tamanho"]))}
    com_placa = {l["nome"] for l in linhas if l.get("placas")}
    candidatos = [tid for tid, t in contados.items() if t["linha"] in com_placa]
    lidas = {tid: votos[tid] for tid in candidatos
             if tid in votos and votos[tid]["n"] >= cp["leituras_min"] and votos[tid]["acordo"] >= cp["acordo_min"]}

    # semáforo
    fluxo_idx = [i for i, s in enumerate(cena.get("semaforo", [])) if s.get("fluxo")]
    fases = []
    if fluxo_idx:
        sl = cena.get("semaforo_leitura", {})
        fases = fases_semaforo([[q["s"][i] for i in fluxo_idx] for q in Q], fps, sl.get("acesa", 230), sl.get("margem", 50))
    completas = fases[1:-1]  # a primeira e a última foram cortadas pelo começo e fim do vídeo
    ciclos = sum(1 for a, b, c in zip(completas, completas[1:], completas[2:])
                 if (a[2], b[2], c[2]) == ("verde", "amarelo", "vermelho"))
    cruzamentos = {l["nome"]: sorted(t["cruza"] for t in contados.values() if t["linha"] == l["nome"]) for l in linhas}

    tempos = {k: np.array(v) for k, v in bruto["tempos"].items()}
    resumo = {
        "video": {"fps": fps, "quadros": len(Q), "largura": bruto["w"], "altura": bruto["h"]},
        "rastros": len(trilhas),
        "veiculos_contados": len(contados),
        "contagem": {l["nome"]: {"rotulo": l.get("rotulo", l["nome"]),
                                 **{c: sum(t["linha"] == l["nome"] and t["classe"] == c for t in trilhas.values()) for c in ORDEM}}
                     for l in linhas},
        "fluxo": {"passo_s": PASSO_FLUXO_S,
                  "por_linha": {n: np.bincount([int(f / fps) // PASSO_FLUXO_S for f in fs],
                                               minlength=int(len(Q) / fps) // PASSO_FLUXO_S + 1).tolist()
                                for n, fs in cruzamentos.items()}},
        "semaforo": {
            "ciclos_completos": ciclos,
            "fases": [{"inicio_s": round(a / fps, 1), "fim_s": round((b + 1) / fps, 1), "estado": e,
                       "cruzamentos": {n: sum(a <= f <= b for f in fs) for n, fs in cruzamentos.items()}} for a, b, e in fases],
            "duracao_s": {e: [round((b - a + 1) / fps, 1) for a, b, ee in completas if ee == e] for e in ESTADOS},
        },
        "placas": {
            "candidatos": len(candidatos),
            "com_placa_detectada": sum(largura.get(tid, 0) > 0 for tid in candidatos),
            "com_voto": sum(tid in votos for tid in candidatos),
            "lidas": len(lidas),
            "por_largura": [{"faixa": rot, "veiculos": len(g := [tid for tid in candidatos if de <= largura.get(tid, 0) < ate]),
                             "lidas": sum(tid in lidas for tid in g)} for de, ate, rot in FAIXAS_LARGURA],
        },
        "desempenho_ms": {k: {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
                          for k, v in tempos.items()},
        "quadros_por_segundo": round(len(Q) / bruto["cpu_total_s"], 2),
    }

    placas_visor = sorted([lidas[tid]["quadro_confirmado"], round(lidas[tid]["conf"], 2), lidas[tid]["n"], tid] for tid in lidas)
    idx = {p[3]: i for i, p in enumerate(placas_visor)}
    t_visor = []
    for tid in sorted(trilhas, key=lambda k: trilhas[k]["f"][0]):
        t = trilhas[tid]
        caixa, ultima, plano = dict(zip(t["f"], t["b"])), t["b"][0], []
        for f in range(t["f"][0], t["f"][-1] + 1, PASSO_VISOR):
            ultima = caixa.get(f, ultima)  # buraco no rastro: repete a última caixa conhecida
            x1, y1, x2, y2 = ultima
            plano += [x1, y1, x2 - x1, y2 - y1]
        t_visor.append([tid, ORDEM.index(t["classe"]), t["f"][0], t["cruza"], idx.get(tid, -1), plano])
    relogio = cena.get("relogio", {}).get("inicio")
    visor = {
        "fps": fps, "passo": PASSO_VISOR, "quadros": len(Q), "largura": bruto["w"], "altura": bruto["h"],
        "classes": ORDEM, "estados": ESTADOS,
        "relogio_inicio_s": sum(int(x) * m for x, m in zip(relogio.split(":"), (3600, 60, 1))) if relogio else None,
        "linhas": [{"nome": l["nome"], "rotulo": l.get("rotulo", l["nome"]), "de": l["de"], "ate": l["ate"]} for l in linhas],
        "trilhas": t_visor,
        "placas": placas_visor,
        "cruzamentos_por_classe": [sorted(t["cruza"] for t in contados.values() if t["classe"] == c) for c in ORDEM],
        "cruzamentos_por_linha": cruzamentos,
        "fases": [[a, ESTADOS.index(e)] for a, _, e in fases if e],
    }
    auditoria = {
        "lidas": {str(tid): v for tid, v in lidas.items()},
        "contados": [{"tid": tid, "classe": t["classe"], "linha": t["linha"], "cruza": t["cruza"],
                      "caixa": t["b"][t["f"].index(t["cruza"])]} for tid, t in contados.items()],
    }
    textos = {v["texto"] for v in votos.values()} | {l[1] for ls in leituras.values() for l in ls}
    textos |= {p[6] for q in Q for p in q["p"] if p[6]}  # inclusive leituras sem veículo dono
    confiaveis = sorted({p[6] for q in Q for p in q["p"] if p[6] and p[7] and p[7] >= 0.8})
    return resumo, visor, auditoria, textos, confiaveis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cena")
    ap.add_argument("saida")
    args = ap.parse_args()
    saida = Path(args.saida)
    resumo, visor, auditoria, textos, confiaveis = resumir(carregar_cena(args.cena), ler_bruto(saida))
    garantir_sem_placa(resumo, textos)
    garantir_sem_placa(visor, textos)
    (saida / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    (saida / "trilhas.json").write_text(json.dumps(visor, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (saida / "auditoria.json").write_text(json.dumps(auditoria, ensure_ascii=False), encoding="utf-8")
    (saida / "textos.json").write_text(json.dumps({"todos": sorted(textos), "confiaveis": confiaveis},
                                                  ensure_ascii=False), encoding="utf-8")
    c = resumo["contagem"]
    print(f"{resumo['veiculos_contados']} veículos contados em {resumo['rastros']} rastros")
    for nome, linha in c.items():
        print(f"  {nome:<20} " + "  ".join(f"{k} {v}" for k, v in linha.items() if k != "rotulo" and v))
    p = resumo["placas"]
    print(f"placas: {p['lidas']} lidas de {p['candidatos']} candidatos ({p['com_placa_detectada']} com placa detectada)")
    if resumo["semaforo"]["fases"]:
        print(f"semáforo: {resumo['semaforo']['ciclos_completos']} ciclos completos")
    print(f"ok -> {saida}/resumo.json, trilhas.json (publicáveis); auditoria.json e textos.json (SENSÍVEIS)")


if __name__ == "__main__":
    main()
