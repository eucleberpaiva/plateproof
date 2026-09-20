"""Testes que não precisam de vídeo nem de modelo:  python -m unittest -v"""
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

from vigia.nucleo import (carregar_cena, cruzamento, dono, fases_semaforo, garantir_sem_placa, lente_acesa, movimento,
                          pixelar, votar)
from vigia.resumo import resumir

RAIZ = Path(__file__).resolve().parent.parent
LINHA_SUBINDO = {"de": [0, 100], "ate": [200, 100], "sentido": "direita_para_esquerda"}


class Contagem(unittest.TestCase):
    def test_conta_quem_cruza_no_sentido_certo(self):
        self.assertEqual(cruzamento([(50, 150), (50, 90)], [0, 1], LINHA_SUBINDO), 1)

    def test_ignora_sentido_contrario(self):
        self.assertIsNone(cruzamento([(50, 90), (50, 150)], [0, 1], LINHA_SUBINDO))

    def test_qualquer_sentido(self):
        self.assertEqual(cruzamento([(50, 90), (50, 150)], [0, 1], {**LINHA_SUBINDO, "sentido": "qualquer"}), 1)

    def test_fora_do_segmento_nao_conta(self):
        self.assertIsNone(cruzamento([(300, 150), (300, 90)], [0, 1], LINHA_SUBINDO))

    def test_primeira_linha_da_lista_ganha_e_deslocamento_filtra(self):
        caixas = [(40, 140 - 10 * i, 60, 160 - 10 * i) for i in range(10)]  # pé sobe de y=160 para y=70
        outra = {**LINHA_SUBINDO, "nome": "outra"}
        exigente = {**LINHA_SUBINDO, "nome": "exigente", "desloc_y": [-np.inf, -500]}
        self.assertEqual(movimento(caixas, list(range(10)), [exigente, outra])[0], "outra")


class Semaforo(unittest.TestCase):
    def test_lente_acesa_precisa_se_destacar(self):
        self.assertEqual(lente_acesa([255, 130, 120]), "vermelho")
        self.assertIsNone(lente_acesa([255, 240, 120]))
        self.assertIsNone(lente_acesa([200, 130, 120]))

    def test_janela_ignora_piscada_e_consenso_ignora_divergencia(self):
        verde, vermelho = [120, 120, 255], [255, 120, 120]
        leituras = [[verde, verde]] * 30 + [[vermelho, verde]] * 2 + [[verde, verde]] * 30
        self.assertEqual([f[2] for f in fases_semaforo(leituras, fps=30)], ["verde"])


class Placas(unittest.TestCase):
    def test_voto_por_posicao(self):
        leituras = [(0, "ABC1234", 0.9, 60), (1, "ABC1Z34", 0.7, 60), (2, "ABC1234", 0.9, 62)]
        v = votar(leituras)
        self.assertEqual((v["texto"], v["n"]), ("ABC1234", 3))
        self.assertLess(v["acordo"], 1.0)

    def test_leitura_fraca_nao_vota(self):
        self.assertIsNone(votar([(0, "ABC1234", 0.9, 60), (1, "ABC1234", 0.3, 60)]))

    def test_dono_e_a_menor_caixa_que_contem_a_placa(self):
        self.assertEqual(dono((45, 45, 55, 50), [(0, 0, 100, 100), (40, 40, 60, 60)], [1, 2]), 2)

    def test_rede_de_seguranca_barra_texto_de_placa(self):
        with self.assertRaises(ValueError):
            garantir_sem_placa({"rotulos": ["carro ABC1234"]}, {"ABC1234"})
        garantir_sem_placa({"coordenadas": [1234, 5678]}, {"1234"})  # número solto não é placa publicada


class Privacidade(unittest.TestCase):
    def test_mosaico_apaga_detalhe(self):
        img = np.zeros((48, 48, 3), np.uint8)
        img[::2, ::2] = 255  # textura fina, como caracteres
        pixelar(img, 0, 0, 48, 48, bloco=24)
        self.assertEqual(len(np.unique(img[:24, :24].reshape(-1, 3), axis=0)), 1)


class ResumoPublicavel(unittest.TestCase):
    def test_arquivos_publicaveis_nao_tem_texto_de_placa(self):
        cena = carregar_cena(RAIZ / "cenas" / "exemplo-cruzamento.toml")
        quadros = []
        for i in range(40):  # um carro sobe cruzando a linha "frente" com a placa lida em todo quadro
            y2 = 900 - i * 10
            quadros.append({"v": [[7, 2, 900, y2 - 150, 1100, y2, 0.9]],
                            "p": [[7, 980, y2 - 40, 1060, y2 - 20, 0.9, "XYZ9876", 0.95]],
                            "s": [[120, 120, 255]] * 3})
        bruto = {"fps": 30, "w": 1920, "h": 1080, "quadros": quadros, "cpu_total_s": 1,
                 "tempos": {"veiculos": [1.0] * 40}}
        resumo, visor, auditoria, textos, confiaveis = resumir(cena, bruto)
        self.assertEqual(resumo["veiculos_contados"], 1)
        self.assertEqual(resumo["placas"]["lidas"], 1)
        for publico in (resumo, visor):
            self.assertNotIn("XYZ9876", json.dumps(publico))
        self.assertIn("XYZ9876", json.dumps(auditoria))  # a auditoria precisa do texto, e por isso é sensível
        self.assertIn("XYZ9876", confiaveis)  # leitura confiável: é com esta lista que a etapa 4 compara


class Repositorio(unittest.TestCase):
    def test_nenhum_video_modelo_ou_resultado_versionado(self):
        try:
            arquivos = subprocess.run(["git", "ls-files"], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("fora de um repositório git")
        proibidos = (".mp4", ".mov", ".avi", ".mkv", ".onnx", ".json.gz", ".png", ".jpg", ".jpeg")
        resultados = {"bruto.json", "auditoria.json", "textos.json", "verificacao.json", "resumo.json", "trilhas.json"}
        self.assertEqual([a for a in arquivos if a.lower().endswith(proibidos)], [])
        self.assertEqual([a for a in arquivos if a.split("/")[0] in ("saida", "saidas", "recortes", "folhas", "modelos")
                          or a.split("/")[-1] in resultados], [])
        self.assertEqual([a for a in arquivos if (RAIZ / a).exists() and (RAIZ / a).stat().st_size > 1_000_000], [])


if __name__ == "__main__":
    unittest.main()
