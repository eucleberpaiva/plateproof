# Vigia

Conta veículos por tipo e por direção, lê o semáforo pela luz acesa e lê placas em vídeo de câmera fixa. Roda em CPU, sem placa de vídeo. Gera um vídeo com as placas escondidas e traz duas conferências para rodar antes de mostrar esse vídeo a alguém.

> **English summary.** Vigia counts vehicles by class and direction, reads traffic-light phases from lens brightness and reads license plates from fixed-camera video, CPU only (ONNX Runtime). Its main contribution is the privacy step: plates are pixelated in layers, and it ships two checks to run before anyone sees the video: an automatic attack with the same plate reader and a whole-frame manual review. Docs are in Portuguese; code identifiers too.

O repositório não traz vídeo nenhum. Você usa o seu.

## O que ele faz

| Etapa | Comando | O que sai |
|---|---|---|
| 0. Modelos | `python -m vigia.modelos` | 3 modelos ONNX baixados da fonte original, com SHA-256 conferido |
| 1. Analisar | `python -m vigia.analisar` | veículos, rastros, placas e semáforo, quadro a quadro, com o tempo de cada etapa |
| 2. Resumir | `python -m vigia.resumo` | contagem por linha e classe, fases do semáforo, taxa de leitura de placa |
| 3. Esconder placas | `python -m vigia.privacidade` | `publico.mp4`, com placas pixeladas |
| 4. Conferir | `python -m vigia.verificar` e `python -m vigia.amostrar` | ataque automático ao vídeo público e quadros inteiros sorteados para olhar |
| Ver | `visor/index.html` | vídeo com caixas, rastros, linhas e contadores, tudo no navegador |

## O que foi medido numa cena real

Sete minutos de uma câmera de trânsito num cruzamento (12.750 quadros, 1920x1080), num notebook com Intel Core i7-13700H e sem GPU. A calibração dessa cena está em [`cenas/exemplo-cruzamento.toml`](cenas/exemplo-cruzamento.toml).

- **316 veículos contados** em 3 movimentos: 168 seguindo em frente, 98 no sentido contrário, 50 pela transversal.
- **3 ciclos completos de semáforo** lidos pela cor das lentes, sem ligação com o controlador.
- **47 placas lidas** em 266 passagens em que a placa ficava de frente para a câmera.
- **5,5 quadros por segundo** com tudo ligado. As duas detecções levam quase todo o tempo: 75 ms (veículos) e 88 ms (placas) por quadro, na mediana.

E o que a conferência manual achou nesses números:

- **Placas:** 32 das 47 estavam exatas. 11 tinham um caractere trocado, quase sempre 8 lido como B, e 4 nem a olho dava para ler.
- **Caminhões:** dos 53 veículos que o modelo chamou de caminhão, 8 eram caminhões. O resto eram 24 picapes, 13 vans, 3 reboques e 5 SUVs. O modelo aprendeu no COCO, onde picape e van costumam cair nessa classe.
- **Contagem dupla:** de 4 a 6 dos 168 que seguiram em frente foram contados a mais. Reboques entram separados do veículo que os puxa, e um poste na frente da câmera faz o rastreador trocar de número.
- **Não medido:** quantos veículos passaram sem ser contados.

O que mais pesou na leitura de placa foi o tamanho dela na imagem. Esta câmera foi instalada para mostrar o cruzamento, e a placa quase nunca passa de 80 px de largura; abaixo de 40 px o OCR nem roda, por configuração. Para uma câmera de portaria feita para ler placa, a Axis pede 130 px. O detalhe está em [`docs/como-funciona.md`](docs/como-funciona.md).

## Instalar

Precisa de Python 3.12 ou mais novo (desenvolvido em 3.14) e do [FFmpeg](https://ffmpeg.org) no PATH, que só a etapa 3 usa.

```bash
git clone https://github.com/eucleberpaiva/vigia.git
cd vigia
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/macOS: source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m vigia.modelos
python -m unittest -v
```

`--require-hashes` faz o pip recusar qualquer pacote cujo hash não seja o registrado em `requirements.txt`. `vigia.modelos` faz o mesmo com os modelos: se a fonte trocar o arquivo, o download falha em vez de rodar outro modelo. Os testes não precisam de vídeo nem de modelo.

## Rodar no seu vídeo

**1. Calibre a cena.** Copie o exemplo e ajuste para a sua câmera:

```bash
cp cenas/exemplo-cruzamento.toml cenas/minha-cena.toml
python -m vigia.amostrar meu-video.mp4 saida --n 3
```

Abra os quadros de `saida/folhas/` num editor de imagem que mostre a posição do cursor e anote as coordenadas: as linhas de contagem, a caixa de cada semáforo e as áreas que não são placa (logotipo, relógio). Cada campo está explicado no próprio arquivo. `cenas/*.toml` fica fora do git (menos o exemplo), então sua calibração não vai parar num commit por engano.

**2. Rode.** Teste com poucos quadros antes de mandar o vídeo inteiro:

```bash
python -m vigia.analisar meu-video.mp4 cenas/minha-cena.toml saida --max 300
python -m vigia.resumo cenas/minha-cena.toml saida
```

**3. Veja.** Abra `visor/index.html` no navegador e escolha o vídeo e o `saida/trilhas.json`. Nenhum arquivo sai do seu computador: a página não faz nenhuma requisição de rede.

**4. Antes de mostrar o vídeo para alguém:**

```bash
python -m vigia.privacidade meu-video.mp4 cenas/minha-cena.toml saida
python -m vigia.verificar saida/publico.mp4 cenas/minha-cena.toml saida
python -m vigia.amostrar saida/publico.mp4 saida --n 24
```

O `verificar` termina com erro sempre que marca alguma leitura, de propósito: quem decide é você, olhando `saida/folhas/verificacao.jpg` recorte a recorte. Na cena de exemplo ele marcou 5, e as 5 eram o letreiro de rua que o canal desenha no canto. Depois abra cada quadro sorteado pelo `amostrar` e procure qualquer placa legível. O motivo de ter os dois testes, e o que falhou até chegar neles, está em [`docs/privacidade.md`](docs/privacidade.md).

## O que é sensível

| Arquivo | Contém | Pode publicar? |
|---|---|---|
| seu vídeo original | pessoas e placas | não |
| `saida/bruto.json.gz` | texto de todas as placas lidas, quadro a quadro | não |
| `saida/recortes/`, `saida/auditoria.json`, `saida/textos.json`, `saida/folhas/` | imagem e texto de placas | não |
| `saida/resumo.json`, `saida/trilhas.json` | números, caixas e cruzamentos, sem texto de placa | sim |
| `saida/publico.mp4` | vídeo com placas pixeladas; **rostos não são tratados** | só depois da etapa 4, e se não houver rosto reconhecível |

`resumo.json` e `trilhas.json` passam por uma checagem antes de serem gravados: se algum texto de placa aparecer neles, nada é gravado. A pasta `saida/`, vídeos, imagens e modelos estão no `.gitignore`, e um teste falha se algum desses arquivos entrar no repositório.

## Uso responsável

Placa de veículo pode identificar uma pessoa. No Brasil vale a LGPD, e outros lugares têm regras próprias para leitura automática de placas. Antes de apontar isto para uma câmera:

- **Direito de uso:** use vídeo que você tem direito de usar. Câmera sua, com aviso a quem passa, ou material com licença que permita.
- **O mínimo necessário:** se você só precisa contar veículos, rode `vigia.analisar` com `--sem-texto`: a placa ainda é detectada para ser pixelada, mas nenhum texto nem recorte é gravado. Se leu placas, apague `bruto.json.gz`, `recortes/`, `auditoria.json` e `textos.json` quando terminar a conferência.
- **Nunca publique o original.** Publique só o que passou pela etapa 4.

Serve para aprender e medir trânsito. A licença não autoriza ninguém a descumprir a lei.

## Limites conhecidos

- **Placas brasileiras:** o leitor de placa (`cct-xs-v2-global`) não foi testado com placas Mercosul, só com as desta cena, dos EUA.
- **Velocidade:** 5,5 quadros por segundo não é tempo real para vídeo a 30 quadros por segundo.
- **Rastreador:** o ByteTrack saiu do `supervision` na versão 0.31, por isso a versão está travada.
- **Privacidade:** a camada que cobre placas não detectadas depende do detector de veículos. Veículo e placa perdidos no mesmo quadro ficam sem cobertura automática, e é por isso que a conferência a olho existe. Rostos não são tratados. O mosaico não foi testado contra reconstrução que combina vários quadros.

## Como contribuir

É um lab: issue é bem-vinda, sem prazo de resposta. Algumas coisas que valem medir:

- **Veículos perdidos:** contar um trecho à mão e comparar com a contagem automática, o número que falta neste README.
- **Rastreador:** trocar o ByteTrack por um mantido e comparar a contagem dupla.
- **Privacidade:** um ataque que combine vários quadros para tentar reconstruir o mosaico, e um jeito de tratar rostos.

Rode `python -m unittest` antes de mandar um PR, e nunca anexe vídeo ou imagem com placa ou rosto reconhecível, nem em issue.

## Créditos

- **Detecção de veículos:** [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), Megvii, Apache-2.0. Treinado no [COCO](https://cocodataset.org) (Lin et al., 2014), anotações CC BY 4.0.
- **Detecção de placa:** [open-image-models](https://github.com/ankandrew/open-image-models), ankandrew, MIT. Arquitetura [YOLOv9](https://arxiv.org/abs/2402.13616) (Wang, Yeh e Liao, 2024).
- **Leitura de placa:** [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) e [fast-alpr](https://github.com/ankandrew/fast-alpr), ankandrew, MIT.
- **Rastreio:** ByteTrack (Zhang et al., 2022) pela biblioteca [supervision](https://github.com/roboflow/supervision), Roboflow, MIT.
- **Execução:** [ONNX Runtime](https://onnxruntime.ai) (MIT), [OpenCV](https://opencv.org) (Apache-2.0), [FFmpeg](https://ffmpeg.org) (LGPL/GPL).

Este repositório não redistribui nenhum peso de modelo. Cada um é baixado da fonte original, e vale conferir a licença de cada modelo para o seu uso.

## Licença

Código sob [MIT](LICENSE). © 2026 Cleber Paiva.
