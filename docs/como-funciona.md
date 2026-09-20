# Como funciona

Cada etapa, por que ela é assim e o que foi medido. Os números vêm da cena de exemplo: 7 minutos, 12.750 quadros em 1920x1080, notebook com Intel Core i7-13700H, sem GPU.

## 1. Detectar veículos · `vigia/yolox.py`

O YOLOX-s acha carros, caminhões, ônibus, motos, bicicletas e pessoas em cada quadro. Ele roda direto no ONNX Runtime, sem PyTorch, então a instalação fica leve e igual em qualquer sistema.

O modelo exportado devolve a grade crua da rede. O código faz o resto: redimensiona o quadro mantendo a proporção, completa com cinza (valor 114, como no treino), converte a grade em caixas e aplica NMS entre classes. Esse NMS existe para que um mesmo carro não vire, ao mesmo tempo, "carro" e "caminhão".

**Medido:** 75 ms por quadro na mediana.

## 2. Rastrear · ByteTrack

Na cena de exemplo, um veículo fica na tela por 136 quadros na mediana (4,5 s). O rastreador liga as caixas de um quadro ao outro para que ele tenha um número só e seja contado uma vez. Rastro com menos de 20 quadros (0,67 s) é tratado como ruído do detector.

A classe de cada rastro é decidida por voto, com a confiança de cada quadro como peso. Um carro que o detector chamou de caminhão por dois quadros continua carro.

## 3. Contar · linhas na cena

O veículo é representado pelo meio da base da caixa, o ponto onde ele toca o chão. Cada linha da cena é um segmento com sentido, e o veículo conta uma vez, na primeira linha da lista que cruzar.

"Cruzar" tem uma definição exata (`nucleo.cruzamento`). Imagine alguém andando de `de` até `ate`. A linha conta quem passa do lado direito dessa pessoa para o esquerdo, e o ponto logo depois precisa cair dentro do comprimento do segmento.

Um carro que faz a curva pode tocar a linha de quem segue em frente, então cada linha pode exigir um deslocamento total da trajetória, medido do início ao fim do rastro. Na cena de exemplo, "seguir em frente" exige subir pelo menos 60 px na tela.

**Onde erra, pela conferência manual:** de 4 a 6 contagens a mais em 168. Reboque entra como veículo separado, e um poste na frente da faixa da direita esconde o carro e faz o rastreador trocar de número.

## 4. Ler o semáforo · pela luz

Sem integração com o controlador: uma caixa cobre cada grupo focal, e o código mede o brilho de cada terço dela: vermelho em cima, amarelo no meio, verde embaixo. A lente acesa passa de 230 de brilho e fica pelo menos 50 acima da segunda mais clara. Na cena de exemplo, acesa dá cerca de 255 e apagada fica entre 110 e 160.

Dois cuidados deixam a leitura estável:

- **Consenso:** quando dois semáforos controlam o mesmo fluxo, o estado só vale se os dois concordarem.
- **Janela de meio segundo:** o estado do quadro é a maioria de uma janela em volta dele. Um caminhão passando na frente do semáforo, ou uma piscada da câmera, não abre uma fase falsa.

**Medido:** 3 ciclos completos. Verde de 53,2 a 62,4 s, amarelo de 5 s, vermelho de 44,3 a 59,5 s. As durações batem entre ciclos, o que é um bom sinal de que a leitura não inventou fase.

## 5. Ler placas · detector + OCR + voto

**Detector de placa (YOLOv9-s-608).** Comparados nos mesmos 5 quadros:

| Modelo | Tempo | Placas achadas por quadro |
|---|---|---|
| t-384 | 18 ms | 1 |
| t-640 | 57 ms | 1,6 |
| s-608 | 64 ms | 3,2 |

O s-608 acha mais que o triplo de placas por 46 ms a mais.

**OCR (cct-xs-v2).** Por configuração, só roda em placa com pelo menos 40 px de largura. Na placa de teste, ele leu os 7 caracteres certos. O modelo maior, cct-s-v2, trocou um 9 por H.

**Voto.** Todas as leituras do mesmo veículo votam, caractere por caractere, com a confiança de cada leitura como peso. A placa só é considerada lida com 3 leituras ou mais e 75% de concordância média.

**Medido:** 47 placas lidas em 266 passagens. Das 47, 32 estavam exatas na conferência a olho.

**O que mais pesa é o tamanho da placa na imagem:**

| Largura máxima da placa | Veículos | Lidas |
|---|---|---|
| até 40 px | 29 | OCR desligado |
| 40 a 60 | 54 | 8 |
| 60 a 80 | 72 | 33 |
| 80 a 100 | 11 | 3 |
| 100 ou mais | 4 | 3 |

Em outras 96 passagens a placa nunca foi detectada, quase sempre carro vindo de longe. Para ler placa de verdade, a câmera precisa ser posta para isso. A Axis pede 130 px de largura de placa, ângulo de até 30° e distância de 2 a 7 m em portaria ([guia](https://help.axis.com/en-us/axis-license-plate-verifier)). A Hikvision pede caracteres com 20 a 30 px de altura e tempo de exposição de 1/250 a 1/500 s para veículos de 30 a 60 km/h ([guia](https://www.hikvision.com/content/dam/hikvision/au/firmware/ids-2cd7a46g0-p-izhs(y)/7xxx-ANPR-Installation-&-Configuration-GuidanceNew.pdf)).

## Desempenho

Na versão final, o vídeo inteiro rodou a 5,5 quadros por segundo. No começo não era assim: o primeiro teste de 300 quadros rodou a 1,3 quadro por segundo, e o processador não estava ocupado com inferência. Três sessões do ONNX Runtime mantinham threads girando em vazio, esperando trabalho, e disputavam os mesmos núcleos com o codificador de vídeo. Duas mudanças juntas levaram aquele teste a 4,8 quadros por segundo, ainda gravando vídeo: desligar essa espera ativa (`session.intra_op.allow_spinning = 0`) e limitar o codificador a 2 threads. Hoje o vídeo nem é gravado na análise, só na etapa de privacidade.

**Tempo por quadro no vídeo inteiro, na mediana:**

| Etapa | ms |
|---|---|
| Decodificar | 4,3 |
| Detectar veículos | 75,0 |
| Rastrear | 2,2 |
| Detectar placas | 87,5 |
| Ler placas (OCR) | 0,1 |

O OCR fica perto de zero na mediana porque só roda quando há placa grande o bastante. O pico dele (p95) é 7,9 ms.
