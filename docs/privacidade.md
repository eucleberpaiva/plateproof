# Privacidade: como esconder placa e provar que escondeu

Pixelar a placa que o detector achou é fácil. O difícil é a placa que ele não achou, e saber se sobrou alguma. Este documento conta o método atual, as tentativas que falharam antes dele e o que as conferências não conseguem provar.

## O método atual · `vigia/privacidade.py`

**Três camadas:**

1. **Placa detectada:** pixelada com margem, em qualquer confiança.
2. **Faixa inferior de todo veículo próximo:** vale para veículo com caixa de 100 px de altura ou mais. A faixa vai de 30% da altura da caixa até a base, na largura toda, e é pixelada em todo quadro, inclusive 5 quadros antes e 5 depois do rastro existir, porque o rastreador demora alguns quadros para confirmar um veículo. Na cena de exemplo, 99% das placas detectadas começam abaixo de 38% da altura da caixa do veículo. Na mesma cena, 99% dos veículos com placa detectada tinham caixa de 133 px de altura ou mais.
3. **Tarjas fixas:** por exemplo, a data queimada no relógio da câmera.

Placa detectada dentro de uma região `ignorar` da cena (logotipo, relógio) não é lida nem contada, mas é pixelada do mesmo jeito: uma placa de verdade pode passar ali.

O mosaico usa blocos de 24 px no original (16 px no vídeo de 1280). Na cena de exemplo, a placa inteira tem 22 px de altura na mediana e 39 px no percentil 99, então vira um ou dois blocos: ilegível, e bem mais limpo que blocos gigantes.

**O que sobra:**

- **Veículo e placa perdidos no mesmo quadro:** nenhuma camada cobre, e é para isso que existe a conferência a olho.
- **Reconstrução por vários quadros:** o mosaico acompanha o veículo, e não testamos um ataque que combine quadros para reconstruir o que está embaixo.
- **Rostos:** não são tratados.

## Tentativas que falharam

| Tentativa | Como falhou | Quem achou |
|---|---|---|
| Pixelar só a placa detectada | O detector falha em alguns quadros e em alguns veículos. A capa do vídeo mostrou uma placa limpa. | revisão da imagem de capa |
| Lembrar onde a placa estava dentro da caixa e continuar pixelando ali | Em veículo meio escondido, a caixa muda de formato e a posição lembrada vai parar no lugar errado. | olho, num quadro específico |
| Pular a faixa quando o veículo já tinha placa detectada | Um caminhão baú tinha duas áreas parecidas com placa. O detector achou a errada, e a placa de verdade vazou. | ataque automático |
| Zona estimada na largura da caixa (de 10 a 90%) | A caixa pegou carro e reboque juntos, e a zona caiu em cima do reboque. | revisão da imagem de capa |

## Conferência 1 · atacar o próprio vídeo · `vigia/verificar.py`

O mesmo leitor de placa, mais sensível que na análise (limiar 0,2), roda em cima do vídeo público, sem pular nenhuma região. O vídeo é ampliado de volta para 1920 px, que é o que alguém faria para tentar ler. Cada leitura com cara de placa é comparada com as leituras confiáveis do original (confiança de 0,8 para cima e 4 caracteres ou mais), e distância de edição de até 2 conta como vazamento. Se o original não tem nenhuma leitura para comparar, o comando para com erro em vez de dizer que passou.

O comando termina com erro sempre que marca alguma coisa, de propósito. O veredito é seu, olhando `folhas/verificacao.jpg` recorte a recorte.

**Resultado na cena de exemplo:** nos 12.750 quadros, o leitor devolveu 8.911 textos com cara de placa, quase todos o letreiro de rua que o canal desenha no canto. Cinco chegaram a ficar a dois caracteres de uma leitura confiável do original, e os cinco recortes mostram o mesmo letreiro, meio coberto. Nenhuma placa de veículo saiu legível.

**Pontos cegos:** o ataque usa o mesmo detector de placa da análise, então uma placa que ele nunca achou no original também não é achada no vídeo público. E placa cuja leitura no original ficou abaixo de 0,8 de confiança não entra na lista de comparação: se ela vazar, este teste não acusa. Passar aqui não prova nada sobre esses dois casos.

## Conferência 2 · quadros inteiros, a olho · `vigia/amostrar.py`

Sorteio com semente fixa, mais os quadros que sempre entram, como a imagem de capa. Olha-se o **quadro inteiro**, procurando qualquer placa legível.

Uma versão anterior sorteava recortes da própria área pixelada. Ela passou com 72 recortes sem nenhuma placa legível e mesmo assim não provava nada: um recorte centrado no mosaico só mostra que o mosaico está borrado, nunca que a placa está dentro dele. A zona fora do lugar nunca entrava na amostra.

**Resultado na cena de exemplo:** 25 quadros inteiros, incluindo o de capa, nenhuma placa legível.

## O que não sai do seu computador

O texto das placas nunca vai para os arquivos publicáveis. `resumo.py` só grava `resumo.json` e `trilhas.json` depois de conferir que nenhum texto lido pelo OCR aparece neles. O visor mostra apenas "placa lida", com o número de leituras e a confiança.
