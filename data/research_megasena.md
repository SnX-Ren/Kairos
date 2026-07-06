# Kairos — Pesquisa: Sistema de Sorteio da Mega-Sena

> Investigação técnica completa. Fontes: Caixa Econômica Federal, Wikipedia, Smartplay International, arXiv, PMC, Olhar Digital, BNLData, AsLoterias.

---

## 1. Equipamento Físico — O Globo

### Duas eras (crítico para ML)

| Era | Concursos | Mecanismo |
|---|---|---|
| 1996–2009 | 1–1140 | Dois globos separados: dígito 0–5 (dezena) + dígito 0–9 (unidade). "00" = 60 |
| 2010–hoje | 1141–3026 | Um único globo acrílico transparente com 60 bolas |

**Implicação direta:** os dois períodos são sistemas físicos distintos e devem ser tratados como datasets separados no modelo.

### Especificações das bolas (confirmado via Caixa)

- **Material:** borracha maciça vulcanizada
- **Peso:** 66 gramas (idêntico entre todas)
- **Diâmetro:** 50 mm (idêntico entre todas)
- **Numeração:** gravada a laser (não pintada — elimina viés de peso por quantidade de tinta)
- **Sistema de cores:** por dígito final — terminadas em 1 (vermelhas), 2 (amarelas), 3 (verdes), 4 (marrons), 5 (azuis), 6 (rosas), 7 (pretas), 8 (cinzas), 9 (laranjas), 0 (brancas)

### Fabricante e manutenção

- Fabricante não divulgado oficialmente. Provável: **Smartplay International** (líder mundial, fornece para Powerball, EuroMillions, UK Lottery)
- Contrato público com **Emibm Engenharia e Inovação** para manutenção preventiva/corretiva — R$ 144.166,32/ano
- Confirmados **dois conjuntos completos** de globos (principal + reserva)
- **Inmetro** realiza verificação anual das bolas (pesagem com balanças miligramétricas)
- Bolas pesadas individualmente antes de cada sorteio pela equipe técnica da Caixa

---

## 2. Processo do Sorteio

### Sequência operacional (confirmado via regulamento oficial)

1. Equipe técnica verifica o equipamento horas antes
2. Auditores populares (2 cidadãos do público) abrem maletas lacradas e conferem lacres
3. Auditores verificam que o globo está vazio
4. Bolas carregadas automaticamente por comando do operador
5. Auditores confirmam carregamento completo
6. Globo inicia circulação por ar comprimido (air-mix)
7. Bolas extraídas **uma a uma** pela caçapa
8. Bola válida = completamente ejetada do interior (se duas saírem juntas, conta a primeira)
9. Ata oficial assinada por auditores e responsáveis técnicos
10. Resultados transmitidos em tempo real ao sistema central

### Ordem de registro — campo confirmado

A API da Caixa registra **ambas** as ordens:
- `dezenasSorteadasOrdemSorteio` → sequência exata em que as bolas saíram
- `listaDezenas` → mesmos números em ordem crescente

Verificado nos concursos 1, 50, 100, 500, 1614 e 3026 — campo preenchido em todos.

**Exemplo (concurso 1, 11/03/1996):**
- Ordem do sorteio: 41, 05, 04, 52, 30, 33
- Ordem crescente: 04, 05, 30, 33, 41, 52

> **Atenção:** Para concursos 1–1140 (dois globos), a confiabilidade do campo de ordem não foi auditada publicamente — pode ter sido reconstruído retroativamente ao migrar os sistemas.

### Local e supervisão

- **Local padrão:** Espaço Loterias Caixa, Terminal Rodoviário do Tietê, São Paulo, SP
- **Sorteios especiais:** Espaço da Sorte (Av. Paulista) ou Auditório da Caixa (Brasília)
- Transmissão ao vivo: YouTube/redes sociais da Caixa; TV Globo na Mega da Virada
- Sorteios arquivados em vídeo para auditoria futura

---

## 3. Componente Eletrônico vs. Físico

### Mega-Sena usa PRNG? Não. (confirmado)

Sistema exclusivamente físico. Aleatoriedade gerada por física caótica (colisões entre bolas + turbulência do ar). Sem conexão com redes externas (air-gapped).

### Loterias que usam PRNG documentado

| Loteria | Sistema | Notas |
|---|---|---|
| Hot Lotto / MUSL (EUA) | PRNG por software | Comprometido por Eddie Tipton (ver seção 5) |
| Danish Spil (Dinamarca) | Szrek2Solutions RNG | Certificado desde 2005 |
| Florida Lottery (EUA) | Szrek2Solutions RNG | |
| Arkansas Scholarship Lottery | Szrek2Solutions RNG | |
| Loterias online / iGaming | PRNG/TRNG certificado | GLI, BMM, iTech Labs |

### Por que loterias grandes resistem ao RNG eletrônico

Razão principal: **percepção pública**. Jogadores exigem o espetáculo visual das esferas como garantia de transparência. Em algumas jurisdições, sorteios mecânicos ao vivo são exigidos por lei. Matematicamente, um TRNG (hardware entropy) seria *mais* aleatório que qualquer sistema físico.

---

## 4. Dados e API

### API oficial da Caixa

- **URL:** `https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena/{concurso}`
- **Cobertura:** concursos 1–3026 (11/03/1996 – 02/07/2026)
- **32 campos** por concurso
- Atualização contínua após cada sorteio

### API alternativa (usada no projeto)

- `https://loteriascaixa-api.herokuapp.com/api/{loteria}`
- Retorna todos os concursos em uma única requisição
- 10 loterias suportadas
- Inclui `dezenasOrdemSorteio` e `dezenas`

### Por que datasets do Kaggle param em ~1614

Simplesmente foram criados e nunca atualizados. A API oficial tem cobertura completa.

### Outras fontes de dados

- **AsLoterias.com.br:** download Excel completo, disponível em duas versões (ordem do sorteio e ordem crescente)
- **guilhermeasn/loteria.json (GitHub):** JSON atualizado diariamente via GitHub Actions
- **BrasilAPI:** wrapper da API oficial com documentação

---

## 5. Casos Documentados de Exploração

### Eddie Tipton — MUSL (2005–2011)

**O maior caso de fraude em loteria da história.**

- **Quem:** Eddie Tipton, Diretor de Segurança da Informação da Multi-State Lottery Association (MUSL)
- **Sistema comprometido:** PRNG por software em computador isolado
- **Método:**
  1. Em 20/11/2010, Tipton teve acesso autorizado à sala do RNG para ajuste de horário de verão
  2. Instalou um **rootkit autodestrutivo** via USB
  3. O código alterava o PRNG para gerar resultados previsíveis em **3 datas específicas**: 27/mai, 23/nov e 29/dez — somente se fossem quarta ou sábado após as 20h
  4. Rootkit se autodestroía sem deixar rastros
- **Estados afetados:** Iowa (2010), Colorado (2005), Wisconsin (2007), Oklahoma (2011), Kansas
- **Descoberta:** filmagem de câmera de segurança mostrou Tipton comprando a aposta vencedora
- **Confissão (2017):** *"Escrevi software que me permitiu tecnicamente prever números vencedores"*
- **Relevância:** confirma que vetores de ataque em PRNG são reais. A Mega-Sena elimina esse risco por ser exclusivamente física.

### Stefan Mandel — Cobertura combinatória (1960–1992)

- **Estratégia:** não detectou padrão — explorou **arbitragem matemática** quando jackpot > custo total de todas as combinações
- **Virgínia, EUA (1992):** pool de 7.059.052 combinações. Mandel comprou ~5 milhões antes do prazo. Ganhou US$27 milhões
- **Resultado:** investigado por FBI, CIA e IRS — sem ilegalidades. Leis alteradas para proibir compra em massa
- **Relevância para ML:** demonstra que a única exploração matemática comprovada não envolve previsão de padrões

### Viés físico em bolas

**Nenhum caso documentado encontrado.** O único vetor teórico (variação de peso por tinta) foi eliminado com a adoção de gravação a laser.

---

## 6. Literatura Científica

### UK Lottery — análise de 2.065 sorteios (PMC5536828)

- Período: 1994–2017
- Métodos: qui-quadrado, simulação Monte Carlo (20.000 iterações), análise por raízes digitais, pares/ímpares, primos
- **Conclusão:** "A distribuição de números vencedores é puramente aleatória. O jogo é justo."

### Statistical Auditing of Lotto k/N Games (arXiv:0806.4595)

- Deriva distribuições teóricas para jogos lotto k/N
- Aplicado ao Melate (México) e Lotto italiano
- Metodologia de referência para detectar desvios da distribuição teórica

### Bayesian Methods for Testing Lottery Randomness

- Compara abordagem bayesiana vs qui-quadrado
- Demonstra que qui-quadrado padrão não segue distribuição simples em sorteios sem reposição

### Canada Lotto 6/49

- Qui-quadrado "mal significativo ao nível 10%" — mecanismo justo
- Números 1–10 são impopulares entre apostadores → estrategicamente melhores para maximizar prêmio quando vencidos (menos divisão), mas **não por frequência maior**

### Análise Mega-Sena (berkurka/blog_mega_sena)

- Números mais frequentes: 10, 5, 53, 4, 23
- Distribuição par/ímpar: 50,4% vs 49,6% — essencialmente uniforme
- **Nenhum padrão exploravelmente previsível detectado**

---

## 7. Loterias Comparáveis

| Loteria | Mecanismo | Fabricante | PRNG? |
|---|---|---|---|
| Mega-Sena (BR) | Air-mix, 1 globo, 60 bolas | Não divulgado | Não |
| Powerball (EUA) | Gravity-pick (paddles), 2 tambores | Smartplay (Halogen) | Não |
| EuroMillions (EU) | Gravity-pick, 2 tambores | — | Não |
| Lotofácil (BR) | Air-mix, 1 globo, 25 bolas | Mesmo que Mega-Sena | Não |
| Danish Spil (DK) | Eletrônico | Szrek2Solutions | Sim |

---

## 8. Implicações para o Projeto Kairos

### O que muda no dataset

- Separar **era 1 (concursos 1–1140)** da **era 2 (1141–3026)** — sistemas físicos diferentes
- Confiabilidade do campo `dezenasOrdemSorteio` nos concursos antigos: não auditada

### O que o ML pode e não pode fazer

| Possível | Não possível |
|---|---|
| Verificar se distribuição desvia da uniforme | Prever números futuros acima do acaso (física) |
| Detectar estrutura condicional intra-sorteio | Explorar viés de bolas (calibração elimina isso) |
| Testar features matemáticas (Fibonacci, soma, φ) | Reproduzir o caso Tipton (sistema é físico, não PRNG) |
| Comparar eras 1 e 2 como experimento natural | |

### Questão em aberto

A confiabilidade histórica do campo `dezenasOrdemSorteio` para os concursos 1–1140 não foi confirmada. Recomendado: usar apenas concursos 1141+ para a análise de ordem de sorteio.
