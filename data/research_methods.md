# Kairos — Pesquisa: Métodos Avançados de Detecção de Sinal

> Investigação sobre testes de aleatoriedade industriais, filtros de ruído de signal processing,
> modelos financeiros e preditores históricos — filtrados pela aplicabilidade ao nosso dataset
> (1886 sorteios Mega-Sena era moderna + ~24.500 draws agregados).

---

## Ranking — Top 10 para o Kairos

| # | Método | Por quê | Ferramenta |
|---|--------|---------|-----------|
| 1 | **E-processo sequencial (SAVI / e-values)** | Transforma o p=0.0095 congelado em monitoramento vivo, anytime-valid — acompanha cada novo sorteio sem inflar erro tipo I; combina evidência entre loterias por média de e-values | `confseq`, `expectation` (pip) |
| 2 | **Teste multivariado hipergeométrico** (Coronel-Brizio, arXiv:0806.4595) | Desenhado especificamente para loterias k/N; corrige a covariância negativa entre frequências que o nosso χ² global ignora (cada sorteio impõe soma fixa) — **pode revisar o p=0.0095** | numpy (fórmulas do paper) |
| 3 | **Higher Criticism** (Donoho-Jin 2004) | Poder ótimo no regime "sinais raros e fracos entre muitos testes" — o teste global da tese; consome todos os p-values que já produzimos | numpy (~20 linhas) |
| 4 | **Bayes factor Dirichlet-multinomial** | Única ferramenta que dá evidência *a favor* da uniformidade — fecha a tese nos dois sentidos; forma fechada | scipy (~15 linhas) |
| 5 | **Testes suaves de Neyman + AD/CvM discretos** | Mais poder que χ² contra desvios *estruturados* (tendência com número da bola — assinatura física plausível); diagnostica a *forma* do desvio | scipy + Legendre |
| 6 | **Scan statistics temporais** (Kulldorff) | Viés físico real é episódico (bola trocada, desgaste); localiza *quando*; controla múltiplas janelas nativamente | implementação própria |
| 7 | **Teste de gaps** (Haigh 1997, UK Lotto) | Dimensão nova: clustering temporal por número (gaps ~ Geométrica(0.1) sob H0); pedigree de auditoria | scipy |
| 8 | **Permutation entropy + missing ordinal patterns** (Bandt-Pompe) | Detector de determinismo independente de frequência; plano complexidade-entropia distingue caos de ruído — responde diretamente à tese | `ordpy` (biblioteca brasileira) |
| 9 | **SumThreshold 2D** (rádio-astronomia, Offringa 2010) | Conexão com RFI mitigation: matriz 60 números × 1886 sorteios como "espectrograma"; flagra regiões bola × época anômalas | ~50 linhas + permutação |
| 10 | **HMM 1-vs-2 regimes** (BIC + bootstrap) | Formaliza "houve épocas diferentes?" como teste, não preditor | `hmmlearn` |

**Menções honrosas:** LZ-complexity/compressão (`antropy`), SampEn, DFA/Hurst (`nolds`), ARCH-LM (`statsmodels`), Monte Carlo SSA (`ssalib`), wavelets com universal threshold (`PyWavelets`).

**Descartados com justificativa:**
- **Dieharder/TestU01** — exigem ~10⁵× mais dados (gigabytes de números)
- **NIST SP 800-22 completo** — 3+ testes exigem ≥10⁶ bits; poder fraco (IACR 2022/169 o considera "obsoleto"); rodar o subconjunto viável só como selo de auditoria
- **Benford** — estruturalmente inaplicável a uniformes limitados [1,60]
- **Kalman** — suavizador, não teste; cria autocorrelação espúria
- **GARCH completo** — só o teste ARCH-LM se justifica

---

## Por que a roleta foi prevista e a loteria não (central para a tese)

**Casos de sucesso na roleta:**
- **Thorp & Shannon (1961):** primeiro wearable computer da história; cronometrava rotor e bola, previa o octante de queda com vantagem de +44%
- **Eudaemons (~1977-80):** física newtoniana em computador no sapato; previam qual dos 8 setores
- **Small & Tse (2012, arXiv:1204.6412):** confirmação formal — conhecendo condições iniciais, retorno esperado de +18%

**A diferença física fundamental:**
1. **Janela de observação:** na roleta, as condições iniciais são observáveis *depois do lançamento e antes do fechamento das apostas*. Na loteria, a aposta fecha antes de o estado físico relevante existir
2. **Grau de caoticidade:** a roleta tem fase laminar longa (bola orbitando de forma previsível) e só fica caótica nos ricochetes finais. O globo opera minutos em mistura caótica plena — o expoente de Lyapunov destrói a informação de condição inicial
3. **Sem memória entre sorteios:** o estado do globo não codifica sorteios anteriores

**Loterias exploradas de verdade — nunca pela física:**
- Stefan Mandel: cobertura combinatória (estrutura de prêmio)
- Selbee / grupo do MIT (Cash WinFall): mecânica de roll-down do prêmio (~$27M)
- Mohan Srivastava (2003): raspadinhas com informação impressa vazando (falha de design)
- **Lição:** todos os ataques bem-sucedidos foram à *estrutura do jogo*, nunca ao sorteio físico

**Renaissance/Simons:** HMMs via Baum-Welch (Leonard Baum foi sócio fundador). Funciona em mercados porque mercados *têm* estados ocultos e autocorrelação real. A loteria é projetada e auditada para não ter.

---

## Síntese estratégica

Os métodos #1–#4 formam o desfecho rigoroso da tese:
1. O **teste hipergeométrico correto** reavalia o único sinal sobrevivente (p=0.0095)
2. O **Higher Criticism** dá o veredito global sobre todas as frentes de uma vez
3. O **e-processo** acumula evidência prospectivamente, sorteio a sorteio, sem p-hacking
4. O **Bayes factor** permite concluir *afirmativamente* que o sistema é uniforme

O resultado publicável provável: auditoria estatística independente confirmando a
uniformidade do sistema de loterias brasileiro, com metodologia reproduzível.
