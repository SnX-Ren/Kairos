"""
Kairos — Conditional Intra-Draw Analysis
Testa se a ordem de sorteio carrega estrutura condicional:
P(bola_k | bolas_anteriores) != P(bola_k)
"""

import csv
import math
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from math import comb

DATA = Path(__file__).parent.parent / "data" / "megasena.csv"
POOL = 60
N    = 6

# ── load com ordem de sorteio ─────────────────────────────────────────────────
ordered_draws = []   # lista de [o1, o2, o3, o4, o5, o6] — ordem real
sorted_draws  = []   # lista de [b1..b6] — ordem crescente

with open(DATA, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        od = [int(r[f"o{i}"]) for i in range(1, N + 1)]
        sd = [int(r[f"b{i}"]) for i in range(1, N + 1)]
        ordered_draws.append(od)
        sorted_draws.append(sd)

n_draws = len(ordered_draws)
print(f"Sorteios carregados: {n_draws}")

# ── 1. FREQUENCIA MARGINAL (baseline) ─────────────────────────────────────────
marginal = Counter(n for draw in sorted_draws for n in draw)
p_marginal = {n: marginal[n] / (n_draws * N) for n in range(1, POOL + 1)}

# ── 2. PROBABILIDADE CONDICIONAL: P(ok | o1) ─────────────────────────────────
# dado que a 1a bola foi X, qual a distribuicao das bolas seguintes?
cond_given_first = defaultdict(Counter)
for draw in ordered_draws:
    first = draw[0]
    for rest in draw[1:]:
        cond_given_first[first][rest] += 1

# para cada primeira bola, calcula desvio medio da distribuicao condicional vs marginal
first_ball_deviations = {}
for first, counts in cond_given_first.items():
    total = sum(counts.values())
    devs = []
    for n in range(1, POOL + 1):
        if n == first: continue
        p_cond = counts.get(n, 0) / total
        p_marg = p_marginal[n]
        devs.append(abs(p_cond - p_marg) / p_marg)
    first_ball_deviations[first] = np.mean(devs)

# ── 3. PARES CONDICIONAIS MAIS FORTES ────────────────────────────────────────
# P(b | a saiu antes no mesmo sorteio) vs P(b) marginal
pair_counts  = defaultdict(Counter)  # pair_counts[a][b] = vezes que b saiu DEPOIS de a
pair_total   = Counter()             # quantas vezes a apareceu em pos nao-ultima

for draw in ordered_draws:
    for i, a in enumerate(draw):
        for b in draw[i+1:]:        # b saiu DEPOIS de a na sequencia
            pair_counts[a][b] += 1
            pair_total[a] += 1

# lift: P(b|a) / P(b) — quanto a presença de a amplifica b
lifts = []
for a, followers in pair_counts.items():
    total_a = pair_total[a]
    for b, count in followers.items():
        p_cond = count / total_a
        p_b    = p_marginal[b]
        lift   = p_cond / p_b if p_b > 0 else 0
        # filtro: minimo de ocorrencias para ser significativo
        if count >= 8:
            lifts.append((lift, a, b, count, p_cond, p_b))

lifts_pos = sorted(lifts, reverse=True)[:15]   # mais atraídos
lifts_neg = sorted(lifts)[:15]                  # mais repelidos

# ── 4. POSICAO DE SORTEIO: cada numero tem preferencia por posicao? ───────────
pos_dist = defaultdict(Counter)   # pos_dist[numero][posicao 1-6] = contagem
for draw in ordered_draws:
    for pos, num in enumerate(draw, 1):
        pos_dist[num][pos] += 1

# chi-quadrado por numero: distribuicao de posicao e uniforme?
expected_pos = n_draws / POOL   # quantas vezes cada num aparece em media por posicao
pos_chi2 = {}
for num in range(1, POOL + 1):
    obs = [pos_dist[num].get(p, 0) for p in range(1, N + 1)]
    total = sum(obs)
    if total < 10: continue
    exp = total / N
    chi2 = sum((o - exp) ** 2 / exp for o in obs if exp > 0)
    pos_chi2[num] = (chi2, obs, total)

top_pos_bias = sorted(pos_chi2.items(), key=lambda x: x[1][0], reverse=True)[:10]

# ── 5. TESTE GLOBAL: ordem vs aleatorio ──────────────────────────────────────
# se a ordem fosse aleatoria dentro do sorteio, P(posicao k | numero X) = 1/6
# chi2 global
all_chi2 = [v[0] for v in pos_chi2.values()]
mean_chi2 = np.mean(all_chi2)
# chi2 critico (5 graus de liberdade, alpha=0.05) = 11.07
chi2_critico = 11.07
numeros_com_vies = sum(1 for c in all_chi2 if c > chi2_critico)

# ── 6. PRIMEIRA BOLA vs ULTIMA: padrao de abertura/fechamento ────────────────
first_freq = Counter(draw[0] for draw in ordered_draws)
last_freq  = Counter(draw[-1] for draw in ordered_draws)

top5_first = first_freq.most_common(5)
top5_last  = last_freq.most_common(5)

# ── REPORT ────────────────────────────────────────────────────────────────────
SEP = "=" * 68

print(f"\n{SEP}")
print("  KAIROS — CONDITIONAL INTRA-DRAW ANALYSIS")
print(f"  {n_draws} sorteios | ordem real (dezenasOrdemSorteio)")
print(SEP)

print(f"\n[1] TESTE GLOBAL DE VIES DE POSICAO")
print(f"    Chi2 medio por numero:  {mean_chi2:.3f}")
print(f"    Chi2 critico (p=0.05):  {chi2_critico}")
print(f"    Numeros com vies sig.:  {numeros_com_vies} de {len(all_chi2)}")
if numeros_com_vies > 3:
    print(f"    >> SINAL: {numeros_com_vies} numeros mostram preferencia de posicao significativa")
else:
    print(f"    >> Sem vies de posicao significativo")

print(f"\n[2] NUMEROS COM MAIOR VIES DE POSICAO (chi2 mais alto)")
print(f"    {'Num':>4}  {'Chi2':>6}  {'Pos1':>5} {'Pos2':>5} {'Pos3':>5} {'Pos4':>5} {'Pos5':>5} {'Pos6':>5}  Total")
for num, (chi2, obs, total) in top_pos_bias:
    obs_str = "  ".join(f"{o:5d}" for o in obs)
    print(f"    #{num:02d}   {chi2:6.2f}  {obs_str}  {total:5d}")

print(f"\n[3] PARES MAIS ATRAIDOS (lift > 1 = aparece mais que o esperado)")
print(f"    {'Seq':>8}  {'Lift':>6}  {'Obs':>5}  {'P(b|a)':>8}  {'P(b)':>8}")
for lift, a, b, count, p_cond, p_b in lifts_pos[:10]:
    print(f"    #{a:02d}->#b{b:02d}   {lift:6.3f}  {count:5d}  {p_cond:8.4f}  {p_b:8.4f}")

print(f"\n[4] PARES MAIS REPELIDOS (lift < 1 = aparece menos que o esperado)")
print(f"    {'Seq':>8}  {'Lift':>6}  {'Obs':>5}  {'P(b|a)':>8}  {'P(b)':>8}")
for lift, a, b, count, p_cond, p_b in lifts_neg[:10]:
    print(f"    #{a:02d}->#b{b:02d}   {lift:6.3f}  {count:5d}  {p_cond:8.4f}  {p_b:8.4f}")

print(f"\n[5] PRIMEIRA BOLA sorteada vs ULTIMA bola sorteada")
print(f"    Top 5 primeiras:  {[(f'#{n}', c) for n,c in top5_first]}")
print(f"    Top 5 ultimas:    {[(f'#{n}', c) for n,c in top5_last]}")

print(f"\n[6] NUMEROS COM MAIOR DESVIO CONDICIONAL DADO A 1a BOLA")
top_first_dev = sorted(first_ball_deviations.items(), key=lambda x: x[1], reverse=True)[:5]
bot_first_dev = sorted(first_ball_deviations.items(), key=lambda x: x[1])[:5]
print(f"    Mais influentes (1a bola que mais muda distribuicao seguinte):")
for n, dev in top_first_dev:
    print(f"      #{n:02d}: desvio medio {dev:.4f}")
print(f"    Menos influentes:")
for n, dev in bot_first_dev:
    print(f"      #{n:02d}: desvio medio {dev:.4f}")

print(f"\n{SEP}\n")
