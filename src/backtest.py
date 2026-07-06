"""
Kairos — Backtest do Gerador
Para cada concurso t dos ultimos N_TEST, gera jogos usando APENAS dados
anteriores a t (sem vazamento temporal) e conta acertos contra o resultado real.

Estrategias comparadas:
  random       — amostragem uniforme pura (baseline)
  kairos       — filtros default do engine (fib_filter + anti_repeat + freq_blend)
  freq_only    — apenas mistura de frequencia (o unico sinal com p<0.05)
  fib_only     — apenas o filtro Fibonacci

Metrica: media de acertos por jogo. Esperado aleatorio (megasena): 6*6/60 = 0.60
"""

import csv
import numpy as np
from collections import Counter
from pathlib import Path
from scipy import stats

DATA = Path(__file__).parent.parent / "data" / "megasena.csv"
POOL_LO, POOL_HI, K = 1, 60, 6
POOL = list(range(POOL_LO, POOL_HI + 1))
FIBS = {1, 2, 3, 5, 8, 13, 21, 34, 55}

N_TEST  = 500   # ultimos N concursos avaliados
N_GAMES = 20    # jogos gerados por concurso por estrategia
rng = np.random.default_rng(42)

REPEAT_FACTOR = 0.85
FREQ_WEIGHT   = 0.30

draws = []
with open(DATA, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        draws.append([int(r[f"b{i}"]) for i in range(1, K + 1)])
n_draws = len(draws)
print(f"Sorteios: {n_draws} | avaliando os ultimos {N_TEST} | {N_GAMES} jogos/concurso/estrategia\n")


def make_weights(history, fib_filter, anti_repeat, freq_blend):
    """Espelha kairos_engine.base_weights, mas restrito ao historico dado."""
    freq = Counter(n for d in history for n in d)
    total = sum(freq.values())
    uniform = 1.0 / len(POOL)
    w = {}
    for n in POOL:
        if freq_blend and total > 0:
            w[n] = (1 - FREQ_WEIGHT) * uniform + FREQ_WEIGHT * (freq.get(n, 0) / total)
        else:
            w[n] = uniform
    prev = set(history[-1])
    for n in prev:
        if fib_filter and n in FIBS:
            w[n] = 0.0
        elif anti_repeat:
            w[n] *= REPEAT_FACTOR
    return w


def sample_game(weights):
    picked = []
    for _ in range(K):
        nums = [n for n in POOL if n not in picked and weights[n] > 0]
        ws = np.array([weights[n] for n in nums])
        if ws.sum() == 0:
            nums = [n for n in POOL if n not in picked]
            ws = np.ones(len(nums))
        ws = ws / ws.sum()
        picked.append(int(rng.choice(nums, p=ws)))
    return set(picked)


STRATEGIES = {
    "random":    dict(fib_filter=False, anti_repeat=False, freq_blend=False),
    "kairos":    dict(fib_filter=True,  anti_repeat=True,  freq_blend=True),
    "freq_only": dict(fib_filter=False, anti_repeat=False, freq_blend=True),
    "fib_only":  dict(fib_filter=True,  anti_repeat=False, freq_blend=False),
}

hits = {s: [] for s in STRATEGIES}
start = n_draws - N_TEST

for t in range(start, n_draws):
    history = draws[:t]
    target = set(draws[t])
    for strat, cfg in STRATEGIES.items():
        w = make_weights(history, **cfg)
        for _ in range(N_GAMES):
            game = sample_game(w)
            hits[strat].append(len(game & target))
    if (t - start + 1) % 100 == 0:
        print(f"  [{t - start + 1}/{N_TEST}] concursos processados...")

EXPECTED = K * K / len(POOL)   # 0.60

print(f"\n{'='*70}")
print("  RESULTADO DO BACKTEST")
print(f"{'='*70}")
print(f"  Esperado teorico (aleatorio): {EXPECTED:.4f} acertos/jogo\n")
print(f"  {'Estrategia':<12} {'Media':>8} {'StdErr':>8} {'vs random':>10} {'p-value':>9}")

base = np.array(hits["random"], dtype=float)
for strat in STRATEGIES:
    h = np.array(hits[strat], dtype=float)
    mean = h.mean()
    se = h.std(ddof=1) / np.sqrt(len(h))
    if strat == "random":
        print(f"  {strat:<12} {mean:>8.4f} {se:>8.4f} {'—':>10} {'—':>9}")
    else:
        diff = mean - base.mean()
        _, pval = stats.ttest_ind(h, base, equal_var=False)
        sig = " *" if pval < 0.05 else ""
        print(f"  {strat:<12} {mean:>8.4f} {se:>8.4f} {diff:>+10.4f} {pval:>9.4f}{sig}")

print(f"\n  Jogos avaliados por estrategia: {len(base):,}")

# distribuicao de acertos da estrategia kairos vs random
print(f"\n  Distribuicao de acertos (freq. relativa):")
print(f"  {'Acertos':<8} {'random':>9} {'kairos':>9}")
for k in range(K + 1):
    fr = float(np.mean(base == k))
    fk = float(np.mean(np.array(hits['kairos']) == k))
    if fr > 0.0001 or fk > 0.0001:
        print(f"  {k:<8} {fr:>9.4f} {fk:>9.4f}")
print(f"{'='*70}")
