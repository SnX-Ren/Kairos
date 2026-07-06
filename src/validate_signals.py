"""
Kairos — Validacao Estatistica dos Sinais
1. Teste de permutacao para os lifts de pares (corrige multiplas comparacoes)
2. Teste de permutacao para o deficit Fibonacci
3. Investigacao temporal do #35 (vies de posicao persiste nas duas metades?)
4. Scan de uniformidade multi-loteria (chi2 + p-value por loteria)
"""

import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from scipy import stats

rng = np.random.default_rng(42)

DATA_DIR = Path(__file__).parent.parent / "data"
POOL, N = 60, 6
FIBS = {1, 2, 3, 5, 8, 13, 21, 34, 55}
SEP = "=" * 68

# ── load megasena ─────────────────────────────────────────────────────────────
ordered, sorted_d = [], []
with open(DATA_DIR / "megasena.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ordered.append([int(r[f"o{i}"]) for i in range(1, N + 1)])
        sorted_d.append([int(r[f"b{i}"]) for i in range(1, N + 1)])
n_draws = len(ordered)

# ── estatistica de lift (mesma da conditional_analysis) ───────────────────────
def lift_stats(draws):
    """Retorna (max_lift, n_pares_lift>=2) com filtro count>=8."""
    marginal = Counter(n for d in draws for n in d)
    p_marg = {n: marginal[n] / (len(draws) * N) for n in range(1, POOL + 1)}
    pair_counts, pair_total = defaultdict(Counter), Counter()
    for d in draws:
        for i, a in enumerate(d):
            for b in d[i + 1:]:
                pair_counts[a][b] += 1
                pair_total[a] += 1
    max_lift, n_strong = 0.0, 0
    for a, followers in pair_counts.items():
        ta = pair_total[a]
        for b, c in followers.items():
            if c < 8:
                continue
            lift = (c / ta) / p_marg[b] if p_marg[b] > 0 else 0
            max_lift = max(max_lift, lift)
            if lift >= 2.0:
                n_strong += 1
    return max_lift, n_strong

def random_draws(n, pool=POOL, k=N):
    return [list(rng.choice(pool, size=k, replace=False) + 1) for _ in range(n)]

print(SEP)
print("  [1] TESTE DE PERMUTACAO — LIFTS DE PARES")
print(SEP)
obs_max, obs_strong = lift_stats(ordered)
print(f"  Observado:  max lift = {obs_max:.3f} | pares com lift>=2: {obs_strong}")

N_PERM = 200
null_max, null_strong = [], []
for i in range(N_PERM):
    m, s = lift_stats(random_draws(n_draws))
    null_max.append(m)
    null_strong.append(s)

p_max    = float(np.mean([m >= obs_max for m in null_max]))
p_strong = float(np.mean([s >= obs_strong for s in null_strong]))
print(f"  Null ({N_PERM} permutacoes): max lift medio = {np.mean(null_max):.3f} "
      f"(range {min(null_max):.2f}-{max(null_max):.2f})")
print(f"  Null: pares lift>=2 medio = {np.mean(null_strong):.1f} "
      f"(range {min(null_strong)}-{max(null_strong)})")
print(f"  p-value (max lift):        {p_max:.3f}")
print(f"  p-value (n pares fortes):  {p_strong:.3f}")
verdict_lift = "RUIDO (esperado por acaso)" if p_max > 0.05 and p_strong > 0.05 else "SINAL POTENCIAL"
print(f"  >> VEREDITO: {verdict_lift}")

# ── 2. permutacao Fibonacci ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("  [2] TESTE DE PERMUTACAO — DEFICIT FIBONACCI")
print(SEP)
obs_fib_mean = float(np.mean([sum(1 for n in d if n in FIBS) for d in sorted_d]))
null_fib = [
    float(np.mean([sum(1 for n in d if n in FIBS) for d in random_draws(n_draws)]))
    for _ in range(N_PERM)
]
# bicaudal: quao extremo e o desvio observado vs esperado 0.9
exp_fib = 9 / 60 * 6
obs_dev = abs(obs_fib_mean - exp_fib)
p_fib = float(np.mean([abs(m - exp_fib) >= obs_dev for m in null_fib]))
print(f"  Observado: media Fibonacci/sorteio = {obs_fib_mean:.4f} (esperado {exp_fib:.3f})")
print(f"  Null: media = {np.mean(null_fib):.4f}, std = {np.std(null_fib):.4f}")
print(f"  p-value (bicaudal): {p_fib:.3f}")
verdict_fib = "RUIDO" if p_fib > 0.05 else "SINAL SIGNIFICATIVO"
print(f"  >> VEREDITO: {verdict_fib}")

# ── 3. #35 nas duas metades ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("  [3] INVESTIGACAO #35 — VIES DE POSICAO PERSISTE?")
print(SEP)

def pos_chi2_for(num, draws):
    obs = [0] * N
    for d in draws:
        if num in d:
            obs[d.index(num)] += 1
    total = sum(obs)
    if total < 10:
        return None, obs, total
    exp = total / N
    chi2 = sum((o - exp) ** 2 / exp for o in obs)
    return chi2, obs, total

half = n_draws // 2
for label, subset in [("1a metade", ordered[:half]), ("2a metade", ordered[half:]),
                      ("Total", ordered)]:
    chi2, obs, total = pos_chi2_for(35, subset)
    pval = 1 - stats.chi2.cdf(chi2, df=N - 1) if chi2 else float("nan")
    print(f"  {label:<10} chi2={chi2:6.2f}  p={pval:.4f}  posicoes={obs}  n={total}")

c1, _, _ = pos_chi2_for(35, ordered[:half])
c2, _, _ = pos_chi2_for(35, ordered[half:])
p1 = 1 - stats.chi2.cdf(c1, df=5)
p2 = 1 - stats.chi2.cdf(c2, df=5)
persists = p1 < 0.05 and p2 < 0.05
print(f"  >> VEREDITO: {'VIES PERSISTENTE nas duas metades' if persists else 'NAO persiste — provavel flutuacao de amostra'}")
# correcao de multiplas comparacoes: testamos 60 numeros
print(f"  Nota: com 60 numeros testados, espera-se ~3 com p<0.05 por acaso (Bonferroni: p critico = {0.05/60:.5f})")

# ── 4. multi-loteria ──────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  [4] SCAN DE UNIFORMIDADE MULTI-LOTERIA (chi2 global)")
print(SEP)
print(f"  {'Loteria':<16} {'Draws':>6} {'Pool':>5} {'Chi2':>9} {'p-value':>9}  Veredito")

for csv_file in sorted(DATA_DIR.glob("*.csv")):
    if csv_file.stem in ("megasena_historico",):
        continue
    draws_l = []
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        bcols = [c for c in reader.fieldnames if c.startswith("b")]
        for r in reader:
            try:
                draws_l.append([int(r[c]) for c in bcols if r[c] != ""])
            except ValueError:
                continue
    if len(draws_l) < 100:
        continue
    nums = [n for d in draws_l for n in d]
    lo, hi = min(nums), max(nums)
    pool = hi - lo + 1
    freq = Counter(nums)
    obs = np.array([freq.get(n, 0) for n in range(lo, hi + 1)])
    exp = np.full(pool, len(nums) / pool)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    pval = float(1 - stats.chi2.cdf(chi2, df=pool - 1))
    verdict = "uniforme" if pval > 0.05 else "DESVIO SIGNIFICATIVO"
    print(f"  {csv_file.stem:<16} {len(draws_l):>6} {pool:>5} {chi2:>9.2f} {pval:>9.4f}  {verdict}")

print(f"\n{SEP}")
