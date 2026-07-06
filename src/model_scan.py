"""
Kairos — Model Scan
Testa multiplos modelos matematicos e rankeia pelo desvio do esperado aleatorio.
"""

import csv, math, statistics
from collections import defaultdict
from pathlib import Path
from math import comb

DATA  = Path(__file__).parent.parent / "data" / "megasena.csv"
POOL  = 60
N     = 6

# ── load ──────────────────────────────────────────────────────────────────────
draws = []
with open(DATA, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        draws.append([int(r[f"b{i}"]) for i in range(1, N + 1)])

n_draws = len(draws)

# ── sets matematicos ──────────────────────────────────────────────────────────
PRIMES    = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59}
TRIANGLES = {1,3,6,10,15,21,28,36,45,55}
LUCAS     = {2,1,3,4,7,11,18,29,47}
FIBS      = {1,2,3,5,8,13,21,34,55}
SQUARES   = {1,4,9,16,25,36,49}

def hypergeometric_expected(K, n_draws):
    """Distribuicao esperada de k acertos em draw de N de pool POOL com K especiais."""
    return {
        k: comb(K, k) * comb(POOL - K, N - k) / comb(POOL, N) * n_draws
        for k in range(min(K, N) + 1)
    }

def chi_square(observed: dict, expected: dict) -> float:
    """Chi-quadrado entre observado e esperado (ignora bins com esperado < 1)."""
    keys = [k for k in expected if expected[k] >= 1]
    return sum((observed.get(k, 0) - expected[k]) ** 2 / expected[k] for k in keys)

# ── 1. SETS DE NUMEROS ESPECIAIS ──────────────────────────────────────────────
results = {}

for name, special_set in [
    ("Primos",     PRIMES),
    ("Fibonacci",  FIBS),
    ("Triangulares", TRIANGLES),
    ("Lucas",      LUCAS),
    ("Quadrados",  SQUARES),
]:
    K = len(special_set & set(range(1, POOL + 1)))
    counts_per_draw = [sum(1 for n in d if n in special_set) for d in draws]
    obs = defaultdict(int)
    for c in counts_per_draw: obs[c] += 1
    exp = hypergeometric_expected(K, n_draws)
    chi2 = chi_square(obs, exp)
    mean_obs = statistics.mean(counts_per_draw)
    mean_exp = K / POOL * N
    results[name] = {
        "chi2": chi2,
        "mean_obs": mean_obs,
        "mean_exp": mean_exp,
        "desvio_pct": abs(mean_obs - mean_exp) / mean_exp * 100,
        "K": K,
    }

# ── 2. ENTROPIA DE SHANNON ────────────────────────────────────────────────────
def entropy(draw):
    total = POOL
    # entropia baseada nas frequencias relativas dos numeros no draw
    return -sum((1/N) * math.log2(1/N) for _ in draw)  # entropia do draw uniforme

# entropia real: baseada em frequencia acumulada ate aquele ponto
entropies = []
freq_so_far = defaultdict(int)
for draw in draws:
    for n in draw: freq_so_far[n] += 1
    total = sum(freq_so_far.values())
    probs = [freq_so_far[n] / total for n in range(1, POOL + 1)]
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    entropies.append(H)

# em sistema uniforme, entropia tende a log2(60) = 5.906
H_max   = math.log2(POOL)
H_mean  = statistics.mean(entropies[-100:])   # ultimos 100 para estabilizar
H_desvio = abs(H_mean - H_max) / H_max * 100
results["Entropia Shannon"] = {
    "chi2": None,
    "mean_obs": H_mean,
    "mean_exp": H_max,
    "desvio_pct": H_desvio,
    "K": None,
}

# ── 3. GAP ANALYSIS ───────────────────────────────────────────────────────────
last_seen = {}
gaps = defaultdict(list)
for i, draw in enumerate(draws):
    for n in draw:
        if n in last_seen:
            gaps[n].append(i - last_seen[n])
        last_seen[n] = i

all_gaps = [g for gs in gaps.values() for g in gs]
# em sistema uniforme, gap medio esperado = POOL / N = 10 sorteios
gap_expected = POOL / N
gap_mean     = statistics.mean(all_gaps) if all_gaps else 0
gap_std_obs  = statistics.stdev(all_gaps) if len(all_gaps) > 1 else 0
gap_std_exp  = math.sqrt(POOL/N * (POOL/N - 1))  # geometric distribution std
gap_desvio   = abs(gap_mean - gap_expected) / gap_expected * 100

# numero mais irregular (maior variancia nos gaps)
gap_variance = {n: statistics.variance(gaps[n]) for n in gaps if len(gaps[n]) > 5}
most_irregular = sorted(gap_variance.items(), key=lambda x: x[1], reverse=True)[:5]

results["Gap Analysis"] = {
    "chi2": None,
    "mean_obs": gap_mean,
    "mean_exp": gap_expected,
    "desvio_pct": gap_desvio,
    "K": None,
    "extra": f"std obs={gap_std_obs:.2f} exp~{gap_std_exp:.2f} | mais irregulares: {[n for n,_ in most_irregular]}",
}

# ── 4. AUTOCORRELACAO ─────────────────────────────────────────────────────────
# para cada numero, calcula correlacao entre aparicao no draw i e draw i+1
presence = {n: [1 if n in d else 0 for d in draws] for n in range(1, POOL + 1)}

def autocorr(series, lag=1):
    n = len(series)
    mean = statistics.mean(series)
    var  = statistics.variance(series)
    if var == 0: return 0
    cov = sum((series[i] - mean) * (series[i - lag] - mean)
              for i in range(lag, n)) / (n - lag)
    return cov / var

autocorrs = {n: autocorr(presence[n], lag=1) for n in range(1, POOL + 1)}
ac_mean   = statistics.mean(autocorrs.values())
ac_max    = max(autocorrs.values())
ac_min    = min(autocorrs.values())
# em sistema i.i.d., autocorrelacao esperada ~ 0
ac_desvio = abs(ac_mean) / (1/POOL) * 100

most_pos = sorted(autocorrs.items(), key=lambda x: x[1], reverse=True)[:5]
most_neg = sorted(autocorrs.items(), key=lambda x: x[1])[:5]

results["Autocorrelacao"] = {
    "chi2": None,
    "mean_obs": ac_mean,
    "mean_exp": 0.0,
    "desvio_pct": abs(ac_mean) * 1000,  # escala diferente — amplificado para comparar
    "K": None,
    "extra": f"max={ac_max:.4f} (#{most_pos[0][0]}) | min={ac_min:.4f} (#{most_neg[0][0]}) | media={ac_mean:.6f}",
}

# ── 5. RSI (Relative Strength Index) ─────────────────────────────────────────
WINDOW = 14
rsi_scores = {}
for n in range(1, POOL + 1):
    series = presence[n]
    if len(series) < WINDOW + 1: continue
    gains, losses = [], []
    for i in range(1, len(series)):
        delta = series[i] - series[i-1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    # RSI final (ultimos WINDOW periodos)
    avg_gain = statistics.mean(gains[-WINDOW:]) if gains else 0
    avg_loss = statistics.mean(losses[-WINDOW:]) if losses else 0
    rs  = avg_gain / avg_loss if avg_loss > 0 else float('inf')
    rsi = 100 - (100 / (1 + rs)) if avg_loss > 0 else 100
    rsi_scores[n] = rsi

rsi_vals  = list(rsi_scores.values())
rsi_mean  = statistics.mean(rsi_vals)
rsi_std   = statistics.stdev(rsi_vals)
# numeros mais overbought / oversold
overbought = sorted(rsi_scores.items(), key=lambda x: x[1], reverse=True)[:5]
oversold   = sorted(rsi_scores.items(), key=lambda x: x[1])[:5]

results["RSI"] = {
    "chi2": None,
    "mean_obs": rsi_mean,
    "mean_exp": 50.0,
    "desvio_pct": abs(rsi_mean - 50) / 50 * 100,
    "K": None,
    "extra": f"overbought: {[n for n,_ in overbought[:3]]} | oversold: {[n for n,_ in oversold[:3]]} | std={rsi_std:.2f}",
}

# ── 6. PARIDADE ───────────────────────────────────────────────────────────────
parity_counts = [sum(1 for n in d if n % 2 == 0) for d in draws]
par_mean  = statistics.mean(parity_counts)
par_exp   = N / 2  # 3.0
par_desvio = abs(par_mean - par_exp) / par_exp * 100

results["Paridade (pares)"] = {
    "chi2": None,
    "mean_obs": par_mean,
    "mean_exp": par_exp,
    "desvio_pct": par_desvio,
    "K": 30,
}

# ── RANKING ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  KAIROS — MODEL SCAN RESULTS")
print("="*65)
print(f"  Dataset: {n_draws} sorteios (era moderna, concurso 1141+)\n")

ranked = sorted(results.items(), key=lambda x: x[1]["desvio_pct"], reverse=True)

for rank, (name, r) in enumerate(ranked, 1):
    signal = "*** SINAL FORTE" if r["desvio_pct"] > 5 else \
             "**  sinal medio" if r["desvio_pct"] > 1 else \
             "    ruido"
    obs_str = f"{r['mean_obs']:.4f}" if r['mean_obs'] is not None else "—"
    exp_str = f"{r['mean_exp']:.4f}" if r['mean_exp'] is not None else "—"
    print(f"  [{rank}] {name:<22} desvio={r['desvio_pct']:6.3f}%  "
          f"obs={obs_str}  exp={exp_str}  {signal}")
    if "extra" in r:
        print(f"       {r['extra']}")

print("\n" + "="*65)
print("  DETALHE: AUTOCORRELACAO POR NUMERO")
print("="*65)
print(f"  Mais positiva (sai -> tende a sair de novo):")
for n, ac in most_pos:
    print(f"    #{n:2d}: {ac:+.5f}")
print(f"  Mais negativa (sai -> tende a nao sair logo):")
for n, ac in most_neg:
    print(f"    #{n:2d}: {ac:+.5f}")

print("\n" + "="*65)
print("  DETALHE: RSI (overbought > 60 / oversold < 40)")
print("="*65)
print(f"  Overbought (quentes):  {[(n, f'{v:.1f}') for n,v in overbought]}")
print(f"  Oversold   (frios):    {[(n, f'{v:.1f}') for n,v in oversold]}")
print("="*65 + "\n")
