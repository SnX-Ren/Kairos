"""
Kairos — Auditoria Estatistica
Bateria de testes da pesquisa (data/research_methods.md), em ordem de ranking:

  1. Teste multivariado hipergeometrico (Coronel-Brizio, arXiv:0806.4595)
     — corrige a covariancia negativa que o chi2 de Pearson ignora.
     Para sorteios k de N: T = chi2_pearson * (N-1)/(N-k), df = N-1.
  2. Higher Criticism (Donoho-Jin 2004) sobre p-values binomiais por numero,
     calibrado por Monte Carlo com sorteios simulados (respeita a dependencia
     intra-sorteio).
  3. Bayes factor Dirichlet-multinomial — evidencia PRO ou CONTRA uniformidade,
     com analise de sensibilidade no prior.
  4. Teste de gaps (Haigh 1997) — tempos de espera ~ Geometrica(k/N).
  5. Testes suaves de Neyman — decompoe o desvio em componentes ortogonais
     (tendencia linear, curvatura...), calibrado por Monte Carlo.
  6. Scan statistics (Kulldorff) — localiza epocas de vies por numero.
  7. Permutation entropy (Bandt-Pompe) — determinismo na serie da soma.

Uso: python audit.py [loteria] [--fast]
"""

import csv
import json
import math
import sys
import time
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import gammaln

DATA_DIR = Path(__file__).parent.parent / "data"
rng = np.random.default_rng(7)


# ── util ──────────────────────────────────────────────────────────────────────

def load(name):
    draws = []
    with open(DATA_DIR / f"{name}.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        bcols = sorted((c for c in reader.fieldnames
                        if c[0] == "b" and c[1:].isdigit()), key=lambda c: int(c[1:]))
        for r in reader:
            try:
                d = [int(r[c]) for c in bcols if r[c] != ""]
            except ValueError:
                continue
            if d:
                draws.append(d)
    nums = [n for d in draws for n in d]
    lo, hi = min(nums), max(nums)
    return draws, lo, hi, len(draws[-1])


def presence_matrix(draws, lo, hi):
    P = np.zeros((len(draws), hi - lo + 1), dtype=np.int8)
    for t, d in enumerate(draws):
        for n in d:
            P[t, n - lo] = 1
    return P


def simulate_presence(n_draws, N, k, n_sims):
    """(n_sims, n_draws, N) matrizes de presenca de sorteios uniformes."""
    r = rng.random((n_sims, n_draws, N))
    idx = np.argpartition(r, k, axis=2)[:, :, :k]
    P = np.zeros((n_sims, n_draws, N), dtype=np.int8)
    s, t = np.meshgrid(np.arange(n_sims), np.arange(n_draws), indexing="ij")
    P[s[..., None], t[..., None], idx] = 1
    return P


# ── 1. hipergeometrico multivariado ──────────────────────────────────────────

def test_hypergeometric(P, k):
    n, N = P.shape
    p = k / N
    freq = P.sum(axis=0)
    chi2_pearson = float(((freq - n * p) ** 2 / (n * p)).sum())
    # correcao pela covariancia negativa intra-sorteio:
    T = chi2_pearson * (N - 1) / (N - k)
    df = N - 1
    pval = float(1 - stats.chi2.cdf(T, df))
    p_pearson = float(1 - stats.chi2.cdf(chi2_pearson, df))
    return {
        "test": "Hipergeometrico multivariado",
        "statistic": round(T, 2),
        "chi2_pearson": round(chi2_pearson, 2),
        "p_pearson_incorreto": round(p_pearson, 5),
        "df": df,
        "p_value": round(pval, 5),
        "verdict": "DESVIO SIGNIFICATIVO" if pval < 0.05 else "uniforme",
        "note": f"correcao (N-1)/(N-k) = {(N-1)/(N-k):.4f} sobre o Pearson",
    }


# ── 2. higher criticism ───────────────────────────────────────────────────────

def _hc_stat(pvals, alpha0=0.5):
    pv = np.sort(pvals)
    Np = len(pv)
    i = np.arange(1, Np + 1)
    mask = pv > 1e-12
    hc = np.sqrt(Np) * (i / Np - pv) / np.sqrt(pv * (1 - pv) + 1e-12)
    hc = hc[mask & (i <= alpha0 * Np)]
    return float(hc.max()) if len(hc) else 0.0


def _binom_pvals(P, k):
    n, N = P.shape
    p = k / N
    freq = P.sum(axis=0)
    # p-value bicaudal exato por numero
    return np.array([stats.binomtest(int(f), n, p).pvalue for f in freq])


def test_higher_criticism(P, k, n_sims=800):
    n, N = P.shape
    obs = _hc_stat(_binom_pvals(P, k))
    sims = simulate_presence(n, N, k, n_sims)
    null = np.array([_hc_stat(_binom_pvals(sims[s], k)) for s in range(n_sims)])
    pval = float((null >= obs).mean())
    return {
        "test": "Higher Criticism (Donoho-Jin)",
        "statistic": round(obs, 3),
        "null_mean": round(float(null.mean()), 3),
        "n_sims": n_sims,
        "p_value": round(pval, 4),
        "verdict": "SINAL RARO-E-FRACO DETECTADO" if pval < 0.05 else "uniforme",
    }


# ── 3. bayes factor dirichlet-multinomial ─────────────────────────────────────

def test_bayes_factor(P):
    n, N = P.shape
    counts = P.sum(axis=0).astype(float)
    T = counts.sum()
    out = {}
    for alpha in (0.5, 1.0, 5.0, 10.0):
        log_uniform = -T * np.log(N)
        log_dirich = (gammaln(N * alpha) - gammaln(N * alpha + T)
                      + (gammaln(alpha + counts) - gammaln(alpha)).sum())
        log10_bf = (log_uniform - log_dirich) / np.log(10)
        out[f"alpha={alpha}"] = round(float(log10_bf), 2)
    best = min(out.values())  # cenario menos favoravel a uniformidade
    verdict = ("EVIDENCIA PRO-UNIFORMIDADE" if best > 0.5 else
               "EVIDENCIA PRO-VIES" if best < -0.5 else "inconclusivo")
    return {
        "test": "Bayes factor Dirichlet-multinomial",
        "log10_bf_por_prior": out,
        "note": "log10 BF > 0 favorece uniformidade; cada unidade = 10x mais provavel",
        "verdict": verdict,
    }


# ── 4. gaps (haigh) ───────────────────────────────────────────────────────────

def test_gaps(P, k):
    n, N = P.shape
    p = k / N
    gaps = []
    for j in range(N):
        idx = np.flatnonzero(P[:, j])
        if len(idx) > 1:
            gaps.extend(np.diff(idx).tolist())
    gaps = np.array(gaps)
    max_bin = 30
    obs = np.array([(gaps == g).sum() for g in range(1, max_bin)] +
                   [(gaps >= max_bin).sum()], dtype=float)
    probs = np.array([p * (1 - p) ** (g - 1) for g in range(1, max_bin)] +
                     [(1 - p) ** (max_bin - 1)])
    exp = probs * len(gaps)
    keep = exp >= 5
    chi2 = float((((obs - exp) ** 2 / exp)[keep]).sum())
    df = int(keep.sum()) - 1
    pval = float(1 - stats.chi2.cdf(chi2, df))
    return {
        "test": "Gaps / tempos de espera (Haigh 1997)",
        "n_gaps": int(len(gaps)),
        "gap_medio": round(float(gaps.mean()), 3),
        "gap_esperado": round(1 / p, 3),
        "statistic": round(chi2, 2),
        "df": df,
        "p_value": round(pval, 4),
        "verdict": "clustering temporal" if pval < 0.05 else "geometrico (ok)",
    }


# ── 5. neyman smooth ──────────────────────────────────────────────────────────

def _neyman_components(P, lo, order=4):
    n, N = P.shape
    freq = P.sum(axis=0).astype(float)
    T = freq.sum()
    u = (np.arange(N) + 0.5) / N          # posicao normalizada de cada numero
    x = 2 * u - 1
    V = []
    for j in range(1, order + 1):
        coef = np.zeros(j + 1); coef[j] = 1
        phi = np.sqrt(2 * j + 1) * np.polynomial.legendre.legval(x, coef)
        V.append(float((freq * phi).sum() / np.sqrt(T)))
    return np.array(V)


def test_neyman(P, lo, k, n_sims=800):
    n, N = P.shape
    obs_V = _neyman_components(P, lo)
    obs_S = float((obs_V ** 2).sum())
    sims = simulate_presence(n, N, k, n_sims)
    null_S = np.array([float((_neyman_components(sims[s], lo) ** 2).sum())
                       for s in range(n_sims)])
    pval = float((null_S >= obs_S).mean())
    comp_names = ["tendencia linear", "curvatura", "assimetria cubica", "ordem 4"]
    return {
        "test": "Neyman smooth (ordem 4)",
        "componentes": {name: round(float(v), 3) for name, v in zip(comp_names, obs_V)},
        "S4": round(obs_S, 3),
        "p_value": round(pval, 4),
        "n_sims": n_sims,
        "verdict": ("DESVIO ESTRUTURADO: " +
                    comp_names[int(np.argmax(obs_V ** 2))]) if pval < 0.05 else "uniforme",
    }


# ── 6. scan statistics ────────────────────────────────────────────────────────

def _scan_max(P, k, windows=(100, 250, 500), stride=25):
    n, N = P.shape
    p = k / N
    cs = np.vstack([np.zeros((1, N), dtype=np.int32), P.cumsum(axis=0)])
    best = 0.0
    for w in windows:
        if w >= n:
            continue
        starts = np.arange(0, n - w, stride)
        counts = cs[starts + w] - cs[starts]          # (n_starts, N)
        z = np.abs(counts - w * p) / np.sqrt(w * p * (1 - p))
        best = max(best, float(z.max()))
    return best


def test_scan(P, k, n_sims=150):
    n, N = P.shape
    obs = _scan_max(P, k)
    sims = simulate_presence(n, N, k, n_sims)
    null = np.array([_scan_max(sims[s], k) for s in range(n_sims)])
    pval = float((null >= obs).mean())
    return {
        "test": "Scan statistics (janelas 100/250/500)",
        "statistic": round(obs, 3),
        "null_mean": round(float(null.mean()), 3),
        "n_sims": n_sims,
        "p_value": round(pval, 4),
        "verdict": "EPISODIO DE VIES LOCALIZADO" if pval < 0.05 else "sem episodios",
    }


# ── 7. permutation entropy ────────────────────────────────────────────────────

def _perm_entropy(series, d=4):
    n = len(series) - d + 1
    pats = Counter()
    for i in range(n):
        pats[tuple(np.argsort(series[i:i + d]))] += 1
    total = sum(pats.values())
    probs = np.array([c / total for c in pats.values()])
    H = float(-(probs * np.log(probs)).sum() / np.log(math.factorial(d)))
    missing = math.factorial(d) - len(pats)
    return H, missing


def test_perm_entropy(draws, n_sims=1000):
    series = np.array([sum(d) for d in draws], dtype=float)
    obs_H, obs_missing = _perm_entropy(series)
    null_H = []
    s = series.copy()
    for _ in range(n_sims):
        rng.shuffle(s)
        null_H.append(_perm_entropy(s)[0])
    null_H = np.array(null_H)
    pval = float((null_H <= obs_H).mean())
    return {
        "test": "Permutation entropy (Bandt-Pompe, d=4, serie da soma)",
        "H_norm": round(obs_H, 4),
        "null_mean": round(float(null_H.mean()), 4),
        "missing_patterns": int(obs_missing),
        "p_value": round(pval, 4),
        "verdict": "DETERMINISMO NA SERIE" if pval < 0.05 else "ruido (ok)",
    }


# ── runner ────────────────────────────────────────────────────────────────────

def run_audit(name, fast=False):
    t0 = time.time()
    draws, lo, hi, k = load(name)
    P = presence_matrix(draws, lo, hi)
    n_sims = 300 if fast else 800
    scan_sims = 80 if fast else 150

    results = [
        test_hypergeometric(P, k),
        test_higher_criticism(P, k, n_sims=n_sims),
        test_bayes_factor(P),
        test_gaps(P, k),
        test_neyman(P, lo, k, n_sims=n_sims),
        test_scan(P, k, n_sims=scan_sims),
        test_perm_entropy(draws),
    ]
    report = {
        "lottery": name,
        "n_draws": len(draws),
        "pool": f"{lo}-{hi}",
        "balls": k,
        "ran_at": time.strftime("%d/%m/%Y %H:%M"),
        "elapsed_s": round(time.time() - t0, 1),
        "tests": results,
    }
    out = DATA_DIR / f"audit_{name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_report(rep):
    print(f"\n{'='*68}")
    print(f"  AUDITORIA — {rep['lottery']} | {rep['n_draws']} sorteios | "
          f"pool {rep['pool']} | {rep['elapsed_s']}s")
    print(f"{'='*68}")
    for t in rep["tests"]:
        pv = t.get("p_value")
        pv_str = f"p={pv}" if pv is not None else ""
        print(f"\n  [{t['test']}]  {pv_str}")
        for key, val in t.items():
            if key in ("test", "p_value", "verdict"):
                continue
            print(f"     {key}: {val}")
        print(f"     >> {t['verdict']}")
    print(f"\n{'='*68}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv
    name = args[0] if args else "megasena"
    rep = run_audit(name, fast=fast)
    print_report(rep)
