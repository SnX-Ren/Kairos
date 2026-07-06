"""
Kairos — Episode Hunt
Localiza e caracteriza os episodios temporais detectados pelo scan statistics:
qual numero, quais concursos, quais datas, e a magnitude do desvio.

Uso: python episode_hunt.py lotofacil
"""

import csv
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
rng = np.random.default_rng(7)


def load(name):
    draws, concursos, datas = [], [], []
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
                concursos.append(int(r["concurso"]))
                datas.append(r.get("data", ""))
    nums = [n for d in draws for n in d]
    return draws, concursos, datas, min(nums), max(nums), len(draws[-1])


def null_threshold(n, N, k, windows, stride, n_sims=100, q=0.999):
    """Percentil q do maximo de z em sorteios uniformes simulados."""
    maxima = []
    for _ in range(n_sims):
        r = rng.random((n, N))
        idx = np.argpartition(r, k, axis=1)[:, :k]
        P = np.zeros((n, N), dtype=np.int8)
        P[np.arange(n)[:, None], idx] = 1
        cs = np.vstack([np.zeros((1, N), dtype=np.int32), P.cumsum(axis=0)])
        p = k / N
        best = 0.0
        for w in windows:
            if w >= n: continue
            starts = np.arange(0, n - w, stride)
            counts = cs[starts + w] - cs[starts]
            z = np.abs(counts - w * p) / np.sqrt(w * p * (1 - p))
            best = max(best, float(z.max()))
        maxima.append(best)
    return float(np.quantile(maxima, 0.95)), float(np.mean(maxima))


def hunt(name):
    draws, concursos, datas, lo, hi, k = load(name)
    n, N = len(draws), hi - lo + 1
    p = k / N
    windows = (100, 250, 500)
    stride = 25

    P = np.zeros((n, N), dtype=np.int8)
    for t, d in enumerate(draws):
        for x in d:
            P[t, x - lo] = 1
    cs = np.vstack([np.zeros((1, N), dtype=np.int32), P.cumsum(axis=0)])

    thresh95, null_mean = null_threshold(n, N, k, windows, stride)
    print(f"\n{'='*70}")
    print(f"  EPISODE HUNT — {name} | {n} sorteios | pool {lo}-{hi} | k={k}")
    print(f"  Limiar do null (95%): z = {thresh95:.3f} | media do null: {null_mean:.3f}")
    print(f"{'='*70}\n")

    # coleta todas as regioes acima do limiar
    hits = []
    for w in windows:
        if w >= n: continue
        starts = np.arange(0, n - w, stride)
        counts = cs[starts + w] - cs[starts]                 # (n_starts, N)
        z = (counts - w * p) / np.sqrt(w * p * (1 - p))
        for si, s in enumerate(starts):
            for j in range(N):
                if abs(z[si, j]) > thresh95:
                    hits.append((abs(float(z[si, j])), float(z[si, j]),
                                 lo + j, int(s), w,
                                 int(counts[si, j]), w * p))

    if not hits:
        print("  Nenhuma regiao acima do limiar — sem episodios.")
        return

    # agrupa por numero, mantendo a janela mais extrema de cada
    best_by_num = {}
    for absz, zval, num, s, w, obs, exp in hits:
        if num not in best_by_num or absz > best_by_num[num][0]:
            best_by_num[num] = (absz, zval, s, w, obs, exp)

    print(f"  {len(hits)} regioes acima do limiar, {len(best_by_num)} numeros envolvidos:\n")
    print(f"  {'Num':>4} {'z':>7} {'dir':>6} {'obs':>5} {'esp':>7}  {'concursos':<16} {'periodo':<24}")
    for num, (absz, zval, s, w, obs, exp) in sorted(
            best_by_num.items(), key=lambda kv: -kv[1][0]):
        c_ini, c_fim = concursos[s], concursos[min(s + w - 1, n - 1)]
        d_ini, d_fim = datas[s], datas[min(s + w - 1, n - 1)]
        direction = "QUENTE" if zval > 0 else "FRIO"
        print(f"  #{num:03d} {zval:>7.2f} {direction:>6} {obs:>5} {exp:>7.1f}  "
              f"{c_ini}-{c_fim:<9} {d_ini} a {d_fim}")

    # detalhe do episodio mais forte: perfil temporal do numero
    top_num, (absz, zval, s, w, obs, exp) = max(
        best_by_num.items(), key=lambda kv: kv[1][0])
    j = top_num - lo
    print(f"\n  PERFIL DO MAIS EXTREMO — #{top_num}:")
    print(f"  frequencia por bloco de 250 sorteios (esperado {250*p:.0f}):")
    for b0 in range(0, n - 250 + 1, 250):
        c = int(P[b0:b0+250, j].sum())
        bar = "#" * int(c / 5)
        flag = "  <<<" if b0 <= s < b0 + 250 else ""
        print(f"    {concursos[b0]:>5}-{concursos[min(b0+249, n-1)]:>5}  {c:>4}  {bar}{flag}")


if __name__ == "__main__":
    hunt(sys.argv[1] if len(sys.argv) > 1 else "lotofacil")
