"""
Kairos — Baseline Analysis
Frequência, co-ocorrência, Fibonacci, soma e golden ratio
"""

import csv
import base64
import io
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "megasena.csv"
OUT  = ROOT / "data" / "dashboard.html"

# ── constants ─────────────────────────────────────────────────────────────────
PHI  = (1 + math.sqrt(5)) / 2          # 1.6180...
FIBS = {1,2,3,5,8,13,21,34,55}         # Fibonacci in 1-60
N_BALLS = 6
POOL    = 60
EXPECTED_SUM = N_BALLS * (POOL + 1) / 2  # 183.0

# ── load ──────────────────────────────────────────────────────────────────────
rows = []
with open(DATA, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        draw = [int(r[f"b{i}"]) for i in range(1, N_BALLS + 1)]
        rows.append(draw)

n_draws     = len(rows)
all_numbers = [n for d in rows for n in d]
freq        = Counter(all_numbers)
numbers     = list(range(1, POOL + 1))
counts      = [freq.get(n, 0) for n in numbers]
expected    = n_draws * N_BALLS / POOL

# ── co-occurrence ─────────────────────────────────────────────────────────────
cooc = np.zeros((POOL, POOL), dtype=int)
for draw in rows:
    for a, b in combinations(draw, 2):
        cooc[a-1][b-1] += 1
        cooc[b-1][a-1] += 1

# ── Fibonacci features ────────────────────────────────────────────────────────
fib_counts_per_draw = [sum(1 for n in d if n in FIBS) for d in rows]
fib_dist = Counter(fib_counts_per_draw)
# expected probability of k Fibonacci numbers in draw of 6 from 60
# hypergeometric: C(9,k)*C(51,6-k)/C(60,6)
from math import comb
fib_expected = {
    k: comb(9, k) * comb(51, 6-k) / comb(60, 6) * n_draws
    for k in range(7)
}

# ── soma features ─────────────────────────────────────────────────────────────
somas = [sum(d) for d in rows]
soma_mean   = np.mean(somas)
soma_std    = np.std(somas)
soma_min    = min(somas)
soma_max    = max(somas)

# ── golden ratio features ─────────────────────────────────────────────────────
ratios = [max(d) / min(d) for d in rows]
ratio_mean = np.mean(ratios)
ratio_std  = np.std(ratios)
phi_diffs  = [abs(r - PHI) for r in ratios]
phi_proximity_mean = np.mean(phi_diffs)

# ── top / bottom ──────────────────────────────────────────────────────────────
sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
top10    = sorted_freq[:10]
bottom10 = sorted_freq[-10:]
flat_cooc = sorted(
    [(cooc[a-1][b-1], a, b) for a in range(1, 61) for b in range(a+1, 61)],
    reverse=True
)

# ── helpers ───────────────────────────────────────────────────────────────────
BG    = "#0f1117"
CARD  = "#1a1d27"
ACC   = "#7c6af7"
GRN   = "#34d399"
RED   = "#f87171"
AMB   = "#fbbf24"
TXT   = "#e2e8f0"
MUTED = "#64748b"

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── plot 1: frequency bar ─────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
ax.set_facecolor(BG)
bar_colors = [
    GRN if c > expected * 1.1 else RED if c < expected * 0.9 else ACC
    for c in counts
]
# mark Fibonacci numbers
fib_mask = [n in FIBS for n in numbers]
for i, (n, c, color, is_fib) in enumerate(zip(numbers, counts, bar_colors, fib_mask)):
    ax.bar(n, c, color=color, width=0.8, zorder=2)
    if is_fib:
        ax.bar(n, c, color=color, width=0.8, zorder=2,
               edgecolor=AMB, linewidth=2)
ax.axhline(expected, color=TXT, linewidth=1, linestyle="--", alpha=0.5,
           label=f"Esperado ({expected:.1f}x)")
ax.set_xlabel("Numero", color=MUTED, fontsize=10)
ax.set_ylabel("Frequencia", color=MUTED, fontsize=10)
ax.set_title("Frequencia por numero (ouro = Fibonacci)", color=TXT, fontsize=13, pad=12)
ax.tick_params(colors=MUTED)
ax.spines[:].set_visible(False)
ax.grid(axis="y", color=MUTED, alpha=0.15, zorder=1)
ax.legend(facecolor=CARD, edgecolor="none", labelcolor=TXT, fontsize=9)
img1 = fig_to_b64(fig1); plt.close(fig1)

# ── plot 2: Fibonacci distribution ───────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
ax.set_facecolor(BG)
ks = [k for k in range(7) if fib_dist.get(k,0) > 0 or fib_expected[k] > 0.5]
x  = np.arange(len(ks))
w  = 0.35
ax.bar(x - w/2, [fib_dist.get(k,0) for k in ks], w, color=AMB, label="Observado", zorder=2)
ax.bar(x + w/2, [fib_expected[k] for k in ks],   w, color=ACC, alpha=0.7,
       label="Esperado (hipergeometrico)", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([f"{k} Fibonacci" for k in ks], color=MUTED, fontsize=9)
ax.set_title("Numeros Fibonacci por sorteio", color=TXT, fontsize=13, pad=12)
ax.set_ylabel("Sorteios", color=MUTED, fontsize=10)
ax.tick_params(colors=MUTED)
ax.spines[:].set_visible(False)
ax.grid(axis="y", color=MUTED, alpha=0.15, zorder=1)
ax.legend(facecolor=CARD, edgecolor="none", labelcolor=TXT, fontsize=9)
img2 = fig_to_b64(fig2); plt.close(fig2)

# ── plot 3: soma distribution ─────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
ax.set_facecolor(BG)
ax.hist(somas, bins=40, color=ACC, alpha=0.85, zorder=2, edgecolor="none")
ax.axvline(EXPECTED_SUM, color=AMB, linewidth=2, linestyle="--",
           label=f"Esperado ({EXPECTED_SUM:.0f})")
ax.axvline(soma_mean, color=GRN, linewidth=2, linestyle="-",
           label=f"Media real ({soma_mean:.1f})")
ax.set_title("Distribuicao da soma dos 6 numeros por sorteio", color=TXT, fontsize=13, pad=12)
ax.set_xlabel("Soma", color=MUTED, fontsize=10)
ax.set_ylabel("Sorteios", color=MUTED, fontsize=10)
ax.tick_params(colors=MUTED)
ax.spines[:].set_visible(False)
ax.grid(axis="y", color=MUTED, alpha=0.15, zorder=1)
ax.legend(facecolor=CARD, edgecolor="none", labelcolor=TXT, fontsize=9)
img3 = fig_to_b64(fig3); plt.close(fig3)

# ── plot 4: golden ratio distribution ────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
ax.set_facecolor(BG)
ax.hist(ratios, bins=40, color=GRN, alpha=0.85, zorder=2, edgecolor="none")
ax.axvline(PHI, color=AMB, linewidth=2, linestyle="--",
           label=f"phi = {PHI:.4f}")
ax.axvline(ratio_mean, color=ACC, linewidth=2, linestyle="-",
           label=f"Media real ({ratio_mean:.3f})")
ax.set_title("Razao max/min por sorteio vs Golden Ratio (phi)", color=TXT, fontsize=13, pad=12)
ax.set_xlabel("max / min", color=MUTED, fontsize=10)
ax.set_ylabel("Sorteios", color=MUTED, fontsize=10)
ax.tick_params(colors=MUTED)
ax.spines[:].set_visible(False)
ax.grid(axis="y", color=MUTED, alpha=0.15, zorder=1)
ax.legend(facecolor=CARD, edgecolor="none", labelcolor=TXT, fontsize=9)
img4 = fig_to_b64(fig4); plt.close(fig4)

# ── plot 5: co-occurrence heatmap ─────────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(9, 8), facecolor=BG)
ax.set_facecolor(BG)
cmap = mcolors.LinearSegmentedColormap.from_list("k", [BG, ACC, GRN])
im = ax.imshow(cooc, cmap=cmap, aspect="auto")
cbar = fig5.colorbar(im, ax=ax, fraction=0.03)
cbar.ax.tick_params(colors=MUTED)
cbar.outline.set_visible(False)
ax.set_title("Co-ocorrencia entre numeros", color=TXT, fontsize=13, pad=12)
ax.set_xlabel("Numero", color=MUTED, fontsize=9)
ax.set_ylabel("Numero", color=MUTED, fontsize=9)
ax.tick_params(colors=MUTED, labelsize=7)
ax.spines[:].set_visible(False)
img5 = fig_to_b64(fig5); plt.close(fig5)

# ── html ──────────────────────────────────────────────────────────────────────
def stat_card(label, value, sub=""):
    return f"""<div class="card stat">
      <div class="stat-val">{value}</div>
      <div class="stat-label">{label}</div>
      {"<div class='stat-sub'>" + sub + "</div>" if sub else ""}
    </div>"""

top10_rows = "".join(
    f'<tr><td class="num">#{n}</td>'
    f'<td class="fib">{"F" if n in FIBS else ""}</td>'
    f'<td><div class="bar-wrap"><div class="bar-fill" style="width:{c/max(counts)*100:.0f}%"></div></div></td>'
    f'<td class="cnt">{c}x</td></tr>'
    for n, c in top10
)
bot10_rows = "".join(
    f'<tr><td class="num">#{n}</td>'
    f'<td class="fib">{"F" if n in FIBS else ""}</td>'
    f'<td><div class="bar-wrap"><div class="bar-fill red" style="width:{c/max(counts)*100:.0f}%"></div></div></td>'
    f'<td class="cnt">{c}x</td></tr>'
    for n, c in bottom10
)

soma_desvio = abs(soma_mean - EXPECTED_SUM)
phi_desvio  = abs(ratio_mean - PHI)

html = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Kairos — Baseline Analysis</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f1117; color: #e2e8f0;
          font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; }}
  h1   {{ font-size: 1.6rem; font-weight: 700; margin-bottom: .25rem; }}
  .subtitle {{ color: #64748b; font-size: .9rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
           gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1a1d27; border-radius: 12px; padding: 1.25rem; }}
  .card.stat {{ text-align: center; }}
  .stat-val   {{ font-size: 1.9rem; font-weight: 800; color: #7c6af7; }}
  .stat-label {{ font-size: .75rem; color: #64748b; margin-top: .25rem;
                 text-transform: uppercase; letter-spacing: .05em; }}
  .stat-sub   {{ font-size: .75rem; color: #94a3b8; margin-top: .2rem; }}
  .section {{ margin-bottom: 2.5rem; }}
  h2 {{ font-size: 1rem; font-weight: 600; color: #94a3b8; margin-bottom: 1rem;
        text-transform: uppercase; letter-spacing: .06em; }}
  img {{ width: 100%; border-radius: 10px; display: block; }}
  .tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th  {{ text-align: left; color: #64748b; font-size: .75rem;
         text-transform: uppercase; letter-spacing: .05em; padding-bottom: .5rem; }}
  td  {{ padding: .35rem 0; vertical-align: middle; }}
  td.num {{ color: #7c6af7; font-weight: 700; font-size: .95rem; width: 3rem; }}
  td.fib {{ color: #fbbf24; font-size: .7rem; font-weight: 700; width: 1.5rem; }}
  td.cnt {{ color: #64748b; font-size: .85rem; width: 3rem; text-align: right; }}
  .bar-wrap {{ background: #0f1117; border-radius: 4px; height: 8px; }}
  .bar-fill {{ background: #7c6af7; height: 8px; border-radius: 4px; }}
  .bar-fill.red {{ background: #f87171; }}
  .insight {{ background: #1a1d27; border-left: 3px solid #7c6af7;
              border-radius: 0 10px 10px 0; padding: 1rem 1.25rem;
              margin-bottom: 1rem; font-size: .9rem; line-height: 1.6; }}
  .insight b {{ color: #e2e8f0; }}
  .insight.warn {{ border-color: #fbbf24; }}
  .insight.ok   {{ border-color: #34d399; }}
  .note {{ color: #64748b; font-size: .8rem; margin-top: 1.5rem;
           border-top: 1px solid #1e293b; padding-top: 1rem; }}
</style>
</head>
<body>
<h1>Kairos — Baseline Analysis</h1>
<p class="subtitle">Mega-Sena (concursos 1141+) &middot; {n_draws} sorteios &middot; features: Fibonacci / Soma / Golden Ratio</p>

<div class="grid">
  {stat_card("Sorteios", n_draws)}
  {stat_card("Mais frequente", f"#{top10[0][0]}", f"{top10[0][1]}x")}
  {stat_card("Menos frequente", f"#{bottom10[-1][0]}", f"{bottom10[-1][1]}x")}
  {stat_card("Soma media", f"{soma_mean:.1f}", f"esperado {EXPECTED_SUM:.0f}")}
  {stat_card("Razao media max/min", f"{ratio_mean:.3f}", f"phi={PHI:.3f}")}
  {stat_card("Par mais comum", f"{flat_cooc[0][1]}&{flat_cooc[0][2]}", f"{flat_cooc[0][0]}x juntos")}
</div>

<div class="section">
  <h2>Insights das features</h2>
  <div class="insight {'ok' if soma_desvio < 2 else 'warn'}">
    <b>Soma:</b> Media real {soma_mean:.2f} vs esperado {EXPECTED_SUM:.0f}
    — desvio de {soma_desvio:.2f} pontos ({soma_desvio/EXPECTED_SUM*100:.2f}%).
    {"Distribuicao praticamente centrada — sem sinal de viés na soma." if soma_desvio < 2
     else "Desvio acima de 2 pontos — vale investigar consistencia temporal."}
  </div>
  <div class="insight {'ok' if phi_desvio < 0.1 else 'warn'}">
    <b>Golden Ratio:</b> Razao max/min media = {ratio_mean:.4f} vs phi = {PHI:.4f}
    — desvio de {phi_desvio:.4f}.
    {"Proximo de phi — interessante, mas pode ser coincidencia matematica." if phi_desvio < 0.15
     else "Distante de phi — golden ratio nao e um atrator neste sistema."}
  </div>
  <div class="insight">
    <b>Fibonacci:</b> {fib_dist.get(0,0)} sorteios sem nenhum Fibonacci,
    {fib_dist.get(1,0)} com 1, {fib_dist.get(2,0)} com 2, {fib_dist.get(3,0)} com 3+.
    Esperado pela distribuicao hipergeometrica: ver grafico abaixo.
  </div>
</div>

<div class="section">
  <h2>Frequencia por numero (borda dourada = Fibonacci)</h2>
  <div class="card"><img src="data:image/png;base64,{img1}"></div>
</div>

<div class="section">
  <h2>Fibonacci — observado vs esperado hipergeometrico</h2>
  <div class="card"><img src="data:image/png;base64,{img2}"></div>
</div>

<div class="section">
  <h2>Soma do sorteio</h2>
  <div class="card"><img src="data:image/png;base64,{img3}"></div>
</div>

<div class="section">
  <h2>Golden Ratio — max/min por sorteio</h2>
  <div class="card"><img src="data:image/png;base64,{img4}"></div>
</div>

<div class="section">
  <h2>Top 10 vs Bottom 10 (F = Fibonacci)</h2>
  <div class="card tables">
    <div>
      <table>
        <thead><tr><th>Numero</th><th></th><th>Freq.</th><th></th></tr></thead>
        <tbody>{top10_rows}</tbody>
      </table>
    </div>
    <div>
      <table>
        <thead><tr><th>Numero</th><th></th><th>Freq.</th><th></th></tr></thead>
        <tbody>{bot10_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<div class="section">
  <h2>Co-ocorrencia entre numeros</h2>
  <div class="card"><img src="data:image/png;base64,{img5}"></div>
</div>

<p class="note">
  Verde = acima do esperado &middot; Roxo = dentro do esperado &middot; Vermelho = abaixo &middot; Borda dourada = Fibonacci.<br>
  Apenas concursos 1141+ (globo unico de 60 bolas). Re-execute para atualizar com novos dados.
</p>
</body>
</html>"""

OUT.write_text(html, encoding="utf-8")
print(f"Dashboard: {OUT}")
print(f"Sorteios: {n_draws}")
print(f"Soma — media: {soma_mean:.2f} | esperado: {EXPECTED_SUM:.0f} | desvio: {soma_desvio:.2f}")
print(f"Golden Ratio — media max/min: {ratio_mean:.4f} | phi: {PHI:.4f} | desvio: {phi_desvio:.4f}")
print(f"Fibonacci — 0 por sorteio: {fib_dist.get(0,0)} | 1: {fib_dist.get(1,0)} | 2: {fib_dist.get(2,0)} | 3+: {sum(v for k,v in fib_dist.items() if k>=3)}")
