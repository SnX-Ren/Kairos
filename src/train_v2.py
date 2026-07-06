"""
Kairos — Modelo v2
Features condicionais cross-draw + filtro Fibonacci + teste temporal (MLP).
Target: Precision@6 vs baseline aleatorio (0.10).
"""

import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).parent.parent / "data" / "megasena.csv"
POOL, N = 60, 6
FIBS = {1, 2, 3, 5, 8, 13, 21, 34, 55}
LAGS = 5   # quantos draws anteriores entram como presenca binaria

draws = []
with open(DATA, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        draws.append([int(r[f"b{i}"]) for i in range(1, N + 1)])
n_draws = len(draws)
print(f"Sorteios: {n_draws}")

# ── split temporal por indice de draw ─────────────────────────────────────────
TRAIN_END = int(n_draws * 0.70)
VAL_END   = int(n_draws * 0.85)

# ── matriz de lift cross-draw (treino apenas, evita leakage) ─────────────────
# lift_cross[m][n] = P(n em draw t | m em draw t-1) / P(n)
marginal_train = Counter(n for d in draws[:TRAIN_END] for n in d)
p_marg = {n: marginal_train[n] / (TRAIN_END * N) for n in range(1, POOL + 1)}

cross_counts = defaultdict(Counter)
cross_total  = Counter()
for t in range(1, TRAIN_END):
    prev, curr = draws[t - 1], set(draws[t])
    for m in prev:
        cross_total[m] += 1
        for n in curr:
            cross_counts[m][n] += 1

lift_cross = np.ones((POOL + 1, POOL + 1), dtype=np.float32)
for m in range(1, POOL + 1):
    tot = cross_total[m]
    if tot == 0:
        continue
    for n in range(1, POOL + 1):
        p_cond = cross_counts[m].get(n, 0) / tot / N  # normaliza por N bolas
        if p_marg[n] > 0:
            lift_cross[m][n] = p_cond / (p_marg[n] / N)

# ── feature builder ───────────────────────────────────────────────────────────
def build(t):
    """Features para prever draw t. Usa apenas draws < t."""
    prev = set(draws[t - 1])
    hist20 = draws[max(0, t - 20): t]
    freq20 = Counter(n for d in hist20 for n in d)

    feats = np.zeros((POOL, 6 + LAGS), dtype=np.float32)
    for i in range(POOL):
        n = i + 1
        feats[i, 0] = 1.0 if n in prev else 0.0                       # saiu no anterior
        feats[i, 1] = 1.0 if (n in FIBS and n in prev) else 0.0       # fib no anterior (filtro)
        feats[i, 2] = float(np.mean([lift_cross[m][n] for m in prev]))  # lift cross-draw medio
        feats[i, 3] = freq20.get(n, 0) / 20.0                         # frequencia janela 20
        feats[i, 4] = 1.0 if n in FIBS else 0.0                       # e fibonacci
        feats[i, 5] = marginal_train.get(n, 0) / (TRAIN_END * N) * POOL  # freq global (chi2 sig!)
        for lag in range(1, LAGS + 1):                                # presenca lag 1..5
            if t - lag >= 0:
                feats[i, 5 + lag] = 1.0 if n in draws[t - lag] else 0.0
    labels = np.array([1.0 if (i + 1) in draws[t] else 0.0 for i in range(POOL)])
    return feats, labels

FEATURE_NAMES = ["saiu_anterior", "fib_no_anterior", "lift_cross_medio",
                 "freq_w20", "is_fib", "freq_global_train"] + \
                [f"lag_{l}" for l in range(1, LAGS + 1)]

X, y, idx = [], [], []
for t in range(LAGS, n_draws):
    f, l = build(t)
    X.append(f); y.append(l); idx.extend([t] * POOL)
X = np.vstack(X); y = np.concatenate(y); idx = np.array(idx)

mask_tr = idx < TRAIN_END
mask_va = (idx >= TRAIN_END) & (idx < VAL_END)
mask_te = idx >= VAL_END

scaler = StandardScaler()
X_tr = scaler.fit_transform(X[mask_tr])
X_va = scaler.transform(X[mask_va])
X_te = scaler.transform(X[mask_te])
y_tr, y_va, y_te = y[mask_tr], y[mask_va], y[mask_te]

def precision_at_6(probs, labels, draw_ids):
    per_draw = defaultdict(lambda: ([], []))
    for p, l, d in zip(probs, labels, draw_ids):
        per_draw[d][0].append(p); per_draw[d][1].append(l)
    precs = []
    for d, (ps, ls) in per_draw.items():
        ps, ls = np.array(ps), np.array(ls)
        precs.append(ls[np.argsort(ps)[-6:]].sum() / 6)
    return float(np.mean(precs)), len(precs)

models = {
    "Logistic Regression v2": LogisticRegression(max_iter=1000, C=0.1),
    "Gradient Boosting v2": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
    "MLP temporal (proxy LSTM)": MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=300, early_stopping=True, random_state=42),
}

print(f"\n{'='*68}")
print("  KAIROS — MODELO V2 (features condicionais + filtro Fibonacci)")
print(f"{'='*68}")
print(f"  Baseline aleatorio: 0.1000\n")

best = ("", 0.0, None)
for name, model in models.items():
    model.fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    p6, nte = precision_at_6(p_te, y_te, idx[mask_te])
    auc = roc_auc_score(y_te, p_te)
    beat = "BATE" if p6 > 0.10 else "nao bate"
    print(f"  {name:<28} Precision@6={p6:.4f}  AUC={auc:.4f}  [{beat}] ({nte} draws teste)")
    if p6 > best[1]:
        best = (name, p6, model)

name, p6, model = best
print(f"\n  Melhor: {name} ({p6:.4f})")
if hasattr(model, "feature_importances_"):
    print("  Importancia das features:")
    for fn, im in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                         key=lambda x: -x[1]):
        print(f"    {fn:<20} {im:.4f}")
elif hasattr(model, "coef_"):
    print("  Coeficientes (abs):")
    for fn, c in sorted(zip(FEATURE_NAMES, np.abs(model.coef_[0])), key=lambda x: -x[1]):
        print(f"    {fn:<20} {c:.4f}")
