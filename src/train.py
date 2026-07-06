"""
Kairos — Training Pipeline
Treina e avalia modelos para predicao de sorteios da Mega-Sena.
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from features import build_dataset, FEATURE_NAMES, POOL, N

# ── config ────────────────────────────────────────────────────────────────────
DATA   = Path(__file__).parent.parent / "data" / "megasena.csv"
WINDOW = 20
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = restante (0.15)

# ── load ──────────────────────────────────────────────────────────────────────
draws = []
with open(DATA, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        draws.append([int(r[f"b{i}"]) for i in range(1, N + 1)])

print(f"Sorteios carregados: {len(draws)}")

# ── features ──────────────────────────────────────────────────────────────────
print("Construindo features...", end=" ", flush=True)
X, y, draw_idx = build_dataset(draws, window=WINDOW)
print(f"ok — {X.shape[0]} amostras, {X.shape[1]} features")

# ── split temporal ────────────────────────────────────────────────────────────
unique_draws = sorted(set(draw_idx))
n_total = len(unique_draws)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)

train_draws = set(unique_draws[:n_train])
val_draws   = set(unique_draws[n_train: n_train + n_val])
test_draws  = set(unique_draws[n_train + n_val:])

mask_train = np.array([d in train_draws for d in draw_idx])
mask_val   = np.array([d in val_draws   for d in draw_idx])
mask_test  = np.array([d in test_draws  for d in draw_idx])

X_train, y_train = X[mask_train], y[mask_train]
X_val,   y_val   = X[mask_val],   y[mask_val]
X_test,  y_test  = X[mask_test],  y[mask_test]

print(f"Split: {len(train_draws)} treino | {len(val_draws)} val | {len(test_draws)} teste")

# ── normalização ──────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── metrica: Precision@6 ──────────────────────────────────────────────────────
def precision_at_6(probs: np.ndarray, labels: np.ndarray, draw_idx_subset) -> float:
    """
    Para cada draw, pega os 6 numeros com maior probabilidade prevista.
    Retorna a fracao media de acertos (esperado baseline: 6/60 = 0.10).
    """
    draws_map = defaultdict(lambda: {"probs": [], "labels": []})
    for prob, label, d in zip(probs, labels, draw_idx_subset):
        draws_map[d]["probs"].append(prob)
        draws_map[d]["labels"].append(label)

    precisions = []
    for d, vals in draws_map.items():
        p = np.array(vals["probs"])
        l = np.array(vals["labels"])
        top6 = np.argsort(p)[-6:]
        precisions.append(l[top6].sum() / 6)

    return float(np.mean(precisions))

# ── baseline: frequencia historica ────────────────────────────────────────────
from collections import Counter
freq_hist = Counter(n for draw in draws[:int(len(draws)*TRAIN_RATIO)] for n in draw)
top6_freq = {n for n, _ in freq_hist.most_common(6)}

def baseline_freq_precision(draws_subset):
    hits = [len(set(d) & top6_freq) / 6 for d in draws_subset]
    return float(np.mean(hits))

test_draws_list = [draws[i] for i in sorted(test_draws)]
baseline_random = N / POOL
baseline_freq   = baseline_freq_precision(test_draws_list)

print(f"\nBaselines:")
print(f"  Aleatorio:  {baseline_random:.4f} ({baseline_random*100:.1f}%)")
print(f"  Frequencia: {baseline_freq:.4f}   ({baseline_freq*100:.1f}%)")

# ── modelos ───────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=0.1, solver="lbfgs", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
}

print("\n" + "="*65)
print("  TREINAMENTO E AVALIACAO")
print("="*65)

best_model     = None
best_precision = 0
best_name      = ""
results        = {}

test_idx_subset = draw_idx[mask_test]

for name, model in models.items():
    print(f"\n  [{name}]")
    print(f"    Treinando...", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("ok")

    # validacao
    val_probs = model.predict_proba(X_val)[:, 1]
    val_p6    = precision_at_6(val_probs, y_val, draw_idx[mask_val])
    val_auc   = roc_auc_score(y_val, val_probs)

    # teste
    test_probs = model.predict_proba(X_test)[:, 1]
    test_p6    = precision_at_6(test_probs, y_test, test_idx_subset)
    test_auc   = roc_auc_score(y_test, test_probs)

    beat_random = "BATE" if test_p6 > baseline_random else "NAO BATE"
    beat_freq   = "BATE" if test_p6 > baseline_freq   else "NAO BATE"

    print(f"    Val   — Precision@6: {val_p6:.4f}  AUC: {val_auc:.4f}")
    print(f"    Teste — Precision@6: {test_p6:.4f}  AUC: {test_auc:.4f}")
    print(f"    vs baseline aleatorio ({baseline_random:.4f}): {beat_random}")
    print(f"    vs baseline frequencia ({baseline_freq:.4f}): {beat_freq}")

    results[name] = {"val_p6": val_p6, "test_p6": test_p6, "test_auc": test_auc}

    if test_p6 > best_precision:
        best_precision = test_p6
        best_model     = model
        best_name      = name

# ── feature importance (melhor modelo) ───────────────────────────────────────
print("\n" + "="*65)
print(f"  MELHOR MODELO: {best_name}  (Precision@6 = {best_precision:.4f})")
print("="*65)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print("\n  Feature importances:")
    for fname, imp in ranked:
        bar = "#" * int(imp * 200)
        print(f"    {fname:<18} {imp:.4f}  {bar}")
elif hasattr(best_model, "coef_"):
    coefs = np.abs(best_model.coef_[0])
    ranked = sorted(zip(FEATURE_NAMES, coefs), key=lambda x: x[1], reverse=True)
    print("\n  Coeficientes (abs):")
    for fname, coef in ranked:
        bar = "#" * int(coef * 20)
        print(f"    {fname:<18} {coef:.4f}  {bar}")

# ── resumo final ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RESUMO")
print("="*65)
print(f"  Baseline aleatorio:          {baseline_random:.4f}")
print(f"  Baseline frequencia:         {baseline_freq:.4f}")
for name, r in results.items():
    delta = r['test_p6'] - baseline_random
    marker = "+" if delta > 0 else ""
    print(f"  {name:<26} {r['test_p6']:.4f}  ({marker}{delta:+.4f} vs random)")
print("="*65)
