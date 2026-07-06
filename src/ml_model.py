"""
Kairos — Modulo ML
Treina, persiste e usa modelos por loteria para pontuar numeros.

O modelo aprende P(numero sai no proximo draw | historico recente).
Features genericas (funcionam para qualquer loteria):
  freq nas janelas 5/10/20, gap normalizado, presenca lag 1-5,
  is_fibonacci, frequencia global historica.

Nota honesta: as validacoes mostraram que nao ha sinal preditivo real —
o modelo existe como componente experimental do gerador, nao como vantagem.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from kairos_engine import get_lottery, FIBS

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MIN_HISTORY = 30   # draws minimos antes de gerar amostras
LAGS = 5


def _features_at(draws, t, lo, hi):
    """Matriz (pool, n_features) para prever o draw t usando draws[:t]."""
    pool = hi - lo + 1
    hist5  = draws[max(0, t - 5): t]
    hist10 = draws[max(0, t - 10): t]
    hist20 = draws[max(0, t - 20): t]

    in5, in10, in20, in_all = {}, {}, {}, {}
    for d in hist5:
        for n in d: in5[n] = in5.get(n, 0) + 1
    for d in hist10:
        for n in d: in10[n] = in10.get(n, 0) + 1
    for d in hist20:
        for n in d: in20[n] = in20.get(n, 0) + 1
    for d in draws[:t]:
        for n in d: in_all[n] = in_all.get(n, 0) + 1
    total_all = sum(in_all.values()) or 1

    # gap: draws desde a ultima aparicao (cap 50)
    gap = {n: 50 for n in range(lo, hi + 1)}
    for back in range(1, min(t, 50) + 1):
        for n in draws[t - back]:
            if gap[n] == 50:
                gap[n] = back - 1

    feats = np.zeros((pool, 6 + LAGS), dtype=np.float32)
    for i in range(pool):
        n = lo + i
        feats[i, 0] = in5.get(n, 0) / 5.0
        feats[i, 1] = in10.get(n, 0) / 10.0
        feats[i, 2] = in20.get(n, 0) / 20.0
        feats[i, 3] = gap[n] / 50.0
        feats[i, 4] = 1.0 if n in FIBS else 0.0
        feats[i, 5] = in_all.get(n, 0) / total_all * pool
        for lag in range(1, LAGS + 1):
            if t - lag >= 0:
                feats[i, 5 + lag] = 1.0 if n in draws[t - lag] else 0.0
    return feats


def _model_path(name):
    return MODELS_DIR / f"{name}.joblib"


def _meta_path(name):
    return MODELS_DIR / f"{name}.meta.json"


def train(name: str) -> dict:
    """Treina o modelo da loteria e persiste. Retorna metricas."""
    t0 = time.time()
    lot = get_lottery(name)
    draws, lo, hi = lot.draws, lot.lo, lot.hi
    k = lot.k

    X, y, idx = [], [], []
    for t in range(MIN_HISTORY, len(draws)):
        f = _features_at(draws, t, lo, hi)
        target = set(draws[t])
        X.append(f)
        y.append(np.array([1.0 if lo + i in target else 0.0 for i in range(hi - lo + 1)]))
        idx.extend([t] * (hi - lo + 1))
    X = np.vstack(X); y = np.concatenate(y); idx = np.array(idx)

    # holdout temporal: ultimos 15%
    cut = int(len(draws) * 0.85)
    tr, te = idx < cut, idx >= cut

    model = HistGradientBoostingClassifier(
        max_iter=120, max_depth=4, learning_rate=0.05,
        early_stopping=True, random_state=42)
    model.fit(X[tr], y[tr])

    probs = model.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y[te], probs))

    # precision@k no holdout
    pool = hi - lo + 1
    per_draw = probs.reshape(-1, pool)
    labels = y[te].reshape(-1, pool)
    topk = np.argsort(per_draw, axis=1)[:, -k:]
    hits = np.take_along_axis(labels, topk, axis=1).sum(axis=1)
    p_at_k = float(hits.mean() / k)
    baseline = k / pool

    joblib.dump(model, _model_path(name))
    meta = {
        "lottery": name,
        "trained_until_concurso": lot.last_concurso,
        "trained_at": time.strftime("%d/%m/%Y %H:%M"),
        "n_draws": len(draws),
        "auc_holdout": round(auc, 4),
        "precision_at_k": round(p_at_k, 4),
        "baseline": round(baseline, 4),
        "train_seconds": round(time.time() - t0, 1),
    }
    _meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def status(name: str) -> dict | None:
    """Meta do modelo salvo, ou None se nunca treinado."""
    p = _meta_path(name)
    if not p.exists() or not _model_path(name).exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def predict_next(name: str) -> dict[int, float] | None:
    """P(numero | proximo draw) para cada numero. None se sem modelo."""
    if not _model_path(name).exists():
        return None
    model = joblib.load(_model_path(name))
    lot = get_lottery(name)
    feats = _features_at(lot.draws, len(lot.draws), lot.lo, lot.hi)
    probs = model.predict_proba(feats)[:, 1]
    return {lot.lo + i: float(p) for i, p in enumerate(probs)}
