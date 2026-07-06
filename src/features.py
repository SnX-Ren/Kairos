"""
Kairos — Feature Engineering
Constroi o vetor de features por (draw, numero) para o modelo ML.
"""

import numpy as np

POOL  = 60
N     = 6
FIBS  = {1,2,3,5,8,13,21,34,55}
LUCAS = {2,1,3,4,7,11,18,29,47}

FEATURE_NAMES = [
    "freq_w5", "freq_w10", "freq_w20",   # frequencia recente em janelas
    "gap",                                # sorteios desde ultima aparicao
    "gap_norm",                           # gap normalizado pelo esperado
    "autocorr_w20",                       # autocorrelacao local
    "is_fib",                             # pertence a Fibonacci
    "is_lucas",                           # pertence a Lucas
    "parity",                             # 0=impar 1=par
    "mod6",                               # numero mod 6
    "rank_freq_w20",                      # rank de frequencia recente (0=mais frio, 1=mais quente)
]

N_FEATURES = len(FEATURE_NAMES)


def build_dataset(draws: list[list[int]], window: int = 20):
    """
    Para cada draw t >= window, cria 60 amostras (uma por numero).
    Retorna X (n_samples, n_features), y (n_samples,), draw_indices.
    """
    X, y, draw_idx = [], [], []

    for t in range(window, len(draws)):
        history = draws[t - window: t]       # janela anterior
        current = set(draws[t])              # draw atual = target

        # matriz binaria: window x POOL
        mat = np.zeros((window, POOL), dtype=np.float32)
        for i, draw in enumerate(history):
            for n in draw:
                mat[i, n - 1] = 1.0

        # frequencias acumuladas por janela
        freq5  = mat[-5:].sum(axis=0)   # ultimos 5
        freq10 = mat[-10:].sum(axis=0)  # ultimos 10
        freq20 = mat.sum(axis=0)        # todos os 20

        # rank de frequencia (0=mais frio, 1=mais quente)
        rank20 = freq20.argsort().argsort().astype(np.float32) / (POOL - 1)

        # gap por numero
        gaps = np.full(POOL, window, dtype=np.float32)
        for i in range(window - 1, -1, -1):
            for n in history[i]:
                if gaps[n - 1] == window:
                    gaps[n - 1] = (window - 1 - i)

        gap_norm = gaps / (POOL / N)  # normalizado pelo gap esperado (10)

        # autocorrelacao local (lag=1) por numero
        autocorr = np.zeros(POOL, dtype=np.float32)
        if window >= 2:
            for n_idx in range(POOL):
                s = mat[:, n_idx]
                m = s.mean()
                v = s.var()
                if v > 0:
                    autocorr[n_idx] = float(
                        np.mean((s[1:] - m) * (s[:-1] - m)) / v
                    )

        # features estaticas por numero
        is_fib    = np.array([1.0 if (i+1) in FIBS  else 0.0 for i in range(POOL)])
        is_lucas  = np.array([1.0 if (i+1) in LUCAS else 0.0 for i in range(POOL)])
        parity    = np.array([float((i+1) % 2 == 0) for i in range(POOL)])
        mod6      = np.array([(i+1) % 6 / 5.0       for i in range(POOL)])

        # empilha features: cada numero vira uma linha
        features = np.column_stack([
            freq5, freq10, freq20,
            gaps, gap_norm,
            autocorr,
            is_fib, is_lucas, parity, mod6,
            rank20,
        ])   # shape: (60, N_FEATURES)

        labels = np.array([1.0 if (i+1) in current else 0.0 for i in range(POOL)])

        X.append(features)
        y.append(labels)
        draw_idx.append(t)

    X = np.vstack(X)                    # (n_draws * 60, N_FEATURES)
    y = np.concatenate(y)               # (n_draws * 60,)
    draw_idx = np.repeat(draw_idx, POOL)

    return X, y, draw_idx
