"""
Kairos — Preditor Unificado
Combina todos os sinais estudados num score por numero e gera jogos ranqueados.

Sinais e pesos (com a evidencia de cada um):
  freq_dev     — desvio de frequencia historica     [hipergeometrico p=0.0018 (megasena)]
  anti_repeat  — penaliza numeros do ultimo sorteio  [autocorr media -0.0035]
  fib_filter   — zera Fibonacci do ultimo sorteio    [deficit p=0.09, escolha do usuario]
  ml           — probabilidades do modelo treinado   [AUC ~0.48-0.50, experimental]

Enquadramento honesto: e um gerador recreativo informado por auditoria
estatistica — nenhum sinal da vantagem real de aposta (backtest: equivalente
ao acaso). O valor esta na transparencia de cada score.

Uso:
  python predictor.py megasena           # mostra scores + 3 jogos
  python predictor.py --export           # exporta cards JSON de todas as loterias
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from kairos_engine import get_lottery, available, FIBS, popularity
import ml_model

EXPORT_DIR = Path(__file__).parent.parent / "data" / "export"

WEIGHTS = {
    "freq_dev": 0.30,       # forca do sinal de frequencia (o unico validado)
    "anti_repeat": 0.85,    # multiplicador para numeros do ultimo sorteio
    "ml_blend": 0.35,       # mistura do modelo ML quando disponivel
    "crowd_strength": 1.0,  # anti-multidao: unica vantagem de EV real
}


class KairosPredictor:
    def __init__(self, name: str, use_ml: bool = True, fib_filter: bool = True,
                 anti_crowd: bool = True):
        self.name = name
        self.lot = get_lottery(name)
        self.fib_filter = fib_filter
        self.anti_crowd = anti_crowd
        self.ml_probs = ml_model.predict_next(name) if use_ml else None

    # ── scores ────────────────────────────────────────────────────────────────
    def scores(self) -> dict[int, dict]:
        """Score final + decomposicao por sinal, para cada numero."""
        lot = self.lot
        total = sum(lot.freq.values())
        uniform = 1.0 / len(lot.pool)
        prev = set(lot.draws[-1])

        out = {}
        for n in lot.pool:
            p_hist = lot.freq.get(n, 0) / total
            fw = WEIGHTS["freq_dev"]
            base = (1 - fw) * uniform + fw * p_hist

            repeat_pen = WEIGHTS["anti_repeat"] if n in prev else 1.0
            fib_block = self.fib_filter and n in FIBS and n in prev

            score = 0.0 if fib_block else base * repeat_pen

            ml_p = None
            if self.ml_probs and not fib_block:
                mw = WEIGHTS["ml_blend"]
                total_ml = sum(self.ml_probs.values())
                ml_p = self.ml_probs.get(n, 0) / total_ml
                score = (1 - mw) * score + mw * ml_p

            pop = popularity(n)
            if self.anti_crowd:
                score *= (1.0 / pop) ** WEIGHTS["crowd_strength"]

            out[n] = {
                "score": score,
                "freq_hist": round(p_hist * len(lot.pool), 4),  # 1.0 = na media
                "no_ultimo": n in prev,
                "fib_bloqueado": fib_block,
                "popularidade": round(pop, 2),
                "ml_prob": round(ml_p * len(lot.pool), 4) if ml_p is not None else None,
            }

        # normaliza scores para somarem 1
        s_total = sum(v["score"] for v in out.values()) or 1.0
        for v in out.values():
            v["score"] = round(v["score"] / s_total, 6)
        return out

    # ── geracao ───────────────────────────────────────────────────────────────
    def predict(self, n_games: int = 3, seed: int | None = None) -> list[dict]:
        """Gera jogos por amostragem ponderada pelos scores, ranqueados."""
        rng = np.random.default_rng(seed)
        sc = self.scores()
        nums = list(sc.keys())
        games = []
        for _ in range(n_games):
            weights = np.array([sc[n]["score"] for n in nums], dtype=float)
            picked = []
            for _ in range(self.lot.k):
                w = weights.copy()
                for i, n in enumerate(nums):
                    if n in picked:
                        w[i] = 0
                if w.sum() == 0:
                    w = np.array([0 if n in picked else 1 for n in nums], dtype=float)
                w /= w.sum()
                picked.append(int(rng.choice(nums, p=w)))
            game = sorted(picked)
            games.append({
                "numbers": game,
                "score_total": round(sum(sc[n]["score"] for n in game), 5),
            })
        games.sort(key=lambda g: -g["score_total"])
        return games

    def top_k(self) -> list[int]:
        """O jogo 'deterministico': os k numeros de maior score."""
        sc = self.scores()
        ranked = sorted(sc.items(), key=lambda kv: -kv[1]["score"])
        return sorted(n for n, _ in ranked[: self.lot.k])

    # ── export mobile ─────────────────────────────────────────────────────────
    def export_card(self) -> dict:
        """Tudo que um app offline precisa para gerar jogos desta loteria."""
        sc = self.scores()
        st = self.lot.stats()
        ml_meta = ml_model.status(self.name)
        return {
            "lottery": self.name,
            "generated_at": time.strftime("%d/%m/%Y %H:%M"),
            "pool_lo": self.lot.lo,
            "pool_hi": self.lot.hi,
            "balls_per_game": self.lot.k,
            "last_concurso": st["last_concurso"],
            "last_data": st["last_data"],
            "last_draw": st["last_draw"],
            "fib_blocked": st["fib_blocked"],
            "scores": {str(n): v["score"] for n, v in sc.items()},
            "top_game": self.top_k(),
            "ml_model": {
                "auc": ml_meta["auc_holdout"],
                "trained_until": ml_meta["trained_until_concurso"],
            } if ml_meta else None,
            "disclaimer": ("Gerador recreativo. Auditoria estatistica do projeto "
                           "Kairos: nenhum metodo aumenta a chance real de acerto."),
        }


def export_all() -> list[str]:
    EXPORT_DIR.mkdir(exist_ok=True)
    written = []
    for name in available():
        try:
            card = KairosPredictor(name).export_card()
        except Exception as e:
            print(f"  {name}: ERRO {e}")
            continue
        path = EXPORT_DIR / f"{name}_card.json"
        path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(path.name)
        print(f"  {name}: {path.name} ({path.stat().st_size} bytes)")
    return written


if __name__ == "__main__":
    if "--export" in sys.argv:
        print("Exportando cards para data/export/ ...")
        export_all()
        sys.exit(0)

    name = next((a for a in sys.argv[1:] if not a.startswith("--")), "megasena")
    pred = KairosPredictor(name)
    print(f"\nKairos Predictor — {name}")
    print(f"Ultimo concurso: #{pred.lot.last_concurso} | {pred.lot.last_draw}\n")

    sc = pred.scores()
    ranked = sorted(sc.items(), key=lambda kv: -kv[1]["score"])
    print("Top 10 numeros por score:")
    for n, v in ranked[:10]:
        flags = []
        if v["no_ultimo"]: flags.append("saiu no ultimo")
        if v["fib_bloqueado"]: flags.append("FIB BLOQUEADO")
        print(f"  #{n:02d}  score={v['score']:.5f}  freq_hist={v['freq_hist']:.3f}"
              f"  ml={v['ml_prob']}  {' '.join(flags)}")

    print(f"\nJogo top-k (deterministico): {pred.top_k()}")
    print("\n3 jogos amostrados (ranqueados por score):")
    for g in pred.predict(3):
        print(f"  {g['numbers']}  (score {g['score_total']})")
