# Kairos

**A statistical audit of Brazilian lottery randomness — and an honest, well-behaved number generator built on top of it.**

Kairos started from a provocative question: *can the unpredictable be predicted?* It ends
with a rigorous, reproducible answer for Brazil's official lotteries — and a small tool that
turns that answer into something useful.

> **TL;DR** — No method predicts a certified ball-draw lottery. Kairos proves this across
> ~24,500 draws using industrial-grade randomness tests, then ships a generator whose only
> real edge is *statistical*: it avoids popular numbers so that **if** you win, you split the
> prize with fewer people.

---

## What's inside

| Area | What it does |
|---|---|
| **Data pipeline** | Downloads the full history of 9 Brazilian lotteries from the public Caixa API and normalizes it |
| **Statistical audit** | 7 tests per lottery: multivariate hypergeometric chi-square, Higher Criticism, Dirichlet–multinomial Bayes factor, gap test, Neyman smooth, scan statistics, permutation entropy — all Monte-Carlo calibrated |
| **Sequential monitor** | An anytime-valid e-process (safe under optional stopping) that keeps accumulating evidence as new draws happen |
| **ML models** | Logistic Regression, Random Forest, Gradient Boosting, MLP — with strict temporal splits and a Precision@k metric |
| **Generator** | A weighted sampler exposing every studied signal as a toggle, each labeled with its real statistical strength |
| **Web panel** | A local Flask + single-page app to explore, generate, train, and audit |

---

## Key findings

- **Every method fails to predict future draws.** ML models sit at AUC ≈ 0.48–0.52 (chance),
  and a 10,000-game backtest of the generator is statistically indistinguishable from random.
- **The physical machines are excellent, but not perfect.** Using the *correct* multivariate
  hypergeometric test (which accounts for the negative covariance a draw-without-replacement
  induces), Mega-Sena shows a small but significant frequency deviation (p ≈ 0.0018). A naive
  chi-square missed or understated this.
- **The deviation is real but useless for betting** — a textbook Lindley's paradox: the Bayes
  factor still strongly favors uniformity. Detectable imperfection, no exploitable edge.
- **A localized temporal episode** was found in Lotofácil (ball #6 ran cold from Dec 2016 to
  Aug 2017) via scan statistics — the signature of a physical change (e.g. a ball-set swap).
- **The only legitimate edge is avoiding the crowd.** Following the literature (Canada 6/49,
  UK Lotto), picking unpopular numbers doesn't change your odds of winning but reduces how
  many people you'd share a jackpot with.

Full write-ups: [`data/research_megasena.md`](data/research_megasena.md) (how the draw system
works) and [`data/research_methods.md`](data/research_methods.md) (the methods surveyed).

---

## Quick start

### 1. The analysis (Python)

```bash
pip install -r requirements.txt

# fetch + normalize data (already included, run to refresh)
python data/fetch_dataset.py
python data/filter_modern.py

# run the audit on any lottery
python src/audit.py megasena
python src/eprocess.py            # sequential monitor, all lotteries
python src/backtest.py            # generator vs. chance
```

### 2. The web panel

```bash
cd src
python server.py
# open http://localhost:7777
```

From the panel you can pick a lottery, choose lucky numbers, generate games, train the ML
model, and run the statistical audit — all with plain-language labels.

---

## Project layout

```
Kairos/
├── src/                  # analysis, models, server, generator
│   ├── audit.py          # 7-test statistical battery
│   ├── eprocess.py       # anytime-valid sequential monitor
│   ├── predictor.py      # unified weighted generator
│   ├── kairos_engine.py  # weighted sampler core
│   ├── server.py         # Flask web panel
│   └── ...
├── data/                 # lottery CSVs + audit results + research notes
└── requirements.txt
```

---

## Honest disclaimer

Kairos is a **research project and a recreational tool**. It does not, and cannot, increase
your probability of winning any lottery — the audit in this very repository is the proof.
Certified physical draws are, for all practical purposes, unpredictable. Play for fun and
responsibly.

## License

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** — see [LICENSE](LICENSE).

You may use, study, modify, and share this project for any **noncommercial** purpose, with
attribution. Commercial use is not permitted without a separate agreement. For commercial
licensing, contact the author.
