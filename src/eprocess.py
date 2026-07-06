"""
Kairos — E-Process de Monitoramento Sequencial
Inferencia anytime-valid (SAVI) para uniformidade dos sorteios.

Construcao: apostador de Krichevsky-Trofimov contra H0.
  - H0: cada bola sorteada e uniforme sobre os numeros restantes do globo
        (dentro do sorteio, sem reposicao).
  - Alternativa: preditiva de Dirichlet(1/2) aprendida com o historico,
        restrita aos numeros restantes.
  - Riqueza W_t = produto dos ratios P_alt(bola)/P_H0(bola).

Propriedade: E[W_t | H0] <= 1 em qualquer tempo de parada — o monitor pode
rodar para sempre, ser consultado a cada novo sorteio, e W >= 20 equivale a
rejeitar H0 com validade alfa=0.05 em qualquer momento.

Usa a ordem real do sorteio (colunas o1..oN) quando disponivel.

Uso: python eprocess.py [loteria]
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
ALPHA_KT = 0.5   # prior de Krichevsky-Trofimov


def load_ordered(name):
    draws = []
    with open(DATA_DIR / f"{name}.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ocols = sorted((c for c in reader.fieldnames
                        if c[0] == "o" and c[1:].isdigit()), key=lambda c: int(c[1:]))
        bcols = sorted((c for c in reader.fieldnames
                        if c[0] == "b" and c[1:].isdigit()), key=lambda c: int(c[1:]))
        # a API anexa numeros extras (ex.: trevos da +Milionaria) ao fim da
        # ordem de sorteio — usa apenas as k primeiras posicoes e valida que
        # correspondem ao conjunto das dezenas principais
        ocols = ocols[:len(bcols)]
        cols = ocols if ocols else bcols
        for r in reader:
            try:
                d = [int(r[c]) for c in cols if r[c] != ""]
                b = [int(r[c]) for c in bcols if r[c] != ""]
            except ValueError:
                continue
            if not d:
                continue
            if set(d) != set(b):   # ordem inconsistente -> usa ordem crescente
                d = b
            draws.append(d)
    nums = [n for d in draws for n in d]
    return draws, min(nums), max(nums), bool(ocols)


def run_eprocess(name):
    draws, lo, hi, has_order = load_ordered(name)
    N = hi - lo + 1
    counts = np.full(N, ALPHA_KT)          # pseudo-contagens KT
    log_w = 0.0
    trajectory = []                        # (draw_idx, log10_W)

    for t, draw in enumerate(draws):
        remaining = np.ones(N, dtype=bool)
        for ball in draw:
            j = ball - lo
            # H0: uniforme sobre os numeros restantes
            p_null = 1.0 / remaining.sum()
            # alternativa: KT preditiva restrita aos restantes
            c = counts[remaining]
            p_alt = counts[j] / c.sum()
            log_w += np.log(p_alt / p_null)
            remaining[j] = False
        for ball in draw:                  # atualiza apos o sorteio completo
            counts[ball - lo] += 1
        if t % 10 == 0 or t == len(draws) - 1:
            trajectory.append((t, round(log_w / np.log(10), 4)))

    log10_w = log_w / np.log(10)
    e_value = 10 ** log10_w if log10_w < 300 else float("inf")
    verdict = ("REJEITA H0 (e >= 20, alfa 0.05 anytime-valid)" if e_value >= 20
               else "sem evidencia contra uniformidade")

    report = {
        "lottery": name,
        "n_draws": len(draws),
        "used_draw_order": has_order,
        "log10_e_value": round(log10_w, 4),
        "e_value": round(e_value, 6) if e_value != float("inf") else "inf",
        "threshold_log10": round(np.log10(20), 3),
        "verdict": verdict,
        "trajectory": trajectory,
        "note": ("Monitor anytime-valid: re-execute apos cada atualizacao de dados; "
                 "a riqueza acumula entre execucoes por construcao."),
    }
    out = DATA_DIR / f"eprocess_{name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    names = ([sys.argv[1]] if len(sys.argv) > 1 else
             ["megasena", "lotofacil", "quina", "lotomania", "timemania",
              "diadesorte", "maismilionaria"])
    print(f"\n{'='*66}")
    print(f"  E-PROCESS — monitor sequencial de uniformidade (KT, alfa=0.05)")
    print(f"  limiar de rejeicao: e >= 20  (log10 >= {np.log10(20):.3f})")
    print(f"{'='*66}\n")
    print(f"  {'loteria':<16} {'draws':>6} {'ordem':>6} {'log10(e)':>10}  veredito")
    for name in names:
        try:
            rep = run_eprocess(name)
        except FileNotFoundError:
            continue
        print(f"  {rep['lottery']:<16} {rep['n_draws']:>6} "
              f"{'sim' if rep['used_draw_order'] else 'nao':>6} "
              f"{rep['log10_e_value']:>10.3f}  {rep['verdict']}")
    print(f"\n  Interpretacao: log10(e) > 0 = evidencia acumulando contra H0;")
    print(f"  negativo = dados comprimem melhor sob uniformidade (pro-H0).")
    print(f"{'='*66}")
