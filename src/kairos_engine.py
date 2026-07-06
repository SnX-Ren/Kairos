"""
Kairos Engine — gerador de jogos baseado nos achados do projeto.

Filtros disponiveis (cada um reflete um resultado das analises):
- fib_filter:    descarta numeros Fibonacci que sairam no concurso anterior
                 (deficit Fibonacci observado, p=0.09 — limite de significancia)
- anti_repeat:   reduz peso de numeros do concurso anterior
                 (autocorrelacao media negativa: -0.0035)
- freq_blend:    mistura frequencia historica no peso
                 (chi2 global da Mega-Sena: p=0.0095 — unico desvio significativo)
- pair_affinity: favorece numeros com alta co-ocorrencia com os ja escolhidos
                 (lifts validados como ruido, p=0.84 — OFF por padrao)
"""

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
FIBS = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}

DEFAULTS = {
    "fib_filter": True,
    "anti_repeat": True,
    "freq_blend": True,
    "pair_affinity": False,
    "use_ml": False,         # mistura probabilidades do modelo ML treinado
    "anti_crowd": True,      # evita numeros populares (reduz divisao de premio)
    "repeat_factor": 0.85,   # multiplicador p/ numeros do concurso anterior
    "freq_weight": 0.30,     # 0 = uniforme puro, 1 = frequencia pura
    "affinity_alpha": 0.25,  # intensidade do boost de co-ocorrencia
    "ml_weight": 0.50,       # 0 = ignora ML, 1 = so ML
    "crowd_strength": 1.0,   # intensidade da penalidade de popularidade
    "favorites": [],         # numeros escolhidos pelo usuario (boost)
    "favorite_boost": 3.0,   # multiplicador de peso dos favoritos
}


def popularity(n: int) -> float:
    """
    Modelo heuristico de popularidade de um numero entre apostadores.
    Base: literatura do Canada 6/49 e UK Lotto — jogadores preferem
    aniversarios (1-31), meses (1-12) e numeros 'de sorte'.
    Nao muda a chance de ganhar; muda quantas pessoas dividem o premio.
    """
    pop = 1.0
    if n <= 31: pop += 0.8    # dia de aniversario
    if n <= 12: pop += 0.4    # mes de aniversario
    if n in (7, 13, 3, 11): pop += 0.3   # "numeros de sorte" classicos
    if n % 10 == 0: pop += 0.1           # numeros redondos
    return pop


class Lottery:
    """Carrega uma loteria a partir do CSV e expoe o gerador."""

    def __init__(self, name: str):
        self.name = name
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset nao encontrado: {path}")

        self.draws: list[list[int]] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            bcols = [c for c in reader.fieldnames if c[0] == "b" and c[1:].isdigit()]
            bcols.sort(key=lambda c: int(c[1:]))
            self.concursos = []
            self.datas = []
            for r in reader:
                try:
                    draw = [int(r[c]) for c in bcols if r[c] != ""]
                except ValueError:
                    continue
                if draw:
                    self.draws.append(draw)
                    self.concursos.append(int(r["concurso"]))
                    self.datas.append(r.get("data", ""))

        nums = [n for d in self.draws for n in d]
        self.lo, self.hi = min(nums), max(nums)
        self.pool = list(range(self.lo, self.hi + 1))
        # duplasena: CSV concatena os 2 sorteios do concurso (12 bolas),
        # mas a aposta e de 6 numeros
        K_OVERRIDE = {"duplasena": 6}
        self.k = K_OVERRIDE.get(name, len(self.draws[-1]))
        self.freq = Counter(nums)

        # matriz de co-ocorrencia (para pair_affinity)
        self.cooc = defaultdict(Counter)
        for d in self.draws:
            for a, b in combinations(d, 2):
                self.cooc[a][b] += 1
                self.cooc[b][a] += 1

    @property
    def last_draw(self) -> list[int]:
        return sorted(self.draws[-1])

    @property
    def last_concurso(self) -> int:
        return self.concursos[-1]

    def base_weights(self, cfg: dict) -> dict[int, float]:
        """Peso inicial de cada numero antes da amostragem."""
        total = sum(self.freq.values())
        uniform = 1.0 / len(self.pool)
        weights = {}
        for n in self.pool:
            w = uniform
            if cfg["freq_blend"]:
                p_hist = self.freq.get(n, 0) / total
                fw = cfg["freq_weight"]
                w = (1 - fw) * uniform + fw * p_hist
            weights[n] = w

        prev = set(self.draws[-1])
        for n in prev:
            if cfg["fib_filter"] and n in FIBS:
                weights[n] = 0.0                      # descarte total
            elif cfg["anti_repeat"]:
                weights[n] *= cfg["repeat_factor"]    # reducao suave

        if cfg["anti_crowd"]:
            s = cfg["crowd_strength"]
            for n in self.pool:
                weights[n] *= (1.0 / popularity(n)) ** s

        # favoritos do usuario: boost explicito que prevalece sobre filtros
        # (se um filtro zerou o numero, o favorito o restaura)
        try:
            favs = {int(x) for x in (cfg.get("favorites") or [])} & set(self.pool)
        except (TypeError, ValueError):
            favs = set()
        if favs:
            uniform = 1.0 / len(self.pool)
            for n in favs:
                base = weights[n] if weights[n] > 0 else uniform
                weights[n] = base * cfg["favorite_boost"]
        return weights

    def generate(self, n_games: int, cfg: dict | None = None,
                 seed: int | None = None,
                 ml_probs: dict[int, float] | None = None) -> list[list[int]]:
        cfg = {**DEFAULTS, **(cfg or {})}
        rng = np.random.default_rng(seed)
        games = []
        for _ in range(n_games):
            weights = self.base_weights(cfg)
            if cfg["use_ml"] and ml_probs:
                mw = cfg["ml_weight"]
                total_ml = sum(ml_probs.values()) or 1.0
                for n in self.pool:
                    p_ml = ml_probs.get(n, 0) / total_ml
                    # blend preservando zeros do fib_filter
                    if weights[n] > 0:
                        weights[n] = (1 - mw) * weights[n] + mw * p_ml
            picked = []
            for _ in range(self.k):
                nums = [n for n in self.pool if n not in picked and weights[n] > 0]
                ws = np.array([weights[n] for n in nums], dtype=float)
                if ws.sum() == 0:  # tudo zerado (edge case) → uniforme
                    nums = [n for n in self.pool if n not in picked]
                    ws = np.ones(len(nums))
                ws /= ws.sum()
                choice = int(rng.choice(nums, p=ws))
                picked.append(choice)
                if cfg["pair_affinity"]:
                    # boost co-ocorrencia com o numero recem escolhido
                    total_c = sum(self.cooc[choice].values()) or 1
                    for n in nums:
                        if n == choice:
                            continue
                        lift = (self.cooc[choice].get(n, 0) / total_c) * len(self.pool)
                        weights[n] *= 1 + cfg["affinity_alpha"] * (lift - 1)
                        weights[n] = max(weights[n], 0.0)
            games.append(sorted(picked))
        return games

    def stats(self) -> dict:
        prev = set(self.draws[-1])
        return {
            "name": self.name,
            "draws": len(self.draws),
            "pool": f"{self.lo}-{self.hi}",
            "balls": self.k,
            "last_concurso": self.last_concurso,
            "last_data": self.datas[-1],
            "last_draw": self.last_draw,
            "fib_blocked": sorted(n for n in prev if n in FIBS),
        }


_cache: dict[str, Lottery] = {}

def get_lottery(name: str) -> Lottery:
    if name not in _cache:
        _cache[name] = Lottery(name)
    return _cache[name]


def available() -> list[str]:
    # supersete excluido: colunas independentes 0-9 com repeticao,
    # incompativel com amostragem sem reposicao deste engine
    skip = {"megasena_historico", "supersete"}
    return sorted(p.stem for p in DATA_DIR.glob("*.csv") if p.stem not in skip)
