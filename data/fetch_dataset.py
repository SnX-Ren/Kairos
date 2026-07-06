"""
Kairos — Dataset Fetcher
Fonte: loteriascaixa-api.herokuapp.com (guto-alves/loterias-api)
Baixa todas as loterias em uma única requisição cada.
"""

import urllib.request
import json
import csv
from pathlib import Path

BASE_URL = "https://loteriascaixa-api.herokuapp.com/api"
DATA_DIR = Path(__file__).parent

LOTERIAS = [
    "megasena",
    "lotofacil",
    "quina",
    "lotomania",
    "duplasena",
    "timemania",
    "diadesorte",
    "supersete",
    "maismilionaria",
]

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_all(loteria: str) -> list:
    url = f"{BASE_URL}/{loteria}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def parse(raw: dict, loteria: str) -> dict | None:
    try:
        dezenas = raw.get("dezenas", [])
        ordem   = raw.get("dezenasOrdemSorteio", dezenas)
        row = {
            "loteria":   loteria,
            "concurso":  raw["concurso"],
            "data":      raw["data"],
            "acumulou":  raw.get("acumulou", False),
        }
        for i, n in enumerate(dezenas, 1):
            row[f"b{i}"] = n
        for i, n in enumerate(ordem, 1):
            row[f"o{i}"] = n
        return row
    except Exception:
        return None


def save(rows: list, path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    for loteria in LOTERIAS:
        print(f"Baixando {loteria}...", end=" ", flush=True)
        try:
            raw_list = fetch_all(loteria)
            rows = [r for raw in raw_list if (r := parse(raw, loteria))]
            rows.sort(key=lambda x: x["concurso"])
            out = DATA_DIR / f"{loteria}.csv"
            save(rows, out)
            print(f"{len(rows)} sorteios -> {out.name}")
        except Exception as e:
            print(f"ERRO: {e}")

    print("\nDone.")
