"""
Kairos — filtra datasets para manter apenas era moderna (concurso >= 1141)
"""
import csv
from pathlib import Path

DATA = Path(__file__).parent
LOTERIAS = [
    "megasena", "lotofacil", "quina", "lotomania",
    "duplasena", "timemania", "diadesorte", "supersete", "maismilionaria",
]

# limites da era moderna por loteria (concurso onde passaram para globo unico)
# megasena: 1141. As demais sempre usaram globo unico, sem corte necessario.
CORTES = {"megasena": 1141}

for loteria in LOTERIAS:
    src = DATA / f"{loteria}.csv"
    if not src.exists():
        continue
    corte = CORTES.get(loteria, 1)
    rows = []
    with open(src, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            if int(r["concurso"]) >= corte:
                rows.append(r)
    with open(src, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{loteria}: {len(rows)} sorteios mantidos (corte >= {corte})")
