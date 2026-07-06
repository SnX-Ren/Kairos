"""
Kairos — Updater
Atualiza os datasets pela API e retreina modelos, em thread de background.
O estado e consultavel via get_status() (polling pela UI).
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import kairos_engine
import ml_model

DATA_DIR = Path(__file__).parent.parent / "data"

_lock = threading.Lock()
_status = {
    "state": "idle",        # idle | running | done | error
    "step": "",
    "log": [],
    "started_at": None,
    "finished_at": None,
}


def get_status() -> dict:
    with _lock:
        return dict(_status)


def _log(msg: str):
    with _lock:
        _status["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        _status["step"] = msg


def _run(train_lotteries: list[str]):
    try:
        _log("Baixando datasets da API...")
        r = subprocess.run(
            [sys.executable, str(DATA_DIR / "fetch_dataset.py")],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"fetch falhou: {r.stderr[-300:]}")
        _log("Datasets baixados.")

        _log("Filtrando era moderna...")
        r = subprocess.run(
            [sys.executable, str(DATA_DIR / "filter_modern.py")],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"filtro falhou: {r.stderr[-300:]}")
        _log("Filtro aplicado.")

        kairos_engine._cache.clear()
        _log("Cache do engine recarregado.")

        for name in train_lotteries:
            _log(f"Treinando modelo ML: {name}...")
            meta = ml_model.train(name)
            _log(f"Modelo {name} treinado — AUC {meta['auc_holdout']} "
                 f"| P@k {meta['precision_at_k']} (baseline {meta['baseline']}) "
                 f"| {meta['train_seconds']}s")

        with _lock:
            _status["state"] = "done"
            _status["finished_at"] = time.strftime("%H:%M:%S")
        _log("Atualizacao concluida.")
    except Exception as e:
        with _lock:
            _status["state"] = "error"
            _status["finished_at"] = time.strftime("%H:%M:%S")
        _log(f"ERRO: {e}")


def start_update(train_lotteries: list[str] | None = None) -> bool:
    """Inicia atualizacao em background. False se ja ha uma rodando."""
    with _lock:
        if _status["state"] == "running":
            return False
        _status.update(state="running", step="iniciando", log=[],
                       started_at=time.strftime("%H:%M:%S"), finished_at=None)
    t = threading.Thread(target=_run, args=(train_lotteries or [],), daemon=True)
    t.start()
    return True


def start_train_only(name: str) -> bool:
    """Treina apenas um modelo, sem baixar dados."""
    with _lock:
        if _status["state"] == "running":
            return False
        _status.update(state="running", step="iniciando", log=[],
                       started_at=time.strftime("%H:%M:%S"), finished_at=None)

    def run():
        try:
            _log(f"Treinando modelo ML: {name}...")
            meta = ml_model.train(name)
            _log(f"Modelo {name} treinado — AUC {meta['auc_holdout']} "
                 f"| P@k {meta['precision_at_k']} (baseline {meta['baseline']}) "
                 f"| {meta['train_seconds']}s")
            with _lock:
                _status["state"] = "done"
                _status["finished_at"] = time.strftime("%H:%M:%S")
        except Exception as e:
            with _lock:
                _status["state"] = "error"
            _log(f"ERRO: {e}")

    threading.Thread(target=run, daemon=True).start()
    return True
