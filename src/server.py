"""
Kairos — Web Server local
Roda em http://localhost:7777

Rotas:
  GET  /                    UI
  GET  /api/lotteries       stats de todas as loterias
  GET  /api/defaults        config default do gerador
  POST /api/generate        gera jogos {lottery, n_games, filtros...}
  GET  /api/system/status   status do updater + modelos ML por loteria
  POST /api/system/update   baixa datasets da API (+ retreina modelos existentes)
  POST /api/ml/train        treina o modelo ML de uma loteria {lottery}
"""

from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path

import json
import threading

from kairos_engine import get_lottery, available, DEFAULTS
import audit
import ml_model
import updater

DATA_DIR = Path(__file__).parent.parent / "data"
_audit_running = set()
_audit_lock = threading.Lock()

STATIC = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC))


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/api/lotteries")
def lotteries():
    out = []
    for name in available():
        try:
            s = get_lottery(name).stats()
            s["ml"] = ml_model.status(name)
            out.append(s)
        except Exception:
            continue
    return jsonify(out)


@app.route("/api/defaults")
def defaults():
    return jsonify(DEFAULTS)


@app.route("/api/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    name = body.get("lottery", "megasena")
    n_games = max(1, min(10, int(body.get("n_games", 1))))
    cfg = {k: body[k] for k in DEFAULTS if k in body}
    try:
        lot = get_lottery(name)
    except FileNotFoundError:
        return jsonify({"error": f"loteria desconhecida: {name}"}), 404

    ml_probs = None
    ml_used = False
    if cfg.get("use_ml"):
        ml_probs = ml_model.predict_next(name)
        ml_used = ml_probs is not None

    games = lot.generate(n_games, cfg, ml_probs=ml_probs)
    return jsonify({
        "lottery": name,
        "games": games,
        "ml_used": ml_used,
        "config_used": {**DEFAULTS, **cfg},
        "stats": lot.stats(),
    })


@app.route("/api/system/status")
def system_status():
    st = updater.get_status()
    st["models"] = {n: ml_model.status(n) for n in available()}
    return jsonify(st)


@app.route("/api/system/update", methods=["POST"])
def system_update():
    body = request.get_json(silent=True) or {}
    # retreina modelos que ja existem (mantidos atualizados com os dados novos)
    retrain = body.get("retrain", True)
    trained = [n for n in available() if ml_model.status(n)] if retrain else []
    if not updater.start_update(trained):
        return jsonify({"error": "atualizacao ja em andamento"}), 409
    return jsonify({"ok": True, "will_retrain": trained})


@app.route("/api/ml/train", methods=["POST"])
def ml_train():
    body = request.get_json(force=True)
    name = body.get("lottery", "megasena")
    if name not in available():
        return jsonify({"error": f"loteria desconhecida: {name}"}), 404
    if not updater.start_train_only(name):
        return jsonify({"error": "operacao ja em andamento"}), 409
    return jsonify({"ok": True})


@app.route("/api/audit/<name>")
def audit_get(name):
    p = DATA_DIR / f"audit_{name}.json"
    running = name in _audit_running
    if not p.exists():
        return jsonify({"exists": False, "running": running})
    rep = json.loads(p.read_text(encoding="utf-8"))
    rep["exists"] = True
    rep["running"] = running
    return jsonify(rep)


@app.route("/api/audit/run", methods=["POST"])
def audit_run():
    body = request.get_json(force=True)
    name = body.get("lottery", "megasena")
    if name not in available():
        return jsonify({"error": f"loteria desconhecida: {name}"}), 404
    with _audit_lock:
        if name in _audit_running:
            return jsonify({"error": "auditoria ja em andamento"}), 409
        _audit_running.add(name)

    def run():
        try:
            audit.run_audit(name, fast=True)
        finally:
            with _audit_lock:
                _audit_running.discard(name)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("\n  Kairos server -> http://localhost:7777\n")
    app.run(host="0.0.0.0", port=7777, debug=False)
