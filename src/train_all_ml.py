"""Treina o modelo ML de todas as loterias suportadas."""
import ml_model
from kairos_engine import available

for name in available():
    print(f"Treinando {name}...", flush=True)
    try:
        meta = ml_model.train(name)
        print(f"  ok — AUC {meta['auc_holdout']} | P@k {meta['precision_at_k']} "
              f"(baseline {meta['baseline']}) | {meta['train_seconds']}s", flush=True)
    except Exception as e:
        print(f"  ERRO: {e}", flush=True)
print("Done.")
