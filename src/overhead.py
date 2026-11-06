"""Partie B : gain d'overhead vs balayage exhaustif + tableau comparatif."""
import pandas as pd
from src.evaluate import topk_accuracy, N_BEAMS


def overhead_gain(k, n=N_BEAMS):
    """% de faisceaux en moins à tester si on n'en balaie que k sur n."""
    return 100.0 * (n - k) / n


def comparison_table(scores, labels, ks=(1, 3, 5)):
    """scores = {"Caméra seule": logits(N,64), "Radar seul": ..., ...}  -> DataFrame."""
    rows = [{"Méthode": "Balayage exhaustif", **{f"Top-{k}": 100.0 for k in ks}, "Faisceaux testés": N_BEAMS, "Gain": "0 %"}]
    for name, lg in scores.items():
        acc = topk_accuracy(lg, labels, ks)
        row = {"Méthode": name}
        row.update({f"Top-{k}": round(100 * acc[f"top{k}"], 1) for k in ks})
        row["Faisceaux testés"] = "/".join(str(k) for k in ks)
        row["Gain"] = "/".join(f"{overhead_gain(k):.1f}" for k in ks) + " %"
        rows.append(row)
    return pd.DataFrame(rows)