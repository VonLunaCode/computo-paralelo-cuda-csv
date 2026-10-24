"""Genera un CSV sintético con el mismo formato que 'eCommerce behavior data
from multi category store' (REES46), para desarrollar sin el archivo de 5 GB.

Los datos son inventados: sirven para probar el código, no para sacar resultados.

Uso: python gen_synthetic.py --rows 2000000 --out data/synthetic.csv
"""
import argparse
import numpy as np
import pandas as pd

# Mismas columnas y mismo orden que el CSV real
COLS = ["event_time", "event_type", "product_id", "category_id",
        "category_code", "brand", "price", "user_id", "user_session"]

# Piezas para armar categorías con forma "electronics.audio.headphone"
TOP = ["electronics", "appliances", "computers", "apparel", "furniture",
       "construction", "kids", "auto", "sport", "accessories", "stationery"]
SUB = ["smartphone", "audio", "video", "kitchen", "notebook", "shoes",
       "bed", "tools", "toys", "carrier", "bag", "printer", "clocks", "light",
       "environment", "personal", "sofa", "cabinet"]
LEAF = ["headphone", "tv", "refrigerator", "washer", "vacuum", "iron",
        "desktop", "tablet", "jeans", "dress", "table", "chair", "drill"]


def build_categories(rng, k=127):
    """Arma k categorías distintas combinando las listas de arriba."""
    cats = set()
    while len(cats) < k:
        parts = [rng.choice(TOP), rng.choice(SUB)]
        if rng.random() < 0.5:                  # la mitad lleva un tercer nivel
            parts.append(rng.choice(LEAF))
        cats.add(".".join(parts))
    return np.array(sorted(cats), dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--out", default="data/synthetic.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pyrng = __import__("random").Random(a.seed)
    cats = build_categories(pyrng)
    brands = np.array([f"brand{i}" for i in range(300)], dtype=object)
    sessions = np.array([f"{rng.integers(0, 2**63):016x}-{i:08x}" for i in range(200_000)], dtype=object)
    events = np.array(["view", "cart", "remove_from_cart", "purchase"], dtype=object)

    # Se escribe por bloques de 500,000 filas para no llenar la RAM
    chunk = 500_000
    first = True                                # la cabecera solo va en el primer bloque
    done = 0
    base = pd.Timestamp("2019-10-01")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        while done < a.rows:
            m = min(chunk, a.rows - done)
            # Probabilidades parecidas al archivo real: casi todo son "view"
            ev = events[rng.choice(4, size=m, p=[0.95, 0.03, 0.005, 0.015])]
            cat = cats[rng.integers(0, len(cats), size=m)]
            cat[rng.random(m) < 0.32] = ""          # ~32 % sin category_code
            brand = brands[rng.integers(0, len(brands), size=m)]
            brand[rng.random(m) < 0.15] = ""
            ts = base + pd.to_timedelta(np.sort(rng.integers(0, 31 * 86400, size=m)), unit="s")
            df = pd.DataFrame({
                "event_time": ts.strftime("%Y-%m-%d %H:%M:%S") + " UTC",
                "event_type": ev,
                "product_id": rng.integers(1_000_000, 60_000_000, size=m),
                "category_id": rng.integers(2_000_000_000_000_000_000, 2_200_000_000_000_000_000, size=m),
                "category_code": cat,
                "brand": brand,
                "price": np.round(rng.gamma(2.0, 80.0, size=m), 2),
                "user_id": rng.integers(100_000_000, 600_000_000, size=m),
                "user_session": sessions[rng.integers(0, len(sessions), size=m)],
            })[COLS]
            df.to_csv(f, index=False, header=first, lineterminator="\n")
            first = False
            done += m
            print(f"{done:,}/{a.rows:,} filas", end="\r")
    print(f"\nListo: {a.out}")


if __name__ == "__main__":
    main()
