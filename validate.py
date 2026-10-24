"""Validación independiente: compara el conteo del núcleo contra pandas.
Usa solo las primeras filas del archivo (rápido). Para el archivo completo ver validate_full.py.
Uso: python validate.py --file data/synthetic.csv [--rows 3000000]
"""
import argparse
import numpy as np
import pandas as pd
import core


def fnv_bucket(s: str) -> int:
    """Misma cuenta que hace el núcleo: hash FNV-1a del texto, módulo 65,536 casillas."""
    h = 2166136261                          # valor inicial de FNV-1a (32 bits)
    for c in s.encode("utf-8"):
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return h & core.MASK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--rows", type=int, default=3_000_000)
    a = ap.parse_args()

    # Referencia: pandas lee solo las dos columnas que nos interesan, como texto
    df = pd.read_csv(a.file, nrows=a.rows, usecols=["event_type", "category_code"],
                     dtype=str, keep_default_na=False)
    # recorta el buffer a exactamente las filas leídas (+ cabecera)
    with open(a.file, "rb") as f:
        n = 0
        for i, line in enumerate(f):
            n += len(line)
            if i == len(df):          # cabecera + len(df) filas
                break
    buf = np.fromfile(a.file, dtype=np.uint8, count=n)
    ev, ca, cp = core.cpu_seq(buf, n)       # nuestro resultado (versión secuencial)

    ok = True
    # 1) Conteo por tipo de evento
    for i, name in enumerate(core.EV):
        ref = int((df.event_type == name).sum())
        good = ref == ev[i]
        ok &= good
        print(f"  {name:<18} pandas={ref:>10,}  núcleo={ev[i]:>10,}  {'OK' if good else 'FALLA'}")

    # 2) Conteo por categoría. Antes se revisa que dos categorías no caigan en la misma casilla.
    cat = df[df.category_code != ""].category_code
    ref_all = cat.value_counts()
    buckets = {}
    for name in ref_all.index:
        buckets.setdefault(fnv_bucket(name), []).append(name)
    collisions = {b: v for b, v in buckets.items() if len(v) > 1}
    print(f"  categorías distintas: {len(ref_all)}  | colisiones de hash: {len(collisions)}")
    for name, cnt in ref_all.items():
        b = fnv_bucket(name)
        if b in collisions:
            continue
        if ca[b] != cnt:
            ok = False
            print(f"  FALLA categoría {name}: pandas={cnt} núcleo={ca[b]}")

    # 3) Líneas sin category_code (casilla extra NB)
    sin = int((df.category_code == "").sum())
    ok &= (ca[core.NB] == sin)
    print(f"  sin category_code  pandas={sin:>10,}  núcleo={ca[core.NB]:>10,}")
    print("\nVALIDACIÓN:", "CORRECTA" if ok else "FALLÓ")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
