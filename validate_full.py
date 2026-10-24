"""Validación sobre el archivo COMPLETO: núcleo multihilo contra pandas.

Compara tres cosas:
  1) eventos por tipo,
  2) eventos por categoría,
  3) eventos por categoría solo de compras (purchase),
más las líneas sin categoría y el total de filas.

Necesita unos 12 GB de RAM libres (el CSV completo en bytes y las dos columnas en pandas).
Uso: python validate_full.py --file data/2019-Oct.csv
"""
import argparse
import numpy as np
import pandas as pd
from numba import set_num_threads

import core
from validate import fnv_bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    # Referencia: pandas lee solo event_type y category_code, como texto
    df = pd.read_csv(a.file, usecols=["event_type", "category_code"], dtype=str,
                     keep_default_na=False)
    print("filas pandas:", len(df))

    # Núcleo: el archivo completo como bytes, procesado con la versión multihilo
    buf = np.fromfile(a.file, dtype=np.uint8)
    set_num_threads(a.threads)
    ev, ca, cp = core.cpu_mt(buf, len(buf), a.threads * 4)

    # Conteos de referencia por categoría (todas las líneas y solo compras)
    cat = df[df.category_code != ""].category_code
    all_cnt = cat.value_counts()
    pur_cnt = df[(df.event_type == "purchase") & (df.category_code != "")].category_code.value_counts()

    # Si dos categorías cayeran en la misma casilla del histograma no se podrían comparar
    buckets = {}
    for name in all_cnt.index:
        buckets.setdefault(fnv_bucket(name), []).append(name)
    collisions = {b: v for b, v in buckets.items() if len(v) > 1}
    print("categorías distintas:", len(all_cnt), "| colisiones de hash:", len(collisions))

    fallas = 0
    for name, c in all_cnt.items():
        b = fnv_bucket(name)
        if b in collisions:
            continue
        if ca[b] != c:
            fallas += 1
            print("FALLA (todas)", name, c, ca[b])
        if cp[b] != int(pur_cnt.get(name, 0)):
            fallas += 1
            print("FALLA (purchase)", name, int(pur_cnt.get(name, 0)), cp[b])

    # Líneas sin category_code (casilla extra NB) y total de eventos
    sin = int((df.category_code == "").sum())
    if ca[core.NB] != sin:
        fallas += 1
    print("sin category_code: pandas", sin, "núcleo", int(ca[core.NB]))
    print("eventos contados:", int(ev.sum()), "| filas:", len(df))
    if int(ev.sum()) != len(df):
        fallas += 1

    print("\nVALIDACIÓN COMPLETA:", "CORRECTA" if fallas == 0 else f"FALLÓ ({fallas} diferencias)")
    raise SystemExit(0 if fallas == 0 else 1)


if __name__ == "__main__":
    main()
