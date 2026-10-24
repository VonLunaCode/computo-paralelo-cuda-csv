"""Núcleo del Proyecto II: procesamiento de texto (CSV) en 3 versiones.

Métrica: frecuencia de eventos por event_type y por category_code
(hash FNV-1a de la cadena -> NB buckets), más el filtrado "solo purchase".

Los tres caminos usan EXACTAMENTE la misma función de parseo (parse_line),
compilada una vez para CPU (njit) y otra para GPU (cuda.jit device).

Particionado maestro-trabajador: el maestro divide los BYTES del archivo en
bloques; cada trabajador alinea su bloque al salto de línea (procesa solo
las líneas que EMPIEZAN dentro de su bloque) y llena su histograma privado;
el maestro reúne los parciales con una reducción (suma).
"""
import numpy as np
from numba import njit, prange, cuda

NB = 1 << 16          # buckets de categoría (la posición NB = "sin category_code")
MASK = NB - 1
R = 64                # réplicas del histograma en GPU (reduce contención de atómicos)
EV = ("view", "cart", "remove_from_cart", "purchase")


# --------------------------------------------------------------------------
# Parseo de una línea (compartido CPU/GPU). Campos del CSV, separados por coma:
# 0 event_time,1 event_type,2 product_id,3 category_id,4 category_code,5 brand,...
# Devuelve (inicio de la siguiente línea, tipo de evento 0..3 o -1, bucket).
# --------------------------------------------------------------------------
def _parse_line(buf, pos, n):
    ev = -1
    commas = 0
    h = 2166136261
    cat_len = 0
    i = pos
    while i < n:
        c = buf[i]
        if c == 10:                      # '\n'
            break
        if c == 44:                      # ','
            commas += 1
            if commas == 1 and i + 1 < n:
                f = buf[i + 1]
                if f == 118:             # v(iew)
                    ev = 0
                elif f == 99:            # c(art)
                    ev = 1
                elif f == 114:           # r(emove_from_cart)
                    ev = 2
                elif f == 112:           # p(urchase)
                    ev = 3
        elif commas == 4:                # dentro de category_code
            h = ((h ^ c) * 16777619) & 0xFFFFFFFF
            cat_len += 1
        i += 1
    cat = NB
    if cat_len > 0:
        cat = h & MASK
    return i + 1, ev, cat


parse_cpu = njit(_parse_line)
parse_gpu = cuda.jit(device=True)(_parse_line)


# --------------------------------------------------------------------------
# CPU: un bloque [s, e) del archivo
# --------------------------------------------------------------------------
@njit
def cpu_block(buf, s, e, n, ev_h, cat_all, cat_pur):
    pos = s
    if pos > 0 and buf[pos - 1] != 10:   # la línea parcial pertenece al bloque anterior
        while pos < n and buf[pos] != 10:
            pos += 1
        pos += 1
    while pos < e and pos < n:
        pos, ev, cat = parse_cpu(buf, pos, n)
        if ev >= 0:                      # ignora cabecera y eventos desconocidos
            ev_h[ev] += 1
            cat_all[cat] += 1
            if ev == 3:
                cat_pur[cat] += 1


@njit
def cpu_seq(buf, n):
    """Versión 1: secuencial (un solo hilo, un solo bloque)."""
    ev = np.zeros(4, np.int64)
    ca = np.zeros(NB + 1, np.int64)
    cp = np.zeros(NB + 1, np.int64)
    cpu_block(buf, 0, n, n, ev, ca, cp)
    return ev, ca, cp


@njit(parallel=True)
def cpu_mt(buf, n, nblocks):
    """Versión 2: maestro-trabajador multihilo. Hilos = numba.set_num_threads."""
    # Un juego de histogramas por bloque: ningún trabajador escribe donde escribe otro,
    # así que no hace falta bloquear ni usar operaciones atómicas.
    ev = np.zeros((nblocks, 4), np.int64)
    ca = np.zeros((nblocks, NB + 1), np.int64)
    cp = np.zeros((nblocks, NB + 1), np.int64)
    for b in prange(nblocks):            # trabajadores (Numba los reparte entre los hilos)
        s = b * n // nblocks
        e = (b + 1) * n // nblocks
        cpu_block(buf, s, e, n, ev[b], ca[b], cp[b])
    return ev.sum(0), ca.sum(0), cp.sum(0)   # reducción del maestro


# --------------------------------------------------------------------------
# GPU: cada hilo procesa un tramo de `chunk` bytes
# --------------------------------------------------------------------------
@cuda.jit
def gpu_kernel(buf, n, chunk, ev_g, ca_g, cp_g):
    t = cuda.grid(1)                     # índice global de este hilo
    s = t * chunk                        # su tramo es el rango de bytes [s, e)
    if s >= n:                           # hilos sobrantes del último bloque: no hacen nada
        return
    e = s + chunk
    if e > n:
        e = n
    rep = cuda.blockIdx.x % R            # réplica del histograma que usa todo su bloque
    pos = s
    # Misma regla que en CPU: si el tramo empieza a la mitad de una línea, se salta hasta
    # el siguiente salto de línea; esa línea la procesa el hilo del tramo anterior.
    if pos > 0 and buf[pos - 1] != 10:
        while pos < n and buf[pos] != 10:
            pos += 1
        pos += 1
    # Contadores por tipo de evento en variables locales (registros): son rápidos y
    # evitan una suma atómica por línea. Se vuelcan una sola vez al final.
    c0 = 0
    c1 = 0
    c2 = 0
    c3 = 0
    while pos < e and pos < n:
        pos, ev, cat = parse_gpu(buf, pos, n)
        if ev >= 0:
            if ev == 0:
                c0 += 1
            elif ev == 1:
                c1 += 1
            elif ev == 2:
                c2 += 1
            else:
                c3 += 1
                cuda.atomic.add(cp_g, (rep, cat), 1)     # histograma solo de compras
            cuda.atomic.add(ca_g, (rep, cat), 1)         # histograma de todas las categorías
    # Volcado final de los contadores locales
    cuda.atomic.add(ev_g, (rep, 0), c0)
    cuda.atomic.add(ev_g, (rep, 1), c1)
    cuda.atomic.add(ev_g, (rep, 2), c2)
    cuda.atomic.add(ev_g, (rep, 3), c3)


def gpu_alloc():
    """Histogramas replicados en GPU (cuentan como transferencia PCIe)."""
    return (cuda.to_device(np.zeros((R, 4), np.int64)),
            cuda.to_device(np.zeros((R, NB + 1), np.int64)),
            cuda.to_device(np.zeros((R, NB + 1), np.int64)))


def gpu_launch(d_buf, n, chunk, ev_g, ca_g, cp_g, tpb=256):
    nthreads = (n + chunk - 1) // chunk      # un hilo por cada tramo de `chunk` bytes (redondeo hacia arriba)
    blocks = (nthreads + tpb - 1) // tpb     # bloques de `tpb` hilos para cubrirlos todos
    gpu_kernel[blocks, tpb](d_buf, n, chunk, ev_g, ca_g, cp_g)


def gpu_collect(ev_g, ca_g, cp_g):
    """Trae los histogramas a la RAM (copia D2H) y suma las R réplicas en el host."""
    return (ev_g.copy_to_host().sum(0),
            ca_g.copy_to_host().sum(0),
            cp_g.copy_to_host().sum(0))


def same(a, b):
    """True si dos resultados (ev, ca, cp) son idénticos. Lo usa bench.py para validar."""
    return all(np.array_equal(x, y) for x, y in zip(a, b))
