"""Benchmark del Proyecto II. Mide secuencial, multihilo y GPU (con PCIe).

Para cada tamaño (fracción del archivo) corre las tres versiones de core.py,
comprueba que den el mismo resultado y guarda los tiempos en un JSON.

Uso:
  python bench.py --file data/2019-Oct.csv --fracs 0.5 0.75 1.0 \
                  --threads 2 4 8 16 32 --reps 5 --pin --out results/results.json
Opcional: --gpu1-bytes 50000000  (referencia: kernel de UN solo hilo de GPU)
"""
import argparse, json, os, platform, statistics, subprocess, time
import numpy as np
from numba import cuda, set_num_threads

import core


def pin_cores(p, logical=32):
    """Windows: fija el proceso a p NÚCLEOS FÍSICOS distintos (lógicos pares 0,2,4..;
    el hermano SMT de cada núcleo queda libre). p >= núcleos físicos -> todos los lógicos.
    Evita que el planificador amontone hilos en hermanos SMT (ver README)."""
    import ctypes
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    phys = logical // 2
    # La máscara es un entero con un bit por procesador lógico.
    # Con p núcleos físicos se encienden los bits pares: 0, 2, 4, ...
    mask = (1 << logical) - 1 if p > phys else sum(1 << (2 * i) for i in range(p))
    assert k.SetProcessAffinityMask(k.GetCurrentProcess(), mask), "no se pudo fijar afinidad"


def best(fn, reps):
    """Mediana de `reps` ejecuciones (devuelve tiempo y último resultado)."""
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


def hw_sheet():
    """Ficha de hardware: CPU y RAM (vía PowerShell), GPU (vía CUDA), SO y Python."""
    cpu = {}
    try:
        ps = ("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
              "NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize | ConvertTo-Json")
        cpu = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", ps], text=True))
    except Exception as e:
        cpu = {"error": str(e)}
    dev = cuda.get_current_device()
    free, total = cuda.current_context().get_memory_info()
    gpu = {"name": dev.name.decode() if isinstance(dev.name, bytes) else dev.name,
           "compute_capability": list(dev.compute_capability),
           "sm_count": dev.MULTIPROCESSOR_COUNT,
           "vram_total_gb": round(total / 2**30, 2),
           "max_threads_per_block": dev.MAX_THREADS_PER_BLOCK}
    ram = None
    try:
        ram = round(int(subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"], text=True)) / 2**30, 1)
    except Exception:
        pass
    return {"cpu": cpu, "gpu": gpu, "ram_gb": ram, "os": platform.platform(),
            "python": platform.python_version()}


def line_aligned(buf, target):
    """Primer índice > target que cae justo después de un '\\n'.
    Sirve para cortar el archivo al 50 %, 75 %... sin partir una línea a la mitad."""
    if target >= len(buf):
        return len(buf)
    j = target
    while buf[j] != 10:           # 10 = '\n'
        j += 1
    return j + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2048, help="bytes por hilo de GPU")
    ap.add_argument("--blocks-per-thread", type=int, default=4)
    ap.add_argument("--gpu1-bytes", type=int, default=0)
    ap.add_argument("--pin", action="store_true",
                    help="fija p núcleos físicos distintos por ejecución (Windows)")
    ap.add_argument("--out", default="results/results.json")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # El archivo se carga completo a RAM como bytes. La lectura de disco NO se mide.
    print(f"Leyendo {a.file} ...")
    t0 = time.perf_counter()
    buf = np.fromfile(a.file, dtype=np.uint8)
    print(f"  {len(buf)/2**30:.2f} GiB en {time.perf_counter()-t0:.1f}s")

    hw = hw_sheet()
    res = {"hw": hw, "file": os.path.basename(a.file), "file_bytes": int(len(buf)),
           "chunk": a.chunk, "reps": a.reps, "pinned": a.pin, "runs": []}

    # Calentamiento: compila todo con un trozo pequeño (JIT fuera de la medición)
    w = buf[:line_aligned(buf, min(1_000_000, len(buf) - 1))]
    core.cpu_seq(w, len(w))
    set_num_threads(2)
    core.cpu_mt(w, len(w), 8)
    d = cuda.to_device(w)
    g = core.gpu_alloc()
    core.gpu_launch(d, len(w), a.chunk, *g)
    cuda.synchronize()
    del d, g

    for frac in a.fracs:
        # Prefijo del archivo que corresponde a esta fracción, cortado en un salto de línea
        n = line_aligned(buf, int(len(buf) * frac) - 1)
        view = buf[:n]
        row = {"frac": frac, "bytes": int(n)}
        print(f"\n=== {frac*100:.0f}%  ({n/2**20:.0f} MiB) ===")

        # --- Versión 1: secuencial (1 solo núcleo). Su resultado `ref` es la referencia.
        if a.pin:
            pin_cores(1)
        t_seq, ref = best(lambda: core.cpu_seq(view, n), a.reps)
        row["t_seq"] = t_seq
        print(f"  secuencial        {t_seq:8.3f} s")

        # --- Versión 2: multihilo con p hilos; cada resultado debe ser igual a `ref`
        row["mt"] = {}
        for p in a.threads:
            if a.pin:
                pin_cores(p)
            set_num_threads(p)
            t_mt, out = best(lambda: core.cpu_mt(view, n, p * a.blocks_per_thread), a.reps)
            assert core.same(ref, out), f"multihilo p={p} difiere del secuencial"
            row["mt"][str(p)] = t_mt
            print(f"  multihilo p={p:<3}    {t_mt:8.3f} s   x{t_seq/t_mt:5.2f}")

        # --- Versión 3: GPU. Se mide por separado:
        #     H2D (copia a la tarjeta, PCIe), kernel, y D2H (copia de regreso + suma de réplicas).
        if a.pin:
            pin_cores(32)
        h2d, kern, d2h = [], [], []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            d_buf = cuda.to_device(view)       # archivo -> VRAM
            g = core.gpu_alloc()               # histogramas en ceros -> VRAM
            cuda.synchronize()
            t1 = time.perf_counter()
            core.gpu_launch(d_buf, n, a.chunk, *g)
            cuda.synchronize()                 # espera a que termine el kernel
            t2 = time.perf_counter()
            out = core.gpu_collect(*g)         # histogramas VRAM -> RAM y suma de réplicas
            t3 = time.perf_counter()
            assert core.same(ref, out), "GPU difiere del secuencial"
            h2d.append(t1 - t0); kern.append(t2 - t1); d2h.append(t3 - t2)
            del d_buf, g
        row["gpu"] = {"h2d": statistics.median(h2d), "kernel": statistics.median(kern),
                      "d2h": statistics.median(d2h)}
        row["gpu"]["pcie"] = row["gpu"]["h2d"] + row["gpu"]["d2h"]       # costo de transferencia
        row["gpu"]["total"] = row["gpu"]["pcie"] + row["gpu"]["kernel"]  # GPU "real", con PCIe
        gg = row["gpu"]
        print(f"  GPU PCIe {gg['pcie']:.3f} s | kernel {gg['kernel']:.3f} s | total {gg['total']:.3f} s"
              f"   x{t_seq/gg['kernel']:.1f} puro / x{t_seq/gg['total']:.1f} real")

        row["events"] = {k: int(v) for k, v in zip(core.EV, ref[0])}
        res["runs"].append(row)

    # Referencia opcional: qué pasa si la GPU usa UN solo hilo (todo el archivo en un tramo)
    if a.gpu1_bytes:
        n1 = line_aligned(buf, min(a.gpu1_bytes, len(buf) - 1))
        d_buf = cuda.to_device(buf[:n1]); g = core.gpu_alloc()
        t0 = time.perf_counter()
        core.gpu_launch(d_buf, n1, n1, *g)        # un solo hilo de GPU
        cuda.synchronize()
        t1 = time.perf_counter() - t0
        t_cpu, _ = best(lambda: core.cpu_seq(buf[:n1], n1), 1)
        res["gpu_1thread"] = {"bytes": int(n1), "t_gpu_1thread": t1, "t_cpu_seq": t_cpu}
        print(f"\nReferencia 1 hilo GPU ({n1/2**20:.0f} MiB): {t1:.2f} s  vs CPU seq {t_cpu:.2f} s")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nGuardado en {a.out}")


if __name__ == "__main__":
    main()
