"""Convierte results/*.json en tablas (Markdown), matrices de aceleración y gráficas.
Uso: python make_report.py --in results/results.json --outdir results
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")                       # sin ventana: solo guarda los PNG
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/results.json")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    r = json.load(open(a.inp, encoding="utf-8"))
    os.makedirs(a.outdir, exist_ok=True)
    runs = r["runs"]
    pcts = [f"{int(x['frac']*100)}%" for x in runs]      # "50%", "75%", "100%"
    mb = [x["bytes"] / 2**20 for x in runs]              # tamaño de cada corrida en MiB
    threads = sorted(int(p) for p in runs[0]["mt"])

    md = []
    h = r["hw"]

    # --- Ficha de hardware ---
    md.append("## Ficha de hardware\n")
    cpu = h["cpu"] if isinstance(h["cpu"], dict) else h["cpu"][0]
    md.append(f"- **CPU:** {cpu.get('Name','?').strip()} — {cpu.get('NumberOfCores','?')} núcleos / "
              f"{cpu.get('NumberOfLogicalProcessors','?')} hilos, L2 {cpu.get('L2CacheSize','?')} KB, "
              f"L3 {cpu.get('L3CacheSize','?')} KB")
    g = h["gpu"]
    md.append(f"- **GPU:** {g['name']} — CC {g['compute_capability'][0]}.{g['compute_capability'][1]}, "
              f"{g['sm_count']} SMs, {g['vram_total_gb']} GB VRAM")
    md.append(f"- **RAM:** {h['ram_gb']} GB · **SO:** {h['os']} · **Python:** {h['python']}")
    md.append(f"- **Archivo:** {r['file']} ({r['file_bytes']/2**30:.2f} GiB), "
              f"mediana de {r['reps']} repeticiones\n")

    # --- Tabla de tiempos: una fila por tamaño ---
    md.append("## Tabla comparativa de tiempos (s)\n")
    md.append("| Tamaño | MiB | Secuencial | " + " | ".join(f"Multihilo p={p}" for p in threads) +
              " | PCIe (H2D+D2H) | Kernel GPU | Total GPU con PCIe |")
    md.append("|" + "---|" * (6 + len(threads)))
    for x, p, m in zip(runs, pcts, mb):
        md.append(f"| {p} | {m:.0f} | {x['t_seq']:.3f} | " +
                  " | ".join(f"{x['mt'][str(t)]:.3f}" for t in threads) +
                  f" | {x['gpu']['pcie']:.3f} | {x['gpu']['kernel']:.3f} | {x['gpu']['total']:.3f} |")

    # --- Aceleración pura: T_secuencial / T_versión, la GPU usa solo el kernel ---
    md.append("\n## Matriz de aceleración pura (T_secuencial / T_versión, sin PCIe)\n")
    cols = [f"Multihilo p={t}" for t in threads] + ["GPU (solo kernel)"]
    md.append("| Tamaño | " + " | ".join(cols) + " |")
    md.append("|" + "---|" * (1 + len(cols)))
    for x, p in zip(runs, pcts):
        vals = [x["t_seq"] / x["mt"][str(t)] for t in threads] + [x["t_seq"] / x["gpu"]["kernel"]]
        md.append(f"| {p} | " + " | ".join(f"{v:.2f}×" for v in vals) + " |")

    # --- Aceleración real: la GPU incluye el tiempo de transferencia PCIe ---
    md.append("\n## Matriz de aceleración real (incluye PCIe en la GPU)\n")
    md.append("| Tamaño | " + " | ".join(f"Multihilo p={t}" for t in threads) + " | GPU total (con PCIe) |")
    md.append("|" + "---|" * (2 + len(threads)))
    for x, p in zip(runs, pcts):
        vals = [x["t_seq"] / x["mt"][str(t)] for t in threads] + [x["t_seq"] / x["gpu"]["total"]]
        md.append(f"| {p} | " + " | ".join(f"{v:.2f}×" for v in vals) + " |")

    # --- Eventos contados en el tamaño completo ---
    md.append("\n## Eventos contados (100 %)\n")
    md.append("| " + " | ".join(runs[-1]["events"]) + " |")
    md.append("|" + "---|" * len(runs[-1]["events"]))
    md.append("| " + " | ".join(f"{v:,}" for v in runs[-1]["events"].values()) + " |")

    # --- Referencia del kernel de un solo hilo (solo si se midió) ---
    if "gpu_1thread" in r:
        o = r["gpu_1thread"]
        md.append(f"\n## Referencia: kernel de 1 solo hilo de GPU\n\n"
                  f"En {o['bytes']/2**20:.0f} MiB: GPU 1 hilo = {o['t_gpu_1thread']:.2f} s, "
                  f"CPU secuencial = {o['t_cpu_seq']:.3f} s "
                  f"({o['t_gpu_1thread']/o['t_cpu_seq']:.0f}× más lento).")

    open(os.path.join(a.outdir, "tablas.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # --- Gráficas ---
    x = mb
    best_mt = {p: [run["mt"][str(p)] for run in runs] for p in threads}

    # Figura 1: tiempo (izquierda) y aceleración (derecha) contra el tamaño.
    # Para no saturar solo se dibujan tres valores de p: el menor, el de en medio y el mayor.
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(x, [run["t_seq"] for run in runs], "o-", label="Secuencial", color="#64748b")
    for p in (threads[0], threads[len(threads)//2], threads[-1]):
        ax[0].plot(x, best_mt[p], "s--", label=f"Multihilo p={p}")
    ax[0].plot(x, [run["gpu"]["kernel"] for run in runs], "^-", label="GPU solo kernel", color="#10b981")
    ax[0].plot(x, [run["gpu"]["total"] for run in runs], "v-", label="GPU total (con PCIe)", color="#ef4444")
    ax[0].set_xlabel("Tamaño del problema (MiB de CSV)"); ax[0].set_ylabel("Tiempo (s)")
    ax[0].set_title("Tiempo de ejecución vs tamaño"); ax[0].grid(alpha=.3); ax[0].legend()

    for p in (threads[0], threads[len(threads)//2], threads[-1]):
        ax[1].plot(x, [run["t_seq"] / run["mt"][str(p)] for run in runs], "s--", label=f"Multihilo p={p}")
    ax[1].plot(x, [run["t_seq"] / run["gpu"]["kernel"] for run in runs], "^-", label="GPU pura", color="#10b981")
    ax[1].plot(x, [run["t_seq"] / run["gpu"]["total"] for run in runs], "v-", label="GPU real (con PCIe)", color="#ef4444")
    ax[1].set_xlabel("Tamaño del problema (MiB de CSV)"); ax[1].set_ylabel("Aceleración (×)")
    ax[1].set_title("Aceleración vs tamaño"); ax[1].grid(alpha=.3); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "aceleracion_vs_tamano.png"), dpi=200); plt.close()

    # Figura 2: escalabilidad del CPU con el tamaño completo, contra la recta ideal (aceleración = p)
    fig, ax = plt.subplots(figsize=(7, 5))
    last = runs[-1]
    ax.plot(threads, [last["t_seq"] / last["mt"][str(p)] for p in threads], "s-", label="Aceleración real CPU")
    ax.plot(threads, threads, "--", color="#ef4444", label="Ideal (p)")
    ax.set_xlabel("Hilos"); ax.set_ylabel("Aceleración (×)")
    ax.set_title(f"Escalabilidad CPU multihilo ({pcts[-1]})"); ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "escalabilidad_cpu.png"), dpi=200); plt.close()
    print("Generado:", os.listdir(a.outdir))


if __name__ == "__main__":
    main()
