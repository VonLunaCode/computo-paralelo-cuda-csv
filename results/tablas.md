## Ficha de hardware

- **CPU:** AMD Ryzen 9 9955HX 16-Core Processor — 16 núcleos / 32 hilos, L2 16384 KB, L3 65536 KB
- **GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU — CC 12.0, 46 SMs, 11.94 GB VRAM
- **RAM:** 31.2 GB · **SO:** Windows-11-10.0.26300-SP0 · **Python:** 3.12.10
- **Archivo:** 2019-Oct.csv (5.28 GiB), mediana de 5 repeticiones

## Tabla comparativa de tiempos (s)

| Tamaño | MiB | Secuencial | Multihilo p=2 | Multihilo p=4 | Multihilo p=8 | Multihilo p=16 | Multihilo p=32 | PCIe (H2D+D2H) | Kernel GPU | Total GPU con PCIe |
|---|---|---|---|---|---|---|---|---|---|---|
| 50% | 2703 | 4.763 | 2.355 | 1.189 | 0.607 | 0.324 | 0.342 | 0.250 | 0.068 | 0.318 |
| 75% | 4055 | 7.145 | 3.545 | 1.781 | 0.906 | 0.488 | 0.481 | 0.352 | 0.103 | 0.454 |
| 100% | 5406 | 9.535 | 4.710 | 2.351 | 1.193 | 0.647 | 0.631 | 0.461 | 0.138 | 0.598 |

## Matriz de aceleración pura (T_secuencial / T_versión, sin PCIe)

| Tamaño | Multihilo p=2 | Multihilo p=4 | Multihilo p=8 | Multihilo p=16 | Multihilo p=32 | GPU (solo kernel) |
|---|---|---|---|---|---|---|
| 50% | 2.02× | 4.01× | 7.85× | 14.72× | 13.94× | 69.56× |
| 75% | 2.02× | 4.01× | 7.89× | 14.63× | 14.86× | 69.56× |
| 100% | 2.02× | 4.06× | 7.99× | 14.74× | 15.11× | 69.33× |

## Matriz de aceleración real (incluye PCIe en la GPU)

| Tamaño | Multihilo p=2 | Multihilo p=4 | Multihilo p=8 | Multihilo p=16 | Multihilo p=32 | GPU total (con PCIe) |
|---|---|---|---|---|---|---|
| 50% | 2.02× | 4.01× | 7.85× | 14.72× | 13.94× | 14.97× |
| 75% | 2.02× | 4.01× | 7.89× | 14.63× | 14.86× | 15.72× |
| 100% | 2.02× | 4.06× | 7.99× | 14.74× | 15.11× | 15.93× |

## Eventos contados (100 %)

| view | cart | remove_from_cart | purchase |
|---|---|---|---|
| 40,779,399 | 926,516 | 0 | 742,849 |

## Referencia: kernel de 1 solo hilo de GPU

En 48 MiB: GPU 1 hilo = 4.06 s, CPU secuencial = 0.080 s (51× más lento).
