# Proyecto II: procesamiento paralelo de texto con CUDA

Cómputo Paralelo y Distribuido (TI155), grupo 7C, Universidad del Valle de Puebla.
Profesor: Jorge Arturo Flores López.

Equipo: Camila Serrano Trinidad, Irais Macuil Chapuli, Misael Reynoso Aguayo y Christian Luna Ortiz.

El programa cuenta frecuencias de eventos en un CSV de 5.28 GiB (tienda en línea, octubre de 2019) y lo hace de tres formas:

1. CPU secuencial (línea base).
2. CPU multihilo, esquema maestro-trabajador con reducción.
3. GPU con CUDA (Numba).

Las tres dan exactamente el mismo resultado. El documento de la entrega explica el diseño, las mediciones y las conclusiones.

## Qué cuenta

- Eventos por tipo: `view`, `cart`, `remove_from_cart`, `purchase`.
- Eventos por categoría (`category_code`).
- Eventos por categoría, solo compras.

## Datos

El archivo `2019-Oct.csv` no está en el repositorio porque pesa 5.28 GiB. Se descarga de Kaggle:
[eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) (REES46 Marketing Platform, 2019).
Se coloca en `data/2019-Oct.csv`.

Para desarrollar sin el archivo real, `gen_synthetic.py` crea un CSV de prueba con el mismo formato.

## Requisitos

- Windows 11 y una GPU NVIDIA con controlador reciente.
- Python 3.12.
- Las librerías de `requirements.txt` (incluyen el compilador y el runtime de CUDA 12.9 como paquetes de pip).
- Unos 12 GB de RAM libres para el archivo completo.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

NumPy está fijado en 2.3.5 porque la versión 2.5 rompe Numba (`np.row_stack`).

## Cómo correrlo

Validar contra pandas (archivo completo):

```
python validate_full.py --file data/2019-Oct.csv
```

Medir (la corrida final del documento):

```
python bench.py --file data/2019-Oct.csv --fracs 0.5 0.75 1.0 --threads 2 4 8 16 32 --reps 5 --pin --out results/results.json
```

Generar tablas y gráficas:

```
python make_report.py --in results/results.json --outdir results
```

Conviene tener la laptop conectada a la corriente, en modo de alto rendimiento y con la GPU NVIDIA activa. No abrir otras aplicaciones pesadas durante la medición.

## Archivos

| Archivo | Qué contiene |
| --- | --- |
| `core.py` | Lectura de una línea y las tres versiones. |
| `bench.py` | Mediciones, ficha de hardware y comparación entre las tres versiones. |
| `validate.py` | Comparación contra pandas con las primeras filas. |
| `validate_full.py` | Comparación contra pandas con el archivo completo. |
| `make_report.py` | Tablas y gráficas a partir de los resultados. |
| `gen_synthetic.py` | CSV sintético para pruebas. |
| `results/` | Resultados de la corrida final, tablas, gráficas y corridas sin afinidad. |

## Resultados (archivo completo)

Mediana de 5 repeticiones, en una laptop con Ryzen 9 9955HX (16 núcleos) y RTX 5070 Ti.

| Versión | Tiempo (s) | Aceleración |
| --- | --- | --- |
| Secuencial | 9.535 | 1.00× |
| Multihilo, 16 hilos | 0.647 | 14.74× |
| Multihilo, 32 hilos | 0.631 | 15.11× |
| GPU, solo kernel | 0.138 | 69.33× |
| GPU, con copia por PCIe | 0.598 | 15.93× |

La copia del archivo a la tarjeta es el 77 % del tiempo total de la GPU.

## Nota sobre la afinidad de hilos (`--pin`)

Sin fijar la afinidad, el multihilo con 2 y 4 hilos rindió de forma irregular (con 2 hilos, entre 1.1× y 2.0× según la corrida). Al fijar cada ejecución a p núcleos físicos distintos, la aceleración salió casi lineal. Sospechamos que el sistema ponía hilos en el mismo núcleo físico, pero no lo comprobamos. Las corridas sin fijar están en `results/` con `sin_pin` en el nombre.

`--pin` usa la API de Windows, así que solo funciona en Windows.
