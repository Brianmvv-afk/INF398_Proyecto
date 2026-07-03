[README.md](https://github.com/user-attachments/files/29642210/README.md)
# ¿Puede el Machine Learning mejorar un portafolio del IPSA frente a estrategias clásicas?

Proyecto del curso **INF398 – Introducción al Aprendizaje Automático**.
Autores: Carlos Tapia, Mauricio Pizarro, Brian Valenzuela.

## Descripción

Se construye y evalúa un portafolio mensual sobre acciones del IPSA (bolsa chilena) comparando
seis estrategias de asignación bajo un esquema de *backtest walk-forward* (2012–2024, ~155 meses):

1. **1/N** — equi-ponderada (benchmark).
2. **Markowitz** — media-varianza clásica.
3. **Mkt_LW** — Markowitz con covarianza encogida (Ledoit-Wolf).
4. **MinVar_LW** — mínima varianza con Ledoit-Wolf.
5. **ML_LW** — predicción de retornos con Ridge + Ledoit-Wolf.
6. **DFL** — red neuronal que asigna pesos directamente (softmax) maximizando el Sharpe, sin estimar µ.

El resultado central del trabajo: **ninguna estrategia supera a 1/N con significancia estadística**, y
los modelos de ML tienen R² fuera de muestra negativo, por lo que el valor proviene del optimizador y la
estructura de covarianza, no de la predicción de retornos.

## Requisitos de software

- **Python 3.12** (probado también en 3.13).
- Se recomienda un entorno aislado (Anaconda o `venv`).
- Bibliotecas principales: `numpy`, `pandas`, `matplotlib`, `yfinance`, `cvxpy`, `scikit-learn`,
  `torch` (build CPU) y `jupyter` para ejecutar el notebook.

### Instalación

Con conda (recomendado en Windows/Anaconda):

```bash
conda create -n portafolio python=3.12
conda activate portafolio
pip install numpy pandas matplotlib yfinance cvxpy scikit-learn jupyter
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> El build **CPU** de PyTorch evita los problemas de DLL que aparecen con la versión CUDA en Windows;
> el proyecto no requiere GPU.

Alternativamente, con `requirements.txt`:

```
numpy
pandas
matplotlib
yfinance
cvxpy
scikit-learn
torch
jupyter
```

```bash
pip install -r requirements.txt
```

## Estructura

```
Proyecto_ML/
├── proyecto_ML.ipynb                 # notebook principal (todo el análisis)
├── datos/
│   └── precios_raw_2026-07-01.csv    # precios cacheados (reproducibilidad)
└── Informe_Proyecto.pdf              # informe escrito
```

## Ejecución

```bash
conda activate portafolio
jupyter notebook proyecto_ML.ipynb
```

Ejecutar las celdas **en orden, de arriba hacia abajo** (`Kernel → Restart & Run All`). El notebook carga
los datos, construye features, entrena los modelos, corre el backtest walk-forward, genera las
visualizaciones y el análisis de significancia por bootstrap.

## Reproducibilidad

`yfinance` descarga datos en vivo, y esas descargas cambian entre corridas (distinto universo de acciones,
métricas que cruzan umbrales de significancia). Para evitarlo, el proyecto **congela los precios en CSV**:

- La función de carga usa automáticamente el CSV más reciente dentro de `datos/`. Mientras
  `precios_raw_2026-07-01.csv` esté presente, los resultados coinciden con los del informe.
- Para forzar una descarga nueva desde Yahoo Finance, **eliminar (o mover) el CSV de `datos/`**. En ese
  caso los números pueden diferir de los reportados.
