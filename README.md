# Detección y Mitigación de Calibration Drift en Modelos de Clasificación

Trabajo Terminal **2026-B080** · Escuela Superior de Cómputo (ESCOM), Instituto Politécnico Nacional

- **Alumnos:** Casas Aguilar José David · Robles Cervantes Pedro Iván
- **Directores:** Efrén Cruz Carlin · Angeles Alejandra Sánchez Manilla

## Descripción

Pipeline reproducible que monitorea por ventanas deslizantes el comportamiento de modelos de clasificación binaria bajo *data drift* y compara políticas de mantenimiento sobre la misma secuencia de ventanas, en tres dominios con estructura temporal.

- **Monitoreo:** PSI, KS, MMD (distribución de entrada); ECE, Brier, AUC-ROC, AUC-PR (desempeño); alertas en dos niveles.
- **Modelos base:** XGBoost y Regresión Logística.
- **Políticas:** Baseline · A (recalibración) · B (reentrenamiento) · C (híbrida con frecuencia mínima por calendario).
- **Metodología:** CRISP-DM. Partición temporal 60/10/30 sin mezcla. Semilla fija `42`.

## Datasets

| Dominio | Fuente | Archivos esperados |
|---|---|---|
| Airline Delay | OpenML, ID 1169 | `data/raw/airline/airline_delay.csv` |
| Polish Companies Bankruptcy | UCI, ID 365 | `data/raw/polish/1year.arff` … `5year.arff` |
| Diabetes 130-US Hospitals | UCI, ID 296 | `data/raw/diabetes/diabetic_data.csv`, `IDS_mapping.csv` |

Los datos no se versionan. Se descargan con `make data` (o `python scripts/download_data.py`).

## Estructura

```
.
├── app.py                  # Dashboard Streamlit (punto de entrada)
├── notebooks/              # Tres notebooks de experimentación (Polish, Airline, Diabetes)
├── src/                    # Módulos reutilizables por dataset: polish_tt2, airline_tt2, diabetes_tt2
├── scripts/
│   └── download_data.py    # Descarga de datasets
├── runs/                   # Artefactos por corrida (runs/<run_id>/: parquet, metadata.json, modelos)
├── docs/                   # Documentación final del TT
├── data/                   # Datos crudos y procesados (ignorados por git)
├── Makefile                # Orquestación de punta a punta
└── requirements.txt
```

## Ejecución

Requiere Python 3.10 o superior.

```bash
git clone <URL-DEL-REPOSITORIO>
cd tt-2026-b080
make setup        # entorno virtual + dependencias
make data         # descarga los tres datasets
make notebooks    # ejecuta los notebooks de punta a punta
make dashboard    # equivale a: streamlit run app.py
```

## Correspondencia con el protocolo

| Producto esperado | Ubicación |
|---|---|
| Repositorio reproducible | Este repositorio (`Makefile`, `requirements.txt`, semilla 42) |
| Pipeline ejecutable | `src/` y `notebooks/` |
| Dashboard interactivo | `app.py` → `streamlit run app.py` |
| Documento técnico | `docs/` |

## Criterios de éxito del protocolo

ECE ≤ 0.05 · Brier ≤ 0.20 · latencia de detección ≤ 500 muestras · falsas alarmas < 15 %.

## Reproducibilidad

- Semilla global `42` en particionado, modelos y órdenes aleatorios.
- Transformaciones ajustadas únicamente con el conjunto de entrenamiento.
- Dependencias listadas en `requirements.txt`.
- Cada resultado del reporte técnico es recomputable desde su notebook.
