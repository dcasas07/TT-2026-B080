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

Los datos crudos no se versionan (son públicos). Enlaces, descarga directa y SHA-256 en [`data/README.md`](data/README.md); se obtienen con `make data` o desde los propios notebooks.

| Dominio | Fuente | Ubicación local |
|---|---|---|
| Airline Delay | [OpenML 1169](https://www.openml.org/d/1169) | `data/airline/Airline_openml.csv` |
| Polish Companies Bankruptcy | [UCI 365](https://archive.ics.uci.edu/dataset/365) | `data/polish/1year.arff` … `5year.arff` |
| Diabetes 130-US Hospitals | [UCI 296](https://archive.ics.uci.edu/dataset/296) | `data/diabetes/diabetic_data.csv`, `IDS_mapping.csv` |

## Estructura

```
.
├── app.py                    # Dashboard Streamlit unificado (punto de entrada)
├── tt2_tablero_prep.py       # Prepara datos_tablero/ para la vista ejecutiva
├── tt2_data_pack.zip         # Tablas de resultados de la pestaña técnica
├── .streamlit/config.toml    # Tema y configuración del dashboard
├── notebooks/                # Tres notebooks ejecutados (Airline, Diabetes, Polish) y el del dashboard
├── src/                      # Módulos por dataset: airline_tt2, diabetes_tt2, polish_tt2
├── runs/                     # Artefactos de las corridas finales (runs/<run_id>/)
├── scripts/                  # download_data.py · preparar_tablero.py
├── docs/                     # Documentación final del TT
├── data/                     # Datos crudos (ignorados por git)
├── Makefile
└── requirements.txt
```

## Ejecución

Python 3.13 (versión de las corridas; el protocolo exige 3.10 o superior).

```bash
git clone https://github.com/dcasas07/TT-2026-B080.git
cd TT-2026-B080
make setup        # entorno virtual + dependencias
make data         # descarga los tres datasets en data/
make notebooks    # re-ejecuta los tres notebooks de experimentación
make tablero      # genera datos_tablero/ (vista ejecutiva) desde runs/ y data/
make dashboard    # equivale a: streamlit run app.py
```

La pestaña técnica del dashboard funciona sin pasos previos (usa `tt2_data_pack.zip`). La vista ejecutiva requiere `make data` y `make tablero`.

## Correspondencia con el protocolo

| Producto esperado | Ubicación |
|---|---|
| Repositorio reproducible | Este repositorio (`Makefile`, `requirements.txt`, semilla 42) |
| Pipeline ejecutable | `src/`, `notebooks/`, `runs/` |
| Dashboard interactivo | `app.py` → `streamlit run app.py` |
| Documento técnico | `docs/Documentacion_Version_Final.pdf` |

## Criterios de éxito del protocolo

ECE ≤ 0.05 · Brier ≤ 0.20 · latencia de detección ≤ 500 muestras · falsas alarmas < 15 %.

## Reproducibilidad

- Semilla global `42` en particionado, modelos y órdenes aleatorios.
- Transformaciones ajustadas únicamente con el conjunto de entrenamiento.
- Versiones de librerías fijadas en `requirements.txt` (tomadas del `metadata.json` de cada corrida).
- Cada corrida registra el SHA-256 de los datos de entrada en su `metadata.json`.
