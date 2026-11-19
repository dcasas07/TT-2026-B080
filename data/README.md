# Datasets

Los datos crudos no se incluyen en el repositorio: son públicos y se descargan de su fuente original. Cada corrida registró el SHA-256 de los archivos usados, para verificar que los datos descargados son los mismos.

## Fuentes

| Dominio | Enlace | Descarga directa |
|---|---|---|
| Airline Delay | https://www.openml.org/d/1169 | `sklearn.datasets.fetch_openml(data_id=1169)` |
| Polish Companies Bankruptcy | https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data | https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip |
| Diabetes 130-US Hospitals (1999–2008) | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 | https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip |

## Ubicación esperada

```
data/airline/Airline_openml.csv
data/polish/1year.arff … 5year.arff
data/diabetes/diabetic_data.csv, IDS_mapping.csv
```

Se obtienen con `make data` (o `python scripts/download_data.py`), o automáticamente al ejecutar cada notebook.

## SHA-256 de los archivos usados en las corridas finales

| Archivo | SHA-256 |
|---|---|
| `airline/Airline_openml.csv` | `e46fc1537a02948846d9e43fc5d5226f3f87de65edf4e409aaf12ba101eefca6` |
| `polish/1year.arff` | `978ce71a4809e964dda43d46a368c27a1a36df1e0b6345f215473ec9c3b12add` |
| `polish/2year.arff` | `ec67460055735857b84a73d961193e2b7313147c1b93d900f08400e4071bdb7c` |
| `polish/3year.arff` | `662617d2f90bfd17223595e50a0ac7007f9ee7915cfb519dee0422b31dd13ebc` |
| `polish/4year.arff` | `ae11434f0823d6208142ac5a4e83b4f3d7bcf4e827b330a798fdd6590bd9f5cb` |
| `polish/5year.arff` | `cb3f6f250ac46bd8d18e9a222f489fe8ee3e396fcec18959f5a0ef8e8169b2fc` |
| `diabetes/diabetic_data.csv` | `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` |
| `diabetes/IDS_mapping.csv` | `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e` |

Verificación: `sha256sum data/diabetes/diabetic_data.csv`
