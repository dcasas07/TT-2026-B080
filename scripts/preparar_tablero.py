"""Genera datos_tablero/ (vista ejecutiva del dashboard) a partir de runs/ y data/.

Requiere los datasets en data/<dataset>/ (ver scripts/download_data.py).
Uso:  python scripts/preparar_tablero.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tt2_tablero_prep as prep  # noqa: E402


def main() -> int:
    artefactos = prep.descubrir_artefactos([ROOT / "runs"], carpeta_zip=ROOT / ".cache" / "artefactos")
    datos = {ds: str(ROOT / "data" / ds) for ds in prep.DATASETS}
    estado = prep.preparar_todo(artefactos, datos, ROOT / "datos_tablero", descargar=True)
    print(estado.to_string(index=False))
    return 0 if estado["estado"].str.startswith("✅").all() else 1


if __name__ == "__main__":
    sys.exit(main())
