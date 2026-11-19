"""Descarga los tres datasets del TT 2026-B080 en data/raw/.

Uso:  python scripts/download_data.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

UCI_ZIPS = {
    "polish": "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip",
    "diabetes": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
}

EXPECTED = {
    "airline": ["airline_delay.csv"],
    "polish": [f"{i}year.arff" for i in range(1, 6)],
    "diabetes": ["diabetic_data.csv", "IDS_mapping.csv"],
}


def extract_zip(data: bytes, dest: Path) -> None:
    """Extrae un zip y, si contiene zips anidados, también los extrae."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)
    for nested in list(dest.rglob("*.zip")):
        extract_zip(nested.read_bytes(), dest)
        nested.unlink()


def download_uci(name: str, url: str) -> None:
    print(f"[{name}] descargando {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        extract_zip(resp.read(), RAW / name)


def download_airline() -> None:
    from sklearn.datasets import fetch_openml

    print("[airline] descargando OpenML 1169")
    frame = fetch_openml(data_id=1169, as_frame=True, parser="auto").frame
    dest = RAW / "airline"
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / "airline_delay.csv", index=False)


def check() -> bool:
    ok = True
    for name, files in EXPECTED.items():
        for fname in files:
            found = any(RAW.joinpath(name).rglob(fname))
            if not found:
                print(f"FALTA: data/raw/{name}/{fname}")
                ok = False
    return ok


def main() -> int:
    download_airline()
    for name, url in UCI_ZIPS.items():
        download_uci(name, url)
    if not check():
        return 1
    print("Datasets completos en", RAW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
