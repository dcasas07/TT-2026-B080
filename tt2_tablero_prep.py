"""tt2_tablero_prep — preparación de datos del Tablero Ejecutivo TT2 (2026-B080).

Qué hace (una sola vez, antes de abrir el tablero):
1. Localiza los artefactos que exportó cada notebook TT2 (carpeta o .zip con bundle.joblib + módulo *_tt2.py).
2. Reconstruye EXACTAMENTE el split de producción de cada notebook a partir de los datos crudos
   (misma lectura, mismas exclusiones, misma partición temporal, misma semilla).
3. Puntúa producción con el modelo recomendado del bundle (modelo + calibrador del notebook) y calcula
   las contribuciones por variable (SHAP nativo de XGBoost / coef × valor en LR).
4. Verifica que los resultados coinciden con los del notebook (tabla `bandas_riesgo`).
5. Escribe en <salida>/<dataset>/:
     filas_produccion.parquet  → una fila por vuelo / paciente / empresa de producción
     contexto.json             → parámetros del modelo, grupos de variables, verificación
     artefactos/               → copia de las tablas del notebook (métricas, políticas, PSI, …)

El tablero (app.py) solo LEE estas tablas: no entrena ni recalcula modelos.
Nada aquí inventa datos: todo proviene de los artefactos y de los datos crudos originales.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import re
import shutil
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------- catálogo de datasets
DATASETS = {
    "airline": {"modulo": "airline_tt2", "titulo": "Aerolínea · Retrasos de vuelos",
                "archivos": ["Airline.arff (o el CSV de OpenML 1169)"]},
    "diabetes": {"modulo": "diabetes_tt2", "titulo": "Hospital · Reingresos de pacientes con diabetes",
                 "archivos": ["diabetic_data.csv", "IDS_mapping.csv"]},
    "polish": {"modulo": "polish_tt2", "titulo": "Finanzas · Riesgo de quiebra empresarial",
               "archivos": [f"{k}year.arff" for k in range(1, 6)]},
}
SCHEMA_VERSION = "1.0"


class PrepError(RuntimeError):
    """Error con mensaje en lenguaje claro para el usuario."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- 1. artefactos
def _descomprimir(zip_path: Path, destino: Path) -> Path | None:
    """Descomprime un .zip de artefactos (el que genera shutil.make_archive en §5.5 de cada notebook)."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            nombres = z.namelist()
            if not any(n.endswith("bundle.joblib") for n in nombres):
                return None
            out = destino / zip_path.stem
            if not (out / "bundle.joblib").exists() and not any(out.rglob("bundle.joblib")):
                out.mkdir(parents=True, exist_ok=True)
                z.extractall(out)
            return out
    except zipfile.BadZipFile:
        return None


def descubrir_artefactos(raices, carpeta_zip=None) -> dict:
    """Busca, en las carpetas indicadas, la corrida más reciente de cada dataset.

    Una carpeta de artefactos válida contiene bundle.joblib y el módulo <dataset>_tt2.py.
    Los .zip se descomprimen en `carpeta_zip`. Devuelve {dataset: Path}.
    """
    raices = [Path(r) for r in raices if r and Path(r).exists()]
    carpeta_zip = Path(carpeta_zip or "/tmp/tt2_artefactos_zip")
    for r in raices:
        for z in r.rglob("*.zip"):
            if "tt2_" in z.name.lower() or "artifact" in str(z).lower() or "artefact" in str(z).lower():
                _descomprimir(z, carpeta_zip)
    raices.append(carpeta_zip)
    encontrados = {}
    for r in raices:
        if not r.exists():
            continue
        for b in r.rglob("bundle.joblib"):
            d = b.parent
            for ds, info in DATASETS.items():
                if (d / f"{info['modulo']}.py").exists():
                    meta = d / "metadata.json"
                    run_id = json.loads(meta.read_text(encoding="utf-8")).get("run_id", d.name) if meta.exists() else d.name
                    encontrados.setdefault(ds, []).append((run_id, d))
    return {ds: sorted(v, key=lambda t: t[0])[-1][1] for ds, v in encontrados.items()}


def _importar_modulo(nombre: str, carpeta: Path):
    """Importa el módulo *_tt2.py que viene junto al bundle (joblib serializó las clases por referencia)."""
    carpeta = str(Path(carpeta).resolve())
    if carpeta in sys.path:
        sys.path.remove(carpeta)
    sys.path.insert(0, carpeta)
    if nombre in sys.modules:
        del sys.modules[nombre]
    return importlib.import_module(nombre)


def _cargar_bundle(art_dir: Path, modulo: str):
    mod = _importar_modulo(modulo, art_dir)
    try:
        bundle = joblib.load(art_dir / "bundle.joblib")
    except Exception as err:  # versión de scikit-learn/xgboost distinta a la del notebook, archivo dañado, etc.
        raise PrepError(f"No se pudo cargar {art_dir / 'bundle.joblib'} ({type(err).__name__}: {err}). "
                        "Usa las mismas versiones de scikit-learn y xgboost con las que se ejecutó el notebook.") from err
    meta_path = art_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return mod, bundle, meta


def _leer_tabla(art_dir: Path, nombre: str):
    for ext in ("parquet", "csv"):
        p = art_dir / f"{nombre}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    return None


def _verificar_bandas(art_dir, bandas_calc: pd.Series) -> dict:
    """Compara el número de casos por banda con la tabla `bandas_riesgo` del notebook."""
    ref = _leer_tabla(art_dir, "bandas_riesgo")
    if ref is None:
        return {"coincide": None, "detalle": "El notebook no exportó bandas_riesgo; no se pudo verificar."}
    col = [c for c in ref.columns if c not in ("banda", "prob_media", "tasa_observada")][0]
    esperado = ref.set_index("banda")[col].astype(int).to_dict()
    obtenido = {k: int(bandas_calc.get(k, 0)) for k in esperado}
    return {"coincide": esperado == obtenido, "esperado_notebook": esperado, "obtenido_tablero": obtenido,
            "detalle": "Conteos por nivel de riesgo idénticos a los del notebook." if esperado == obtenido else
            "Los conteos difieren: los datos crudos o las versiones de librerías no son los mismos que en la corrida del notebook."}


def _copiar_artefactos(art_dir: Path, out_dir: Path):
    dest = out_dir / "artefactos"
    dest.mkdir(parents=True, exist_ok=True)
    for p in art_dir.iterdir():
        if p.suffix in (".parquet", ".csv") or p.name == "metadata.json":   # el modelo .json no se necesita
            shutil.copy2(p, dest / p.name)


def _contribuciones(scorer, X, prepared=None) -> pd.DataFrame:
    C = scorer.explain(X) if prepared is None else scorer.explain(X, prepared=prepared)
    return C.astype("float32").add_prefix("c__")


# ----------------------------------------------------------------------------- 2. datos crudos
def _buscar(carpeta, patrones):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        return []
    out = []
    for pat in patrones:
        out += sorted(carpeta.rglob(pat))
    return out


def _descargar_zip(url, carpeta: Path):
    carpeta.mkdir(parents=True, exist_ok=True)
    zpath = carpeta / "descarga.zip"
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(carpeta)
    for inner in carpeta.rglob("*.zip"):
        if inner != zpath:
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.parent)


# ============================================================================= AIRLINE
GRUPOS_AIRLINE = {
    "Historial de retrasos del vuelo y su ruta": ["h_fl", "p_fl", "n_fl", "h_route", "p_route", "n_route"],
    "Retrasos del día anterior (aerolínea y aeropuertos)": ["p_airline", "p_orig", "p_dest", "prev_day_rate"],
    "Desempeño habitual de aerolínea y aeropuertos": ["h_airline", "h_orig", "h_dest", "Airline", "AirportFrom", "AirportTo"],
    "Hora de salida": ["Time", "sin_t", "cos_t"],
    "Día de la semana": ["DayOfWeek"],
    "Duración programada del vuelo": ["Length"],
}


def _leer_airline(path, M):
    """Misma lectura que §1.5 del notebook Airline (ARFF con parser rápido o CSV de OpenML)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            attrs = [ln.split()[1].strip("'\"") for ln in lines if ln.strip().lower().startswith("@attribute")]
            start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "@data") + 1
            df = pd.read_csv(io.StringIO("\n".join(lines[start:])), header=None, names=attrs,
                             skipinitialspace=True, quotechar="'", na_values="?")
        except Exception:
            from scipy.io import arff
            data, _ = arff.loadarff(str(path))
            df = pd.DataFrame(data)
    faltan = [c for c in M.RAW_COLUMNS + ["Delay"] if c not in df.columns]
    if faltan:
        raise PrepError(f"El archivo {path.name} no tiene las columnas {faltan}.")
    for c in ["Airline", "AirportFrom", "AirportTo"]:
        df[c] = df[c].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v).astype(str).str.strip().str.strip("'")
    for c in ["Flight", "DayOfWeek", "Time", "Length", "Delay"]:
        df[c] = pd.to_numeric(df[c].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v), errors="coerce")
    df["Delay"] = df["Delay"].astype(int)
    return df[M.RAW_COLUMNS + ["Delay"]]


def _datos_airline(data_dir, descargar=True):
    found = _buscar(data_dir, ["Airline*.arff", "airline*.arff", "Airline*.csv", "airline*.csv"])
    if found:
        return found[0]
    if descargar:
        try:
            from sklearn.datasets import fetch_openml
            print("  Descargando Airline (OpenML 1169)…")
            df = fetch_openml(data_id=1169, as_frame=True, parser="auto").frame
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            path = Path(data_dir) / "Airline_openml.csv"
            df.to_csv(path, index=False)
            return path
        except Exception as err:
            print(f"  Descarga no disponible ({err})")
    raise PrepError(f"Falta el archivo de Airline en {data_dir}: {DATASETS['airline']['archivos']}.")


def preparar_airline(art_dir, data_dir, out_dir, descargar=True):
    art_dir, out_dir = Path(art_dir), Path(out_dir)
    M, bundle, meta = _cargar_bundle(art_dir, "airline_tt2")
    cfg = meta.get("config", {})
    d_tr = int(cfg.get("train_days", 18))
    d_ca = d_tr + int(cfg.get("cal_days", 3))
    m_hist = float(cfg.get("history_smoothing_m", 20))
    path = _datos_airline(data_dir, descargar)
    raw = _leer_airline(path, M)
    # §1.6 reconstrucción de días: cada cambio de DayOfWeek avanza un día
    dow = raw["DayOfWeek"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(dow) != 0) + 1]
    data = raw.copy()
    data["day"] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(raw)]))
    prior = float(data.loc[data["day"] < d_tr, "Delay"].mean())
    lags = M.causal_lags(data, prior=prior, m=m_hist)                        # §2.3 historial causal (sin fuga)
    prod = data[data["day"] >= d_ca]
    best = bundle["best"]
    scorer = bundle["scorers"][best]
    X = scorer.featurizer.transform(prod, lags.loc[prod.index])
    p = scorer.predict_proba(X)
    C = _contribuciones(scorer, X)
    franja = pd.cut(prod["Time"] // 60, bins=[-1, 5, 11, 17, 24], labels=["Madrugada (00–05 h)", "Mañana (06–11 h)",
                                                                           "Tarde (12–17 h)", "Noche (18–23 h)"]).astype(str)
    filas = pd.DataFrame({
        "dia": prod["day"].astype(int) + 1,                                  # días 1–31 (el notebook usa 0–30)
        "dia_semana": prod["DayOfWeek"].astype(int),
        "aerolinea": prod["Airline"].astype(str), "vuelo": prod["Airline"].astype(str) + " " + prod["Flight"].astype(int).astype(str),
        "origen": prod["AirportFrom"].astype(str), "destino": prod["AirportTo"].astype(str),
        "ruta": prod["AirportFrom"].astype(str) + " → " + prod["AirportTo"].astype(str),
        "hora": (prod["Time"] // 60).astype(int), "minuto_salida": prod["Time"].astype(int), "franja": franja,
        "duracion_min": prod["Length"].astype(float),
        "retraso_ayer_origen": X["p_orig"].astype("float32"), "retraso_ayer_aerolinea": X["p_airline"].astype("float32"),
        "historial_vuelo": X["h_fl"].astype("float32"), "operaciones_previas_vuelo": np.round(np.expm1(X["n_fl"])).astype(int),
        "y": prod["Delay"].astype(int), "p": p.astype("float32"),
        "banda": scorer.risk_band(p), "alerta": (p >= scorer.threshold).astype(int),
    }, index=prod.index)
    filas = pd.concat([filas, C.set_index(filas.index)], axis=1).reset_index(names="fila_original")
    ver = _verificar_bandas(art_dir, filas["banda"].value_counts())
    sha = _sha256(path)
    ctx = {
        "dataset": "airline", "titulo": DATASETS["airline"]["titulo"], "run_id": meta.get("run_id"),
        "modelo": scorer.name, "tipo_modelo": scorer.kind, "umbral_alerta": float(scorer.threshold),
        "cortes_riesgo": list(scorer.risk_cuts), "regla_umbral": "Youden (equilibrio entre retrasos detectados y falsas alertas)",
        "unidad": "vuelo", "unidad_plural": "vuelos", "evento": "retraso", "dias_totales": int(data["day"].nunique()),
        "dias_train": [1, d_tr], "dias_calibracion": [d_tr + 1, d_ca], "dias_produccion": [d_ca + 1, int(data["day"].max()) + 1],
        "tasa_train": prior, "tasa_calibracion": float(data.loc[(data["day"] >= d_tr) & (data["day"] < d_ca), "Delay"].mean()),
        "grupos": {g: [v for v in vs if f"c__{v}" in filas.columns] for g, vs in GRUPOS_AIRLINE.items()},
        "etiquetas": bundle.get("feature_labels", {}), "verificacion": ver,
        "datos": {"archivo": Path(path).name, "sha256": sha, "sha256_notebook": meta.get("sha256_datos"),
                  "coincide_hash": (sha == meta.get("sha256_datos")) if meta.get("sha256_datos") else None},
        "schema_version": SCHEMA_VERSION,
    }
    _escribir(out_dir / "airline", filas, ctx, art_dir)
    return ctx


# ============================================================================= DIABETES
GRUPOS_DIABETES = {
    "Uso previo de servicios hospitalarios": ["number_inpatient", "number_emergency", "number_outpatient", "prior_visits"],
    "Destino al alta": ["discharge_disposition"],
    "Diagnósticos (cantidad y tipo)": ["number_diagnoses", "diag_1_group", "diag_2_group", "diag_3_group"],
    "Estancia y procedimientos": ["time_in_hospital", "num_lab_procedures", "num_procedures"],
    "Medicación y control de la diabetes": ["num_medications", "n_meds_active", "n_meds_changed", "insulin", "metformin",
                                            "change", "diabetesMed", "A1Cresult", "max_glu_serum"],
    "Ingreso (tipo, origen y especialidad)": ["admission_type", "admission_source", "medical_specialty"],
    "Edad, sexo y grupo étnico": ["age_mid", "race", "gender"],
}

TRAD_ALTA = {
    "Discharged to home": "Alta a domicilio",
    "Discharged/transferred to another short term hospital": "Traslado a otro hospital (corta estancia)",
    "Discharged/transferred to SNF": "Traslado a centro de enfermería especializada (SNF)",
    "Discharged/transferred to ICF": "Traslado a centro de cuidados intermedios (ICF)",
    "Discharged/transferred to another type of inpatient care institution": "Traslado a otra institución de hospitalización",
    "Discharged/transferred to home with home health service": "Alta a domicilio con atención domiciliaria",
    "Left AMA": "Alta voluntaria contra indicación médica",
    "Discharged/transferred to home under care of Home IV provider": "Alta a domicilio con terapia intravenosa",
    "Admitted as an inpatient to this hospital": "Ingresado como hospitalizado en este hospital",
    "Still patient or expected to return for outpatient services": "Continúa en atención o regresa a consulta externa",
    "Discharged/transferred within this institution to Medicare approved swing bed": "Traslado interno a cama de transición (swing bed)",
    "Discharged/transferred/referred another institution for outpatient services": "Referido a otra institución (consulta externa)",
    "Discharged/transferred/referred to this institution for outpatient services": "Referido a consulta externa de esta institución",
    "Discharged/transferred to another rehab fac including rehab units of a hospital .": "Traslado a centro de rehabilitación",
    "Discharged/transferred to a long term care hospital.": "Traslado a hospital de larga estancia",
    "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.": "Traslado a residencia de enfermería (Medicaid)",
    "Discharged/transferred to a federal health care facility.": "Traslado a institución federal de salud",
    "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital": "Traslado a hospital psiquiátrico",
    "Discharged/transferred to a Critical Access Hospital (CAH).": "Traslado a hospital de acceso crítico",
    "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere": "Traslado a otra institución de salud",
    "Neonate discharged to another hospital for neonatal aftercare": "Neonato trasladado a otro hospital",
    "Desconocido": "No registrado",
}
TRAD_ORIGEN = {
    "Physician Referral": "Referencia médica", "Clinic Referral": "Referencia de clínica", "HMO Referral": "Referencia de aseguradora (HMO)",
    "Transfer from a hospital": "Traslado desde otro hospital", "Transfer from a Skilled Nursing Facility (SNF)": "Traslado desde centro de enfermería",
    "Transfer from another health care facility": "Traslado desde otra institución de salud", "Emergency Room": "Sala de urgencias",
    "Court/Law Enforcement": "Orden judicial / autoridad", "Transfer from critial access hospital": "Traslado desde hospital de acceso crítico",
    "Transfer from hospital inpt/same fac reslt in a sep claim": "Traslado interno (reclamación separada)",
    "Transfer from Ambulatory Surgery Center": "Traslado desde cirugía ambulatoria", "Transfer From Another Home Health Agency": "Traslado desde atención domiciliaria",
    "Readmission to Same Home Health Agency": "Reingreso a la misma agencia domiciliaria", "Transfer from Hospice": "Traslado desde cuidados paliativos",
    "Normal Delivery": "Parto normal", "Premature Delivery": "Parto prematuro", "Sick Baby": "Recién nacido enfermo",
    "Extramural Birth": "Nacimiento extrahospitalario", "Born inside this hospital": "Nacido en este hospital",
    "Born outside this hospital": "Nacido fuera de este hospital", "Desconocido": "No registrado",
}
TRAD_TIPO = {"Emergency": "Emergencia", "Urgent": "Urgente", "Elective": "Programada", "Newborn": "Recién nacido",
             "Trauma Center": "Centro de trauma", "Desconocido": "No registrado"}
TRAD_ESPECIALIDAD = {
    "InternalMedicine": "Medicina interna", "Family/GeneralPractice": "Medicina familiar / general", "Cardiology": "Cardiología",
    "Surgery-General": "Cirugía general", "Orthopedics": "Ortopedia", "Orthopedics-Reconstructive": "Ortopedia reconstructiva",
    "Emergency/Trauma": "Urgencias / trauma", "Nephrology": "Nefrología", "Radiologist": "Radiología", "Psychiatry": "Psiquiatría",
    "Pulmonology": "Neumología", "ObstetricsandGynecology": "Ginecología y obstetricia", "Gastroenterology": "Gastroenterología",
    "Surgery-Cardiovascular/Thoracic": "Cirugía cardiotorácica", "Urology": "Urología", "Surgery-Vascular": "Cirugía vascular",
    "Oncology": "Oncología", "Hematology/Oncology": "Hematología / oncología", "Neurology": "Neurología",
    "Endocrinology": "Endocrinología", "PhysicalMedicineandRehabilitation": "Medicina física y rehabilitación",
    "Surgery-Neuro": "Neurocirugía", "Pediatrics": "Pediatría", "Otra": "Otra especialidad", "No registrada": "No registrada",
}
TRAD_RAZA = {"Caucasian": "Blanca", "AfricanAmerican": "Afroamericana", "Hispanic": "Hispana", "Asian": "Asiática",
             "Other": "Otra", "Desconocido": "No registrado"}
TRAD_A1C = {">8": "HbA1c > 8 %", ">7": "HbA1c 7–8 %", "Norm": "HbA1c normal", "No medido": "HbA1c no medida"}
TRAD_INSULINA = {"No": "Sin insulina", "Steady": "Insulina sin cambio", "Up": "Insulina: dosis aumentada", "Down": "Insulina: dosis reducida"}


def _datos_diabetes(data_dir, descargar=True):
    def find():
        return {p.name: p for p in _buscar(data_dir, ["*.csv"]) if p.name in ("diabetic_data.csv", "IDS_mapping.csv")}
    found = find()
    if len(found) < 2 and descargar:
        try:
            print("  Descargando Diabetes (UCI 296)…")
            _descargar_zip("https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip", Path(data_dir))
            found = find()
        except Exception as err:
            print(f"  Descarga no disponible ({err})")
    faltan = [f for f in ("diabetic_data.csv", "IDS_mapping.csv") if f not in found]
    if faltan:
        raise PrepError(f"Faltan {faltan} en {data_dir}.")
    return found


def preparar_diabetes(art_dir, data_dir, out_dir, descargar=True):
    art_dir, out_dir = Path(art_dir), Path(out_dir)
    M, bundle, meta = _cargar_bundle(art_dir, "diabetes_tt2")
    cfg = meta.get("config", {})
    files = _datos_diabetes(data_dir, descargar)
    raw = pd.read_csv(files["diabetic_data.csv"], keep_default_na=False, na_values=["?"], low_memory=False)   # §1.5
    ids = M.parse_ids_mapping(files["IDS_mapping.csv"])                                                       # §1.6
    excl = M.excluded_discharge_codes(ids)
    # §1.7 cohorte: sin defunción/hospicio, primer encuentro por paciente, sexo registrado, sin cola censurada
    d = raw.sort_values("encounter_id")
    d = d[~d["discharge_disposition_id"].isin(excl)]
    d = d.drop_duplicates("patient_nbr", keep="first")
    d = d[d["gender"].isin(["Female", "Male"])].reset_index(drop=True)
    d["y"] = (d["readmitted"] == "<30").astype(int)
    d["period"] = pd.qcut(np.arange(len(d)), int(cfg.get("n_periods", 20)), labels=False)
    rates = d.groupby("period")["y"].mean().to_numpy()
    censored = []
    for k in range(len(rates) - 1, 5, -1):
        if rates[k] < float(cfg.get("censor_ratio", 0.70)) * np.median(rates[k - 5:k]):
            censored.append(k)
        else:
            break
    cohort = d[~d["period"].isin(censored)].reset_index(drop=True)
    n = len(cohort)
    i_tr = int(n * float(cfg.get("train_frac", 0.60)))
    i_ca = int(n * (float(cfg.get("train_frac", 0.60)) + float(cfg.get("cal_frac", 0.10))))
    prod = cohort.iloc[i_ca:]
    best = bundle["best"]
    scorer = bundle["scorers"][best]
    fz = scorer.featurizer
    X = fz.transform(prod)
    B = fz.base(prod)
    p = scorer.predict_proba(X)
    C = _contribuciones(scorer, X)
    pos = np.arange(len(prod))
    block = int(cfg.get("block_size", 1000))
    edad = prod["age"].astype(str).str.extract(r"\[(\d+)-(\d+)\)").astype(float)
    filas = pd.DataFrame({
        "encuentro": prod["encounter_id"].astype("int64"), "paciente": prod["patient_nbr"].astype("int64"),
        "bloque": pos // block + 1, "periodo": prod["period"].astype(int),
        "edad": (edad[0].astype(int).astype(str) + "–" + (edad[1] - 1).astype(int).astype(str) + " años").to_numpy(),
        "edad_media": B["age_mid"].to_numpy(), "sexo": B["gender"].map({"Female": "Mujer", "Male": "Hombre"}).fillna("No registrado").to_numpy(),
        "grupo_etnico": B["race"].map(lambda s: TRAD_RAZA.get(s, s)).to_numpy(),
        "destino_alta": B["discharge_disposition"].map(lambda s: TRAD_ALTA.get(s, s)).to_numpy(),
        "origen_admision": B["admission_source"].map(lambda s: TRAD_ORIGEN.get(s, s)).to_numpy(),
        "tipo_admision": B["admission_type"].map(lambda s: TRAD_TIPO.get(s, s)).to_numpy(),
        "diagnostico_principal": B["diag_1_group"].to_numpy(),
        "especialidad": pd.Series(B["medical_specialty"]).map(lambda s: TRAD_ESPECIALIDAD.get(s, s)).to_numpy(),
        "hospitalizaciones_previas": B["number_inpatient"].astype(int).to_numpy(),
        "urgencias_previas": B["number_emergency"].astype(int).to_numpy(),
        "visitas_previas": B["prior_visits"].astype(int).to_numpy(),
        "dias_estancia": B["time_in_hospital"].astype(int).to_numpy(),
        "n_diagnosticos": B["number_diagnoses"].astype(int).to_numpy(),
        "n_medicamentos": B["num_medications"].astype(int).to_numpy(),
        "hba1c": B["A1Cresult"].map(lambda s: TRAD_A1C.get(s, s)).to_numpy(),
        "insulina": B["insulin"].map(lambda s: TRAD_INSULINA.get(s, s)).to_numpy(),
        "aseguradora_registrada": B["payer_code"].notna().to_numpy(),
        "y": prod["y"].astype(int).to_numpy(), "p": p.astype("float32"),
        "banda": scorer.risk_band(p), "alerta": (p >= scorer.threshold).astype(int),
    }, index=prod.index)
    filas = pd.concat([filas, C.set_index(filas.index)], axis=1).reset_index(drop=True)
    ver = _verificar_bandas(art_dir, filas["banda"].value_counts())
    shas = {k: _sha256(v) for k, v in files.items()}
    y_tr, y_ca = cohort["y"].iloc[:i_tr], cohort["y"].iloc[i_tr:i_ca]
    ctx = {
        "dataset": "diabetes", "titulo": DATASETS["diabetes"]["titulo"], "run_id": meta.get("run_id"),
        "modelo": scorer.name, "tipo_modelo": scorer.kind, "umbral_alerta": float(scorer.threshold),
        "cortes_riesgo": list(scorer.risk_cuts), "regla_umbral": "F2 (prioriza no omitir pacientes que reingresarán)",
        "unidad": "paciente", "unidad_plural": "pacientes", "evento": "reingreso en menos de 30 días",
        "tasa_train": float(y_tr.mean()), "tasa_calibracion": float(y_ca.mean()),
        "n_cohorte": int(n), "periodos_censurados": [int(c) for c in censored], "tamano_bloque": block,
        "grupos": {g: [v for v in vs if f"c__{v}" in filas.columns] for g, vs in GRUPOS_DIABETES.items()},
        "etiquetas": bundle.get("feature_labels", {}), "verificacion": ver,
        "datos": {"archivos": list(files), "sha256": shas, "sha256_notebook": meta.get("sha256_datos"),
                  "coincide_hash": (shas == meta.get("sha256_datos")) if meta.get("sha256_datos") else None},
        "schema_version": SCHEMA_VERSION,
    }
    _escribir(out_dir / "diabetes", filas, ctx, art_dir)
    return ctx


# ============================================================================= POLISH
CATEGORIAS_POLISH = {
    "Rentabilidad": ["Attr1", "Attr6", "Attr7", "Attr11", "Attr13", "Attr14", "Attr18", "Attr19", "Attr22", "Attr23", "Attr24",
                     "Attr31", "Attr35", "Attr39", "Attr42", "Attr45", "Attr48", "Attr49", "Attr56", "Attr58"],
    "Endeudamiento y solvencia": ["Attr2", "Attr8", "Attr10", "Attr15", "Attr16", "Attr17", "Attr25", "Attr26", "Attr27",
                                  "Attr30", "Attr38", "Attr41", "Attr51", "Attr53", "Attr54", "Attr59"],
    "Liquidez": ["Attr3", "Attr4", "Attr5", "Attr12", "Attr28", "Attr33", "Attr34", "Attr37", "Attr40", "Attr46", "Attr50",
                 "Attr55", "Attr57"],
    "Eficiencia operativa y rotación": ["Attr9", "Attr20", "Attr32", "Attr36", "Attr43", "Attr44", "Attr47", "Attr52",
                                        "Attr60", "Attr61", "Attr62", "Attr63", "Attr64"],
    "Tamaño de la empresa": ["Attr29"],
    "Crecimiento de ventas": ["Attr21"],
    "Información financiera incompleta": ["n_missing"],
}


def _read_arff_polish(path):
    """Misma lectura que §1.5 del notebook Polish."""
    from scipy.io import arff
    try:
        data, _ = arff.loadarff(str(path))
        df = pd.DataFrame(data)
    except Exception:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        attrs = [ln.split()[1] for ln in lines if ln.strip().lower().startswith("@attribute")]
        start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "@data") + 1
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])), header=None, names=attrs, na_values="?")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    return df


def _datos_polish(data_dir, descargar=True):
    esperados = [f"{k}year.arff" for k in range(1, 6)]

    def find():
        return {p.name: p for p in _buscar(data_dir, ["*year.arff"]) if p.name in esperados}
    found = find()
    if len(found) < 5 and descargar:
        try:
            print("  Descargando Polish (UCI 365)…")
            _descargar_zip("https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip", Path(data_dir))
            found = find()
        except Exception as err:
            print(f"  Descarga no disponible ({err})")
    faltan = [f for f in esperados if f not in found]
    if faltan:
        raise PrepError(f"Faltan {faltan} en {data_dir}.")
    return {f: found[f] for f in esperados}


def preparar_polish(art_dir, data_dir, out_dir, descargar=True):
    art_dir, out_dir = Path(art_dir), Path(out_dir)
    M, bundle, meta = _cargar_bundle(art_dir, "polish_tt2")
    cfg = meta.get("config", {})
    seed = int(cfg.get("seed", 42))
    files = _datos_polish(data_dir, descargar)
    frames = []
    for name, path in sorted(files.items()):                                  # §1.5 unificación (cohorte = nombre de archivo)
        k = int(re.match(r"(\d)year", name).group(1))
        d = _read_arff_polish(path)
        d[M.RAW_FEATURES] = d[M.RAW_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        d["class"] = pd.to_numeric(d["class"]).astype(int)
        d.insert(0, "cohort", k)
        d.insert(1, "horizon_years", 6 - k)
        d.insert(2, "row_in_file", np.arange(len(d)))
        d.insert(3, "source_file", name)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    # §1.7 deduplicación + orden aleatorio con semilla DENTRO de cada cohorte
    dedup = raw.loc[~raw.duplicated(subset=M.RAW_FEATURES, keep="first")]
    rng = np.random.default_rng(seed)
    parts = [rng.permutation(dedup.index[dedup["cohort"] == k].to_numpy()) for k in sorted(dedup["cohort"].unique())]
    data = dedup.loc[np.concatenate(parts)].reset_index(drop=True)
    data["t_index"] = np.arange(len(data))
    n = len(data)
    i_tr = int(n * float(cfg.get("train_frac", 0.60)))
    i_ca = int(n * (float(cfg.get("train_frac", 0.60)) + float(cfg.get("cal_frac", 0.10))))
    prod = data[data["t_index"] >= i_ca]
    best = bundle["best"]
    scorer = bundle["scorers"][best]
    X = M.build_features(prod)
    p = scorer.predict_proba(X, prepared=True)
    C = _contribuciones(scorer, X, prepared=True)
    filas = pd.DataFrame({
        "empresa": prod["source_file"].str.replace(".arff", "", regex=False) + " · fila " + (prod["row_in_file"] + 1).astype(str),
        "archivo": prod["source_file"], "horizonte_anios": prod["horizon_years"].astype(int),
        "y": prod["class"].astype(int), "p": p.astype("float32"),
        "banda": scorer.risk_band(p), "alerta": (p >= scorer.threshold).astype(int),
        "n_faltantes": X["n_missing"].astype(int),
    }, index=prod.index)
    filas = pd.concat([filas, X[M.RAW_FEATURES].astype("float32"), C.set_index(filas.index)], axis=1).reset_index(drop=True)
    ver = _verificar_bandas(art_dir, filas["banda"].value_counts())
    shas = {k: _sha256(v) for k, v in files.items()}
    ctx = {
        "dataset": "polish", "titulo": DATASETS["polish"]["titulo"], "run_id": meta.get("run_id"),
        "modelo": scorer.name, "tipo_modelo": scorer.kind, "umbral_alerta": float(scorer.threshold),
        "cortes_riesgo": list(scorer.risk_cuts), "regla_umbral": "F2 (prioriza no omitir quiebras)",
        "unidad": "empresa", "unidad_plural": "empresas", "evento": "quiebra",
        "tasa_train": float(data["class"].iloc[:i_tr].mean()), "tasa_calibracion": float(data["class"].iloc[i_tr:i_ca].mean()),
        "duplicados_eliminados": int(len(raw) - len(dedup)), "variables_lr": bundle.get("lr_features"),
        "grupos": {g: [v for v in vs if f"c__{v}" in filas.columns] for g, vs in CATEGORIAS_POLISH.items()},
        "etiquetas": bundle.get("feature_labels", {}), "verificacion": ver,
        "datos": {"archivos": list(files), "sha256": shas, "sha256_notebook": meta.get("sha256_datos"),
                  "coincide_hash": (shas == meta.get("sha256_datos")) if meta.get("sha256_datos") else None},
        "schema_version": SCHEMA_VERSION,
    }
    _escribir(out_dir / "polish", filas, ctx, art_dir)
    return ctx


# ----------------------------------------------------------------------------- salida
def _escribir(out_dir: Path, filas: pd.DataFrame, ctx: dict, art_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    filas.to_parquet(out_dir / "filas_produccion.parquet", index=False)
    ctx["n_filas"] = int(len(filas))
    ctx["carpeta_artefactos_origen"] = str(art_dir)
    (out_dir / "contexto.json").write_text(json.dumps(ctx, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _copiar_artefactos(art_dir, out_dir)


PREPARADORES = {"airline": preparar_airline, "diabetes": preparar_diabetes, "polish": preparar_polish}


def preparar_todo(artefactos: dict, datos: dict, salida, descargar=True) -> pd.DataFrame:
    """Prepara los datasets disponibles. Un fallo en uno no detiene a los demás; se reporta en la tabla."""
    salida = Path(salida)
    filas = []
    for ds, fn in PREPARADORES.items():
        info = {"dataset": ds, "titulo": DATASETS[ds]["titulo"]}
        if ds not in artefactos:
            filas.append({**info, "estado": "❌ sin artefactos",
                          "detalle": f"No se encontró la carpeta/zip con bundle.joblib y {DATASETS[ds]['modulo']}.py"})
            continue
        try:
            print(f"▶ {DATASETS[ds]['titulo']}  ←  {artefactos[ds]}")
            ctx = fn(artefactos[ds], datos.get(ds, f"/content/{ds}_data"), salida, descargar=descargar)
            v = ctx["verificacion"]
            estado = "✅ listo" if v.get("coincide") else ("⚠️ listo, sin verificar" if v.get("coincide") is None else "⚠️ listo, con diferencias")
            filas.append({**info, "estado": estado, "detalle": f"{ctx['n_filas']:,} registros de producción · {v['detalle']}"})
        except PrepError as err:
            filas.append({**info, "estado": "❌ error", "detalle": str(err)})
        except Exception as err:  # error inesperado: se reporta sin detener los demás
            filas.append({**info, "estado": "❌ error", "detalle": f"{type(err).__name__}: {err}"})
    return pd.DataFrame(filas)
