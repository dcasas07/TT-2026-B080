"""diabetes_tt2 — componentes reutilizables del TT2 (2026-B080) para Diabetes 130-US Hospitals.

Lo importan el notebook y el futuro dashboard: lectura correcta de IDS_mapping, agrupación de
diagnósticos ICD-9, construcción de variables (DiabetesFeaturizer), pipeline lineal, calibradores,
PSI (numérico y categórico), métricas y el objeto ReadmissionScorer. Al vivir en un módulo, joblib
serializa estas clases por referencia a `diabetes_tt2` y el dashboard las carga con `import diabetes_tt2`.
"""
from __future__ import annotations

import csv
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
        "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone"]
ID_COLUMNS = {"admission_type_id": "admission_type", "discharge_disposition_id": "discharge_disposition",
              "admission_source_id": "admission_source"}
UNKNOWN_DESCRIPTIONS = {"NULL", "Not Mapped", "Not Available", "Unknown/Invalid"}
NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
           "number_emergency", "number_inpatient", "number_diagnoses"]
ENGINEERED = ["age_mid", "prior_visits", "n_meds_active", "n_meds_changed"]
NUM_FEATURES = NUMERIC + ENGINEERED
LOG_FEATURES = ["num_procedures", "number_outpatient", "number_emergency", "number_inpatient", "prior_visits"]
CAT_FEATURES = ["race", "gender", "admission_type", "discharge_disposition", "admission_source", "medical_specialty",
                "diag_1_group", "diag_2_group", "diag_3_group", "A1Cresult", "max_glu_serum", "insulin", "metformin",
                "change", "diabetesMed"]
MODEL_FEATURES = NUM_FEATURES + CAT_FEATURES
MONITOR_FEATURES = ["age_mid", "time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses",
                    "number_inpatient", "admission_type", "discharge_disposition", "admission_source",
                    "medical_specialty", "A1Cresult", "insulin", "payer_code"]
RAW_INPUT = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id",
             "time_in_hospital", "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications",
             "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
             "number_diagnoses", "max_glu_serum", "A1Cresult", *MEDS, "change", "diabetesMed"]
RISK_LABELS = ("Bajo", "Medio", "Alto")
EPS = 1e-6

FEATURE_LABELS = {
    "time_in_hospital": "Días de hospitalización", "num_lab_procedures": "N.º de pruebas de laboratorio",
    "num_procedures": "N.º de procedimientos (no laboratorio)", "num_medications": "N.º de medicamentos distintos",
    "number_outpatient": "Consultas externas en el año previo", "number_emergency": "Visitas a urgencias en el año previo",
    "number_inpatient": "Hospitalizaciones en el año previo", "number_diagnoses": "N.º de diagnósticos registrados",
    "age_mid": "Edad (punto medio del rango)", "prior_visits": "Total de visitas en el año previo",
    "n_meds_active": "N.º de antidiabéticos recetados", "n_meds_changed": "N.º de antidiabéticos con dosis ajustada",
    "race": "Grupo étnico registrado", "gender": "Sexo", "admission_type": "Tipo de admisión",
    "discharge_disposition": "Destino al alta", "admission_source": "Origen de la admisión",
    "medical_specialty": "Especialidad del médico tratante", "diag_1_group": "Diagnóstico principal (grupo ICD-9)",
    "diag_2_group": "Diagnóstico secundario (grupo ICD-9)", "diag_3_group": "Diagnóstico adicional (grupo ICD-9)",
    "A1Cresult": "Resultado de HbA1c", "max_glu_serum": "Glucosa sérica máxima", "insulin": "Insulina (cambio de dosis)",
    "metformin": "Metformina (cambio de dosis)", "change": "Hubo cambio de medicación", "diabetesMed": "Recibe antidiabéticos",
    "payer_code": "Aseguradora (solo monitoreo de captura)",
}


# ----------------------------------------------------------------------------- IDS_mapping
def parse_ids_mapping(path) -> pd.DataFrame:
    """Lee IDS_mapping.csv (3 tablas separadas por filas vacías, CRLF, descripciones con comas) en formato ordenado."""
    rows, var = [], None
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for rec in csv.reader(f):
            rec = [c.strip() for c in rec]
            if not any(rec):
                continue
            if rec[0].endswith("_id") and len(rec) > 1 and rec[1].lower() == "description":
                var = rec[0]
                continue
            if var and rec[0].isdigit():
                desc = ",".join(rec[1:]).strip()
                rows.append({"variable": var, "id": int(rec[0]), "descripcion": desc,
                             "es_desconocido": desc in UNKNOWN_DESCRIPTIONS})
    out = pd.DataFrame(rows)
    out["etiqueta"] = np.where(out["es_desconocido"], "Desconocido", out["descripcion"])
    return out


def excluded_discharge_codes(ids: pd.DataFrame):
    """Altas que impiden una readmisión: defunción u hospicio (derivado del propio IDS_mapping)."""
    d = ids[ids["variable"] == "discharge_disposition_id"]
    return sorted(d.loc[d["descripcion"].str.contains("Expired|Hospice", case=False), "id"].tolist())


# ----------------------------------------------------------------------------- diagnósticos ICD-9
def _icd9_group(code) -> str:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Desconocido"
    s = str(code).strip()
    if s in ("", "?", "nan"):
        return "Desconocido"
    if s[0] in "VE":
        return "Otros"
    try:
        v = float(s)
    except ValueError:
        return "Otros"
    i = int(v)
    if i == 250:
        return "Diabetes"
    if 390 <= i <= 459 or i == 785:
        return "Circulatorio"
    if 460 <= i <= 519 or i == 786:
        return "Respiratorio"
    if 520 <= i <= 579 or i == 787:
        return "Digestivo"
    if 800 <= i <= 999:
        return "Lesiones"
    if 710 <= i <= 739:
        return "Musculoesquelético"
    if 580 <= i <= 629 or i == 788:
        return "Genitourinario"
    if 140 <= i <= 239:
        return "Neoplasias"
    return "Otros"


def icd9_group(s: pd.Series) -> pd.Series:
    """Agrupa códigos ICD-9 en 9 categorías clínicas (esquema de Strack et al., 2014)."""
    u = pd.unique(s.astype(object))
    return s.astype(object).map({v: _icd9_group(v) for v in u})


# ----------------------------------------------------------------------------- variables
class DiabetesFeaturizer:
    """Esquema estático: IDs → descripciones del IDS_mapping, ICD-9 → grupos, medicamentos agregados,
    edad ordinal, especialidades frecuentes. Vocabularios congelados en train (no vistos → NaN en XGBoost)."""

    def __init__(self, ids_table: pd.DataFrame, top_specialties=15):
        self.ids_table, self.top_specialties = ids_table, top_specialties
        self.maps_ = {v: dict(zip(g["id"], g["etiqueta"])) for v, g in ids_table.groupby("variable")}

    @staticmethod
    def _col(df, name, default):
        return df[name] if name in df.columns else pd.Series(default, index=df.index, dtype=object)

    def base(self, df):
        """Variables con categorías como texto (útil para EDA y PSI)."""
        c = lambda n, d=np.nan: self._col(df, n, d)
        X = pd.DataFrame(index=df.index)
        for n in NUMERIC:
            X[n] = pd.to_numeric(c(n), errors="coerce").astype(float)
        age = c("age").astype(str).str.extract(r"\[(\d+)-(\d+)\)").astype(float)
        X["age_mid"] = age.mean(axis=1)
        X["prior_visits"] = X[["number_outpatient", "number_emergency", "number_inpatient"]].sum(axis=1, min_count=1)
        meds = pd.DataFrame({m: c(m, "No").fillna("No").astype(str) for m in MEDS}, index=df.index)
        X["n_meds_active"] = (meds != "No").sum(axis=1).astype(float)
        X["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)
        X["race"] = c("race").fillna("Desconocido").astype(str).replace({"?": "Desconocido"})
        g = c("gender").astype(str)
        X["gender"] = g.where(g.isin(["Female", "Male"]), "Desconocido")
        for raw, out in ID_COLUMNS.items():
            X[out] = pd.to_numeric(c(raw), errors="coerce").map(self.maps_.get(raw, {})).fillna("Desconocido")
        spec = c("medical_specialty")
        X["medical_specialty"] = np.where(spec.isna(), "No registrada",
                                          np.where(spec.isin(getattr(self, "specialties_", [])), spec, "Otra"))
        for d in ["diag_1", "diag_2", "diag_3"]:
            X[f"{d}_group"] = icd9_group(c(d))
        for n in ["A1Cresult", "max_glu_serum"]:
            v = c(n).astype(object)
            X[n] = v.where(v.notna() & ~v.astype(str).isin(["None", "nan", ""]), "No medido").astype(str)
        for n in ["insulin", "metformin"]:
            X[n] = c(n, "No").fillna("No").astype(str)
        X["change"] = c("change", "No").fillna("No").astype(str)
        X["diabetesMed"] = c("diabetesMed", "No").fillna("No").astype(str)
        pc = c("payer_code")
        X["payer_code"] = pc.where(pc.notna() & (pc.astype(str) != "?"), np.nan)
        return X

    def fit(self, df):
        spec = df["medical_specialty"].dropna().value_counts()
        self.specialties_ = list(spec.index[: self.top_specialties])
        b = self.base(df)
        self.categories_ = {n: sorted(b[n].astype(str).unique().tolist()) for n in CAT_FEATURES}
        return self

    def transform(self, df):
        b = self.base(df)
        for n in CAT_FEATURES:
            v = b[n].astype(str)
            b[n] = pd.Categorical(v.where(v.isin(self.categories_[n])), categories=self.categories_[n])
        return b[MODEL_FEATURES]

    def monitor_frame(self, df):
        b = self.base(df)
        for n in MONITOR_FEATURES:
            if not pd.api.types.is_numeric_dtype(b[n]):
                b[n] = b[n].astype(object)
        return b[MONITOR_FEATURES]


def make_lr_pipeline(C=0.1, class_weight=None, min_frequency=100, categorical=None, numeric=None):
    """One-hot (categorías raras agrupadas) + log de conteos + numéricas escaladas → LR."""
    categorical = list(CAT_FEATURES if categorical is None else categorical)
    numeric = list(NUM_FEATURES if numeric is None else numeric)
    logc = [c for c in LOG_FEATURES if c in numeric]
    other = [c for c in numeric if c not in logc]
    tr = [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency), categorical)]
    if logc:
        tr.append(("log", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                    ("sc", StandardScaler())]), logc))
    if other:
        tr.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), other))
    return Pipeline([("prep", ColumnTransformer(tr, verbose_feature_names_out=False)),
                     ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=3000))])


# ----------------------------------------------------------------------------- calibradores
def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class IdentityCalibrator:
    name = "none"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return _clip(p)


class TemperatureScaler:
    """p' = σ(logit(p)/T). Un parámetro; corrige sobre/subconfianza pero no un sesgo de prevalencia."""
    name = "temperature"

    def fit(self, p, y):
        z, y = logit(p), np.asarray(y, dtype=float)

        def nll(T):
            q = _clip(sigmoid(z / T))
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        self.T_ = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
        return self

    def predict(self, p):
        return _clip(sigmoid(logit(p) / self.T_))


class PlattScaler:
    """p' = σ(a·logit(p) + b). Escala (a) y sesgo (b): corrige también cambios de prevalencia."""
    name = "platt"

    def fit(self, p, y):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), np.asarray(y).astype(int))
        self.a_, self.b_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict(self, p):
        return _clip(sigmoid(self.a_ * logit(p) + self.b_))


class IsotonicCalibrator:
    """Función monótona no paramétrica; flexible, requiere muchos datos."""
    name = "isotonic"

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
        self.iso_.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, p):
        return _clip(self.iso_.predict(np.asarray(p, dtype=float)))


CALIBRATORS = {"none": IdentityCalibrator, "temperature": TemperatureScaler, "platt": PlattScaler, "isotonic": IsotonicCalibrator}


def make_calibrator(name):
    return CALIBRATORS[name]()


# ----------------------------------------------------------------------------- PSI
class PSIReference:
    """PSI por variable. Numéricas: deciles de la referencia + bin NaN. Categóricas: top-K categorías + 'otras' + NaN."""

    def __init__(self, n_bins=10, max_categories=30, eps=1e-4):
        self.n_bins, self.max_categories, self.eps = n_bins, max_categories, eps

    @staticmethod
    def _is_num(s):
        return pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype)

    def fit(self, X: pd.DataFrame):
        self.features_, self.edges_, self.cats_, self.ref_ = list(X.columns), {}, {}, {}
        for c in self.features_:
            s = X[c]
            if self._is_num(s):
                v = s.to_numpy(dtype=float)
                fin = v[~np.isnan(v)]
                inner = np.unique(np.quantile(fin, np.linspace(0, 1, self.n_bins + 1))[1:-1]) if fin.size else np.array([])
                self.edges_[c], self.cats_[c] = np.concatenate([[-np.inf], inner, [np.inf]]), None
            else:
                vc = s.astype(object).dropna().value_counts()
                self.cats_[c], self.edges_[c] = list(vc.index[: self.max_categories]), None
            self.ref_[c] = self._props(s, c)
        return self

    def _props(self, s, c):
        if self.cats_[c] is None:
            v = s.to_numpy(dtype=float)
            nan = np.isnan(v)
            counts = np.append(np.histogram(v[~nan], bins=self.edges_[c])[0], nan.sum()).astype(float)
        else:
            v = s.astype(object)
            nan = v.isna().to_numpy()
            code = v.map({cat: i for i, cat in enumerate(self.cats_[c])}).to_numpy(dtype=float)
            known = ~np.isnan(code)
            counts = np.bincount(code[known].astype(int), minlength=len(self.cats_[c])).astype(float)
            counts = np.append(counts, [(~known & ~nan).sum(), nan.sum()])
        return counts / max(len(s), 1)

    def transform(self, X: pd.DataFrame) -> pd.Series:
        out = {}
        for c in self.features_:
            q, p = self._props(X[c], c) + self.eps, self.ref_[c] + self.eps
            out[c] = float(np.sum((q - p) * np.log(q / p)))
        return pd.Series(out)


def psi_null_thresholds(ref_df, psi_ref, n, B=200, seed=42, qs=(0.95, 0.99)):
    """Distribución nula de PSI_máx para muestras de tamaño n sin drift (bootstrap)."""
    rng, m = np.random.default_rng(seed), len(ref_df)
    vals = np.array([psi_ref.transform(ref_df.iloc[rng.integers(0, m, n)]).max() for _ in range(B)])
    return {"media": float(vals.mean()), "p95": float(np.quantile(vals, qs[0])), "p99": float(np.quantile(vals, qs[1]))}


def fit_psi_monitor(ref_df, sizes, B=200, seed=42, n_bins=10, max_categories=30):
    """Referencia con una mitad aleatoria de ref_df; distribución nula muestreada de la otra mitad (independiente)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ref_df))
    half = len(idx) // 2
    ref = PSIReference(n_bins=n_bins, max_categories=max_categories).fit(ref_df.iloc[idx[:half]])
    null_src = ref_df.iloc[idx[half:]]
    return ref, {int(n): psi_null_thresholds(null_src, ref, n=int(n), B=B, seed=seed) for n in sizes}


def calibration_z(y, p):
    """z = (observados − esperados) / sqrt(Σ p(1−p)): calibración en lo grande."""
    p = np.asarray(p, dtype=float)
    var = float(np.sum(p * (1 - p)))
    return (float(np.sum(y)) - float(p.sum())) / np.sqrt(var) if var > 0 else float("nan")


def z_null_thresholds(y, p, n, B=500, seed=42, qs=(0.95, 0.99)):
    """Percentiles de |z| en muestras de tamaño n sin drift, con predicciones fuera de muestra."""
    rng, y, p = np.random.default_rng(seed), np.asarray(y), np.asarray(p, dtype=float)
    zs = np.abs([calibration_z(y[i], p[i]) for i in (rng.integers(0, len(y), n) for _ in range(B))])
    return {"media": float(np.nanmean(zs)), "p95": float(np.nanquantile(zs, qs[0])), "p99": float(np.nanquantile(zs, qs[1]))}


# ----------------------------------------------------------------------------- métricas
def ece_uniform(y, p, n_bins=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(np.abs(np.bincount(b, y, n_bins) - np.bincount(b, p, n_bins)).sum() / len(y))


def ace_equal_mass(y, p, n_bins=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    chunks = np.array_split(np.argsort(p, kind="mergesort"), n_bins)
    return float(sum(len(c) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)) / len(y))


def mce_uniform(y, p, n_bins=10, min_count=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    cnt, sp, sy = np.bincount(b, minlength=n_bins), np.bincount(b, p, n_bins), np.bincount(b, y, n_bins)
    m = cnt >= min_count
    return float(np.max(np.abs(sy[m] - sp[m]) / cnt[m])) if m.any() else float("nan")


def calibration_intercept_slope(y, p):
    """Calibración en lo grande (CITL, ideal 0) y pendiente de calibración (ideal 1)."""
    z, y = logit(p), np.asarray(y, dtype=float)
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y.astype(int)).coef_[0, 0])

    def nll(a):
        q = _clip(sigmoid(a + z))
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    return float(minimize_scalar(nll, bounds=(-10, 10), method="bounded").x), slope


def ks_separation(y, p):
    """KS entre las distribuciones de score de pacientes readmitidos y no readmitidos."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    r = stats.ks_2samp(p[y == 1], p[y == 0])
    return float(r.statistic), float(r.pvalue)


def evaluate_probs(y, p, prior=None):
    y, p = np.asarray(y).astype(int), _clip(p)
    both = len(np.unique(y)) == 2
    brier = brier_score_loss(y, p)
    prior = y.mean() if prior is None else prior
    brier_ref = np.mean((prior - y) ** 2)
    out = {"n": len(y), "positivos": int(y.sum()), "prevalencia": float(y.mean()), "p_media": float(p.mean()),
           "AUC": roc_auc_score(y, p) if both else np.nan, "AP": average_precision_score(y, p) if both else np.nan,
           "Brier": brier, "BSS": 1 - brier / brier_ref if brier_ref > 0 else np.nan,
           "LogLoss": log_loss(y, p, labels=[0, 1]), "ECE": ece_uniform(y, p), "ACE": ace_equal_mass(y, p),
           "MCE": mce_uniform(y, p), "O/E": float(y.sum() / p.sum()),
           "CITL": np.nan, "Pendiente": np.nan, "KS": np.nan, "KS_p": np.nan}
    if both:
        out["CITL"], out["Pendiente"] = calibration_intercept_slope(y, p)
        out["KS"], out["KS_p"] = ks_separation(y, p)
    return out


def reliability_table(y, p, n_bins=10, strategy="uniform"):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if strategy == "quantile":
        g = np.empty(len(p), dtype=int)
        for i, chunk in enumerate(np.array_split(np.argsort(p, kind="mergesort"), n_bins)):
            g[chunk] = i
    else:
        g = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return (pd.DataFrame({"bin": g, "y": y, "p": p}).groupby("bin")
            .agg(n=("y", "size"), p_media=("p", "mean"), frecuencia_observada=("y", "mean")).reset_index())


# ----------------------------------------------------------------------------- scorer para el dashboard
def _base_name(name):
    hits = [b for b in MODEL_FEATURES if name == b or name.startswith(b + "_")]
    return max(hits, key=len) if hits else name


class ReadmissionScorer:
    """Modelo + esquema de variables + calibrador + umbral + bandas de riesgo + explicación local."""

    def __init__(self, name, kind, model, calibrator, featurizer, threshold, risk_cuts=(0.08, 0.15)):
        self.name, self.kind, self.model, self.calibrator, self.featurizer = name, kind, model, calibrator, featurizer
        self.threshold, self.risk_cuts = float(threshold), tuple(risk_cuts)

    def features(self, df):
        return self.featurizer.transform(df)

    def raw_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        return self.calibrator.predict(self.raw_proba(X))

    def risk_band(self, p):
        return pd.cut(np.asarray(p), bins=[0.0, *self.risk_cuts, 1.0 + 1e-9], labels=list(RISK_LABELS),
                      right=False, include_lowest=True).astype(str)

    def explain(self, X):
        """Contribuciones en log-momios (SHAP nativo en XGBoost; coef×valor en LR, agregadas por variable)."""
        if self.kind == "xgb":
            import xgboost as xgb
            contrib = self.model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[:, :-1]
            return pd.DataFrame(contrib, columns=list(X.columns), index=X.index)
        prep = self.model[:-1]
        Z = prep.transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        names = [_base_name(n) for n in prep.get_feature_names_out()]
        C = pd.DataFrame(Z * self.model[-1].coef_[0], columns=names, index=X.index)
        return C.T.groupby(level=0).sum().T

    def predict(self, df, top_k=3, explain=True, labels=None):
        labels = labels or FEATURE_LABELS
        X = self.features(df)
        p = self.predict_proba(X)
        out = pd.DataFrame({"prob_readmision_30d": p, "banda_riesgo": self.risk_band(p),
                            "alerta": (p >= self.threshold).astype(int)}, index=df.index)
        if explain:
            C = self.explain(X)
            tops = []
            for _, row in C.iterrows():
                r = row.reindex(row.abs().sort_values(ascending=False).index[:top_k])
                tops.append("; ".join(f"{k} {'↑' if v > 0 else '↓'} ({labels.get(k, k)[:40]})" for k, v in r.items()))
            out["factores_principales"] = tops
        return out


def score_batch(bundle, df, model="best", explain=True, top_k=3, drift_check=True, min_rows_drift=200):
    """Función que consumirá el dashboard: predicciones + reporte de drift + advertencias de esquema."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas.DataFrame")
    warns = []
    absent = [c for c in RAW_INPUT if c not in df.columns]
    if absent:
        warns.append(f"{len(absent)} columna(s) ausente(s) → se tratan como desconocidas/No: {absent[:8]}")
    fz = bundle["scorers"][bundle["best"]].featurizer
    for raw in ID_COLUMNS:
        if raw in df.columns:
            unk = int((~pd.to_numeric(df[raw], errors="coerce").isin(list(fz.maps_.get(raw, {})))).sum())
            if unk:
                warns.append(f"{unk} fila(s) con {raw} fuera de IDS_mapping → 'Desconocido'")
    name = bundle["best"] if model == "best" else model
    if name not in bundle["scorers"]:
        raise ValueError(f"Modelo '{name}' no disponible; opciones: {list(bundle['scorers'])}")
    preds = bundle["scorers"][name].predict(df, top_k=top_k, explain=explain, labels=bundle.get("feature_labels"))
    drift = None
    if drift_check and len(df) >= min_rows_drift:
        psi = bundle["psi_reference"].transform(fz.monitor_frame(df))
        sizes = np.array(sorted(bundle["psi_null"]))
        n_key = int(sizes[np.argmin(np.abs(sizes - len(df)))])
        thr = bundle["psi_null"][n_key]
        lvl = 2 if psi.max() >= thr["p99"] else (1 if psi.max() >= thr["p95"] else 0)
        drift = {"n": int(len(df)), "psi_max": float(psi.max()), "nivel": lvl, "umbral_p95": float(thr["p95"]),
                 "umbral_p99": float(thr["p99"]), "n_referencia_umbral": n_key,
                 "top_variables": {k: round(float(v), 4) for k, v in psi.sort_values(ascending=False).head(6).items()}}
    elif drift_check:
        warns.append(f"Verificación de drift omitida: se requieren ≥ {min_rows_drift} filas")
    return {"modelo": name, "predicciones": preds, "drift": drift, "advertencias": warns}
