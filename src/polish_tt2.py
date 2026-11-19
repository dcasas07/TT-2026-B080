"""polish_tt2 — componentes reutilizables del TT2 (2026-B080) para Polish Companies Bankruptcy.

Lo importan el notebook y el futuro dashboard: construcción de variables, pipeline lineal,
calibradores, PSI, métricas y el objeto BankruptcyScorer (modelo + calibrador + umbrales).
Al vivir en un módulo, joblib serializa estas clases por referencia a `polish_tt2` y el
dashboard puede cargarlas con `import polish_tt2`.
"""
from __future__ import annotations

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
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RAW_FEATURES = [f"Attr{i}" for i in range(1, 65)]
ENGINEERED_FEATURES = ["n_missing"]
MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
RISK_LABELS = ("Bajo", "Medio", "Alto")
EPS = 1e-6

# Diccionario de variables según la descripción del dataset en UCI (id 365). Verificar contra la fuente oficial.
FEATURE_LABELS = {
    "Attr1": "Utilidad neta / activos totales",
    "Attr2": "Pasivos totales / activos totales",
    "Attr3": "Capital de trabajo / activos totales",
    "Attr4": "Activo circulante / pasivos a corto plazo",
    "Attr5": "[(Efectivo + valores CP + cuentas por cobrar − pasivos CP) / (gastos operativos − depreciación)] × 365",
    "Attr6": "Utilidades retenidas / activos totales",
    "Attr7": "EBIT / activos totales",
    "Attr8": "Valor contable del capital / pasivos totales",
    "Attr9": "Ventas / activos totales",
    "Attr10": "Capital contable / activos totales",
    "Attr11": "(Utilidad bruta + partidas extraordinarias + gastos financieros) / activos totales",
    "Attr12": "Utilidad bruta / pasivos a corto plazo",
    "Attr13": "(Utilidad bruta + depreciación) / ventas",
    "Attr14": "(Utilidad bruta + intereses) / activos totales",
    "Attr15": "(Pasivos totales × 365) / (utilidad bruta + depreciación)",
    "Attr16": "(Utilidad bruta + depreciación) / pasivos totales",
    "Attr17": "Activos totales / pasivos totales",
    "Attr18": "Utilidad bruta / activos totales",
    "Attr19": "Utilidad bruta / ventas",
    "Attr20": "(Inventario × 365) / ventas",
    "Attr21": "Ventas (n) / ventas (n−1)",
    "Attr22": "Utilidad de operación / activos totales",
    "Attr23": "Utilidad neta / ventas",
    "Attr24": "Utilidad bruta (3 años) / activos totales",
    "Attr25": "(Capital contable − capital social) / activos totales",
    "Attr26": "(Utilidad neta + depreciación) / pasivos totales",
    "Attr27": "Utilidad de operación / gastos financieros",
    "Attr28": "Capital de trabajo / activos fijos",
    "Attr29": "Logaritmo de activos totales",
    "Attr30": "(Pasivos totales − efectivo) / ventas",
    "Attr31": "(Utilidad bruta + intereses) / ventas",
    "Attr32": "(Pasivos circulantes × 365) / costo de ventas",
    "Attr33": "Gastos operativos / pasivos a corto plazo",
    "Attr34": "Gastos operativos / pasivos totales",
    "Attr35": "Utilidad en ventas / activos totales",
    "Attr36": "Ventas totales / activos totales",
    "Attr37": "(Activo circulante − inventarios) / pasivos a largo plazo",
    "Attr38": "Capital permanente / activos totales",
    "Attr39": "Utilidad en ventas / ventas",
    "Attr40": "(Activo circulante − inventario − cuentas por cobrar) / pasivos a corto plazo",
    "Attr41": "Pasivos totales / ((utilidad de operación + depreciación) × 12/365)",
    "Attr42": "Utilidad de operación / ventas",
    "Attr43": "Rotación de cuentas por cobrar + rotación de inventario (días)",
    "Attr44": "(Cuentas por cobrar × 365) / ventas",
    "Attr45": "Utilidad neta / inventario",
    "Attr46": "(Activo circulante − inventario) / pasivos a corto plazo",
    "Attr47": "(Inventario × 365) / costo de ventas",
    "Attr48": "EBITDA (utilidad de operación − depreciación) / activos totales",
    "Attr49": "EBITDA (utilidad de operación − depreciación) / ventas",
    "Attr50": "Activo circulante / pasivos totales",
    "Attr51": "Pasivos a corto plazo / activos totales",
    "Attr52": "(Pasivos a corto plazo × 365) / costo de ventas",
    "Attr53": "Capital contable / activos fijos",
    "Attr54": "Capital permanente / activos fijos",
    "Attr55": "Capital de trabajo",
    "Attr56": "(Ventas − costo de ventas) / ventas",
    "Attr57": "(Activo circulante − inventario − pasivos CP) / (ventas − utilidad bruta − depreciación)",
    "Attr58": "Costos totales / ventas totales",
    "Attr59": "Pasivos a largo plazo / capital contable",
    "Attr60": "Ventas / inventario",
    "Attr61": "Ventas / cuentas por cobrar",
    "Attr62": "(Pasivos a corto plazo × 365) / ventas",
    "Attr63": "Ventas / pasivos a corto plazo",
    "Attr64": "Ventas / activos fijos",
    "n_missing": "Número de ratios faltantes (variable construida)",
}


# ----------------------------------------------------------------------------- variables
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Esquema estático: 64 ratios (float, ±inf→NaN) + n_missing. Columnas ausentes → NaN."""
    X = df.reindex(columns=RAW_FEATURES).apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X["n_missing"] = X[RAW_FEATURES].isna().sum(axis=1).astype(float)
    return X


def make_lr_pipeline(columns, C=0.1, imputer="median", class_weight=None, clip=8.0, seed=42):
    """Pipeline lineal: selección → asinh (log con signo) → imputación → escalado → winsorización ±clip → LR."""
    columns = list(columns)
    if imputer == "median":
        imp = SimpleImputer(strategy="median", add_indicator=True)
    elif imputer == "median_sin_indicadores":
        imp = SimpleImputer(strategy="median")
    elif imputer == "iterativo":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=5, n_nearest_features=15, random_state=seed, add_indicator=True, skip_complete=True)
    else:
        raise ValueError(f"imputer desconocido: {imputer}")
    return Pipeline([
        ("select", ColumnTransformer([("sel", "passthrough", columns)], remainder="drop", verbose_feature_names_out=False)),
        ("asinh", FunctionTransformer(np.arcsinh, feature_names_out="one-to-one")),
        ("impute", imp),
        ("scale", StandardScaler()),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -clip, "a_max": clip}, feature_names_out="one-to-one")),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)),
    ])


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
    """p' = σ(logit(p)/T). Un parámetro; corrige sobre/subconfianza pero no un sesgo de prior."""
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
    """p' = σ(a·logit(p) + b). Escala (a) y sesgo (b): corrige también desplazamientos de prevalencia."""
    name = "platt"

    def fit(self, p, y):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), np.asarray(y).astype(int))
        self.a_, self.b_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict(self, p):
        return _clip(sigmoid(self.a_ * logit(p) + self.b_))


class IsotonicCalibrator:
    """Función monótona no paramétrica; flexible pero requiere muchos positivos."""
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
    """PSI con bins por cuantiles de la referencia + bin explícito para NaN (captura drift de faltantes)."""

    def __init__(self, n_bins=10, eps=1e-4):
        self.n_bins, self.eps = n_bins, eps

    def fit(self, X: pd.DataFrame):
        self.features_, self.edges_, self.ref_ = list(X.columns), {}, {}
        for c in self.features_:
            v = X[c].to_numpy(dtype=float)
            fin = v[~np.isnan(v)]
            inner = np.unique(np.quantile(fin, np.linspace(0, 1, self.n_bins + 1))[1:-1]) if fin.size else np.array([])
            edges = np.concatenate([[-np.inf], inner, [np.inf]])
            self.edges_[c], self.ref_[c] = edges, self._props(v, edges)
        return self

    @staticmethod
    def _props(v, edges):
        nan = np.isnan(v)
        counts = np.append(np.histogram(v[~nan], bins=edges)[0], nan.sum()).astype(float)
        return counts / max(len(v), 1)

    def transform(self, X: pd.DataFrame) -> pd.Series:
        out = {}
        for c in self.features_:
            q = self._props(X[c].to_numpy(dtype=float), self.edges_[c]) + self.eps
            p = self.ref_[c] + self.eps
            out[c] = float(np.sum((q - p) * np.log(q / p)))
        return pd.Series(out)


def psi_null_thresholds(ref_df, psi_ref, n, B=200, seed=42, qs=(0.95, 0.99)):
    """Distribución nula de PSI_máx para muestras de tamaño n sin drift (bootstrap desde la referencia)."""
    rng, m = np.random.default_rng(seed), len(ref_df)
    vals = np.array([psi_ref.transform(ref_df.iloc[rng.integers(0, m, n)]).max() for _ in range(B)])
    return {"media": float(vals.mean()), "p95": float(np.quantile(vals, qs[0])), "p99": float(np.quantile(vals, qs[1]))}


def calibration_z(y, p):
    """z = (observados − esperados) / sqrt(Σ p(1−p)): calibración en lo grande de un bloque."""
    p = np.asarray(p, dtype=float)
    var = float(np.sum(p * (1 - p)))
    return (float(np.sum(y)) - float(p.sum())) / np.sqrt(var) if var > 0 else float("nan")


def z_null_thresholds(y, p, n, B=500, seed=42, qs=(0.95, 0.99)):
    """Percentiles de |z| en bloques de tamaño n SIN drift, con predicciones fuera de muestra.

    Los umbrales teóricos 2/3 suponen probabilidades exactas fila por fila; con modelos reales
    el estadístico está sobredisperso y esos umbrales producen demasiadas falsas alarmas.
    """
    rng, y, p = np.random.default_rng(seed), np.asarray(y), np.asarray(p, dtype=float)
    zs = np.abs([calibration_z(y[i], p[i]) for i in (rng.integers(0, len(y), n) for _ in range(B))])
    return {"media": float(np.nanmean(zs)), "p95": float(np.nanquantile(zs, qs[0])), "p99": float(np.nanquantile(zs, qs[1]))}


def fit_psi_monitor(ref_df, sizes, B=200, seed=42, n_bins=10):
    """Referencia PSI + umbrales nulos con muestras INDEPENDIENTES de la referencia.

    La referencia se ajusta con una mitad aleatoria de ref_df y la distribución nula se
    muestrea de la otra mitad. Muestrear de la propia referencia subestima el PSI bajo
    'sin drift' y dispara falsas alarmas.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ref_df))
    half = len(idx) // 2
    ref = PSIReference(n_bins=n_bins).fit(ref_df.iloc[idx[:half]])
    null_src = ref_df.iloc[idx[half:]]
    thr = {int(n): psi_null_thresholds(null_src, ref, n=int(n), B=B, seed=seed) for n in sizes}
    return ref, thr


# ----------------------------------------------------------------------------- métricas
def ece_uniform(y, p, n_bins=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    sp, sy = np.bincount(b, p, n_bins), np.bincount(b, y, n_bins)
    return float(np.abs(sy - sp).sum() / len(y))


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
    """KS entre las distribuciones de score de quiebras y sanas (capacidad de separación)."""
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
class BankruptcyScorer:
    """Modelo + calibrador + umbral + bandas de riesgo + explicación local."""

    def __init__(self, name, kind, model, calibrator, threshold, risk_cuts=(0.05, 0.20)):
        self.name, self.kind, self.model, self.calibrator = name, kind, model, calibrator
        self.threshold, self.risk_cuts = float(threshold), tuple(risk_cuts)

    def _prep(self, df, prepared):
        return df if prepared else build_features(df)

    def raw_proba(self, df, prepared=False):
        X = self._prep(df, prepared)
        return self.model.predict_proba(X[MODEL_FEATURES] if self.kind == "xgb" else X)[:, 1]

    def predict_proba(self, df, prepared=False):
        return self.calibrator.predict(self.raw_proba(df, prepared))

    def risk_band(self, p):
        return pd.cut(np.asarray(p), bins=[0.0, *self.risk_cuts, 1.0 + 1e-9], labels=list(RISK_LABELS),
                      right=False, include_lowest=True).astype(str)

    def explain(self, df, prepared=False):
        """Contribuciones en log-momios (SHAP nativo en XGBoost; coef×valor en LR)."""
        X = self._prep(df, prepared)
        if self.kind == "xgb":
            import xgboost as xgb
            contrib = self.model.get_booster().predict(xgb.DMatrix(X[MODEL_FEATURES]), pred_contribs=True)[:, :-1]
            return pd.DataFrame(contrib, columns=MODEL_FEATURES, index=X.index)
        pre = self.model[:-1]
        Z = pre.transform(X)
        names = [n.replace("missingindicator_", "") for n in pre.get_feature_names_out()]
        C = pd.DataFrame(np.asarray(Z) * self.model[-1].coef_[0], columns=names, index=X.index)
        return C.T.groupby(level=0).sum().T

    def predict(self, df, top_k=3, explain=True, labels=None):
        labels = labels or FEATURE_LABELS
        X = build_features(df)
        p = self.predict_proba(X, prepared=True)
        out = pd.DataFrame({"prob_quiebra": p, "banda_riesgo": self.risk_band(p),
                            "alerta": (p >= self.threshold).astype(int),
                            "n_faltantes": X["n_missing"].astype(int).to_numpy()}, index=df.index)
        if explain:
            C = self.explain(X, prepared=True)
            tops = []
            for _, row in C.iterrows():
                r = row.reindex(row.abs().sort_values(ascending=False).index[:top_k])
                tops.append("; ".join(f"{k} {'↑' if v > 0 else '↓'} ({labels.get(k, k)[:45]})" for k, v in r.items()))
            out["factores_principales"] = tops
        return out


def score_batch(bundle, df, model="best", explain=True, top_k=3, drift_check=True, min_rows_drift=200):
    """Función que consumirá el dashboard: predicciones + reporte de drift + advertencias de esquema."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas.DataFrame")
    warns = []
    unknown = [c for c in df.columns if c not in RAW_FEATURES]
    if unknown:
        warns.append(f"Columnas ignoradas (no son Attr1–Attr64): {unknown[:10]}")
    absent = [c for c in RAW_FEATURES if c not in df.columns]
    if absent:
        warns.append(f"{len(absent)} de 64 ratios ausentes → se tratan como faltantes (NaN)")
    name = bundle["best"] if model == "best" else model
    if name not in bundle["scorers"]:
        raise ValueError(f"Modelo '{name}' no disponible; opciones: {list(bundle['scorers'])}")
    preds = bundle["scorers"][name].predict(df, top_k=top_k, explain=explain, labels=bundle.get("feature_labels"))
    drift = None
    if drift_check and len(df) >= min_rows_drift:
        psi = bundle["psi_reference"].transform(build_features(df)[MODEL_FEATURES])
        sizes = np.array(sorted(bundle["psi_null"]))
        n_key = int(sizes[np.argmin(np.abs(sizes - len(df)))])
        thr = bundle["psi_null"][n_key]
        lvl = 2 if psi.max() >= thr["p99"] else (1 if psi.max() >= thr["p95"] else 0)
        drift = {"n": int(len(df)), "psi_max": float(psi.max()), "nivel": lvl, "umbral_p95": float(thr["p95"]),
                 "umbral_p99": float(thr["p99"]), "n_referencia_umbral": n_key,
                 "top_variables": {k: round(float(v), 4) for k, v in psi.sort_values(ascending=False).head(5).items()}}
    elif drift_check:
        warns.append(f"Verificación de drift omitida: se requieren ≥ {min_rows_drift} filas")
    return {"modelo": name, "predicciones": preds, "drift": drift, "advertencias": warns}
