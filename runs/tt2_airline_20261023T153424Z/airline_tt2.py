"""airline_tt2 — componentes reutilizables del TT2 (2026-B080) para Airline Delay.

Lo importan el notebook y el futuro dashboard: llaves e historial causal de retrasos,
construcción de variables, pipeline lineal, calibradores, PSI (numérico y categórico),
métricas y el objeto DelayScorer. Al vivir en un módulo, joblib serializa estas clases por
referencia a `airline_tt2` y el dashboard las carga con `import airline_tt2`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAW_COLUMNS = ["Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length"]
CAT_FEATURES = ["Airline", "AirportFrom", "AirportTo"]
LAG_KEYS = {"fl": "vuelo (aerolínea + número)", "route": "ruta (origen → destino)", "orig": "aeropuerto de origen",
            "dest": "aeropuerto de destino", "airline": "aerolínea"}
RATE_FEATURES = [f"h_{k}" for k in LAG_KEYS] + [f"p_{k}" for k in LAG_KEYS] + ["prev_day_rate"]
COUNT_FEATURES = ["n_fl", "n_route"]
LAG_FEATURES = RATE_FEATURES + COUNT_FEATURES
BASE_NUMERIC = ["Time", "Length", "sin_t", "cos_t"]
XGB_FEATURES = CAT_FEATURES + ["DayOfWeek"] + BASE_NUMERIC + LAG_FEATURES
MONITOR_FEATURES = ["Airline", "AirportFrom", "AirportTo", "Time", "Length"]
RISK_LABELS = ("Bajo", "Medio", "Alto")
EPS = 1e-6

FEATURE_LABELS = {
    "Airline": "Aerolínea", "Flight": "Número de vuelo (identificador)", "AirportFrom": "Aeropuerto de origen",
    "AirportTo": "Aeropuerto de destino", "DayOfWeek": "Día de la semana", "Time": "Hora de salida (min desde 00:00)",
    "Length": "Duración programada (min)", "sin_t": "Hora de salida (componente seno)", "cos_t": "Hora de salida (componente coseno)",
    "prev_day_rate": "Tasa global de retraso del día anterior",
    "n_fl": "Historial del vuelo (n.º de operaciones previas, log)", "n_route": "Historial de la ruta (n.º de vuelos previos, log)",
    **{f"h_{k}": f"Tasa histórica de retraso · {v}" for k, v in LAG_KEYS.items()},
    **{f"p_{k}": f"Tasa de retraso del día anterior · {v}" for k, v in LAG_KEYS.items()},
}


# ----------------------------------------------------------------------------- historial causal
def make_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Llaves de historial: vuelo, ruta, origen, destino y aerolínea (como texto)."""
    flight = pd.to_numeric(df["Flight"], errors="coerce").astype("Int64").astype(str)
    return pd.DataFrame({"fl": df["Airline"].astype(str) + "_" + flight,
                         "route": df["AirportFrom"].astype(str) + "-" + df["AirportTo"].astype(str),
                         "orig": df["AirportFrom"].astype(str), "dest": df["AirportTo"].astype(str),
                         "airline": df["Airline"].astype(str)}, index=df.index)


def causal_lags(df: pd.DataFrame, prior: float, m: float = 20.0) -> pd.DataFrame:
    """Variables de historial para CADA fila usando solo etiquetas de días anteriores (sin fuga).

    h_k: tasa histórica suavizada hasta el día d−1 · p_k: tasa del día d−1 · n_k: log(1 + n.º de vuelos previos).
    Suavizado bayesiano: (retrasos + m·prior) / (vuelos + m).
    """
    keys, day, y = make_keys(df), df["day"].to_numpy(), df["Delay"].to_numpy(dtype=float)
    out = pd.DataFrame(index=df.index)
    for k in LAG_KEYS:
        g = (pd.DataFrame({"key": keys[k].to_numpy(), "day": day, "y": y})
             .groupby(["key", "day"], sort=True)["y"].agg(["sum", "count"]).reset_index())
        gk = g.groupby("key")
        g["cs"], g["cc"] = gk["sum"].cumsum() - g["sum"], gk["count"].cumsum() - g["count"]
        ok = gk["day"].shift(1) == g["day"] - 1
        ps, pc = gk["sum"].shift(1), gk["count"].shift(1)
        g["h"] = (g["cs"] + m * prior) / (g["cc"] + m)
        g["p"] = np.where(ok, (ps + m * prior) / (pc + m), prior)
        g["n"] = np.log1p(g["cc"])
        mg = pd.DataFrame({"key": keys[k].to_numpy(), "day": day}).merge(g[["key", "day", "h", "p", "n"]], on=["key", "day"], how="left")
        out[f"h_{k}"], out[f"p_{k}"] = mg["h"].to_numpy(), mg["p"].to_numpy()
        if f"n_{k}" in COUNT_FEATURES:
            out[f"n_{k}"] = mg["n"].to_numpy()
    rate = pd.Series(y).groupby(day).mean()
    out["prev_day_rate"] = pd.Series(day - 1).map(rate).fillna(prior).to_numpy()
    return out[LAG_FEATURES]


class HistoryStore:
    """Estado del historial para inferencia en línea: acumulados por llave hasta el último día etiquetado."""

    def __init__(self, prior, m=20.0):
        self.prior, self.m = float(prior), float(m)

    def fit(self, df):
        keys, y = make_keys(df), df["Delay"].astype(float)
        self.last_day_ = int(df["day"].max())
        last = (df["day"] == self.last_day_).to_numpy()
        self.cum_ = {k: y.groupby(keys[k]).agg(["sum", "count"]) for k in LAG_KEYS}
        self.last_ = {k: y[last].groupby(keys[k][last]).agg(["sum", "count"]) for k in LAG_KEYS}
        self.last_rate_ = float(y[last].mean())
        return self

    def update(self, df_new):
        """Agrega uno o más días nuevos ya etiquetados (el dashboard lo llamaría al cierre de cada día)."""
        keys, y = make_keys(df_new), df_new["Delay"].astype(float)
        new_last = int(df_new["day"].max())
        last = (df_new["day"] == new_last).to_numpy()
        for k in LAG_KEYS:
            self.cum_[k] = self.cum_[k].add(y.groupby(keys[k]).agg(["sum", "count"]), fill_value=0)
            self.last_[k] = y[last].groupby(keys[k][last]).agg(["sum", "count"])
        self.last_day_, self.last_rate_ = new_last, float(y[last].mean())
        return self

    def lookup(self, df):
        """Variables de historial para vuelos del día siguiente al último día etiquetado."""
        keys, m, pr = make_keys(df), self.m, self.prior
        out = pd.DataFrame(index=df.index)
        for k in LAG_KEYS:
            c = self.cum_[k].reindex(keys[k].to_numpy())
            l = self.last_[k].reindex(keys[k].to_numpy())
            cs, cc = c["sum"].fillna(0).to_numpy(), c["count"].fillna(0).to_numpy()
            out[f"h_{k}"] = (cs + m * pr) / (cc + m)
            ls, lc = l["sum"].to_numpy(dtype=float), l["count"].to_numpy(dtype=float)
            out[f"p_{k}"] = np.where(np.isnan(lc), pr, (np.nan_to_num(ls) + m * pr) / (np.nan_to_num(lc) + m))
            if f"n_{k}" in COUNT_FEATURES:
                out[f"n_{k}"] = np.log1p(cc)
        out["prev_day_rate"] = self.last_rate_
        return out[LAG_FEATURES]


# ----------------------------------------------------------------------------- variables
class AirlineFeaturizer:
    """Esquema estático: vocabularios categóricos congelados en train (no vistos → NaN) + hora cíclica + historial."""

    def fit(self, df):
        self.categories_ = {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT_FEATURES}
        return self

    def transform(self, df, lags):
        X = pd.DataFrame(index=df.index)
        for c in CAT_FEATURES:
            v = df[c].astype(str)
            X[c] = pd.Categorical(v.where(v.isin(self.categories_[c])), categories=self.categories_[c])
        for c in ["DayOfWeek", "Time", "Length"]:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        X["sin_t"] = np.sin(2 * np.pi * X["Time"] / 1440)
        X["cos_t"] = np.cos(2 * np.pi * X["Time"] / 1440)
        X = pd.concat([X, pd.DataFrame(np.asarray(lags[LAG_FEATURES], dtype=float), columns=LAG_FEATURES, index=df.index)], axis=1)
        return X[XGB_FEATURES]


def monitor_frame(df):
    """Variables crudas que vigila el PSI (el día de la semana se excluye: es calendario, no drift)."""
    X = pd.DataFrame({c: df[c].astype(str) for c in ["Airline", "AirportFrom", "AirportTo"]}, index=df.index)
    for c in ["Time", "Length"]:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return X[MONITOR_FEATURES]


def _logit(Z):
    Z = np.clip(np.asarray(Z, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(Z / (1 - Z))


def make_lr_pipeline(C=0.1, class_weight=None, min_frequency=200, lags=True):
    """One-hot (categorías raras agrupadas) + hora cíclica + logit de tasas históricas → LR."""
    cat = CAT_FEATURES + ["DayOfWeek"]
    tr = [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency), cat),
          ("num", StandardScaler(), ["Length", "sin_t", "cos_t"])]
    if lags:
        tr += [("rates", Pipeline([("logit", FunctionTransformer(_logit, feature_names_out="one-to-one")),
                                   ("sc", StandardScaler())]), RATE_FEATURES),
               ("counts", StandardScaler(), COUNT_FEATURES)]
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
    """KS entre las distribuciones de score de vuelos retrasados y puntuales."""
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
_EXPLAIN_BASES = CAT_FEATURES + ["DayOfWeek"] + BASE_NUMERIC + LAG_FEATURES


def _base_name(name):
    hits = [b for b in _EXPLAIN_BASES if name == b or name.startswith(b + "_")]
    return max(hits, key=len) if hits else name


class DelayScorer:
    """Modelo + calibrador + esquema de variables + umbral + bandas de riesgo + explicación local."""

    def __init__(self, name, kind, model, calibrator, featurizer, threshold, risk_cuts=(0.35, 0.60)):
        self.name, self.kind, self.model, self.calibrator, self.featurizer = name, kind, model, calibrator, featurizer
        self.threshold, self.risk_cuts = float(threshold), tuple(risk_cuts)

    def features(self, df, history):
        return self.featurizer.transform(df, history.lookup(df))

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

    def predict(self, df, history, top_k=3, explain=True, labels=None):
        labels = labels or FEATURE_LABELS
        X = self.features(df, history)
        p = self.predict_proba(X)
        out = pd.DataFrame({"prob_retraso": p, "banda_riesgo": self.risk_band(p), "alerta": (p >= self.threshold).astype(int),
                            "historial_vuelo": np.round(np.expm1(X["n_fl"].to_numpy())).astype(int)}, index=df.index)
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
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    warns = []
    fz = bundle["scorers"][bundle["best"]].featurizer
    for c in CAT_FEATURES:
        unseen = int((~df[c].astype(str).isin(fz.categories_[c])).sum())
        if unseen:
            warns.append(f"{unseen} fila(s) con {c} no visto en entrenamiento → se trata como desconocido")
    name = bundle["best"] if model == "best" else model
    if name not in bundle["scorers"]:
        raise ValueError(f"Modelo '{name}' no disponible; opciones: {list(bundle['scorers'])}")
    preds = bundle["scorers"][name].predict(df, bundle["history"], top_k=top_k, explain=explain, labels=bundle.get("feature_labels"))
    drift = None
    if drift_check and len(df) >= min_rows_drift:
        psi = bundle["psi_reference"].transform(monitor_frame(df))
        sizes = np.array(sorted(bundle["psi_null"]))
        n_key = int(sizes[np.argmin(np.abs(sizes - len(df)))])
        thr = bundle["psi_null"][n_key]
        lvl = 2 if psi.max() >= thr["p99"] else (1 if psi.max() >= thr["p95"] else 0)
        drift = {"n": int(len(df)), "psi_max": float(psi.max()), "nivel": lvl, "umbral_p95": float(thr["p95"]),
                 "umbral_p99": float(thr["p99"]), "n_referencia_umbral": n_key,
                 "psi_por_variable": {k: round(float(v), 4) for k, v in psi.sort_values(ascending=False).items()}}
    elif drift_check:
        warns.append(f"Verificación de drift omitida: se requieren ≥ {min_rows_drift} filas")
    return {"modelo": name, "predicciones": preds, "drift": drift, "advertencias": warns}
