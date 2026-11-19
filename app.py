"""
Calibration Drift Monitor · Trabajo Terminal 2026-B080 (ESCOM-IPN)
"Detección y Mitigación de Calibration Drift en Modelos de Clasificación"
Casas Aguilar José David · Robles Cervantes Pedro Iván

Tablero unificado con dos pestañas:
  1. Resumen Ejecutivo ........................ lectura gerencial: situación, indicadores, alertas y recomendaciones
                                                (antes: TT2_Tablero_Ejecutivo.ipynb)
  2. Monitoreo, resultados y detalles ......... evidencia técnica: métricas, calibración, deriva, estrategias y bitácora
                                                (antes: TT2_Dashboard_v2.ipynb)

Estructura del archivo
  0. Identidad visual común (paleta, CSS, plantilla de Plotly)
  Pestaña 2 · secciones 2–7 (capa de datos, semáforos, gráficas, módulos por dataset, exploración)
  Pestaña 1 · secciones 1–11 (reglas, utilidades, cálculos, análisis por dataset, orquestación)
  Página principal (encabezado y pestañas).

Fuentes de datos (el tablero no entrena ni recalcula modelos: solo presenta la evidencia de las corridas finales)
  · Pestaña 2: tt2_data_pack.zip (tablas parquet/JSON de los notebooks TT2), escrito por la celda 4 del notebook.
  · Pestaña 1: datos_tablero/ (registros de producción puntuados por el modelo), generado por tt2_tablero_prep.py
    en la celda 7 a partir de los artefactos de los notebooks TT2 (mismo flujo que TT2_Tablero_Ejecutivo.ipynb).

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Calibration Drift Monitor · TT 2026-B080", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

# =====================================================================================================
# 0. IDENTIDAD VISUAL COMÚN (paleta, CSS y plantilla de gráficas de ambas pestañas)
# =====================================================================================================
INK, INK_2, MUTED = "#1b1b1a", "#52514e", "#7a7974"
SURFACE, PAGE, RULE, GRID = "#ffffff", "#f5f5f2", "#dcdbd5", "#ecebe7"
ACCENT, ACCENT_DARK, ACCENT_SOFT = "#1c5cab", "#104281", "#e8f0fb"
# Paleta categórica validada (orden fijo, nunca ciclado) y paleta de estado reservada
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#8a8984"
STATUS = {"good": "#0ca30c", "warn": "#fab219", "bad": "#d03b3b", "info": ACCENT}
EMOJI = {"good": "🟢", "warn": "🟡", "bad": "🔴", "info": "🔵"}
WORD = {"good": "Bueno", "warn": "Regular", "bad": "Malo", "info": "Informativo"}
POS_COLOR, NEG_COLOR = CAT[1], CAT[0]          # evento (naranja) vs no evento (azul), fijo en todo el tablero

POLICIES = ["Sin calibrar (TT1)", "Baseline calibrado", "A · Recalibración", "B · Reentrenamiento", "C · Híbrida",
            "Oráculo (in-sample)"]
POLICY_STYLE = {  # color por entidad (no por rango) + patrón de línea como codificación secundaria
    "Sin calibrar (TT1)": (NEUTRAL, "dot", "circle-open"),
    "Baseline calibrado": (CAT[0], "solid", "circle"),
    "A · Recalibración": (CAT[1], "dash", "triangle-up"),
    "B · Reentrenamiento": (CAT[2], "dashdot", "square"),
    "C · Híbrida": (CAT[3], "longdash", "diamond"),
    "Oráculo (in-sample)": (CAT[4], "dot", "x"),
}
POLICY_PLAIN = {
    "Sin calibrar (TT1)": "El modelo tal como salió del entrenamiento, sin ajuste de probabilidades (esquema de TT1).",
    "Baseline calibrado": "El modelo con su ajuste inicial de probabilidades y sin ningún mantenimiento posterior.",
    "A · Recalibración": "Cuando hay alerta, solo se reajusta la escala de las probabilidades (rápido y barato).",
    "B · Reentrenamiento": "Ante una alerta crítica, el modelo se vuelve a entrenar con los datos recientes (más caro).",
    "C · Híbrida": "Recalibra como regla; reentrena solo si el problema persiste o si pasó demasiado tiempo.",
    "Oráculo (in-sample)": "Referencia imposible en la práctica: usa las respuestas del mismo lote que evalúa. Sirve como techo.",
}
# Paleta de la vista ejecutiva (azul = real/observado, naranja = estimado por el modelo)
AZUL, NARANJA = "#2a78d6", "#eb6834"
INK2, AXIS = INK_2, "#c3c2b7"          # INK, MUTED y GRID se comparten con la vista técnica
ESTADO = {"bajo": "#0ca30c", "medio": "#fab219", "alto": "#ec835a", "critico": "#d03b3b", "info": "#5b6b7f"}
NIVEL_TXT = {"bajo": "Bajo", "medio": "Medio", "alto": "Alto", "critico": "Crítico"}
SEVERIDAD = {"Crítica": "critico", "Alta": "alto", "Media": "medio", "Informativa": "info"}
ORDEN_SEV = {"Crítica": 0, "Alta": 1, "Media": 2, "Informativa": 3}
BANDA_COLOR = {"Bajo": ESTADO["bajo"], "Medio": ESTADO["medio"], "Alto": ESTADO["critico"]}
FUENTE = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

POLITICAS_TXT = {
    "Sin calibrar (TT1)": "Modelo sin ajuste de probabilidades (enfoque de TT1)",
    "Baseline calibrado": "Ajuste inicial único, sin mantenimiento",
    "A · Recalibración": "Reajustar probabilidades cuando el monitor alerta",
    "B · Reentrenamiento": "Reentrenar el modelo ante alertas fuertes",
    "C · Híbrida": "Reajustar ante alertas y reentrenar periódicamente",
    "Oráculo (in-sample)": "Referencia teórica (usa información no disponible en operación)",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif !important; }}
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1380px; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; }} h2, h3 {{ font-weight: 600; }}
section[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid {RULE}; }}
section[data-testid="stSidebar"] .stRadio label p {{ font-size: 0.95rem; }}
/* Tarjetas KPI (st.metric) con borde definido */
div[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {RULE}; border-radius: 10px;
  padding: 14px 16px 12px 16px; box-shadow: 0 1px 0 rgba(0,0,0,0.02); min-height: 118px; }}
div[data-testid="stMetric"] label p {{ font-size: 0.78rem !important; color: {INK_2} !important; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.04em; }}
div[data-testid="stMetricValue"] {{ font-size: 1.65rem !important; color: {INK}; font-weight: 600; }}
div[data-testid="stMetricDelta"] {{ font-size: 0.82rem !important; }}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {{ white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; line-height: 1.25 !important; }}
div[data-testid="stMetricDelta"] div {{ white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
/* Pestañas */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 16px; font-weight: 500; color: {INK_2}; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT_DARK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; }}
/* Pestañas principales (primer nivel) más visibles que las internas */
div[data-testid="stTabs"] [data-testid="stTab"] p, div[data-testid="stTabs"] [data-baseweb="tab"] p {{
  font-size: 1.1rem !important; font-weight: 650 !important; }}
div[data-testid="stTabs"] div[data-testid="stTabs"] [data-testid="stTab"] p,
div[data-testid="stTabs"] div[data-testid="stTabs"] [data-baseweb="tab"] p {{ font-size: 0.93rem !important; font-weight: 500 !important; }}
/* Encabezado general y guía de lectura de cada gráfica */
.app-head {{ margin: 0 0 6px 0; }}
.app-head .t {{ font-size: 1.7rem; font-weight: 700; color: {INK}; line-height: 1.2; }}
.app-head .d {{ color: {INK_2}; font-size: 0.95rem; margin-top: 2px; }}
.lee {{ font-size: 0.86rem; color: {INK_2}; background: #f6f9fd; border-left: 3px solid {ACCENT}; border-radius: 6px;
  padding: 7px 11px; margin: 2px 0 8px 0; line-height: 1.45; }}
.lee b {{ color: {INK}; }}
/* Expanders y contenedores */
div[data-testid="stExpander"] {{ background: {SURFACE}; border: 1px solid {RULE}; border-radius: 10px; }}
div[data-testid="stExpander"] summary p {{ font-weight: 500; color: {INK_2}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {SURFACE}; border-radius: 10px; }}
/* Componentes propios */
.hero {{ background: {SURFACE}; border: 1px solid {RULE}; border-left: 5px solid {ACCENT}; border-radius: 10px;
  padding: 18px 22px; margin-bottom: 14px; }}
.hero .k {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; color: {ACCENT}; font-weight: 600; }}
.hero .t {{ font-size: 1.55rem; font-weight: 700; color: {INK}; margin: 2px 0 6px 0; }}
.hero .d {{ color: {INK_2}; font-size: 0.98rem; line-height: 1.5; }}
.hero .chips {{ margin-top: 10px; }}
.chip {{ display: inline-block; border: 1px solid {RULE}; background: {PAGE}; color: {INK_2}; border-radius: 999px;
  padding: 2px 10px; font-size: 0.78rem; margin: 2px 6px 2px 0; }}
.sec {{ margin: 22px 0 6px 0; }}
.sec .t {{ font-size: 1.12rem; font-weight: 600; color: {INK}; }}
.sec .d {{ color: {MUTED}; font-size: 0.9rem; }}
.take {{ background: {SURFACE}; border: 1px solid {RULE}; border-radius: 8px; padding: 8px 12px; margin: 4px 0 10px 0;
  font-size: 0.9rem; color: {INK_2}; }}
.pill {{ display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: 0.76rem; font-weight: 600; margin-right: 6px; }}
.pill.good {{ background: #e5f4e5; color: #0b6b0b; }} .pill.warn {{ background: #fff3d6; color: #7a5200; }}
.pill.bad {{ background: #fbe4e4; color: #9e2424; }} .pill.info {{ background: {ACCENT_SOFT}; color: {ACCENT_DARK}; }}
.guide p {{ margin: 0 0 6px 0; }}
.scorecard {{ width: 100%; border-collapse: collapse; background: {SURFACE}; border: 1px solid {RULE}; border-radius: 10px;
  overflow: hidden; font-size: 0.9rem; }}
.scorecard th {{ background: #f0efeb; color: {INK_2}; font-weight: 600; text-align: left; padding: 9px 12px; border-bottom: 1px solid {RULE}; }}
.scorecard td {{ padding: 9px 12px; border-bottom: 1px solid {GRID}; color: {INK}; vertical-align: top; }}
.scorecard td.num {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem; }}
.card {{ background: {SURFACE}; border: 1px solid {RULE}; border-radius: 10px; padding: 14px 16px; height: 100%; }}
.card .h {{ font-weight: 600; color: {INK}; margin-bottom: 4px; }}
.card .b {{ color: {INK_2}; font-size: 0.9rem; line-height: 1.45; }}
.log {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; background: #fbfbf9; border: 1px solid {RULE};
  border-radius: 8px; padding: 10px 12px; max-height: 460px; overflow-y: auto; }}
.log .row {{ padding: 3px 0; border-bottom: 1px dashed {GRID}; color: {INK}; }}
.log .tag {{ display: inline-block; min-width: 98px; font-weight: 600; }}
.foot {{ color: {MUTED}; font-size: 0.8rem; margin-top: 30px; border-top: 1px solid {RULE}; padding-top: 10px; }}
/* ---- Vista ejecutiva ---- */
:root {{ --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --line:rgba(11,11,11,.10); }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
.tt-head h1 {{ font-size: 1.65rem; font-weight: 700; margin: 0 0 .15rem 0; color: var(--ink); }}
.tt-head p {{ color: var(--ink2); margin: 0; font-size: .95rem; }}
.tt-sec {{ margin: 1.6rem 0 .55rem 0; display:flex; align-items:baseline; gap:.6rem; border-bottom:1px solid var(--line); padding-bottom:.35rem; }}
.tt-sec .num {{ font-size:.78rem; font-weight:700; color:var(--muted); letter-spacing:.06em; }}
.tt-sec .tit {{ font-size:1.12rem; font-weight:700; color:var(--ink); }}
.tt-sec .sub {{ font-size:.85rem; color:var(--ink2); margin-left:auto; }}
.tt-card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; height:100%; }}
.tt-resumen {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:6px 18px; }}
.tt-resumen .fila {{ display:grid; grid-template-columns: 170px 1fr; gap:14px; padding:10px 0; border-bottom:1px solid #f0efec; }}
.tt-resumen .fila:last-child {{ border-bottom:none; }}
.tt-resumen .lab {{ font-size:.8rem; font-weight:700; color:var(--ink2); text-transform:uppercase; letter-spacing:.04em; padding-top:2px; }}
.tt-resumen .txt {{ font-size:.98rem; color:var(--ink); line-height:1.45; }}
.tt-kpi {{ min-height: 238px; }}
.tt-kpi .nom {{ font-size:.85rem; font-weight:600; color:var(--ink2); display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }}
.tt-kpi .nom .tt-badge {{ flex-shrink:0; }}
.tt-kpi .val {{ font-size:1.85rem; font-weight:700; color:var(--ink); line-height:1.15; margin-top:6px; }}
.tt-kpi .val small {{ font-size:.9rem; font-weight:500; color:var(--ink2); }}
.tt-kpi .int {{ font-size:.88rem; color:var(--ink); margin-top:6px; line-height:1.4; }}
.tt-kpi .cmp {{ font-size:.8rem; color:var(--ink2); margin-top:8px; padding-top:8px; border-top:1px solid #f0efec; }}
.tt-badge {{ display:inline-flex; align-items:center; gap:6px; font-size:.74rem; font-weight:700; color:var(--ink);
            padding:2px 9px; border-radius:999px; border:1px solid var(--line); background:#fafaf8; white-space:nowrap; }}
.tt-dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.tt-alerta {{ background:#fff; border:1px solid var(--line); border-left-width:5px; border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.tt-alerta .tit {{ font-weight:700; color:var(--ink); font-size:.98rem; margin:4px 0 6px 0; }}
.tt-alerta .grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; }}
.tt-alerta .k {{ font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
.tt-alerta .v {{ font-size:.87rem; color:var(--ink); line-height:1.4; }}
.tt-rec {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:10px; }}
.tt-rec .acc {{ font-weight:700; font-size:1rem; color:var(--ink); margin:2px 0 8px 0; }}
.tt-rec .meta {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; font-size:.8rem; color:var(--ink2); }}
.tt-rec .grid {{ display:grid; grid-template-columns: 1.3fr 1fr 1fr; gap:14px; margin-top:8px; }}
.tt-rec .k {{ font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
.tt-rec .v {{ font-size:.87rem; color:var(--ink); line-height:1.4; }}
.tt-diag {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin-bottom:10px; }}
.tt-diag .q {{ font-weight:700; color:var(--ink); font-size:.95rem; margin-bottom:4px; }}
.tt-diag ul {{ margin:0 0 0 1.1rem; padding:0; }} .tt-diag li {{ font-size:.9rem; color:var(--ink); margin:3px 0; line-height:1.42; }}
.tt-gtit {{ font-weight:700; font-size:1rem; color:var(--ink); margin:2px 0 2px 0; line-height:1.35; }}
.tt-gnota {{ font-size:.82rem; color:var(--ink2); line-height:1.4; margin:2px 0; }}
.tt-gnota b {{ color:var(--ink); }}
.tt-filtros {{ font-size:.82rem; color:var(--ink2); margin:-.2rem 0 .4rem 0; }}
.tt-aviso {{ font-size:.85rem; color:var(--ink); background:#fff8e6; border:1px solid #f3dca0; border-radius:8px; padding:8px 12px; margin:6px 0; }}
.tt-nota {{ font-size:.8rem; color:var(--ink2); }}
.tt-ov {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px 18px; height:100%; }}
.tt-ov .t {{ font-weight:700; font-size:1.05rem; color:var(--ink); margin-bottom:2px; }}
.tt-ov .q {{ font-size:.85rem; color:var(--ink2); margin-bottom:10px; }}
.tt-ov .k {{ font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin-top:10px; }}
.tt-ov .v {{ font-size:.9rem; color:var(--ink); line-height:1.42; }}
@media (max-width: 900px) {{ .tt-alerta .grid, .tt-rec .grid {{ grid-template-columns: 1fr; }}
  .tt-resumen .fila {{ grid-template-columns: 1fr; gap:2px; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

pio.templates["ejecutivo"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, Segoe UI, Arial, sans-serif", size=13, color=INK),
    colorway=CAT, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    title=dict(font=dict(size=15, color=INK), x=0.0, xanchor="left", y=0.98, yanchor="top"),
    xaxis=dict(gridcolor=GRID, linecolor="#c9c8c2", zerolinecolor=GRID, ticks="outside", tickcolor="#c9c8c2",
               title=dict(font=dict(size=12, color=INK_2))),
    yaxis=dict(gridcolor=GRID, linecolor="#c9c8c2", zerolinecolor=GRID, ticks="outside", tickcolor="#c9c8c2",
               title=dict(font=dict(size=12, color=INK_2))),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    hoverlabel=dict(bgcolor="white", bordercolor=RULE, font=dict(color=INK, size=12)),
    margin=dict(l=56, r=20, t=92, b=48),
))
pio.templates.default = "ejecutivo"


# #####################################################################################################
# PESTAÑA 2 · MONITOREO, RESULTADOS Y DETALLES (vista técnica)
# #####################################################################################################
# =====================================================================================================
# 2. CAPA DE DATOS
# =====================================================================================================
# Paquete de tablas de las corridas finales (lo escribe la celda 4 del notebook junto a app.py).
DATA_PACK = os.environ.get("TT2_DATA_PACK", str(Path(__file__).resolve().parent / "tt2_data_pack.zip"))

DS = {
    "polish": dict(
        nombre="Polish Companies Bankruptcy", corto="Polish", sector="Riesgo financiero", icono="🏭",
        evento="quiebra", eventos="quiebras", fila="empresa", filas="empresas", pos="Quebró", neg="Siguió operando",
        unidad="lote", unidades="lotes", tam_unidad="500 empresas", col_t="bloque", tabla_t="politicas_bloques",
        proposito="Estimar la probabilidad de que una empresa manufacturera polaca quiebre, a partir de 64 indicadores "
                  "financieros de sus estados contables (rentabilidad, endeudamiento, liquidez, eficiencia).",
        industria="Bancos, aseguradoras y áreas de crédito: aprobar o rechazar financiamiento, fijar tasas de interés, "
                  "calcular reservas para pérdidas y vigilar la cartera de proveedores.",
        drift="La proporción de quiebras crece conforme las empresas se acercan a su desenlace: un modelo entrenado con "
              "empresas lejanas al evento subestima el riesgo de las cercanas.",
        costo_fn=10.0, costo_fp=1.0, tt1_interv=14),
    "airline": dict(
        nombre="Airline Delay", corto="Airline", sector="Transporte aéreo", icono="✈️",
        evento="retraso", eventos="retrasos", fila="vuelo", filas="vuelos", pos="Llegó con retraso", neg="Llegó a tiempo",
        unidad="día", unidades="días", tam_unidad="≈ 17 400 vuelos", col_t="día", tabla_t="politicas_dias",
        proposito="Anticipar qué vuelos comerciales en Estados Unidos llegarán con retraso, a partir de la aerolínea, la "
                  "ruta, la hora de salida, la duración y el historial reciente de retrasos de cada vuelo y aeropuerto.",
        industria="Aerolíneas y aeropuertos: asignar personal y puertas, proteger conexiones, avisar a pasajeros y "
                  "planear la operación del día.",
        drift="No cambian los vuelos sino cuántos se retrasan cada día (clima, congestión): como un restaurante con el "
              "mismo menú pero con días de mucha y de poca demanda.",
        costo_fn=1.0, costo_fp=1.0, tt1_interv=807),
    "diabetes": dict(
        nombre="Diabetes 130-US Hospitals", corto="Diabetes", sector="Salud", icono="🏥",
        evento="readmisión < 30 días", eventos="readmisiones", fila="paciente", filas="pacientes", pos="Regresó al hospital",
        neg="No regresó", unidad="lote", unidades="lotes", tam_unidad="1 000 pacientes", col_t="bloque",
        tabla_t="politicas_bloques",
        proposito="Identificar a los pacientes con diabetes que tienen mayor riesgo de volver a hospitalizarse en menos "
                  "de 30 días después de su alta, en una red de 130 hospitales de Estados Unidos.",
        industria="Hospitales y aseguradoras de salud: priorizar llamadas de seguimiento, citas tempranas y educación "
                  "al alta; reducir readmisiones evitables y sus penalizaciones.",
        drift="Cambia sobre todo la forma de registrar la información (aseguradora, especialidad) y, poco a poco, la tasa "
              "de readmisión.",
        costo_fn=5.0, costo_fp=1.0, tt1_interv=103),
}


@st.cache_data(show_spinner="Cargando evidencia de las corridas…")
def load_pack(path: str) -> dict:
    """Lee el paquete zip: <ds>/meta.json, <ds>/<tabla>.parquet y <ds>/x_*.parquet (extras)."""
    out = {ds: {"t": {}} for ds in DS}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            ds, fn = name.split("/", 1)
            raw = z.read(name)
            if fn == "meta.json":
                out[ds]["meta"] = json.loads(raw)
            elif fn == "x_info.json":
                out[ds]["info"] = json.loads(raw)
            elif fn.endswith(".parquet"):
                out[ds]["t"][fn[:-8]] = pd.read_parquet(io.BytesIO(raw))
    return out


def T(ds: str, name: str) -> pd.DataFrame:
    return PACK[ds]["t"].get(name, pd.DataFrame()).copy()


def M(ds: str) -> dict:
    return PACK[ds]["meta"]


def prod_row(ds: str, prefix: str) -> pd.Series:
    t = T(ds, "metricas_produccion")
    return t[t["modelo"].str.startswith(prefix)].iloc[0]


def valor_ic(x) -> float:
    """'0.9636 [0.95, 0.97]' -> 0.9636"""
    if isinstance(x, str):
        return float(x.split(" ")[0])
    return float(x)


def ci(x) -> str:
    return x.split(" ", 1)[1] if isinstance(x, str) and "[" in x else ""


def calibrated_label(ds: str, kind: str = "xgb") -> str:
    return f"{kind.upper()} · calibrada ({M(ds)['modelos'][kind]['calibrador']})"


def policy_summary(ds: str, model: str = "XGB") -> pd.DataFrame:
    t = T(ds, "politicas_resumen")
    return t[t["modelo"] == model].set_index("política")


def best_active(ds: str, model: str = "XGB") -> str:
    s = policy_summary(ds, model)
    return s.loc[["A · Recalibración", "B · Reentrenamiento", "C · Híbrida"], "LogLoss"].idxmin()


def timeline(ds: str, model: str = "XGB") -> pd.DataFrame:
    t = T(ds, DS[ds]["tabla_t"])
    t = t[t["modelo"] == model].rename(columns={DS[ds]["col_t"]: "t"})
    return t


def far_value(ds: str, col: str = "falsas_alarmas_nivel2_%") -> float:
    t = T(ds, "monitor_falsas_alarmas")
    return float(t[t["señal"].str.startswith("ALERTA")][col].iloc[0])


def latency(ds: str) -> pd.DataFrame:
    return T(ds, "monitor_latencia")


# =====================================================================================================
# 3. SEMÁFOROS Y CAPA NARRATIVA
# =====================================================================================================
def st_ece(v):
    return "good" if v <= 0.03 else ("warn" if v <= 0.05 else "bad")


def st_brier(v):
    return "good" if v <= 0.15 else ("warn" if v <= 0.20 else "bad")


def st_auc(v):
    return "good" if v >= 0.80 else ("warn" if v >= 0.65 else "bad")


def st_ap_norm(ap, prev):
    g = (ap - prev) / (1 - prev)
    return "good" if g >= 0.5 else ("warn" if g >= 0.2 else "bad")


def st_ks(v):
    return "good" if v >= 0.40 else ("warn" if v >= 0.20 else "bad")


def st_far(v):
    return "good" if v < 10 else ("warn" if v < 15 else "bad")


def st_lat(v):
    return "good" if v <= 500 else ("warn" if v <= 1000 else "bad")


def st_oe(v):
    return "good" if 0.9 <= v <= 1.1 else ("warn" if 0.8 <= v <= 1.25 else "bad")


def pill(status: str, text: str | None = None) -> str:
    return f'<span class="pill {status}">{EMOJI[status]} {text or WORD[status]}</span>'


def hero(kicker: str, title: str, desc: str, chips: list[str] | None = None):
    ch = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    st.markdown(f'<div class="hero"><div class="k">{kicker}</div><div class="t">{title}</div><div class="d">{desc}</div>'
                f'<div class="chips">{ch}</div></div>', unsafe_allow_html=True)


def section(title: str, desc: str = ""):
    st.markdown(f'<div class="sec"><div class="t">{title}</div><div class="d">{desc}</div></div>', unsafe_allow_html=True)


def takeaway(status: str, text: str):
    """Lectura de una línea bajo cada gráfica: semáforo + conclusión."""
    st.markdown(f'<div class="take">{pill(status)} {text}</div>', unsafe_allow_html=True)


def lectura(que: str, significa: str | None = None):
    """Guía breve ANTES de una gráfica: qué se está viendo y cómo interpretarlo, en una o dos frases."""
    html = f"<b>Qué muestra:</b> {que}"
    if significa:
        html += f"<br><b>Cómo interpretarla:</b> {significa}"
    st.markdown(f'<div class="lee">{html}</div>', unsafe_allow_html=True)


def guide(title: str, que: str, valor: str, status: str, juicio: str, decision: str, analogia: str | None = None,
          expanded: bool = False):
    """Capa narrativa estándar: las cuatro preguntas que pidieron los sinodales (+ analogía opcional)."""
    with st.expander(f"{EMOJI[status]}  Guía de lectura: {title}", expanded=expanded):
        html = (f'<div class="guide"><p><b>¿Qué mide?</b> {que}</p>'
                f'<p><b>¿Qué significa el valor actual?</b> {valor}</p>'
                f'<p><b>¿Es bueno, regular o malo?</b> {pill(status)} {juicio}</p>'
                f'<p><b>¿Qué decisión implica?</b> {decision}</p>')
        if analogia:
            html += f'<p><b>Analogía:</b> {analogia}</p>'
        st.markdown(html + "</div>", unsafe_allow_html=True)


def kpi(col, label: str, value: str, delta: str | None, status: str, help_: str, delta_color: str = "normal"):
    col.metric(f"{EMOJI[status]} {label}", value, delta=delta, delta_color=delta_color, help=help_)


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_pct(x, nd=1):
    return f"{100 * x:.{nd}f} %"


def technical() -> bool:
    return st.session_state.get("tecnico", False)


# =====================================================================================================
# 4. UTILIDADES DE GRÁFICAS
# =====================================================================================================
def show(fig, height: int | None = None, key: str | None = None):
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def table(df: pd.DataFrame, height: int | None = None):
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def download(df: pd.DataFrame, name: str, label: str = "Descargar CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=f"{name}.csv", mime="text/csv",
                       key=f"dl_{name}")


def bar_rate(df: pd.DataFrame, x: str, y: str, n: str | None = None, title: str = "", horizontal: bool = False,
             color: str = CAT[0], yfmt: str = ".0%", ref: float | None = None, ref_label: str = "promedio"):
    """Barras de una sola serie (tasa por categoría) con extremo redondeado y referencia opcional."""
    fig = go.Figure()
    hover = f"%{{{'y' if horizontal else 'x'}}}<br>tasa: %{{{'x' if horizontal else 'y'}:{yfmt}}}"
    if n:
        hover += "<br>casos: %{customdata:,}"
    kw = dict(marker=dict(color=color, cornerradius=4), customdata=df[n] if n else None, hovertemplate=hover + "<extra></extra>")
    if horizontal:
        fig.add_bar(x=df[y], y=df[x].astype(str), orientation="h", **kw)
        fig.update_yaxes(type="category")
        fig.update_xaxes(tickformat=yfmt)
        fig.update_yaxes(autorange="reversed")
        if ref is not None:
            fig.add_vline(x=ref, line_dash="dot", line_color=INK_2, annotation_text=ref_label, annotation_position="top")
    else:
        fig.add_bar(x=df[x].astype(str), y=df[y], **kw)
        fig.update_xaxes(type="category")
        fig.update_yaxes(tickformat=yfmt)
        if ref is not None:
            fig.add_hline(y=ref, line_dash="dot", line_color=INK_2, annotation_text=ref_label, annotation_position="top left")
    fig.update_layout(title=title, bargap=0.25, showlegend=False)
    return fig


def rate_by(df: pd.DataFrame, key: str, min_n: int = 1) -> pd.DataFrame:
    g = df.groupby(key, observed=True)["y"].agg(["mean", "size"]).reset_index().rename(columns={"mean": "tasa", "size": "casos"})
    return g[g["casos"] >= min_n]


def pct_clip(s: pd.Series, lo: float, hi: float) -> pd.Series:
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


# =====================================================================================================
# 5. MÓDULO · RESUMEN GENERAL
# =====================================================================================================
def module_overview():
    hero("Trabajo Terminal 2026-B080 · ESCOM-IPN · Resumen general",
         "¿Podemos confiar en las probabilidades que da un modelo a lo largo del tiempo?",
         "Un modelo de clasificación no solo responde <b>sí</b> o <b>no</b>: estima <b>qué tan probable</b> es un evento. "
         "Esta sección verifica que esas probabilidades sigan coincidiendo con la frecuencia real cuando los datos cambian "
         "(<i>calibration drift</i>) y compara qué conviene hacer cuando dejan de coincidir: recalibrar, reentrenar o combinar ambas.",
         ["3 conjuntos de datos reales", "2 modelos (regresión logística y XGBoost)", "6 estrategias de mantenimiento",
          "4 criterios cuantitativos del protocolo", "Cifras de las corridas finales"])

    with st.container(border=True):
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("#### La idea en un minuto")
            st.markdown(
                "Piensa en el **pronóstico del clima**. Si cada vez que anuncia *70 % de probabilidad de lluvia* llueve en "
                "7 de cada 10 de esos días, el pronóstico está **calibrado** y puedes decidir con él si llevas paraguas. Si "
                "con el tiempo llueve solo 4 de cada 10 veces, el pronóstico **se descalibró**: ya no puedes tomar sus números "
                "al pie de la letra, aunque siga distinguiendo bien los días lluviosos de los secos.\n\n"
                "Con los modelos pasa igual: una empresa con *8 % de riesgo de quiebra*, un vuelo con *60 % de probabilidad de "
                "retraso* o un paciente con *15 % de riesgo de reingreso* solo sirven para decidir si esos porcentajes se "
                "cumplen en la realidad.")
        with c2:
            x = np.linspace(0.02, 0.98, 12)
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1.5),
                            name="calibración perfecta", hoverinfo="skip")
            fig.add_scatter(x=x, y=x * 0.98 + 0.01, mode="lines+markers", line=dict(color=CAT[0], width=2),
                            marker=dict(size=8), name="modelo calibrado",
                            hovertemplate="estima %{x:.0%} · ocurre %{y:.0%}<extra></extra>")
            fig.add_scatter(x=x, y=np.clip(x * 0.62, 0, 1), mode="lines+markers", line=dict(color=CAT[1], width=2, dash="dash"),
                            marker=dict(size=8, symbol="triangle-up"), name="modelo descalibrado (sobreestima el riesgo)",
                            hovertemplate="estima %{x:.0%} · ocurre %{y:.0%}<extra></extra>")
            fig.update_layout(title="Probabilidad estimada vs. frecuencia real (ejemplo ilustrativo)",
                              xaxis=dict(title="probabilidad estimada por el modelo", tickformat=".0%", range=[0, 1]),
                              yaxis=dict(title="frecuencia real", tickformat=".0%", range=[0, 1]))
            lectura("cada punto agrupa casos con la misma probabilidad estimada (eje horizontal) y marca con qué frecuencia "
                    "ocurrió el evento en ellos (eje vertical).",
                    "sobre la diagonal punteada, el modelo está calibrado; por debajo, sobreestima el riesgo; por encima, lo subestima.")
            show(fig, 330, key="ov_ilus")

    section("Cumplimiento de los criterios del protocolo",
            "Los cuatro criterios cuantitativos del protocolo, más la capacidad de discriminación, para el modelo recomendado "
            "(XGBoost calibrado) en cada conjunto de datos.")
    rows, status_count = [], {"good": 0, "warn": 0, "bad": 0}
    crit = [
        ("Error de calibración (ECE ≤ 0.05)",
         "Diferencia promedio entre la probabilidad estimada y la frecuencia observada del evento.",
         lambda ds: valor_ic(prod_row(ds, calibrated_label(ds))["ECE"]), st_ece, lambda v: f"{v:.4f}"),
        ("Puntaje de Brier (≤ 0.20)", "Error cuadrático medio de las probabilidades (0 = pronóstico perfecto).",
         lambda ds: valor_ic(prod_row(ds, calibrated_label(ds))["Brier"]), st_brier, lambda v: f"{v:.4f}"),
        ("Tasa de falsas alarmas (< 15 %)",
         "Porcentaje de lotes sin cambio real en los que el monitor emitió una alerta crítica.",
         lambda ds: far_value(ds), st_far, lambda v: f"{v:.1f} %"),
        ("Volumen de observaciones para detección (≤ 500)",
         "Observaciones nuevas que el monitor procesa, desde que inicia un cambio fuerte, hasta emitir la primera alerta.",
         lambda ds: float(latency(ds)["latencia_mediana_N1"].min()), st_lat, lambda v: f"{v:,.0f} obs."),
        ("Capacidad de discriminación (AUC)",
         "Probabilidad de que un caso con evento reciba mayor riesgo que un caso sin evento (0.5 = azar).",
         lambda ds: valor_ic(prod_row(ds, calibrated_label(ds))["AUC"]), st_auc, lambda v: f"{v:.3f}"),
    ]
    head = "<tr><th>Criterio</th><th>Qué significa</th>" + "".join(
        f"<th>{DS[d]['icono']} {DS[d]['corto']}</th>" for d in DS) + "</tr>"
    for name, desc, getter, fn, f in crit:
        cells = ""
        for d in DS:
            v = getter(d)
            s = fn(v)
            status_count[s] += 1
            cells += f'<td class="num">{EMOJI[s]} {f(v)}</td>'
        rows.append(f"<tr><td><b>{name}</b></td><td>{desc}</td>{cells}</tr>")
    st.markdown(f'<table class="scorecard">{head}{"".join(rows)}</table>', unsafe_allow_html=True)
    takeaway("good" if status_count["bad"] <= 3 else "warn",
             f"{status_count['good']} indicadores en verde, {status_count['warn']} en amarillo y {status_count['bad']} en rojo. "
             "La calibración se cumple en los tres conjuntos; el volumen de observaciones para detección y la capacidad de "
             "discriminación en Diabetes son las áreas de mejora identificadas.")
    guide("cumplimiento de criterios",
          "Cinco preguntas de control por conjunto de datos: ¿las probabilidades coinciden con la realidad (ECE y Brier)?, ¿el "
          "monitor evita alertas sin cambio real?, ¿cuántas observaciones necesita para detectar un cambio? y ¿el modelo "
          "distingue los casos con evento de los casos sin evento?",
          "Cada celda muestra el valor medido en producción (el 30 % más reciente de cada conjunto) con su semáforo.",
          "info",
          "🟢 cumple con holgura · 🟡 cumple pero cerca del límite, o no cumple por poco · 🔴 no cumple o es débil. El Brier "
          "de Airline no se compara en igualdad: con la mitad de los vuelos retrasados, hasta un pronóstico perfecto de la "
          "tasa promedio obtiene 0.25.",
          "Los criterios en verde sostienen que el sistema puede usarse para decidir con probabilidades. Los amarillos y "
          "rojos son el plan de trabajo: detectar cambios con menos observaciones, sin esperar a conocer el resultado real "
          "de cada caso, y enriquecer los datos clínicos (Diabetes).",
          "Es la boleta de calificaciones del sistema: no todas las materias tienen la misma escala, por eso cada una "
          "trae su propio criterio de aprobación.")

    section("¿Cuánto mejoró la calibración y cuánto cuesta mantenerla?")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        series = [("Sin calibrar (TT1)", NEUTRAL), ("Baseline calibrado", CAT[0]), ("Mejor estrategia de mantenimiento", CAT[3])]
        for i, (lab, col) in enumerate(series):
            ys = []
            for d in DS:
                s = policy_summary(d)
                ys.append(s.loc[best_active(d), "ECE"] if lab.startswith("Mejor") else s.loc[lab, "ECE"])
            fig.add_bar(x=[DS[d]["corto"] for d in DS], y=ys, name=lab, marker=dict(color=col, cornerradius=4),
                        text=[f"{v:.3f}" for v in ys], textposition="outside", textfont=dict(size=11, color=INK_2),
                        hovertemplate="%{x}<br>" + lab + ": %{y:.4f}<extra></extra>")
        fig.add_hline(y=0.05, line_dash="dot", line_color=INK_2, annotation_text="criterio 0.05", annotation_position="top right")
        fig.update_layout(title="Error de calibración (ECE) en producción · menor es mejor", barmode="group", bargap=0.28,
                          bargroupgap=0.08, yaxis=dict(title="ECE"))
        lectura("el error de calibración de cada conjunto en tres escenarios: modelo sin calibrar (TT1), calibrado una sola "
                "vez y calibrado con la mejor estrategia de mantenimiento.",
                "barras más bajas = probabilidades más cercanas a la realidad; debajo de la línea de 0.05 se cumple el criterio.")
        show(fig, 380, key="ov_ece")
        takeaway("good", "Calibrar desde el inicio reduce el error entre 2 y 4 veces; la mejor política lo reduce aún más.")
    with c2:
        fig = go.Figure()
        tt1 = [DS[d]["tt1_interv"] for d in DS]
        tt2 = []
        for d in DS:
            s = policy_summary(d).loc[best_active(d)]
            tt2.append(int(s["recalibraciones"] + s["reentrenamientos"]))
        fig.add_bar(x=[DS[d]["corto"] for d in DS], y=tt1, name="TT1 (intervenciones)", marker=dict(color=NEUTRAL, cornerradius=4),
                    text=tt1, textposition="outside", hovertemplate="%{x}<br>TT1: %{y}<extra></extra>")
        fig.add_bar(x=[DS[d]["corto"] for d in DS], y=tt2, name="TT2 · mejor política", marker=dict(color=CAT[0], cornerradius=4),
                    text=tt2, textposition="outside", hovertemplate="%{x}<br>TT2: %{y}<extra></extra>")
        fig.update_layout(title="Intervenciones al modelo en producción (escala logarítmica)", barmode="group",
                          yaxis=dict(type="log", title="intervenciones"), bargap=0.28)
        lectura("cuántas veces hubo que recalibrar o reentrenar el modelo durante producción en TT1 y con la mejor estrategia de TT2.",
                "cada marca del eje vertical multiplica por 10; menos intervenciones = menor costo de cómputo y de revisión humana.")
        show(fig, 380, key="ov_int")
        takeaway("good", "Con umbrales de alerta bien calculados, el sistema pasa de cientos de intervenciones a menos de diez.")
    guide("gráficas de mejora",
          "Izquierda: el error de calibración en tres escenarios (sin ajuste, con ajuste inicial y con la mejor estrategia de "
          "mantenimiento). Derecha: cuántas veces hubo que intervenir el modelo en la fase anterior (TT1) y ahora.",
          "Las barras azules y amarillas quedan muy por debajo de la línea de 0.05; las intervenciones bajan de 807 a 9 en "
          "Airline y de 103 a 5 en Diabetes.", "good",
          "Bueno: se cumple el criterio y el mantenimiento es mucho más barato.",
          "Adoptar la calibración inicial como práctica estándar y reservar el reentrenamiento para cuando el monitor lo "
          "justifique: menos horas de cómputo y de revisión humana.",
          "Antes se llevaba el coche al taller cada vez que se encendía una luz del tablero, aunque fuera un falso contacto; "
          "ahora se revisa solo cuando la falla es real.")

    section("¿Qué estrategia de mantenimiento conviene en cada conjunto?")
    cols = st.columns(3)
    recs = {
        "polish": ("C · Híbrida", "El riesgo cambia conforme las empresas se acercan al desenlace y el modelo aprende de los "
                                  "nuevos casos: conviene recalibrar como regla y reentrenar de forma periódica."),
        "airline": ("C · Híbrida", "La tasa de retrasos cambia todos los días por clima y congestión; solo la combinación de "
                                   "recalibración y reentrenamiento cada tres días mejora al modelo de referencia."),
        "diabetes": ("A · Recalibración", "El cambio es sobre todo de proporción y de captura; recalibrar logra casi toda la "
                                          "mejora con el menor costo (fracciones de segundo)."),
    }
    for c, d in zip(cols, DS):
        pol, why = recs[d]
        s = policy_summary(d).loc[pol]
        c.markdown(f'<div class="card"><div class="h">{DS[d]["icono"]} {DS[d]["nombre"]}</div>'
                   f'<div class="b"><span class="pill info">Recomendación: {pol}</span><br><br>{why}<br><br>'
                   f'ECE con esta política: <b>{s["ECE"]:.4f}</b> · intervenciones: <b>{int(s["recalibraciones"] + s["reentrenamientos"])}</b>'
                   f' · cómputo: <b>{s["seg_en_intervenciones"]:.1f} s</b></div></div>', unsafe_allow_html=True)

    section("Cómo usar esta pestaña según tu perfil")
    cols = st.columns(3)
    prof = [("Dirección y áreas de negocio", "Empieza por la pestaña <b>Resumen Ejecutivo</b>. Aquí, lee el <b>Resumen</b> "
             "de cada conjunto: seis indicadores con semáforo y las guías de lectura. No se requieren conocimientos técnicos."),
            ("Analista de negocio o de operación", "Usa <b>Exploración</b> con los filtros para ver qué segmentos tienen "
             "más riesgo, y <b>Modelo</b> para ajustar el umbral de alerta según el costo de cada tipo de error."),
            ("Ingeniería de datos y de ML", "Revisa <b>Deriva y estrategias</b> y la <b>Bitácora de alertas</b>; activa "
             "«Mostrar detalle técnico» (arriba) para ver benchmark, ablaciones y tiempos.")]
    for c, (h, b) in zip(cols, prof):
        c.markdown(f'<div class="card"><div class="h">{h}</div><div class="b">{b}</div></div>', unsafe_allow_html=True)

    with st.expander("📘 Glosario en lenguaje claro"):
        gl = [("Calibración", "Que las probabilidades coincidan con la realidad: de 100 casos con 20 % de riesgo, el evento ocurre en unos 20."),
              ("Calibration drift (deriva de calibración)", "Pérdida gradual de esa coincidencia porque los datos cambiaron después del entrenamiento."),
              ("ECE (error de calibración esperado)", "Diferencia promedio entre la probabilidad estimada y la frecuencia observada. 0 es perfecto; el criterio es ≤ 0.05."),
              ("Brier", "Error cuadrático medio de las probabilidades; 0 es perfecto. Su valor depende de qué tan frecuente es el evento."),
              ("AUC", "Probabilidad de que el modelo asigne más riesgo a un caso con evento que a uno sin evento. 0.5 = azar; 1 = perfecto."),
              ("AP (precisión promedio)", "Proporción de alertas correctas promediada sobre todos los umbrales posibles; su referencia es la tasa base del evento."),
              ("Umbral de alerta", "Probabilidad a partir de la cual se emite una alerta. Se ajusta según el costo de cada tipo de error."),
              ("Recalibrar", "Ajustar la escala de las probabilidades sin cambiar lo aprendido por el modelo. Como ajustar una báscula."),
              ("Reentrenar", "Volver a entrenar el modelo con datos recientes. Como cambiar la báscula por una nueva."),
              ("PSI (índice de estabilidad poblacional)", "Cuánto cambió una distribución respecto de la de referencia. < 0.10 estable; > 0.25 cambio fuerte."),
              ("Falsa alarma", "Alerta emitida cuando en realidad no hubo un cambio."),
              ("Volumen de observaciones para detección", "Número de observaciones nuevas que procesa el monitor, desde que inicia un cambio real, "
               "hasta emitir la primera alerta. Se expresa en observaciones (casos), no en unidades de tiempo."),
              ("pp (puntos porcentuales)", "Diferencia entre dos porcentajes: de 20 % a 23 % hay 3 pp.")]
        table(pd.DataFrame(gl, columns=["Término", "Significado"]))


# =====================================================================================================
# 6. MÓDULO POR DATASET
# =====================================================================================================
def module_dataset(ds: str):
    info = DS[ds]
    meta = M(ds)
    part = meta["particion"]
    hero(f"{info['icono']} {info['sector']} · {info['nombre']}", f"¿Qué {info['fila']} tiene más riesgo de {info['evento']}?",
         f"<b>Propósito.</b> {info['proposito']}<br><b>Aplicación en la industria.</b> {info['industria']}<br>"
         f"<b>Qué cambia con el tiempo.</b> {info['drift']}",
         [f"Modelo recomendado: {meta['modelo_recomendado'].upper()} + calibrador {meta['modelos']['xgb']['calibrador']}",
          f"Producción evaluada: {sum(v.get('filas', v.get('vuelos', v.get('encuentros', 0))) for k, v in part.items() if k == 'production'):,} {info['filas']}",
          f"Unidad de monitoreo: {info['unidad']} ({info['tam_unidad']})", f"Corrida: {meta['run_id']}"])
    tabs = st.tabs(["📌 Resumen", "🔎 Exploración (EDA)", "🤖 Modelo predictivo", "📈 Deriva y estrategias", "🚨 Bitácora de alertas"])
    with tabs[0]:
        tab_summary(ds)
    with tabs[1]:
        {"polish": eda_polish, "airline": eda_airline, "diabetes": eda_diabetes}[ds](ds)
    with tabs[2]:
        tab_model(ds)
    with tabs[3]:
        tab_drift(ds)
    with tabs[4]:
        tab_alerts(ds)


# ------------------------------------------------------------------------------------------- Resumen
def tab_summary(ds: str):
    info = DS[ds]
    r = prod_row(ds, calibrated_label(ds))
    r0 = prod_row(ds, "XGB · sin calibrar")
    prev = float(np.mean([v for v in [M(ds)["particion"]["production"].get(k) for k in
                                       ("prevalencia", "tasa_retraso", "tasa_readmision_30d")] if v is not None]))
    auc, ece, brier, bss = valor_ic(r["AUC"]), valor_ic(r["ECE"]), valor_ic(r["Brier"]), float(r["BSS"])
    ap = valor_ic(r["AP"]) if "AP" in r.index and pd.notna(r["AP"]) else np.nan
    fa, lat = far_value(ds), float(latency(ds)["latencia_mediana_N1"].min())

    section("Indicadores principales", f"Medidos sobre los {info['filas']} más recientes (producción), que el modelo no vio "
                                       "durante su entrenamiento.")
    c = st.columns(3) + st.columns(3)
    kpi(c[0], "Capacidad de discriminación (AUC)", f"{auc:.3f}", f"+{auc - 0.5:.3f} sobre el azar", st_auc(auc),
        "0.5 = equivale a lanzar una moneda; 1 = ordena todos los casos sin error.")
    kpi(c[1], "Precisión promedio (AP)", f"{ap:.3f}" if ap == ap else "—", f"tasa base {prev:.1%}", st_ap_norm(ap, prev) if ap == ap else "info",
        "Proporción de alertas correctas promediada sobre todos los umbrales. Su referencia es la tasa base del evento.", "off")
    kpi(c[2], "Error de calibración (ECE)", f"{ece:.4f}", f"{ece - valor_ic(r0['ECE']):+.4f} vs. sin calibrar", st_ece(ece),
        "Diferencia promedio entre la probabilidad estimada y la frecuencia observada. Criterio ≤ 0.05.", delta_color="inverse")
    kpi(c[3], "Puntaje de Brier", f"{brier:.4f}", f"criterio ≤ 0.20 · BSS {bss:.2f}", st_brier(brier),
        "Error cuadrático medio de las probabilidades; BSS = mejora frente a predecir siempre la tasa promedio.", delta_color="off")
    kpi(c[4], "Tasa de falsas alarmas críticas", f"{fa:.1f} %", f"{fa - 15:+.1f} pp vs. criterio 15 %", st_far(fa),
        "Porcentaje de lotes sin cambio real en los que el monitor emitió una alerta crítica.", delta_color="inverse")
    kpi(c[5], "Volumen para detección", f"{lat:,.0f} obs.", "observaciones · criterio ≤ 500", st_lat(lat),
        "Observaciones nuevas que el monitor procesa, desde que inicia un cambio fuerte en la tasa del evento, hasta emitir "
        "la primera alerta.", delta_color="off")

    with st.expander("🧭 ¿Cómo leer estos seis indicadores? (guía para cualquier perfil)"):
        g = [
            ("AUC", "Si se toma al azar un caso que sí tuvo el evento y uno que no, es la probabilidad de que el modelo le asigne "
                    "más riesgo al primero.", f"{auc:.3f}: ordena correctamente {auc:.0%} de las parejas.", st_auc(auc),
             "Define si vale la pena usar el modelo para priorizar. Por debajo de 0.65 el orden es poco confiable.",
             "Como un catador que distingue el vino bueno del malo en 9 de cada 10 pruebas a ciegas."),
            ("AP", "Qué proporción de las alertas es correcta (de los casos marcados como riesgosos, cuántos tuvieron el evento), "
                   "promediada sobre todos los umbrales.", f"{ap:.3f} frente a una tasa base de {prev:.1%}.", st_ap_norm(ap, prev) if ap == ap else "info",
             "Si es baja, cada alerta real viene acompañada de muchas falsas: hay que dimensionar al equipo que las revisa.",
             "Como un detector de metales en la playa: AP alta significa que casi siempre que suena hay algo enterrado."),
            ("ECE", "Diferencia promedio entre la probabilidad que estima el modelo y la frecuencia con que ocurre el evento.",
             f"{ece:.4f}: si el modelo estima 10 %, la frecuencia real difiere en promedio {ece * 100:.1f} pp.", st_ece(ece),
             "Con ECE bajo los porcentajes pueden usarse directamente para presupuestar, reservar o asignar recursos.",
             "Como una báscula: un ECE de 0.01 es una báscula que se equivoca por 10 gramos en un kilo."),
            ("Brier", "Error cuadrático entre la probabilidad y el resultado (1 si ocurrió, 0 si no).",
             f"{brier:.4f}; la mejora sobre predecir siempre el promedio (BSS) es {bss:.0%}.", st_brier(brier),
             "Resume la calidad global de las probabilidades; compárese entre modelos del mismo conjunto, no entre conjuntos.",
             "Como calificar un pronóstico del clima: castiga más decir 90 % de lluvia en un día soleado que decir 60 %."),
            ("Falsas alarmas", "Porcentaje de lotes sin cambio real en los que el monitor emitió una alerta crítica.",
             f"{fa:.1f} % de los lotes sin cambio.", st_far(fa),
             "Cada falsa alarma cuesta horas de revisión; por debajo de 15 % el monitor es sostenible.",
             "Como la alarma de un coche: si suena con cada camión que pasa, nadie le hace caso."),
            ("Volumen para detección", "Número de observaciones nuevas que el monitor procesa, desde que inicia un cambio real, "
                                       "hasta emitir la primera alerta.",
             f"{lat:,.0f} observaciones en el mejor escenario.", st_lat(lat),
             "Mientras mayor sea, más decisiones se toman con probabilidades desactualizadas. 1 000 observaciones equivalen a "
             "detectar al cierre del segundo lote, porque el resultado real de cada lote se conoce un lote después.",
             "Como el detector de humo: importa que suene, pero también que suene pronto."),
        ]
        for name, que, val, s, dec, an in g:
            st.markdown(f"**{EMOJI[s]} {name}** — {que}  \n*Valor actual:* {val} · *Evaluación:* {pill(s)}  \n*Decisión:* {dec}  \n"
                        f"*Analogía:* {an}", unsafe_allow_html=True)
            st.markdown("")

    section("De dónde vienen los datos", "Partición temporal: el modelo aprende del periodo más antiguo, ajusta sus "
                                         "probabilidades con un periodo intermedio y se evalúa en el periodo más reciente.")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        p = M(ds)["particion"]
        names = {"train": "Entrenamiento (aprende)", "calibration": "Calibración (ajusta probabilidades)",
                 "production": "Producción (se evalúa)"}
        rows = []
        for k in ["train", "calibration", "production"]:
            v = p[k]
            n = v.get("filas", v.get("vuelos", v.get("encuentros")))
            rate = v.get("prevalencia", v.get("tasa_retraso", v.get("tasa_readmision_30d")))
            rows.append({"etapa": names[k], "casos": n, "tasa": rate})
        dfp = pd.DataFrame(rows)
        fig = go.Figure()
        for i, rr in dfp.iterrows():
            fig.add_bar(y=["casos"], x=[rr["casos"]], orientation="h", name=rr["etapa"],
                        marker=dict(color=[CAT[0], CAT[2], CAT[1]][i], line=dict(color=SURFACE, width=2)),
                        text=f"{rr['casos']:,}", textposition="inside", insidetextanchor="middle",
                        textfont=dict(color="white"), hovertemplate=f"{rr['etapa']}<br>%{{x:,}} {info['filas']}<extra></extra>")
        fig.update_layout(barmode="stack", legend=dict(traceorder="normal"), title=f"{info['filas'].capitalize()} por etapa (en orden temporal)",
                          yaxis=dict(visible=False), xaxis=dict(title=info["filas"]), height=220, margin=dict(t=70, b=40))
        lectura(f"cuántos {info['filas']} se usaron para entrenar, para calibrar y para evaluar (de izquierda a derecha, del más "
                "antiguo al más reciente).")
        show(fig, key=f"{ds}_part")
    with c2:
        fig = bar_rate(dfp.assign(etapa=dfp["etapa"].str.split(" ").str[0]), "etapa", "tasa", "casos",
                       title=f"Tasa de {info['evento']} por etapa", yfmt=".1%", color=CAT[1])
        lectura(f"el porcentaje de {info['filas']} con {info['evento']} en cada etapa.")
        show(fig, 220, key=f"{ds}_part_rate")
    t0, t2 = rows[0]["tasa"], rows[2]["tasa"]
    s = "warn" if abs(t2 / t0 - 1) > 0.15 else "good"
    takeaway(s, f"La tasa de {info['evento']} pasa de {t0:.1%} en entrenamiento a {t2:.1%} en producción "
                f"({(t2 / t0 - 1):+.0%}). Este cambio de proporción es justamente lo que descalibra a un modelo que no se vigila.")

    section("Validación del dataset", "Requisitos medibles que el dataset debe cumplir para estudiar calibration drift.")
    v = T(ds, "validacion_dataset")
    last = v.columns[-1]
    vv = v[["requisito", "criterio", last]].rename(columns={last: "resultado (versión corregida)"})
    ok = vv["resultado (versión corregida)"].str.contains("✅").sum()
    warn_ = vv["resultado (versión corregida)"].str.contains("⚠").sum()
    takeaway("good" if ok + warn_ == len(vv) else "warn",
             f"{ok} de {len(vv)} requisitos se cumplen y {warn_} con advertencia: el dataset es apto para el estudio.")
    with st.expander("Ver los requisitos uno por uno"):
        table(vv)


# ------------------------------------------------------------------------------------------- Modelo
def tab_model(ds: str):
    info = DS[ds]
    meta = M(ds)
    x = PACK[ds]["info"]
    r = prod_row(ds, calibrated_label(ds))
    section("Ficha del modelo")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            f'<div class="card"><div class="h">XGBoost con {meta["xgb_n_arboles"]} árboles + calibrador '
            f'{meta["modelos"]["xgb"]["calibrador"]}</div><div class="b">'
            f"<b>Qué es, en términos simples:</b> un conjunto de {meta['xgb_n_arboles']} árboles de decisión pequeños. Cada árbol hace "
            f"preguntas simples (¿el endeudamiento es alto?, ¿el vuelo sale de noche?) y corrige los errores de los anteriores; "
            f"la suma de sus respuestas se convierte en una probabilidad, y el <b>calibrador</b> ajusta esa escala para que los "
            f"porcentajes coincidan con la frecuencia real.<br><br>"
            f"<b>Por qué este modelo:</b> fue el de menor error probabilístico (log-loss) frente a regresión logística, Random "
            f"Forest, HistGradientBoosting y una red neuronal, maneja datos faltantes y se reentrena en segundos.<br><br>"
            f"<b>Umbral de alerta:</b> {x['umbral']:.3f} — elegido en el periodo de calibración "
            f"({'priorizando no dejar pasar eventos (F2)' if ds != 'airline' else 'equilibrando aciertos y falsas alarmas (Youden)'}).</div></div>",
            unsafe_allow_html=True)
    with c2:
        cc = st.columns(2)
        kpi(cc[0], "AUC", f"{valor_ic(r['AUC']):.3f}", ci(r["AUC"]) or None, st_auc(valor_ic(r["AUC"])),
            "Capacidad de discriminación: ordenar los casos con evento por encima de los casos sin evento.", "off")
        kpi(cc[1], "ECE", f"{valor_ic(r['ECE']):.4f}", ci(r["ECE"]) or None, st_ece(valor_ic(r["ECE"])),
            "Error de calibración: diferencia promedio entre probabilidad estimada y frecuencia observada.", "off")
        cc = st.columns(2)
        oe = float(r["O/E"])
        kpi(cc[0], "Observados / esperados (O/E)", f"{oe:.2f}", "ideal 1.00", st_oe(oe),
            "Eventos observados entre eventos esperados por el modelo. < 1: sobreestima el riesgo; > 1: lo subestima.", "off")
        kpi(cc[1], "Separación (KS)", f"{float(r['KS']):.3f}", "0 = nula · 1 = total", st_ks(float(r["KS"])),
            "Distancia máxima entre las distribuciones de riesgo de los casos con y sin evento (estándar en crédito).", "off")
        st.caption("Rangos entre corchetes: intervalo de confianza al 95 % por bootstrap.")

    # ---------------- Matriz de confusión interactiva
    section("Matriz de confusión explicada paso a paso",
            "Ajusta el umbral, es decir, la probabilidad a partir de la cual el sistema emite una alerta. Las cifras corresponden a "
            "todo el periodo de producción.")
    grid = T(ds, "x_umbrales")
    cA, cB, cC = st.columns([1.3, 1, 1])
    use_rec = cA.toggle(f"Usar el umbral recomendado ({x['umbral']:.3f})", value=True, key=f"{ds}_rec")
    thr = cA.slider("Umbral de alerta (probabilidad)", 0.0, 1.0, float(round(x["umbral"], 2)), 0.005, key=f"{ds}_thr",
                    disabled=use_rec, help="Por encima de este valor el caso se marca como riesgoso.")
    if use_rec:
        thr = float(x["umbral"])
    cfn = cB.number_input(f"Costo de NO detectar un(a) {info['evento']}", 0.0, 1000.0, info["costo_fn"], 1.0, key=f"{ds}_cfn")
    cfp = cC.number_input("Costo de una falsa alarma", 0.0, 1000.0, info["costo_fp"], 1.0, key=f"{ds}_cfp")
    row = grid.iloc[(grid["umbral"] - thr).abs().argmin()]
    TP, FP, FN, TN = int(row["TP"]), int(row["FP"]), int(row["FN"]), int(row["TN"])
    P, N = TP + FN, TN + FP
    rec = TP / P if P else 0
    prec = TP / (TP + FP) if TP + FP else 0
    spec = TN / N if N else 0
    alert = (TP + FP) / (P + N)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        z = [[TN, FP], [FN, TP]]
        labels = [[f"<b>Tranquilos correctos</b><br>{TN:,}", f"<b>Falsas alarmas</b><br>{FP:,}"],
                  [f"<b>Omisiones</b><br>{FN:,}", f"<b>Detecciones</b><br>{TP:,}"]]
        colors = [[0, 1], [1, 0]]      # 0 = acierto (azul pálido), 1 = error (naranja pálido)
        fig = go.Figure(go.Heatmap(z=colors, x=[f"Predicho: {info['neg'].lower()}", f"Predicho: {info['pos'].lower()}"],
                                   y=[f"Real: {info['neg'].lower()}", f"Real: {info['pos'].lower()}"],
                                   colorscale=[[0, "#e3eefb"], [1, "#fde3d8"]], showscale=False, zmin=0, zmax=1,
                                   text=labels, texttemplate="%{text}", textfont=dict(size=14, color=INK),
                                   hovertemplate="%{y}<br>%{x}<extra></extra>", xgap=3, ygap=3))
        fig.update_layout(title=f"Umbral {row['umbral']:.3f} · {alert:.1%} de los {info['filas']} reciben alerta",
                          yaxis=dict(autorange="reversed"), margin=dict(l=150, t=60))
        lectura("cruza lo que predijo el sistema (columnas) con lo que ocurrió (renglones). Azul = aciertos; naranja = errores.",
                "subir el umbral reduce falsas alarmas pero aumenta omisiones; bajarlo hace lo contrario.")
        show(fig, 340, key=f"{ds}_cm")
    with c2:
        st.markdown(
            f"**Paso 1 · Los que sí tuvieron {info['evento']} ({P:,}).** El sistema detectó a **{TP:,}** y dejó pasar a "
            f"**{FN:,}**. De cada 100 casos reales detecta **{rec * 100:.0f}** (*sensibilidad* o *recall*).\n\n"
            f"**Paso 2 · Los que no lo tuvieron ({N:,}).** Dejó tranquilos a **{TN:,}** y alertó por error a **{FP:,}**. "
            f"De cada 100 casos sin evento, deja tranquilos a **{spec * 100:.0f}** (*especificidad*).\n\n"
            f"**Paso 3 · Calidad de las alertas.** De cada 100 alertas, **{prec * 100:.0f}** son reales (*precisión*).\n\n"
            f"**Paso 4 · Costo.** Con los costos capturados, este umbral cuesta "
            f"**{(FN * cfn + FP * cfp):,.0f}** unidades ({FN:,} omisiones × {cfn:g} + {FP:,} falsas alarmas × {cfp:g}).")
        k = st.columns(4)
        k[0].metric("Sensibilidad", f"{rec:.0%}")
        k[1].metric("Precisión", f"{prec:.0%}")
        k[2].metric("Especificidad", f"{spec:.0%}")
        k[3].metric("Alertados", f"{alert:.0%}")
    gg = grid.assign(costo=grid["FN"] * cfn + grid["FP"] * cfp)
    best = gg.loc[gg["costo"].idxmin()]
    fig = go.Figure()
    fig.add_scatter(x=gg["umbral"], y=gg["costo"], mode="lines", line=dict(color=CAT[0], width=2), name="costo total",
                    hovertemplate="umbral %{x:.3f}<br>costo %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=[row["umbral"]], y=[FN * cfn + FP * cfp], mode="markers", marker=dict(size=11, color=CAT[1],
                    line=dict(color=SURFACE, width=2)), name="umbral actual", hovertemplate="actual: %{x:.3f}<extra></extra>")
    fig.add_scatter(x=[best["umbral"]], y=[best["costo"]], mode="markers", marker=dict(size=11, color=CAT[2], symbol="diamond",
                    line=dict(color=SURFACE, width=2)), name="umbral de menor costo", hovertemplate="óptimo: %{x:.3f}<extra></extra>")
    fig.update_layout(title="Costo total según el umbral (con los costos capturados arriba)",
                      xaxis=dict(title="umbral de alerta", range=[0, 1]), yaxis=dict(title="costo"))
    lectura("el costo total (omisiones × su costo + falsas alarmas × su costo) para cada umbral posible.",
            "el punto más bajo de la curva (rombo verde) es el umbral más económico con los costos que capturaste.")
    show(fig, 300, key=f"{ds}_cost")
    s = "good" if abs(best["umbral"] - row["umbral"]) < 0.05 else "warn"
    takeaway(s, f"El umbral de menor costo con estos supuestos es {best['umbral']:.3f}; el actual es {row['umbral']:.3f}. "
                "Si el costo de omitir un evento sube, el umbral óptimo baja (se alerta a más casos).")
    guide("la matriz de confusión",
          "Cruza lo que predijo el sistema con lo que pasó realmente. Hay cuatro casillas: detecciones, omisiones, falsas "
          "alarmas y tranquilos correctos.",
          f"Con el umbral {row['umbral']:.3f}, se detecta {rec:.0%} de los casos reales y {prec:.0%} de las alertas son correctas.",
          "good" if rec >= 0.6 and prec >= max(2 * P / (P + N), 0.1) else "warn",
          "Depende del costo de cada error: en salud o crédito omitir un caso es más caro que una falsa alarma, por eso se "
          "acepta más ruido a cambio de detectar más.",
          "El umbral es una decisión de negocio, no del modelo: con la calculadora de costos se elige el punto de operación "
          "y se dimensiona al equipo que atenderá las alertas.",
          "Es el detector de un aeropuerto: si lo pones muy sensible suena con hebillas y llaves (falsas alarmas); si lo pones "
          "poco sensible deja pasar objetos prohibidos (omisiones).", expanded=False)

    # ---------------- Importancia de variables
    section("¿Qué variables pesan más en la predicción?",
            "Importancia global: promedio del efecto absoluto de cada variable sobre la predicción (valores SHAP) en 4 000 casos "
            "de producción.")
    sh = T(ds, "x_shap").head(12).iloc[::-1]
    lab = [s if len(s) <= 58 else s[:56] + "…" for s in sh["etiqueta"]]
    fig = go.Figure(go.Bar(x=sh["mean_abs_shap"], y=lab, orientation="h", marker=dict(color=CAT[0], cornerradius=4),
                           customdata=np.stack([sh["variable"], sh["etiqueta"]], axis=1),
                           hovertemplate="%{customdata[1]}<br>(%{customdata[0]})<br>peso medio: %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Las 12 variables con más peso en la predicción", xaxis=dict(title="peso medio en la predicción (log-momios)"),
                      margin=dict(l=330))
    lectura("cuánto mueve cada variable, en promedio, la predicción del modelo (barra más larga = más influencia).",
            "indica en qué se apoya el modelo, no si la variable sube o baja el riesgo ni que lo cause.")
    show(fig, 440, key=f"{ds}_shap")
    top = T(ds, "x_shap").iloc[0]
    takeaway("info", f"La variable más influyente es «{top['etiqueta']}». Las barras indican cuánto pesa cada variable, no "
                     "si sube o baja el riesgo: ese sentido depende de la combinación con las demás.")
    guide("la importancia de variables",
          "Cuánto mueve cada variable la probabilidad final, en promedio, considerando todas sus interacciones.",
          f"Las tres variables principales concentran {T(ds, 'x_shap')['mean_abs_shap'].head(3).sum() / T(ds, 'x_shap')['mean_abs_shap'].sum():.0%} del peso total.",
          "info", "Informativo: no hay un valor bueno o malo; sirve para validar que el modelo usa señales con sentido de negocio.",
          "Si una variable importante deja de capturarse o cambia su forma de registro, la predicción se degrada: son las "
          "primeras que hay que vigilar y documentar con el área dueña del dato.",
          "Es la receta de un platillo: dice cuánto aporta cada ingrediente al sabor final, no si lo hace más dulce o más salado.")

    # ---------------- Calibración y distribución del riesgo
    section("¿Las probabilidades coinciden con la frecuencia real?")
    c1, c2 = st.columns(2)
    with c1:
        rb = T(ds, "confiabilidad_bins")
        fig = go.Figure()
        mx = float(max(rb["p_media"].max(), rb["frecuencia_observada"].max())) * 1.05
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1.5),
                        name="calibración perfecta", hoverinfo="skip")
        for lab_, col, dash, sym in [("XGB · sin calibrar", NEUTRAL, "dash", "circle-open"), (calibrated_label(ds), CAT[0], "solid", "circle")]:
            g = rb[rb["modelo"] == lab_]
            fig.add_scatter(x=g["p_media"], y=g["frecuencia_observada"], mode="lines+markers", name=lab_.replace("XGB · ", ""),
                            line=dict(color=col, width=2, dash=dash), marker=dict(size=8, symbol=sym),
                            customdata=g["n"], hovertemplate="estima %{x:.1%}<br>ocurre %{y:.1%}<br>casos %{customdata:,}<extra></extra>")
        fig.update_layout(title="Diagrama de confiabilidad (10 grupos de igual tamaño)",
                          xaxis=dict(title="probabilidad media estimada", tickformat=".0%"),
                          yaxis=dict(title="frecuencia real", tickformat=".0%"))
        lectura("los casos se agrupan en 10 grupos según su probabilidad estimada; cada punto compara esa probabilidad con la "
                "frecuencia real del grupo. Gris = modelo sin calibrar; azul = calibrado.",
                "mientras más pegada a la diagonal esté la curva, más confiables son las probabilidades.")
        show(fig, 380, key=f"{ds}_rel")
    with c2:
        b = T(ds, "bandas_riesgo")
        fig = go.Figure()
        fig.add_bar(x=b["banda"], y=b["prob_media"], name="probabilidad estimada", marker=dict(color=CAT[0], cornerradius=4),
                    hovertemplate="%{x}<br>estimada %{y:.1%}<extra></extra>")
        fig.add_bar(x=b["banda"], y=b["tasa_observada"], name="frecuencia real", marker=dict(color=CAT[1], cornerradius=4),
                    hovertemplate="%{x}<br>real %{y:.1%}<extra></extra>")
        cuts = M(ds)["bandas_riesgo"]
        fig.update_layout(title="Bandas de riesgo: " + " · ".join(f"{k} {v}" for k, v in cuts.items()),
                          barmode="group", yaxis=dict(tickformat=".0%"), bargap=0.3)
        lectura("para cada banda de riesgo (bajo, medio, alto), la probabilidad promedio estimada (azul) junto a la frecuencia "
                "real del evento (naranja).",
                "barras de igual altura = la banda refleja fielmente el riesgo real.")
        show(fig, 380, key=f"{ds}_bands")
    gap = float((b["prob_media"] - b["tasa_observada"]).abs().max())
    takeaway("good" if gap < 0.05 else "warn",
             f"En cada banda, la probabilidad estimada y la frecuencia real difieren como máximo {gap * 100:.1f} pp.")
    guide("el diagrama de confiabilidad",
          "Agrupa los casos según la probabilidad que estimó el modelo y la compara con lo que ocurrió en cada grupo.",
          f"La curva calibrada está pegada a la diagonal; el ECE en producción es {valor_ic(r['ECE']):.4f}.", st_ece(valor_ic(r["ECE"])),
          "Bueno si los puntos están sobre la diagonal; por encima significa que el modelo subestima el riesgo y por debajo que lo exagera.",
          "Las bandas de riesgo pueden usarse como reglas de operación (por ejemplo, seguimiento obligatorio en banda alta).",
          "Es como verificar un termómetro contra uno de referencia en varias temperaturas, no solo en una.")

    hist = T(ds, "x_hist_riesgo")
    fig = go.Figure()
    for col, name, color in [("negativos", info["neg"], NEG_COLOR), ("positivos", info["pos"], POS_COLOR)]:
        fig.add_bar(x=(hist["desde"] + hist["hasta"]) / 2, y=hist[col] / hist[col].sum(), name=name, width=0.022,
                    marker=dict(color=color, opacity=0.85), hovertemplate="riesgo ≈ %{x:.0%}<br>%{y:.1%} del grupo<extra></extra>")
    fig.add_vline(x=x["umbral"], line_dash="dot", line_color=INK_2, annotation_text="umbral", annotation_position="top")
    fig.update_layout(title="Distribución del riesgo estimado: casos con evento vs. sin evento", barmode="overlay",
                      xaxis=dict(title="probabilidad estimada", tickformat=".0%"), yaxis=dict(title="% del grupo", tickformat=".0%"))
    lectura(f"cómo se reparte la probabilidad estimada entre los {info['filas']} con evento (naranja) y sin evento (azul); "
            "la línea punteada es el umbral de alerta.",
            "mientras menos se superpongan los dos colores, mejor separa el modelo ambos grupos.")
    show(fig, 320, key=f"{ds}_hist")
    takeaway(st_auc(valor_ic(r["AUC"])), "Entre menos se enciman los dos colores, mejor separa el modelo a quienes tendrán el evento.")

    if technical():
        with st.expander("🛠 Detalle técnico: benchmark, ablaciones, selección del calibrador y tiempos", expanded=False):
            b = T(ds, "benchmark").sort_values("LogLoss")
            fig = go.Figure()
            for i, (_, rr) in enumerate(b.iterrows()):
                fig.add_scatter(x=[rr["ajuste_s"]], y=[rr["LogLoss"]], mode="markers+text", text=[rr["variante"].split(" (")[0]],
                                textposition="top center", marker=dict(size=12, color=CAT[i % 8], line=dict(color=SURFACE, width=2)),
                                showlegend=False, hovertemplate=rr["variante"] + "<br>%{x:.1f} s · log-loss %{y:.4f}<extra></extra>")
            fig.update_layout(title="Benchmark: calidad (log-loss, menor es mejor) vs. tiempo de ajuste",
                              xaxis=dict(type="log", title="segundos de ajuste (log)"), yaxis=dict(title="log-loss"))
            lectura("cada punto es un modelo candidato: su error probabilístico (log-loss) frente al tiempo que tarda en entrenarse.",
                    "los mejores candidatos quedan abajo (menor error) y a la izquierda (menor tiempo).")
            show(fig, 360, key=f"{ds}_bench")
            st.markdown("**Ablaciones** (cada fila cambia una sola decisión de diseño)")
            table(T(ds, "ablaciones").round(4))
            st.markdown("**Selección del calibrador** (validación cruzada de 5 pliegues)")
            table(T(ds, "seleccion_calibrador").round(4))
            st.markdown("**Protocolo de tiempos** (mediana de repeticiones)")
            table(T(ds, "tiempos_protocolo").round(4))


# ------------------------------------------------------------------------------------------- Deriva
def tab_drift(ds: str):
    info = DS[ds]
    model = st.session_state.get("modelo", "XGB")
    tl = timeline(ds, model)
    summ = policy_summary(ds, model)
    hero("Deriva de calibración", "¿Las probabilidades siguen coincidiendo con la realidad a lo largo del tiempo?",
         f"Cada punto es un {info['unidad']} de producción ({info['tam_unidad']}). El monitor revisa dos señales —cuánto cambió "
         f"la distribución del riesgo estimado y cuántos eventos ocurrieron frente a los esperados— y cada estrategia decide si "
         f"interviene. Como en operación real, el resultado de cada caso se conoce un {info['unidad']} después.",
         [f"Modelo: {model}", f"{tl['t'].nunique()} {info['unidades']} evaluados"])

    pols = st.multiselect("Estrategias a comparar", POLICIES, default=["Baseline calibrado", "A · Recalibración",
                                                                      "B · Reentrenamiento", "C · Híbrida"], key=f"{ds}_pols")
    with st.expander("¿Qué hace cada estrategia?"):
        for p in POLICIES:
            st.markdown(f"- **{p}.** {POLICY_PLAIN[p]}")

    fig = go.Figure()
    for p in [q for q in POLICIES if q in pols]:
        g = tl[tl["política"] == p]
        col, dash, sym = POLICY_STYLE[p]
        fig.add_scatter(x=g["t"], y=g["ECE"], name=p, mode="lines+markers", line=dict(color=col, dash=dash, width=2),
                        marker=dict(size=8, symbol=sym, line=dict(color=SURFACE, width=1)),
                        hovertemplate=p + f"<br>{info['unidad']} %{{x}}<br>ECE %{{y:.4f}}<extra></extra>")
    fig.add_hline(y=0.05, line_dash="dot", line_color=INK_2, annotation_text="criterio 0.05", annotation_position="top right")
    fig.update_layout(title=f"Error de calibración (ECE) por {info['unidad']}", xaxis=dict(title=info["unidad"]), yaxis=dict(title="ECE"))
    lectura(f"la evolución del error de calibración en cada {info['unidad']} de producción, una línea por estrategia de mantenimiento.",
            "líneas más bajas = probabilidades más confiables; los puntos sobre la línea de 0.05 incumplen el criterio del protocolo.")
    show(fig, 400, key=f"{ds}_ece_t")
    base = tl[tl["política"] == "Baseline calibrado"]
    share = float((base["ECE"] > 0.05).mean())
    takeaway("good" if share < 0.2 else ("warn" if share < 0.5 else "bad"),
             f"Sin mantenimiento, {share:.0%} de los {info['unidades']} superan el criterio de 0.05. Con muestras pequeñas, el ECE "
             f"de un solo {info['unidad']} incluye variación aleatoria; la lectura confiable es la tendencia y el valor agregado.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=base["t"], y=base["prevalencia"], name=f"tasa real de {info['eventos']}",
                    marker=dict(color="#cde2fb", cornerradius=3), hovertemplate="%{x}: real %{y:.1%}<extra></extra>")
        fig.add_scatter(x=base["t"], y=base["p_media"], name="probabilidad media estimada", mode="lines+markers",
                        line=dict(color=CAT[0], width=2), marker=dict(size=7),
                        hovertemplate="%{x}: estimada %{y:.1%}<extra></extra>")
        fig.update_layout(title="Tasa esperada vs. tasa observada (modelo calibrado, sin mantenimiento)", yaxis=dict(tickformat=".0%"),
                          xaxis=dict(title=info["unidad"]))
        lectura(f"barras = porcentaje real de {info['eventos']} en cada {info['unidad']}; línea = porcentaje que el modelo esperaba.",
                "cuando la línea se separa de las barras, el modelo se descalibró en ese periodo.")
        show(fig, 340, key=f"{ds}_prev")
    with c2:
        pol_sel = st.selectbox("Estrategia para el panel de alertas", [p for p in POLICIES if p != "Oráculo (in-sample)"],
                               index=4, key=f"{ds}_polsel")
        g = tl[tl["política"] == pol_sel]
        colors = [STATUS["good"] if a == 0 else (STATUS["warn"] if a == 1 else STATUS["bad"]) for a in g["alerta"]]
        fig = go.Figure(go.Bar(x=g["t"], y=g["alerta"] + 0.15, marker=dict(color=colors, cornerradius=3),
                               customdata=np.stack([g["acción"], g["psi_score"], g["z_calibracion"].fillna(0)], axis=1),
                               hovertemplate=f"{info['unidad']} %{{x}}<br>acción: %{{customdata[0]}}<br>PSI del riesgo %{{customdata[1]:.3f}}"
                                             "<br>z de calibración %{customdata[2]:.2f}<extra></extra>"))
        acts = g[g["acción"] != "ninguna"]
        fig.add_scatter(x=acts["t"], y=[2.6] * len(acts), mode="markers", name="intervención",
                        marker=dict(symbol=["diamond" if a.startswith("reent") else "triangle-down" for a in acts["acción"]],
                                    size=11, color=INK), hovertemplate="%{x}: intervención<extra></extra>")
        fig.update_layout(title=f"Nivel de alerta por {info['unidad']} · ▼ recalibración · ◆ reentrenamiento",
                          yaxis=dict(tickvals=[0.15, 1.15, 2.15], ticktext=["🟢 sin alerta", "🟡 nivel 1", "🔴 nivel 2"], range=[0, 2.9]),
                          xaxis=dict(title=info["unidad"]), showlegend=False)
        lectura(f"la altura y el color de cada barra indican el nivel de alerta del monitor en ese {info['unidad']}; los símbolos "
                "de arriba marcan cuándo la estrategia elegida intervino el modelo.",
                "un sistema sano alerta solo donde la tasa real se separa de la esperada, no todo el tiempo.")
        show(fig, 300, key=f"{ds}_lvl")
    guide("las señales del monitor",
          f"Izquierda: la proporción real de {info['eventos']} frente a la que esperaba el modelo. Derecha: el nivel de alerta "
          f"que emitió el monitor en cada {info['unidad']} y las intervenciones de la estrategia seleccionada.",
          f"Con la estrategia «{pol_sel}» hubo alerta en {(g['alerta'] >= 1).mean():.0%} de los {info['unidades']} y "
          f"{len(acts)} intervenciones.",
          "good" if (g["alerta"] == 2).mean() < 0.3 else "warn",
          "Bueno si las alertas se concentran donde la línea se separa de las barras; malo si se activan todo el tiempo.",
          "Cada alerta de nivel 1 dispara una recalibración automática; el nivel 2 sostenido justifica reentrenar.",
          "Es el tablero de un coche: la luz amarilla pide revisar la presión de las llantas (recalibrar); la roja persistente "
          "pide ir al taller (reentrenar).")

    section("Comparación de estrategias de mantenimiento", "Resultado agregado en todo el periodo de producción.")
    view = summ.copy()
    base_ll = view.loc["Baseline calibrado", "LogLoss"]
    view["Δ log-loss vs baseline"] = view["LogLoss"] - base_ll
    view["intervenciones"] = view["recalibraciones"] + view["reentrenamientos"]
    view["Estado ECE"] = [f"{EMOJI[st_ece(v)]} {WORD[st_ece(v)]}" for v in view["ECE"]]
    cols = ["Estado ECE", "ECE", "Brier", "AUC", "LogLoss", "Δ log-loss vs baseline", "intervenciones", "seg_en_intervenciones"]
    show_df = view[cols].reset_index().rename(columns={"política": "Estrategia", "seg_en_intervenciones": "segundos de cómputo"})
    table(show_df.round(4))
    best = best_active(ds, model)
    bs = summ.loc[best]
    best_ece = summ.loc[["A · Recalibración", "B · Reentrenamiento", "C · Híbrida"], "ECE"].idxmin()
    txt = (f"Menor error probabilístico (log-loss): «{best}», con ECE {bs['ECE']:.4f}, "
           f"{int(bs['recalibraciones'] + bs['reentrenamientos'])} intervenciones y {bs['seg_en_intervenciones']:.1f} s de cómputo.")
    if best_ece != best:
        be = summ.loc[best_ece]
        txt += f" Menor ECE: «{best_ece}» ({be['ECE']:.4f}). La elección final pondera calidad y costo."
    takeaway("good", txt)
    c1, c2 = st.columns(2)
    with c1:
        act = view.loc[["A · Recalibración", "B · Reentrenamiento", "C · Híbrida"]]
        fig = go.Figure()
        for p, rr in act.iterrows():
            col, _, sym = POLICY_STYLE[p]
            fig.add_scatter(x=[rr["seg_en_intervenciones"] + 0.01], y=[rr["Δ log-loss vs baseline"]], mode="markers+text", text=[p],
                            textposition="top center", marker=dict(size=14, color=col, symbol=sym, line=dict(color=SURFACE, width=2)),
                            showlegend=False, hovertemplate=p + "<br>%{x:.1f} s<br>Δ log-loss %{y:+.4f}<extra></extra>")
        fig.add_hline(y=0, line_dash="dot", line_color=INK_2)
        fig.update_layout(title="Costo (segundos de cómputo) vs. beneficio (abajo = mejora)",
                          xaxis=dict(title="segundos en intervenciones", type="log", dtick=1),
                          yaxis=dict(title="Δ log-loss respecto del baseline"))
        lectura("cada punto es una estrategia: a la derecha, más tiempo de cómputo invertido; abajo de la línea, menor error "
                "probabilístico que el modelo sin mantenimiento.",
                "la estrategia ideal está abajo y a la izquierda: mejora la calidad con poco cómputo.")
        show(fig, 360, key=f"{ds}_cb")
    with c2:
        fa = T(ds, "monitor_falsas_alarmas")
        fig = go.Figure()
        fig.add_bar(y=fa["señal"], x=fa["falsas_alarmas_nivel≥1_%"], orientation="h", name="nivel ≥ 1",
                    marker=dict(color="#9ec5f4", cornerradius=3), hovertemplate="%{y}<br>nivel ≥ 1: %{x:.1f} %<extra></extra>")
        fig.add_bar(y=fa["señal"], x=fa["falsas_alarmas_nivel2_%"], orientation="h", name="nivel 2 (crítica)",
                    marker=dict(color=CAT[0], cornerradius=3), hovertemplate="%{y}<br>nivel 2: %{x:.1f} %<extra></extra>")
        fig.add_vline(x=15, line_dash="dot", line_color=INK_2, annotation_text="criterio 15 %")
        fig.update_layout(title="Falsas alarmas en flujos sin cambio (escenarios de prueba)", barmode="group",
                          xaxis=dict(title="% de lotes sin cambio con alerta"), yaxis=dict(autorange="reversed"), margin=dict(l=230))
        lectura("en datos simulados sin ningún cambio real, qué porcentaje de lotes generó una alerta, para cada señal del monitor.",
                "barras a la izquierda de la línea de 15 % cumplen el criterio: el monitor no alerta por ruido.")
        show(fig, 360, key=f"{ds}_fa")
    lat = latency(ds)
    xcol = lat.columns[0]
    lat_v = lat.rename(columns={xcol: "magnitud del cambio"})
    lat_v["magnitud del cambio"] = [f"× {v:g}" if "multiplicador" in xcol else f"+ {v:g} pp" for v in lat[xcol]]
    lat_v["estado"] = [f"{EMOJI[st_lat(v)]} {WORD[st_lat(v)]}" for v in lat["latencia_mediana_N1"]]
    lat_v = lat_v.rename(columns={
        "detección_N1_%": "% de escenarios detectados (nivel ≥ 1)",
        "latencia_mediana_N1": "observaciones hasta la 1.ª alerta, nivel ≥ 1 (mediana)",
        "detección_N2_%": "% de escenarios detectados (nivel 2)",
        "latencia_mediana_N2": "observaciones hasta la 1.ª alerta, nivel 2 (mediana)"})
    st.markdown(f"**Escenarios de prueba con cambio conocido:** se aumenta artificialmente la tasa de {info['eventos']} y se "
                "mide cuántas observaciones nuevas necesita el monitor para emitir la primera alerta (criterio ≤ 500).")
    table(lat_v)
    guide("la validación del monitor",
          "Se crean flujos de prueba con datos reales en los que sabemos exactamente cuándo empieza un cambio, para medir "
          "falsas alarmas (antes del cambio) y rapidez de detección (después).",
          f"Falsas alarmas críticas: {far_value(ds):.1f} %; volumen mínimo para detección: "
          f"{lat['latencia_mediana_N1'].min():,.0f} observaciones.",
          st_lat(float(lat["latencia_mediana_N1"].min())),
          "Las falsas alarmas cumplen. Un volumen de 1 000 observaciones significa que la alerta llega al cierre del segundo "
          "lote, porque el resultado real de cada lote se conoce un lote después.",
          "Para reaccionar antes se propone estimar la proporción de eventos sin esperar los resultados reales (siguiente versión).",
          "Es un simulacro de incendio: se provoca el humo a propósito para comprobar que la alarma suena y cuánto tarda.")


# ------------------------------------------------------------------------------------------- Alertas
SEV = {"CRÍTICO": ("bad", "#9e2424"), "ADVERTENCIA": ("warn", "#7a5200"), "ACCIÓN": ("info", ACCENT_DARK),
       "CALIBRACIÓN": ("warn", "#7a5200"), "SESGO": ("warn", "#7a5200"), "DATOS": ("info", ACCENT_DARK), "OK": ("good", "#0b6b0b")}


def build_events(ds: str, policy: str, model: str = "XGB") -> pd.DataFrame:
    info = DS[ds]
    g = timeline(ds, model)
    g = g[g["política"] == policy].sort_values("t")
    ev = []
    size = 500 if ds == "polish" else 1000

    def clock(t):
        if ds == "airline":
            return f"Día {int(t):02d}"
        return f"Lote {int(t):02d} · casos {int(t) * size + 1:,}–{(int(t) + 1) * size:,}"

    prev_lv = 0
    for _, r in g.iterrows():
        t = r["t"]
        z = r["z_calibracion"]
        zt = f"|z| = {abs(z):.2f}" if pd.notna(z) else "|z| n/d (primer lote)"
        if r["alerta"] == 2:
            ev.append((t, "CRÍTICO", f"Señal crítica · PSI del riesgo = {r['psi_score']:.3f} · {zt}"))
        elif r["alerta"] == 1:
            ev.append((t, "ADVERTENCIA", f"Señal de cambio · PSI del riesgo = {r['psi_score']:.3f} · {zt}"))
        if r["acción"] != "ninguna":
            ev.append((t, "ACCIÓN", f"{r['acción'].capitalize()} ejecutada en {r['seg_acción']:.2f} s"))
        if r["ECE"] > 0.05:
            ev.append((t, "CALIBRACIÓN", f"ECE del {info['unidad']} = {r['ECE']:.3f} supera el criterio 0.05"))
        oe = r.get("O/E", np.nan)
        if pd.notna(oe) and (oe < 0.8 or oe > 1.25):
            ev.append((t, "SESGO", f"Ocurrieron {oe:.2f} veces los {info['eventos']} esperados "
                                   f"({'el modelo subestimó el riesgo' if oe > 1 else 'el modelo sobreestimó el riesgo'})"))
        lv = int(r["nivel_variables_info"])
        if lv != prev_lv:
            txt = ("Inicia un cambio fuerte" if lv == 2 else ("Cambio moderado" if lv == 1 else "Se normaliza")) + \
                  f" en la composición de los datos (PSI de variables = {r['psi_variables_max']:.3f})"
            ev.append((t, "DATOS", txt))
        prev_lv = lv
        if r["alerta"] == 0 and r["acción"] == "ninguna" and r["ECE"] <= 0.05:
            ev.append((t, "OK", f"{info['unidad'].capitalize()} estable · tasa real {r['prevalencia']:.1%} · estimada {r['p_media']:.1%}"))
    df = pd.DataFrame(ev, columns=["t", "severidad", "mensaje"])
    df.insert(1, "momento", df["t"].map(clock))
    return df


def render_log(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        s, col = SEV[r["severidad"]]
        rows.append(f'<div class="row"><span class="tag" style="color:{col}">{EMOJI[s]} {r["severidad"]}</span> '
                    f'<span style="color:{MUTED}">[{r["momento"]}]</span> {r["mensaje"]}</div>')
    return '<div class="log">' + "".join(rows) + "</div>"


def tab_alerts(ds: str):
    info = DS[ds]
    model = st.session_state.get("modelo", "XGB")
    hero("Bitácora de alertas", "¿Qué detectó el sistema y qué hizo al respecto?",
         f"Bitácora generada a partir del registro real de monitoreo de cada {info['unidad']} de producción. El modo «en vivo» "
         "reproduce la secuencia como la vería un operador en tiempo real (reproducción simulada con datos históricos).")
    c = st.columns([1.3, 1.6, 1])
    pol = c[0].selectbox("Estrategia activa", [p for p in POLICIES if p != "Oráculo (in-sample)"], index=4, key=f"{ds}_alpol")
    sev = c[1].multiselect("Severidad", list(SEV), default=["CRÍTICO", "ADVERTENCIA", "ACCIÓN", "CALIBRACIÓN", "SESGO"],
                           key=f"{ds}_sev")
    speed = c[2].select_slider("Velocidad", ["lenta", "normal", "rápida"], value="normal", key=f"{ds}_spd")
    ev = build_events(ds, pol, model)
    evf = ev[ev["severidad"].isin(sev)]
    k = st.columns(5)
    for i, s in enumerate(["CRÍTICO", "ADVERTENCIA", "ACCIÓN", "CALIBRACIÓN", "SESGO"]):
        n = int((ev["severidad"] == s).sum())
        k[i].metric(f"{EMOJI[SEV[s][0]]} {s.capitalize()}", n)

    fig = go.Figure()
    order = ["OK", "DATOS", "SESGO", "CALIBRACIÓN", "ACCIÓN", "ADVERTENCIA", "CRÍTICO"]
    col = {"OK": STATUS["good"], "DATOS": CAT[0], "SESGO": STATUS["warn"], "CALIBRACIÓN": "#ec835a", "ACCIÓN": ACCENT_DARK,
           "ADVERTENCIA": STATUS["warn"], "CRÍTICO": STATUS["bad"]}
    for s in order:
        g = evf[evf["severidad"] == s]
        if len(g):
            fig.add_scatter(x=g["t"], y=[s] * len(g), mode="markers", name=s, marker=dict(size=11, color=col[s],
                            line=dict(color=SURFACE, width=2)), customdata=g["mensaje"],
                            hovertemplate="%{customdata}<extra></extra>")
    fig.update_layout(title=f"Línea de tiempo de eventos por {info['unidad']}", xaxis=dict(title=info["unidad"]),
                      yaxis=dict(categoryorder="array", categoryarray=order), showlegend=False)
    lectura(f"cada punto es un evento registrado en un {info['unidad']}; los renglones van de lo normal (abajo) a lo crítico (arriba).",
            "los eventos críticos deben ir seguidos de una ACCIÓN (recalibración o reentrenamiento) en el mismo periodo o el siguiente.")
    show(fig, 300, key=f"{ds}_evt")

    b1, b2 = st.columns([1, 4])
    play = b1.button("▶ Reproducir en vivo", key=f"{ds}_play", type="primary")
    b2.caption("Muestra los eventos en el orden en que habrían aparecido, con una pausa por "
               f"{info['unidad']}. Sin reproducir, se muestra la bitácora completa.")
    slot = st.empty()
    if play:
        delay = {"lenta": 0.9, "normal": 0.45, "rápida": 0.15}[speed]
        prog = st.progress(0.0)
        ts = sorted(evf["t"].unique())
        for i, t in enumerate(ts):
            slot.markdown(render_log(evf[evf["t"] <= t].iloc[::-1]), unsafe_allow_html=True)
            prog.progress((i + 1) / len(ts), text=f"Reproduciendo {info['unidad']} {int(t)} de {int(ts[-1])}")
            time.sleep(delay)
        prog.empty()
    else:
        slot.markdown(render_log(evf.iloc[::-1]), unsafe_allow_html=True)
    download(evf.drop(columns="t"), f"bitacora_{ds}_{pol[:1]}")
    guide("la bitácora de eventos",
          "Cada renglón es un evento detectado en un lote o día: alertas del monitor, intervenciones ejecutadas, lotes con "
          "calibración fuera de criterio, desviaciones entre eventos esperados y observados, y cambios en los datos.",
          f"Con «{pol}» hubo {int((ev['severidad'] == 'CRÍTICO').sum())} eventos críticos y {int((ev['severidad'] == 'ACCIÓN').sum())} "
          "intervenciones.",
          "good" if (ev["severidad"] == "CRÍTICO").sum() <= len(ev["t"].unique()) * 0.3 else "warn",
          "Bueno si los eventos críticos son pocos y van seguidos de una acción; regular si se acumulan sin respuesta.",
          "La bitácora es la evidencia para auditoría: qué vio el sistema, cuándo y qué decidió. Se exporta en CSV.",
          "Es la bitácora de vuelo de un avión: registra cada turbulencia y cada corrección del piloto.")


# =====================================================================================================
# 7. EXPLORACIÓN ESPECÍFICA DE CADA DOMINIO
# =====================================================================================================
HORIZ = {1: "5 años antes (1year)", 2: "4 años antes (2year)", 3: "3 años antes (3year)", 4: "2 años antes (4year)",
         5: "1 año antes (5year)"}
RATIOS = {"Attr1": "Rentabilidad: utilidad neta / activos", "Attr2": "Endeudamiento: pasivos / activos",
          "Attr3": "Capital de trabajo / activos", "Attr9": "Rotación: ventas / activos", "Attr10": "Capital contable / activos",
          "Attr21": "Crecimiento de ventas (n / n−1)", "Attr27": "Utilidad de operación / gastos financieros",
          "Attr29": "Tamaño: log de activos totales", "Attr39": "Margen: utilidad en ventas / ventas"}


def filter_box():
    return st.container(border=True)


def eda_polish(ds: str):
    info = DS[ds]
    df = T(ds, "x_eda")
    df["horizonte"] = df["cohort"].map(HORIZ)
    with filter_box():
        st.markdown("**Filtros**")
        c = st.columns([2, 1.2, 1.5, 1.3])
        hs = c[0].multiselect("Horizonte (distancia al desenlace)", list(HORIZ.values()), default=list(HORIZ.values()), key="pl_h")
        est = c[1].radio("Estado", ["Todas", info["pos"], info["neg"]], key="pl_e", horizontal=False)
        var = c[2].selectbox("Indicador financiero", list(RATIOS), format_func=lambda k: RATIOS[k], key="pl_v")
        clip = c[3].slider("Recortar extremos (percentiles)", 0, 10, 2, key="pl_c",
                           help="Los ratios financieros tienen valores extremos; se recortan para poder ver la forma de la distribución.")
    d = df[df["horizonte"].isin(hs)]
    if est != "Todas":
        d = d[d["y"] == (1 if est == info["pos"] else 0)]
    if d.empty:
        st.warning("No hay datos con estos filtros.")
        return
    k = st.columns(4)
    k[0].metric("Empresas en la muestra", f"{len(d):,}")
    k[1].metric("Tasa de quiebra", fmt_pct(d["y"].mean()))
    k[2].metric("Faltantes promedio por empresa", f"{d['n_missing'].mean():.2f}")
    k[3].metric("Riesgo medio estimado", fmt_pct(d["p"].mean()))
    st.caption("Muestra aleatoria de 8 000 empresas de las 43 004 (tras eliminar duplicados) para que los filtros respondan al instante.")

    c1, c2 = st.columns(2)
    with c1:
        g = rate_by(d, "cohort").assign(h=lambda x: x["cohort"].map(HORIZ))
        fig = bar_rate(g, "h", "tasa", "casos", "Tasa de quiebra según la distancia al desenlace", yfmt=".1%", color=CAT[1],
                       ref=float(d["y"].mean()))
        lectura("el porcentaje de empresas que quebraron según cuántos años faltaban para el desenlace al tomar sus datos.",
                "la línea punteada es el promedio; las barras por encima indican horizontes con más quiebras de lo normal.")
        show(fig, 360, key="pl_rate")
        takeaway("warn", "Mientras más cerca está el desenlace, mayor es la proporción de quiebras: este es el cambio que "
                         "descalibra a un modelo entrenado con empresas lejanas.")
    with c2:
        g = d.groupby("cohort").agg(real=("y", "mean"), predicho=("p", "mean")).reset_index()
        g["h"] = g["cohort"].map(HORIZ)
        fig = go.Figure()
        fig.add_bar(x=g["h"], y=g["real"], name="tasa real", marker=dict(color=CAT[1], cornerradius=4),
                    hovertemplate="%{x}<br>real %{y:.1%}<extra></extra>")
        fig.add_bar(x=g["h"], y=g["predicho"], name="riesgo medio estimado", marker=dict(color=CAT[0], cornerradius=4),
                    hovertemplate="%{x}<br>estimado %{y:.1%}<extra></extra>")
        fig.update_layout(title="¿El modelo estima bien la tasa de cada horizonte?", barmode="group", yaxis=dict(tickformat=".1%"),
                          bargap=0.3)
        lectura("por horizonte, la tasa real de quiebra (naranja) junto al riesgo promedio que estimó el modelo (azul).",
                "si la barra azul queda por debajo de la naranja, el modelo subestima el riesgo en ese horizonte.")
        show(fig, 360, key="pl_pred")
        last = g.sort_values("cohort").iloc[-1]
        gap = last["real"] - last["predicho"]
        takeaway("warn" if abs(gap) > 0.01 else "good",
                 f"En el horizonte más cercano al desenlace la tasa real es {last['real']:.1%} y el modelo base estima "
                 f"{last['predicho']:.1%}: esa brecha es el calibration drift que las estrategias de mantenimiento corrigen. "
                 "Los horizontes de 5 a 3 años se usaron para entrenar; la calibración se ajustó con 3 y 2 años.")

    c1, c2 = st.columns(2)
    with c1:
        lo, hi = clip / 100, 1 - clip / 100
        fig = go.Figure()
        for val, name, col in [(0, info["neg"], NEG_COLOR), (1, info["pos"], POS_COLOR)]:
            s = d[d["y"] == val][var].dropna()
            if len(s):
                s = s.clip(df[var].quantile(lo), df[var].quantile(hi))
                fig.add_histogram(x=s, name=name, histnorm="percent", nbinsx=40, marker=dict(color=col), opacity=0.75,
                                  hovertemplate=name + "<br>valor %{x}<br>%{y:.1f} % del grupo<extra></extra>")
        fig.update_layout(title=f"Distribución: {RATIOS[var]}", barmode="overlay", xaxis=dict(title=RATIOS[var]),
                          yaxis=dict(title="% de empresas del grupo"))
        lectura("cómo se distribuye el indicador elegido entre empresas que quebraron (naranja) y que siguieron operando (azul).",
                "mientras menos se superpongan las dos formas, más útil es el indicador por sí solo para anticipar quiebras.")
        show(fig, 360, key="pl_dist")
        m1, m0 = d[d["y"] == 1][var].median(), d[d["y"] == 0][var].median()
        takeaway("info", f"Mediana en empresas que quebraron: {m1:.3f} · en empresas sanas: {m0:.3f}. Entre más separadas "
                         "estén las dos formas, más útil es el indicador por sí solo.")
    with c2:
        mt = T(ds, "faltantes_top").head(8).iloc[::-1]
        fig = go.Figure()
        fig.add_bar(y=mt["variable"] + " · " + mt["descripción"].str.slice(0, 38), x=mt["faltante_si_sana_%"] / 100, orientation="h",
                    name="empresas sanas", marker=dict(color=NEG_COLOR, cornerradius=3), hovertemplate="%{x:.1%}<extra>sanas</extra>")
        fig.add_bar(y=mt["variable"] + " · " + mt["descripción"].str.slice(0, 38), x=mt["faltante_si_quiebra_%"] / 100, orientation="h",
                    name="empresas que quebraron", marker=dict(color=POS_COLOR, cornerradius=3),
                    hovertemplate="%{x:.1%}<extra>quebraron</extra>")
        fig.update_layout(title="Datos faltantes: ¿no reportar un indicador es una señal?", barmode="group",
                          xaxis=dict(tickformat=".0%", title="% de empresas sin el dato"), margin=dict(l=300))
        lectura("el porcentaje de empresas que no reportó cada indicador, separando empresas sanas y empresas que quebraron.",
                "si el faltante es mucho más común entre las que quebraron, no reportar ese dato ya es una señal de riesgo.")
        show(fig, 360, key="pl_miss")
        takeaway("warn", "Attr27 falta en 38 % de las empresas que quebraron y solo en 5 % de las sanas: no reportar un dato "
                         "es información. Por eso el modelo conserva la ausencia en lugar de rellenarla a ciegas.")
    guide("la exploración de Polish",
          "Cómo se reparten las quiebras por horizonte y cómo se comportan los indicadores financieros entre empresas que "
          "quebraron y empresas sanas.",
          f"Con los filtros actuales hay {len(d):,} empresas y una tasa de quiebra de {d['y'].mean():.1%}.",
          "warn", "Regular: la tasa cambia mucho entre horizontes, lo que obliga a vigilar la calibración del modelo.",
          "Un área de crédito debe usar el modelo con el horizonte correcto y revisar con más cuidado a las empresas que dejan "
          "de reportar indicadores.",
          "Es como comparar los análisis clínicos de pacientes sanos y enfermos: algunos valores, y también los estudios que "
          "el paciente no quiso hacerse, dicen mucho.")
    with st.expander("Mapa de cambios entre horizontes (PSI por indicador)"):
        ps = T(ds, "psi_cohortes").set_index("variable")
        ps = ps.loc[ps.max(axis=1).sort_values(ascending=False).index[:15]]
        fig = go.Figure(go.Heatmap(z=ps.values, x=[HORIZ[int(c[0])] for c in ps.columns], y=ps.index, colorscale=[[i / 7, c] for i, c in enumerate(SEQ)],
                                   zmin=0, zmax=float(min(1.0, ps.values.max())), text=np.round(ps.values, 2), texttemplate="%{text}",
                                   hovertemplate="%{y} · %{x}<br>PSI %{z:.3f}<extra></extra>", colorbar=dict(title="PSI")))
        fig.update_layout(title="Cambio de distribución frente a 5 años antes (PSI · > 0.25 = cambio fuerte)", yaxis=dict(autorange="reversed"))
        lectura("cuánto cambió la distribución de cada indicador (renglón) en cada horizonte (columna) frente a 5 años antes.",
                "celdas más oscuras = cambios más fuertes; por encima de 0.25 el cambio puede descalibrar al modelo.")
        show(fig, 460, key="pl_psi")


def eda_airline(ds: str):
    info = DS[ds]
    df = T(ds, "x_eda")
    df["hora"] = (df["Time"] // 60).clip(0, 23).astype(int)
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    with filter_box():
        st.markdown("**Filtros**")
        c = st.columns([2.2, 1.4, 1.4])
        als = sorted(df["Airline"].dropna().unique())
        al = c[0].multiselect("Aerolíneas", als, default=als, key="al_a")
        dr = c[1].slider("Días (0 = primer día del registro)", 0, 30, (0, 30), key="al_d")
        hr = c[2].slider("Hora de salida", 0, 23, (0, 23), key="al_h")
    d = df[df["Airline"].isin(al) & df["day"].between(*dr) & df["hora"].between(*hr)]
    if d.empty:
        st.warning("No hay datos con estos filtros.")
        return
    k = st.columns(4)
    k[0].metric("Vuelos en la muestra", f"{len(d):,}")
    k[1].metric("Tasa de retraso", fmt_pct(d["y"].mean()))
    k[2].metric("Duración media", f"{d['Length'].mean():.0f} min")
    k[3].metric("Aeropuertos de origen", f"{d['AirportFrom'].nunique()}")
    st.caption("Muestra aleatoria de 25 000 de los 539 383 vuelos de 31 días consecutivos.")

    c1, c2 = st.columns(2)
    with c1:
        g = rate_by(d, "day")
        split = np.where(g["day"] <= 17, "entrenamiento", np.where(g["day"] <= 20, "calibración", "producción"))
        fig = go.Figure()
        for name, col in [("entrenamiento", CAT[0]), ("calibración", CAT[2]), ("producción", CAT[1])]:
            gg = g[split == name]
            fig.add_bar(x=gg["day"], y=gg["tasa"], name=name, marker=dict(color=col, cornerradius=3), customdata=gg["casos"],
                        hovertemplate="día %{x}<br>%{y:.1%} retrasados<br>%{customdata:,} vuelos<extra></extra>")
        fig.update_layout(title="Tasa de retraso por día", yaxis=dict(tickformat=".0%"), xaxis=dict(title="día"), bargap=0.15)
        lectura("el porcentaje de vuelos retrasados en cada día del mes, coloreado según la etapa del modelo en la que se usó ese día.",
                "la altura cambia mucho entre días: es el cambio de proporción que obliga a vigilar la calibración.")
        show(fig, 360, key="al_day")
        takeaway("warn", f"La tasa diaria va de {g['tasa'].min():.0%} a {g['tasa'].max():.0%}: el clima y la congestión cambian "
                         "la proporción de retrasos de un día a otro aunque los vuelos sean los mismos.")
    with c2:
        g = rate_by(d, "hora", 30)
        fig = bar_rate(g, "hora", "tasa", "casos", "Tasa de retraso por hora de salida", yfmt=".0%", color=CAT[0],
                       ref=float(d["y"].mean()))
        fig.update_xaxes(title="hora del día", dtick=2)
        lectura("el porcentaje de vuelos retrasados según la hora de salida programada.",
                "barras sobre la línea punteada (promedio) señalan franjas con más retrasos de lo normal.")
        show(fig, 360, key="al_hour")
        hmax = g.loc[g["tasa"].idxmax()]
        takeaway("warn", f"Los retrasos se acumulan a lo largo del día: la hora más crítica es {int(hmax['hora'])}:00 con "
                         f"{hmax['tasa']:.0%}. Los vuelos de la mañana son los más puntuales.")
    c1, c2 = st.columns(2)
    with c1:
        g = rate_by(d, "Airline", 50).sort_values("tasa", ascending=False)
        fig = bar_rate(g, "Airline", "tasa", "casos", "Tasa de retraso por aerolínea (≥ 50 vuelos)", horizontal=True,
                       yfmt=".0%", color=CAT[0], ref=float(d["y"].mean()))
        lectura("el porcentaje de vuelos retrasados de cada aerolínea con al menos 50 vuelos en la muestra.",
                "la línea punteada es el promedio; las aerolíneas a su derecha se retrasan más que el resto.")
        show(fig, 420, key="al_air")
        takeaway("info", f"La diferencia entre la aerolínea más y menos puntual es de {(g['tasa'].max() - g['tasa'].min()) * 100:.0f} "
                         "puntos porcentuales.")
    with c2:
        h = d.assign(franja=pd.cut(d["hora"], [-1, 5, 9, 13, 17, 20, 23], labels=["0–5", "6–9", "10–13", "14–17", "18–20", "21–23"]))
        pv = h.pivot_table(index="DayOfWeek", columns="franja", values="y", aggfunc="mean", observed=False)
        fig = go.Figure(go.Heatmap(z=pv.values, x=[str(c) for c in pv.columns], y=[dias[int(i) - 1] for i in pv.index],
                                   colorscale=[[i / 7, c] for i, c in enumerate(SEQ)], text=np.round(pv.values * 100, 0),
                                   texttemplate="%{text:.0f}%", hovertemplate="%{y} · %{x} h<br>%{z:.1%} retrasados<extra></extra>",
                                   colorbar=dict(title="tasa", tickformat=".0%"), xgap=2, ygap=2))
        fig.update_layout(title="Mapa de calor: día de la semana × franja horaria", xaxis=dict(title="franja horaria"))
        lectura("el porcentaje de retrasos para cada combinación de día de la semana (renglón) y franja horaria (columna).",
                "las celdas más oscuras concentran más retrasos; ahí conviene reforzar personal y conexiones.")
        show(fig, 420, key="al_heat")
        takeaway("info", "Las celdas más oscuras marcan combinaciones de día y hora donde conviene reforzar personal y conexiones.")
    c1, c2 = st.columns(2)
    with c1:
        dd = d.dropna(subset=["h_fl"]).copy()
        dd["decil"] = pd.qcut(dd["h_fl"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
        g = dd.groupby("decil", observed=True).agg(tasa=("y", "mean"), casos=("y", "size"), h=("h_fl", "mean")).reset_index()
        fig = bar_rate(g, "decil", "tasa", "casos", "Historial del vuelo: tasa real por decil de su retraso histórico", yfmt=".0%",
                       color=CAT[2])
        fig.update_xaxes(title="decil del historial (D1 = vuelos históricamente puntuales)")
        lectura("los vuelos ordenados en 10 grupos (deciles) según su historial de retrasos en días anteriores, y la tasa real de cada grupo.",
                "si las barras crecen de D1 a D10, el historial anticipa bien el retraso.")
        show(fig, 360, key="al_hist")
        takeaway("good", f"Los vuelos con peor historial (D10) se retrasan {g['tasa'].iloc[-1]:.0%} de las veces frente a "
                         f"{g['tasa'].iloc[0]:.0%} de los más puntuales (D1): el historial de días anteriores es la señal más "
                         "fuerte del modelo.")
    with c2:
        top = d["AirportFrom"].value_counts().head(12).index
        g = rate_by(d[d["AirportFrom"].isin(top)], "AirportFrom").sort_values("tasa", ascending=False)
        fig = bar_rate(g, "AirportFrom", "tasa", "casos", "Aeropuertos de origen con más vuelos: tasa de retraso",
                       horizontal=True, yfmt=".0%", color=CAT[0], ref=float(d["y"].mean()))
        lectura("el porcentaje de vuelos retrasados en los 12 aeropuertos de origen con más vuelos.",
                "la línea punteada es el promedio; los aeropuertos a su derecha concentran más retrasos.")
        show(fig, 360, key="al_apt")
        takeaway("info", "Los grandes centros de conexión concentran la congestión y propagan retrasos al resto de la red.")
    guide("la exploración de Airline",
          "Cuándo, dónde y con quién se concentran los retrasos, y qué tan útil es el historial reciente de cada vuelo.",
          f"Con los filtros actuales, {d['y'].mean():.1%} de {len(d):,} vuelos llegó con retraso.",
          "warn", "Regular: la tasa es alta (cerca de la mitad) y cambia de un día a otro, lo que limita la precisión posible.",
          "Planear personal y puertas por franja horaria y priorizar los vuelos con peor historial; incorporar el pronóstico "
          "del clima elevaría la calidad del modelo.",
          "Es como predecir el tráfico: la hora y la ruta ayudan, pero un día de lluvia cambia todo para todos.")


def eda_diabetes(ds: str):
    info = DS[ds]
    df = T(ds, "x_eda")
    with filter_box():
        st.markdown("**Filtros**")
        c = st.columns([1.4, 1.2, 2, 1.4])
        ag = c[0].slider("Edad", 0, 100, (0, 100), 10, key="db_a")
        sx = c[1].multiselect("Sexo", sorted(df["gender"].dropna().unique()), default=sorted(df["gender"].dropna().unique()), key="db_s")
        ats = sorted(df["admission_type"].dropna().unique())
        at = c[2].multiselect("Tipo de admisión", ats, default=ats, key="db_t")
        pr = c[3].slider("Periodo (orden de registro)", 0, 18, (0, 18), key="db_p")
    d = df[df["age_mid"].between(*ag) & df["gender"].isin(sx) & df["admission_type"].isin(at) & df["period"].between(*pr)]
    if d.empty:
        st.warning("No hay datos con estos filtros.")
        return
    k = st.columns(4)
    k[0].metric("Pacientes en la muestra", f"{len(d):,}")
    k[1].metric("Readmisión < 30 días", fmt_pct(d["y"].mean()))
    k[2].metric("Días hospitalizado (media)", f"{d['time_in_hospital'].mean():.1f}")
    k[3].metric("Riesgo medio estimado", fmt_pct(d["p"].mean()))
    st.caption("Muestra aleatoria de 12 000 de los 66 487 pacientes de la cohorte (primer ingreso de cada paciente).")
    avg = float(d["y"].mean())
    c1, c2 = st.columns(2)
    with c1:
        g = rate_by(d, "discharge_disposition", 80).sort_values("tasa", ascending=False).head(10)
        g["discharge_disposition"] = g["discharge_disposition"].str.slice(0, 48)
        fig = bar_rate(g, "discharge_disposition", "tasa", "casos", "Readmisión según el destino al alta (≥ 80 pacientes)",
                       horizontal=True, yfmt=".0%", color=CAT[1], ref=avg)
        fig.update_layout(margin=dict(l=320))
        lectura("el porcentaje de pacientes que regresó al hospital en menos de 30 días según su destino al alta (destinos con ≥ 80 pacientes).",
                "la línea punteada es el promedio; los destinos a su derecha requieren más seguimiento.")
        show(fig, 400, key="db_dis")
        takeaway("warn", "Los pacientes transferidos a otra institución (rehabilitación, cuidados especializados) regresan "
                         "mucho más que los que van a casa: el destino al alta es una de las señales más fuertes.")
    with c2:
        d2 = d.assign(prev=pd.cut(d["number_inpatient"], [-1, 0, 1, 2, 3, 100], labels=["0", "1", "2", "3", "4 o más"]))
        g = rate_by(d2, "prev")
        fig = bar_rate(g, "prev", "tasa", "casos", "Readmisión según hospitalizaciones en el año previo", yfmt=".0%",
                       color=CAT[1], ref=avg)
        fig.update_xaxes(title="hospitalizaciones previas")
        lectura("el porcentaje de readmisión según cuántas hospitalizaciones tuvo el paciente en el año previo.",
                "si las barras crecen hacia la derecha, el historial del paciente anticipa la readmisión.")
        show(fig, 400, key="db_prev")
        if len(g) > 1:
            takeaway("warn", f"Con 4 o más hospitalizaciones previas la tasa llega a {g['tasa'].iloc[-1]:.0%}, frente a "
                             f"{g['tasa'].iloc[0]:.0%} sin ninguna: el historial del paciente pesa más que su diagnóstico.")
    c1, c2 = st.columns(2)
    with c1:
        g = rate_by(d, "diag_1_group", 50).sort_values("tasa", ascending=False)
        fig = bar_rate(g, "diag_1_group", "tasa", "casos", "Readmisión por diagnóstico principal", horizontal=True, yfmt=".0%",
                       color=CAT[0], ref=avg)
        lectura("el porcentaje de readmisión según el grupo del diagnóstico principal (grupos con ≥ 50 pacientes).",
                "barras parecidas entre sí indican que el diagnóstico, por sí solo, separa poco el riesgo.")
        show(fig, 380, key="db_diag")
    with c2:
        g = rate_by(d, "age_mid", 50)
        fig = go.Figure(go.Scatter(x=g["age_mid"], y=g["tasa"], mode="lines+markers", line=dict(color=CAT[0], width=2),
                                   marker=dict(size=9), customdata=g["casos"],
                                   hovertemplate="edad ≈ %{x}<br>%{y:.1%}<br>%{customdata:,} pacientes<extra></extra>"))
        fig.update_layout(title="Readmisión por edad", yaxis=dict(tickformat=".0%"), xaxis=dict(title="edad (punto medio del rango)"))
        lectura("el porcentaje de readmisión por grupo de edad (punto medio de cada rango de 10 años).")
        show(fig, 380, key="db_age")
    takeaway("info", "El diagnóstico y la edad separan poco a los pacientes: la readmisión depende de una mezcla de factores, "
                     "por eso el modelo alcanza una capacidad de discriminación moderada.")
    per = T(ds, "periodos")
    per = per[per["period"] <= 18]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=per["period"], y=per["readm_30d"], mode="lines+markers", line=dict(color=CAT[1], width=2),
                                   marker=dict(size=8), hovertemplate="periodo %{x}<br>%{y:.1%}<extra></extra>"))
        fig.update_layout(title="Tasa de readmisión < 30 días por periodo (cohorte completa)", yaxis=dict(tickformat=".1%"),
                          xaxis=dict(title="periodo"))
        lectura("la tasa de readmisión en menos de 30 días en cada periodo, en orden de registro, para toda la cohorte.",
                "una tendencia a la baja hace que un modelo sin mantenimiento sobreestime el riesgo.")
        show(fig, 340, key="db_per")
        takeaway("warn", "La tasa desciende en los periodos recientes: un modelo sin mantenimiento tiende a exagerar el riesgo.")
    with c2:
        fig = go.Figure()
        for col_, name, c_, dash in [("payer_faltante", "aseguradora no registrada", CAT[0], "solid"),
                                     ("especialidad_faltante", "especialidad no registrada", CAT[1], "dash"),
                                     ("hba1c_medida", "HbA1c medida", CAT[2], "dashdot")]:
            fig.add_scatter(x=per["period"], y=per[col_], name=name, mode="lines+markers", line=dict(color=c_, width=2, dash=dash),
                            marker=dict(size=7), hovertemplate=name + "<br>periodo %{x}: %{y:.0%}<extra></extra>")
        fig.update_layout(title="Cómo cambia la forma de capturar la información", yaxis=dict(tickformat=".0%"),
                          xaxis=dict(title="periodo"))
        lectura("en cada periodo, el porcentaje de registros sin aseguradora, sin especialidad médica y con HbA1c medida.",
                "cambios bruscos reflejan cambios en la forma de capturar datos, no en los pacientes.")
        show(fig, 340, key="db_cap")
        takeaway("warn", "La aseguradora pasa de no registrarse nunca a registrarse casi siempre: es un cambio de captura, no "
                         "de pacientes. Por eso esa variable se vigila pero no se usa para predecir.")
    guide("la exploración de Diabetes",
          "Qué características del paciente y de su alta se asocian con volver al hospital en menos de 30 días, y cómo cambió "
          "el registro de la información con el tiempo.",
          f"Con los filtros actuales, {avg:.1%} de {len(d):,} pacientes regresó en menos de 30 días.",
          "warn", "Regular: las señales son moderadas y la forma de capturar datos cambió, lo que exige vigilar el modelo.",
          "Priorizar el seguimiento al alta de pacientes con hospitalizaciones previas o transferidos a otra institución; "
          "estandarizar la captura de datos clínicos.",
          "Es como predecir qué alumnos reprobarán: el historial de materias reprobadas dice más que la carrera que estudian.")


# =====================================================================================================
# 8. NAVEGACIÓN DE LA PESTAÑA (antes en la barra lateral; ahora dentro de la pestaña para no mezclar vistas)
# =====================================================================================================
MODULOS_TECNICOS = {"📋 Resumen general": None, "🏭 Polish Bankruptcy": "polish", "✈️ Airline Delay": "airline",
                    "🏥 Diabetes 130-US": "diabetes"}


def vista_tecnica():
    with st.container(border=True):
        c1, c2, c3 = st.columns([2.6, 1.2, 1])
        choice = c1.radio("Conjunto de datos", list(MODULOS_TECNICOS), key="modulo", horizontal=True)
        c2.selectbox("Modelo para deriva y alertas", ["XGB", "RL"], key="modelo",
                     help="XGB = XGBoost  RL = Regresión Logística")
        c3.toggle("Mostrar detalle técnico", key="tecnico",
                  help="Agrega benchmark, ablaciones, selección del calibrador y tiempos en «Modelo predictivo».")
        st.caption("Semáforo: 🟢 bueno · 🟡 regular · 🔴 malo · 🔵 informativo. Todas las cifras provienen de las ejecuciones "
                   "finales de los notebooks TT2 (corridas " + ", ".join(PACK[d]["meta"]["run_id"] for d in DS) +
                   "); el tablero no recalcula modelos.")
    ds_sel = MODULOS_TECNICOS[choice]
    if ds_sel is None:
        module_overview()
    else:
        module_dataset(ds_sel)


# #####################################################################################################
# PESTAÑA 1 · RESUMEN EJECUTIVO (vista gerencial)
# #####################################################################################################
# =============================================================================
# 1. REGLAS DEL TABLERO (umbrales de atención). Son criterios de presentación,
#    no resultados del modelo: se muestran en "Metodología" y pueden ajustarse.
# =============================================================================
REGLAS = {
    # mínimo de casos para hablar de un segmento (aerolínea, aeropuerto, destino al alta, …)
    "min_segmento": {"airline": 400, "diabetes": 150, "polish": 100},
    # muestra mínima para conclusiones estables con filtros
    "min_muestra": {"airline": 1000, "diabetes": 500, "polish": 300},
    "min_eventos": 30,
    # diferencia estimado-vs-real (en puntos porcentuales para Airline; relativa para eventos raros)
    "brecha_pp": (2.0, 5.0, 10.0),            # medio, alto, crítico (Airline)
    "brecha_relativa": (0.05, 0.15, 0.30),    # medio, alto, crítico (Diabetes, Polish)
    # segmento con tasa muy superior al promedio
    "segmento_pp": (10.0, 20.0),              # alta, crítica (Airline, en puntos)
    "segmento_veces": (1.5, 2.0),             # alta, crítica (Diabetes/Polish, veces la tasa general)
    # cambio de la tasa del evento frente al periodo de entrenamiento
    "cambio_tasa_pp": {"airline": (3.0, 8.0, 15.0), "diabetes": (0.5, 1.5, 3.0), "polish": (0.5, 1.5, 3.0)},
    # proporción de casos en riesgo alto
    "share_alto": {"airline": (0.10, 0.25, 0.40), "diabetes": (0.05, 0.12, 0.20), "polish": (0.02, 0.05, 0.10)},
    # capacidad de detección de las alertas: min(sensibilidad, precisión) — mayor es mejor
    "anticipacion": (0.60, 0.45, 0.30),       # ≥ bajo · ≥ medio · ≥ alto · menor = crítico
    "psi_relevante": 0.25,                    # cambio fuerte en la forma de capturar/presentar datos
    "psi_estable": 0.10,
}

# Paleta (validada para daltonismo en la guía de visualización): azul = real/observado, naranja = estimado por el modelo


# =============================================================================
# 3. UTILIDADES DE FORMATO
# =============================================================================
def pct(x, d=0):
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def num(x):
    return "—" if x is None or pd.isna(x) else f"{int(round(x)):,}"


def pp(x, d=1):
    """Diferencia entre dos proporciones, con signo, en puntos porcentuales (p. ej., 0.032 → '+3.2 pp')."""
    return "—" if x is None or pd.isna(x) else f"{100 * x:+.{d}f} pp"


def pts(x, d=1):
    """Magnitud expresada en puntos porcentuales (p. ej., ECE 0.0228 → '2.3 pp')."""
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{d}f} pp"


def de_cada(x, base=100):
    return f"{int(round(x * base))} de cada {base}"


def nivel_mayor_peor(v, cortes):
    """cortes = (medio, alto, crítico): valores mayores implican más atención."""
    if v is None or pd.isna(v):
        return "info"
    return "critico" if v >= cortes[2] else "alto" if v >= cortes[1] else "medio" if v >= cortes[0] else "bajo"


def min_valido(*vals):
    v = [x for x in vals if x is not None and not pd.isna(x)]
    return min(v) if v else np.nan


def nivel_mayor_mejor(v, cortes):
    """cortes = (bajo, medio, alto): valores mayores implican menos atención."""
    if v is None or pd.isna(v):
        return "info"
    return "bajo" if v >= cortes[0] else "medio" if v >= cortes[1] else "alto" if v >= cortes[2] else "critico"


def badge(nivel, texto=None):
    color = ESTADO.get(nivel, ESTADO["info"])
    texto = texto or (f"Atención {NIVEL_TXT[nivel].lower()}" if nivel in NIVEL_TXT else "Informativa")
    return f'<span class="tt-badge"><span class="tt-dot" style="background:{color}"></span>{texto}</span>'


def sin_parentesis(s):
    import re
    return re.sub(r"\s*\([^)]*\)", "", str(s)).strip()


def esc(s):
    return str(s).replace("<", "&lt;").replace(">", "&gt;")


# =============================================================================
# 4. CARGA DE DATOS (solo lectura, en caché)
# =============================================================================
@st.cache_data(show_spinner=False)
def cargar(ds: str, raiz: str):
    carpeta = Path(raiz) / ds
    if not (carpeta / "filas_produccion.parquet").exists():
        raise FileNotFoundError(f"No existe {carpeta / 'filas_produccion.parquet'}")
    filas = pd.read_parquet(carpeta / "filas_produccion.parquet")
    ctx = json.loads((carpeta / "contexto.json").read_text(encoding="utf-8"))
    art = {}
    for p in sorted((carpeta / "artefactos").glob("*")):
        try:
            if p.suffix == ".parquet":
                art[p.stem] = pd.read_parquet(p)
            elif p.suffix == ".csv":
                art[p.stem] = pd.read_csv(p)
            elif p.name == "metadata.json":
                art["metadata"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return filas, ctx, art


def disponible(ds):
    if EX_RAIZ is None:
        return False, "No se encontraron los datos de la vista ejecutiva."
    try:
        cargar(ds, str(EX_RAIZ))
        return True, ""
    except Exception as err:
        return False, f"{type(err).__name__}: {err}"


# =============================================================================
# 5. CÁLCULOS COMUNES (todos sobre los registros filtrados)
# =============================================================================
def metricas(df):
    n = len(df)
    if n == 0:
        return {"n": 0}
    y, p = df["y"].to_numpy(), df["p"].to_numpy(dtype=float)
    alto, medio, alerta = (df["banda"] == "Alto").to_numpy(), (df["banda"] == "Medio").to_numpy(), (df["alerta"] == 1).to_numpy()
    ev = int(y.sum())
    m = {"n": n, "eventos": ev, "tasa": ev / n, "p_media": float(p.mean()), "esperados": float(p.sum()),
         "oe": ev / p.sum() if p.sum() > 0 else np.nan,
         "n_alto": int(alto.sum()), "n_medio": int(medio.sum()), "share_alto": alto.mean(), "share_medio": medio.mean(),
         "tasa_alto": y[alto].mean() if alto.any() else np.nan, "tasa_medio": y[medio].mean() if medio.any() else np.nan,
         "tasa_bajo": y[~alto & ~medio].mean() if (~alto & ~medio).any() else np.nan,
         "cob_alto": y[alto].sum() / ev if ev else np.nan, "cob_alto_medio": y[alto | medio].sum() / ev if ev else np.nan,
         "n_alerta": int(alerta.sum()), "precision": y[alerta].mean() if alerta.any() else np.nan,
         "sensibilidad": alerta[y == 1].mean() if ev else np.nan}
    return m


def tasa_por(df, col, min_n=0, orden=None):
    g = (df.groupby(col, observed=True)
         .agg(n=("y", "size"), eventos=("y", "sum"), tasa=("y", "mean"), estimado=("p", "mean")).reset_index())
    g = g[g["n"] >= min_n]
    if orden is not None:
        g[col] = pd.Categorical(g[col], categories=[o for o in orden if o in set(g[col])], ordered=True)
        return g.sort_values(col)
    return g.sort_values("tasa", ascending=False)


def influencia(df, grupos):
    """Peso de cada grupo de factores en la estimación: promedio de |contribución| (SHAP agrupado), en % del total."""
    filas = {}
    for g, cols in grupos.items():
        cc = [f"c__{c}" for c in cols if f"c__{c}" in df.columns]
        if not cc or df.empty:
            continue
        s = df[cc].sum(axis=1)
        filas[g] = {"abs": float(s.abs().mean()), "media": float(s.mean())}
    imp = pd.DataFrame(filas).T
    if imp.empty:
        return imp
    imp["peso"] = imp["abs"] / imp["abs"].sum()
    return imp.sort_values("peso", ascending=False)


def influencia_variables(df, etiquetas, top=10):
    cols = [c for c in df.columns if c.startswith("c__")]
    if not cols or df.empty:
        return pd.DataFrame()
    a = df[cols].abs().mean()
    t = (a / a.sum()).sort_values(ascending=False).head(top)
    return pd.DataFrame({"Variable": [etiquetas.get(c[3:], c[3:]) for c in t.index], "Código": [c[3:] for c in t.index],
                         "Peso en la estimación": t.to_numpy()})


def top_casos(df, k=15):
    """Casos con mayor probabilidad; empates (mesetas del calibrador) se rompen con el puntaje interno del modelo."""
    cols = [c for c in df.columns if c.startswith("c__")]
    return df.assign(_s=df[cols].sum(axis=1) if cols else 0.0).sort_values(["p", "_s"], ascending=False).head(k)


def razones(fila, grupos, frases, k=2):
    vals = {g: sum(float(fila.get(f"c__{c}", 0.0)) for c in cols) for g, cols in grupos.items()}
    pos = sorted(((v, g) for g, v in vals.items() if v > 0), reverse=True)[:k]
    return " · ".join(frases.get(g, g) for _, g in pos) if pos else "Sin un factor dominante que eleve el riesgo"


def politicas(art, ctx):
    """Resume la comparación de estrategias de mantenimiento del notebook (§4.6) para el modelo del tablero.

    Criterio de la estrategia recomendada (definido para el tablero): mejor posición conjunta en log-loss
    (exactitud de la probabilidad) y ECE (calibración) entre las estrategias implementables A, B y C;
    desempate por AUC.
    """
    res = art.get("politicas_resumen")
    if res is None:
        return None
    mod = str(ctx.get("tipo_modelo", "xgb")).upper()
    r = res[res["modelo"] == mod].set_index("política")
    if r.empty:
        return None
    activos = [p for p in r.index if p[:2] in ("A ", "B ", "C ")]
    rank = (r.loc[activos, "LogLoss"].rank() + r.loc[activos, "ECE"].rank()).to_frame("rank")
    rank["auc"] = -r.loc[activos, "AUC"]
    best = rank.sort_values(["rank", "auc"]).index[0]
    bloques = art.get("politicas_bloques", art.get("politicas_dias"))
    oe = None
    if bloques is not None and {"prevalencia", "p_media", "n"} <= set(bloques.columns):
        b = bloques[bloques["modelo"] == mod]
        obs = (b["prevalencia"] * b["n"]).groupby(b["política"]).sum()
        esp = (b["p_media"] * b["n"]).groupby(b["política"]).sum()
        oe = (obs / esp).to_dict()
    alert_col = [c for c in r.columns if c.startswith("%") and "≥1" in c and "PSI" not in c]
    return {"tabla": r, "mejor": best, "oe": oe or {}, "base": "Baseline calibrado", "sin": "Sin calibrar (TT1)",
            "col_alerta": alert_col[0] if alert_col else None, "bloques": bloques, "modelo": mod}


def fila_metricas_modelo(art, ctx):
    mp = art.get("metricas_produccion")
    if mp is None:
        return None
    mod = str(ctx.get("tipo_modelo", "xgb")).upper()
    r = mp[mp["modelo"].str.startswith(mod) & mp["modelo"].str.contains("calibrada") & ~mp["modelo"].str.contains("INCLUYENDO")]
    return r.iloc[0] if len(r) else None


def valor_num(v):
    """Las métricas con intervalo se exportaron como '0.7062 [0.7041, 0.7084]'."""
    try:
        return float(str(v).split()[0])
    except Exception:
        return np.nan


def falsas_alarmas(art):
    t = art.get("monitor_falsas_alarmas")
    if t is None:
        return None
    r = t[t["señal"].str.startswith("ALERTA")]
    return None if r.empty else r.iloc[0]


# =============================================================================
# 6. GRÁFICAS (Plotly, estilo ejecutivo: marcas delgadas, rejilla tenue, un solo eje)
# =============================================================================
def estilo(fig, alto=330, pct_y=True, leyenda=True, x_pct=False):
    fig.update_layout(
        template="none", height=alto, margin=dict(l=8, r=12, t=36 if leyenda else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FUENTE, size=12, color=INK2), showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK, size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FUENTE, size=12, color=INK)),
        bargap=0.35, bargroupgap=0.12,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="outside", ticklen=4, automargin=True,
                     tickfont=dict(color=INK2), title_font=dict(color=INK2, size=12))
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS, automargin=True,
                     tickfont=dict(color=INK2), title_font=dict(color=INK2, size=12))
    if pct_y:
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    if x_pct:
        fig.update_xaxes(tickformat=".0%", rangemode="tozero")
    else:
        fig.update_xaxes(tickformat=",")
    return fig


def fig_real_vs_estimado(x, real, est, x_titulo, tipo="lineas", ref=None, ref_txt=None, alto=330, hover_x="%{x}", categorico=None):
    x = list(x)
    if categorico is None:
        categorico = any(isinstance(v, str) for v in x)
    fig = go.Figure()
    if tipo == "lineas":
        fig.add_scatter(x=x, y=real, mode="lines+markers", name="Real (observado)", line=dict(color=AZUL, width=2),
                        marker=dict(size=8, line=dict(color="white", width=2)), hovertemplate=f"{hover_x}<br>Real: %{{y:.1%}}<extra></extra>")
        fig.add_scatter(x=x, y=est, mode="lines+markers", name="Estimado por el modelo", line=dict(color=NARANJA, width=2),
                        marker=dict(size=8, symbol="diamond", line=dict(color="white", width=2)),
                        hovertemplate=f"{hover_x}<br>Estimado: %{{y:.1%}}<extra></extra>")
    else:
        fig.add_bar(x=x, y=real, name="Real (observado)", marker=dict(color=AZUL, cornerradius=4),
                    hovertemplate=f"{hover_x}<br>Real: %{{y:.1%}}<extra></extra>")
        fig.add_scatter(x=x, y=est, mode="markers", name="Estimado por el modelo",
                        marker=dict(color=NARANJA, size=11, symbol="diamond", line=dict(color="white", width=2)),
                        hovertemplate=f"{hover_x}<br>Estimado: %{{y:.1%}}<extra></extra>")
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=MUTED, width=1), annotation_text=ref_txt, annotation_position="bottom right",
                      annotation_font=dict(color=INK2, size=11))
    fig.update_xaxes(title_text=x_titulo)
    fig = estilo(fig, alto=alto)
    if categorico:
        fig.update_xaxes(type="category")
    return fig


def fig_barras_h(etiquetas, valores, texto=None, hover=None, color=AZUL, alto=None, x_pct=True, x_titulo=None):
    alto = alto or max(220, 34 * len(etiquetas) + 60)
    fig = go.Figure(go.Bar(y=list(etiquetas)[::-1], x=list(valores)[::-1], orientation="h",
                           marker=dict(color=color, cornerradius=4),
                           text=None if texto is None else list(texto)[::-1], textposition="outside", cliponaxis=False,
                           textfont=dict(color=INK2, size=12),
                           hovertext=None if hover is None else list(hover)[::-1],
                           hovertemplate="%{hovertext}<extra></extra>" if hover is not None else "%{y}: %{x}<extra></extra>"))
    fig = estilo(fig, alto=alto, pct_y=False, leyenda=False, x_pct=x_pct)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", ticks="", tickfont=dict(color=INK, size=12))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title_text=x_titulo)
    if valores is not None and len(valores):
        fig.update_xaxes(range=[0, max(valores) * 1.25])
    return fig


def fig_concentracion(bandas, pct_casos, pct_eventos, unidad, evento):
    fig = go.Figure()
    fig.add_bar(x=bandas, y=pct_casos, name=f"% de {unidad}", marker=dict(color=AZUL, cornerradius=4),
                text=[pct(v) for v in pct_casos], textposition="outside", cliponaxis=False, textfont=dict(color=INK2),
                hovertemplate="Nivel %{x}<br>" + f"{unidad.capitalize()}: " + "%{y:.1%}<extra></extra>")
    fig.add_bar(x=bandas, y=pct_eventos, name=f"% de {evento}", marker=dict(color=NARANJA, cornerradius=4),
                text=[pct(v) for v in pct_eventos], textposition="outside", cliponaxis=False, textfont=dict(color=INK2),
                hovertemplate="Nivel %{x}<br>" + f"{evento.capitalize()}: " + "%{y:.1%}<extra></extra>")
    fig.update_xaxes(title_text="Nivel de riesgo estimado por el modelo")
    fig = estilo(fig, alto=330)
    fig.update_yaxes(range=[0, max(max(pct_casos), max(pct_eventos)) * 1.2])
    return fig


def fig_influencia(imp):
    return fig_barras_h(imp.index, imp["peso"].to_numpy(), texto=[pct(v) for v in imp["peso"]],
                        hover=[f"{g}: {pct(v)} del peso total" for g, v in zip(imp.index, imp["peso"])],
                        x_titulo="Peso relativo en la estimación del riesgo")


def mostrar_grafica(g, key):
    """Título que comunica la conclusión + cómo leerla + hallazgo principal + tabla de datos (accesibilidad)."""
    with st.container(border=True):
        st.markdown(f'<div class="tt-gtit">{esc(g["titulo"])}</div>', unsafe_allow_html=True)
        st.plotly_chart(g["fig"], config={"displayModeBar": False, "responsive": True}, key=key)
        st.markdown(f'<div class="tt-gnota"><b>Cómo leerla:</b> {esc(g["como_leer"])}</div>'
                    f'<div class="tt-gnota"><b>Hallazgo:</b> {esc(g["hallazgo"])}</div>', unsafe_allow_html=True)
        if g.get("tabla") is not None:
            with st.expander("Ver datos de la gráfica"):
                st.dataframe(g["tabla"], hide_index=True)


# =============================================================================
# 7. COMPONENTES DE LA VISTA EJECUTIVA
# =============================================================================
def encabezado_seccion(num_, titulo, sub=""):
    st.markdown(f'<div class="tt-sec"><span class="num">{num_}</span><span class="tit">{titulo}</span>'
                f'<span class="sub">{sub}</span></div>', unsafe_allow_html=True)


def tarjeta_resumen(r, nivel):
    filas = [("Situación general", r["situacion"]), ("Hallazgo principal", r["hallazgo"]), ("Principal riesgo", r["riesgo"]),
             ("Principal oportunidad", r["oportunidad"]), ("Acción prioritaria", r["accion"])]
    html = "".join(f'<div class="fila"><div class="lab">{k}</div><div class="txt">{esc(v)}</div></div>' for k, v in filas)
    st.markdown(f'<div style="display:flex;justify-content:flex-end;margin:-.2rem 0 .35rem 0">{badge(nivel, "Nivel de atención general: " + NIVEL_TXT.get(nivel, "—").lower())}</div>'
                f'<div class="tt-resumen">{html}</div>', unsafe_allow_html=True)


def tarjetas_kpi(kpis):
    cols = st.columns(len(kpis))
    for c, k in zip(cols, kpis):
        with c:
            st.markdown(
                f'<div class="tt-card tt-kpi" title="{esc(k.get("ayuda", ""))}">'
                f'<div class="nom"><span>{esc(k["nombre"])}</span>{badge(k["nivel"], NIVEL_TXT.get(k["nivel"], "Info"))}</div>'
                f'<div class="val">{esc(k["valor"])} <small>{esc(k.get("unidad", ""))}</small></div>'
                f'<div class="int">{esc(k["interpretacion"])}</div>'
                f'<div class="cmp">{esc(k["comparacion"])}</div></div>', unsafe_allow_html=True)
    ayudas = [f"**{k['nombre']}:** {k['ayuda']}" for k in kpis if k.get("ayuda")]
    if ayudas:
        with st.expander("¿Qué significa cada indicador?"):
            st.markdown("\n\n".join(ayudas))


def bloque_diagnostico(items):
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for c, (pregunta, bullets) in zip(cols, items[i:i + 2]):
            html = "".join(f"<li>{esc(b)}</li>" for b in bullets if b)
            with c:
                st.markdown(f'<div class="tt-diag"><div class="q">{esc(pregunta)}</div><ul>{html}</ul></div>', unsafe_allow_html=True)


def tarjetas_alertas(alertas):
    if not alertas:
        st.markdown('<div class="tt-nota">No se detectaron hallazgos que requieran atención con los filtros actuales.</div>',
                    unsafe_allow_html=True)
        return
    for a in sorted(alertas, key=lambda a: ORDEN_SEV[a["sev"]]):
        nivel = SEVERIDAD[a["sev"]]
        st.markdown(
            f'<div class="tt-alerta" style="border-left-color:{ESTADO[nivel]}">{badge(nivel, a["sev"])}'
            f'<div class="tit">{esc(a["titulo"])}</div><div class="grid">'
            f'<div><div class="k">Qué se detectó</div><div class="v">{esc(a["que"])}</div></div>'
            f'<div><div class="k">Por qué importa</div><div class="v">{esc(a["por_que"])}</div></div>'
            f'<div><div class="k">Si no se atiende</div><div class="v">{esc(a["si_no"])}</div></div>'
            f'<div><div class="k">Acción recomendada</div><div class="v">{esc(a["accion"])}</div></div>'
            f'</div></div>', unsafe_allow_html=True)


def tarjetas_recomendaciones(recs):
    prio = {"Alta": "alto", "Crítica": "critico", "Media": "medio", "Baja": "bajo"}
    for i, r in enumerate(recs, 1):
        st.markdown(
            f'<div class="tt-rec"><div class="meta">{badge(prio.get(r["prioridad"], "info"), "Prioridad " + r["prioridad"].lower())}'
            f'<span>Responsable sugerido: <b>{esc(r["responsable"])}</b></span></div>'
            f'<div class="acc">{i}. {esc(r["accion"])}</div><div class="grid">'
            f'<div><div class="k">Justificación con datos</div><div class="v">{esc(r["justificacion"])}</div></div>'
            f'<div><div class="k">Resultado esperado</div><div class="v">{esc(r["resultado"])}</div></div>'
            f'<div><div class="k">Precauciones</div><div class="v">{esc(r["precaucion"])}</div></div>'
            f'</div></div>', unsafe_allow_html=True)


def nivel_general(alertas, kpis):
    """Nivel general = el mayor nivel de atención entre los indicadores clave (las alertas conservan su propia severidad)."""
    orden = ["critico", "alto", "medio", "bajo"]
    niveles = [k["nivel"] for k in kpis]
    for n in orden:
        if n in niveles:
            return n
    return "bajo"


def aviso_muestra(m, ds):
    if m["n"] < REGLAS["min_muestra"][ds] or m["eventos"] < REGLAS["min_eventos"]:
        st.markdown(f'<div class="tt-aviso">Muestra pequeña con estos filtros ({num(m["n"])} registros, {num(m["eventos"])} eventos): '
                    'las cifras pueden variar mucho y deben interpretarse con cautela.</div>', unsafe_allow_html=True)


# =============================================================================
# 8. AEROLÍNEA · RETRASOS
# =============================================================================
FRASES_AIRLINE = {
    "Historial de retrasos del vuelo y su ruta": "historial frecuente de retrasos en este vuelo o ruta",
    "Retrasos del día anterior (aerolínea y aeropuertos)": "muchos retrasos ayer en su aerolínea o aeropuertos",
    "Desempeño habitual de aerolínea y aeropuertos": "aerolínea o aeropuertos con tasa habitual de retraso alta",
    "Hora de salida": "sale en una franja horaria con más retrasos",
    "Día de la semana": "día de la semana con más retrasos",
    "Duración programada del vuelo": "duración del vuelo asociada a más retrasos",
}
FRANJAS = ["Madrugada (00–05 h)", "Mañana (06–11 h)", "Tarde (12–17 h)", "Noche (18–23 h)"]
DIAS_SEMANA = {1: "lunes", 2: "martes", 3: "miércoles", 4: "jueves", 5: "viernes", 6: "sábado", 7: "domingo"}


def filtrar_airline(df, f):
    m = (df["dia"] >= f["dias"][0]) & (df["dia"] <= f["dias"][1])
    for col, key in [("aerolinea", "aerolineas"), ("origen", "origenes"), ("franja", "franjas"), ("banda", "niveles")]:
        if f.get(key):
            m &= df[col].isin(f[key])
    if f.get("buscar"):
        q = f["buscar"].strip().upper().replace("->", "→")
        m &= df["vuelo"].str.upper().str.contains(q, regex=False) | df["ruta"].str.upper().str.contains(q, regex=False)
    return df[m]


def analizar_airline(df, todo, ctx, art, desc):
    R = REGLAS
    m, mt = metricas(df), metricas(todo)
    if m["n"] == 0:
        return None
    train = ctx["tasa_train"]
    cut_lo, cut_hi = ctx["cortes_riesgo"]
    min_seg = min(R["min_segmento"]["airline"], max(100, int(0.01 * m["n"])))
    pol = politicas(art, ctx)
    dias = df.groupby("dia").agg(n=("y", "size"), tasa=("y", "mean"), est=("p", "mean"))
    dias = dias[dias["n"] >= 200]
    dias["brecha"] = dias["est"] - dias["tasa"]
    n_dias = df["dia"].nunique()
    aer = tasa_por(df, "aerolinea", min_seg)
    ori = tasa_por(df, "origen", min_seg)
    rut = tasa_por(df, "ruta", max(100, min_seg // 3))
    fr = tasa_por(df, "franja", min_seg, orden=FRANJAS)
    hora = tasa_por(df, "hora", 100, orden=list(range(24)))
    imp = influencia(df, ctx["grupos"])
    brecha = m["p_media"] - m["tasa"]
    # persistencia día a día: tasa de hoy según cuántos retrasos tuvo ayer el aeropuerto de origen (terciles)
    pers = None
    if df["retraso_ayer_origen"].nunique() > 3 and m["n"] >= 300:
        q = df["retraso_ayer_origen"].quantile([1 / 3, 2 / 3]).to_numpy()
        if q[1] > q[0]:
            cat = pd.cut(df["retraso_ayer_origen"], [-np.inf, q[0], q[1], np.inf],
                         labels=[f"Pocos (< {pct(q[0])})", f"Intermedios ({pct(q[0])}–{pct(q[1])})", f"Muchos (≥ {pct(q[1])})"])
            pers = df.assign(t=cat).groupby("t", observed=True).agg(n=("y", "size"), tasa=("y", "mean"), est=("p", "mean"))
    dif_pers = (pers["tasa"].iloc[-1] - pers["tasa"].iloc[0]) if pers is not None and len(pers) == 3 else np.nan
    dif_fr = (fr["tasa"].max() - fr["tasa"].min()) if len(fr) >= 2 else np.nan
    fr_peor = fr.sort_values("tasa").iloc[-1] if len(fr) else None
    fr_mejor = fr.sort_values("tasa").iloc[0] if len(fr) else None
    top_ori = ori[ori["tasa"] >= m["tasa"] + R["segmento_pp"][0] / 100].head(3)
    top_aer = aer[aer["tasa"] >= m["tasa"] + R["segmento_pp"][0] / 100].head(3)
    psi = art.get("psi_splits")
    psi_max = float(psi["train→producción"].max()) if psi is not None else np.nan
    fa = falsas_alarmas(art)

    # ---------------- KPIs
    kpis = [
        {"nombre": "Vuelos con riesgo alto de retraso", "valor": num(m["n_alto"]), "unidad": f"({pct(m['share_alto'])} de los vuelos)",
         "interpretacion": f"El modelo estima ≥ {pct(cut_hi)} de probabilidad de retraso; en la práctica se retrasó el {pct(m['tasa_alto'])} de ellos.",
         "nivel": nivel_mayor_peor(m["share_alto"], R["share_alto"]["airline"]),
         "comparacion": f"Periodo completo de evaluación: {pct(mt['share_alto'])} de los vuelos en riesgo alto.",
         "ayuda": f"Vuelos cuya probabilidad estimada de retraso es de {pct(cut_hi)} o más (nivel alto)."},
        {"nombre": "Vuelos que llegaron con retraso", "valor": pct(m["tasa"], 1), "unidad": f"({num(m['eventos'])} de {num(m['n'])})",
         "interpretacion": f"{de_cada(m['tasa'], 10).capitalize()} vuelos llegaron tarde en el periodo seleccionado.",
         "nivel": nivel_mayor_peor((m["tasa"] - train) * 100, R["cambio_tasa_pp"]["airline"]),
         "comparacion": f"Semanas con las que aprendió el modelo: {pct(train, 1)} ({pp(m['tasa'] - train)}).",
         "ayuda": "Proporción real de vuelos con retraso. Se compara con el periodo con el que se entrenó el modelo porque un cambio grande exige mantenerlo."},
        {"nombre": "Retrasos detectados con alerta", "valor": pct(m["sensibilidad"]), "unidad": "de los vuelos retrasados (sensibilidad)",
         "interpretacion": f"De los vuelos que recibieron alerta, {pct(m['precision'])} sí se retrasó (precisión).",
         "nivel": nivel_mayor_mejor(min_valido(m["sensibilidad"], m["precision"]), R["anticipacion"]),
         "comparacion": f"Si las alertas se asignaran al azar, acertarían {pct(m['tasa'])} de las veces.",
         "ayuda": f"Sensibilidad: % de vuelos retrasados que tenían alerta (probabilidad estimada ≥ {pct(ctx['umbral_alerta'])}). Precisión: % de alertas que terminaron en retraso."},
        {"nombre": "Desajuste entre lo estimado y lo real", "valor": f"{abs(brecha) * 100:.1f}", "unidad": "pp de diferencia",
         "interpretacion": (f"Estimado {pct(m['p_media'], 1)} vs. real {pct(m['tasa'], 1)}: "
                            + ("las probabilidades coinciden con lo observado." if abs(brecha) < R["brecha_pp"][0] / 100 else
                               f"el modelo {'sobreestima' if brecha > 0 else 'subestima'} el retraso.")),
         "nivel": nivel_mayor_peor(abs(brecha) * 100, R["brecha_pp"]),
         "comparacion": (f"Día con mayor desajuste: {abs(dias['brecha']).max() * 100:.1f} pp (día {int(abs(dias['brecha']).idxmax())})." if len(dias) else "—"),
         "ayuda": "Diferencia, en puntos porcentuales (pp), entre el % de retrasos que esperaba el modelo y el % real. Mide la calibración: si el modelo estima 60 %, de cada 100 vuelos así deberían retrasarse unos 60."},
    ]

    # ---------------- diagnóstico
    seg_txt = []
    if len(top_aer):
        seg_txt.append("Aerolíneas con más retrasos: " + ", ".join(f"{r.aerolinea} ({pct(r.tasa)}, {num(r.n)} vuelos)" for r in top_aer.itertuples()) + ".")
    if len(top_ori):
        seg_txt.append("Aeropuertos de origen más afectados: " + ", ".join(f"{r.origen} ({pct(r.tasa)}, {num(r.n)} vuelos)" for r in top_ori.itertuples()) + ".")
    if len(rut):
        r0 = rut.iloc[0]
        seg_txt.append(f"Ruta con mayor tasa (≥ {num(max(100, min_seg // 3))} vuelos): {r0.ruta} ({pct(r0.tasa)} de {num(r0.n)} vuelos).")
    if fr_peor is not None:
        seg_txt.append(f"Franja con más retrasos: {fr_peor.franja.lower()} ({pct(fr_peor.tasa)}); la de menos: {fr_mejor.franja.lower()} ({pct(fr_mejor.tasa)}).")
    if not seg_txt:
        seg_txt.append("Con estos filtros, ningún segmento con volumen suficiente supera la tasa promedio en 10 pp o más.")
    imp_txt = [f"{g}: {pct(r.peso)} del peso total en la estimación." for g, r in imp.head(3).iterrows()]
    por_que = []
    if not pd.isna(dif_pers):
        por_que.append(f"Persistencia entre días: cuando el aeropuerto de origen tuvo muchos retrasos ayer, hoy se retrasó el {pct(pers['tasa'].iloc[-1])} "
                       f"de sus vuelos; cuando tuvo pocos, el {pct(pers['tasa'].iloc[0])} ({'+' if dif_pers >= 0 else ''}{dif_pers * 100:.0f} pp).")
    if len(hora) >= 6:
        h_ini = hora[hora["hora"].astype(int) <= 8]
        h_fin = hora[hora["hora"].astype(int) >= 17]
        if len(h_ini) and len(h_fin):
            a_, b_ = h_ini["eventos"].sum() / h_ini["n"].sum(), h_fin["eventos"].sum() / h_fin["n"].sum()
            por_que.append(f"Acumulación durante el día: salidas antes de las 9 h se retrasan {pct(a_)}; desde las 17 h, {pct(b_)}.")
    if not pd.isna(psi_max) and psi_max < R["psi_estable"]:
        por_que.append("La mezcla de vuelos (aerolíneas, aeropuertos, horarios y duraciones) prácticamente no cambió respecto del entrenamiento; "
                       "lo que cambia es la proporción de retrasos.")
    por_que.append("Estos patrones son compatibles con congestión acumulada y efecto dominó, pero los datos no incluyen clima, rotación de aeronaves "
                   "ni tripulaciones: muestran asociación, no prueban la causa.")
    impacto = [f"{num(m['eventos'])} vuelos retrasados en {n_dias} día(s) (≈ {num(m['eventos'] / max(n_dias, 1))} por día).",
               f"Los vuelos de riesgo alto ({pct(m['share_alto'])} de la operación) concentran el {pct(m['cob_alto'])} de los retrasos.",
               "No hay datos de costos, pasajeros ni minutos de retraso: el impacto se expresa en número de vuelos."]
    continua = []
    if not pd.isna(dif_pers) and dif_pers > 0.05:
        continua.append("Si el patrón se mantiene, los aeropuertos y aerolíneas con muchos retrasos hoy tenderían a repetir mañana (asociación observada).")
    if len(dias) >= 2:
        continua.append(f"La tasa diaria varió entre {pct(dias['tasa'].min())} y {pct(dias['tasa'].max())}: el entorno cambia día con día "
                        "y el modelo necesita mantenimiento continuo.")
    if pol:
        t = pol["tabla"]
        continua.append(f"Sin mantenimiento, el error de calibración (diferencia promedio entre probabilidad estimada y frecuencia real) "
                        f"fue de {pts(t.loc[pol['base'], 'ECE'])}; con la estrategia '{pol['mejor']}' bajó a {pts(t.loc[pol['mejor'], 'ECE'])} "
                        "(10 días de evaluación, sin filtros).")
    diagnostico = [
        ("¿Qué está ocurriendo?", [f"En {desc}, {pct(m['tasa'])} de {num(m['n'])} vuelos llegó con retraso, frente a {pct(train)} en las semanas "
                                   "con las que se entrenó el modelo.",
                                   (f"La tasa diaria osciló entre {pct(dias['tasa'].min())} (día {int(dias['tasa'].idxmin())}) y "
                                    f"{pct(dias['tasa'].max())} (día {int(dias['tasa'].idxmax())})." if len(dias) >= 2 else None)]),
        ("¿Dónde se concentra?", seg_txt),
        ("¿Qué factores pesan más en la estimación?", imp_txt + ["El peso indica cuánto usa el modelo cada factor para estimar el riesgo; no implica causa."]),
        ("¿Por qué podría estar ocurriendo?", por_que),
        ("¿Qué impacto puede tener?", impacto),
        ("¿Qué podría ocurrir si continúa?", continua),
    ]

    # ---------------- alertas
    alertas = []
    for nombre, tabla, col in [("aeropuertos de origen", top_ori, "origen"), ("aerolíneas", top_aer, "aerolinea")]:
        if len(tabla):
            exceso = (tabla["tasa"].iloc[0] - m["tasa"]) * 100
            sev = "Crítica" if exceso >= R["segmento_pp"][1] else "Alta"
            lista = ", ".join(f"{getattr(r, col)} ({pct(r.tasa)})" for r in tabla.itertuples())
            alertas.append({"sev": sev, "titulo": f"Retrasos muy por encima del promedio en {nombre}: {lista}",
                            "que": f"Tasa de retraso hasta {exceso:.0f} pp mayor que el promedio del filtro ({pct(m['tasa'])}), con volumen ≥ {num(min_seg)} vuelos.",
                            "por_que": f"Concentran {num(tabla['eventos'].sum())} retrasos; su desempeño arrastra la puntualidad global.",
                            "si_no": "Los retrasos tienden a repetirse al día siguiente (persistencia observada); los datos no permiten medir si se propagan a otros vuelos.",
                            "accion": "Revisar capacidad, holguras de conexión y recursos en tierra de estos puntos; confirmar causas con datos operativos."})
    cambio = m["tasa"] - train
    if cambio * 100 >= R["cambio_tasa_pp"]["airline"][1]:
        alertas.append({"sev": "Crítica" if cambio * 100 >= R["cambio_tasa_pp"]["airline"][2] else "Alta",
                        "titulo": "El entorno operativo cambió: hay más retrasos que cuando se entrenó el modelo",
                        "que": f"Tasa actual {pct(m['tasa'])} vs. {pct(train)} en entrenamiento ({pp(cambio)}).",
                        "por_que": "Un modelo entrenado en condiciones distintas pierde exactitud si no se actualiza.",
                        "si_no": "Las probabilidades dejarían de reflejar la realidad y las prioridades operativas serían menos confiables.",
                        "accion": f"Mantener el modelo con la estrategia '{pol['mejor'] if pol else 'híbrida'}' (reajuste ante alertas y reentrenamiento periódico)."})
    if abs(brecha) * 100 >= R["brecha_pp"][0]:
        alertas.append({"sev": "Alta" if abs(brecha) * 100 >= R["brecha_pp"][1] else "Media",
                        "titulo": f"El modelo {'sobreestima' if brecha > 0 else 'subestima'} los retrasos en {abs(brecha) * 100:.1f} pp",
                        "que": f"Estimado {pct(m['p_media'], 1)} vs. real {pct(m['tasa'], 1)} en el filtro actual.",
                        "por_que": "Decisiones basadas en probabilidades desajustadas asignan recursos de más o de menos.",
                        "si_no": "El desajuste puede crecer si el nivel de retrasos sigue cambiando.",
                        "accion": "Recalibrar con los días más recientes y vigilar el desajuste diario."})
    if not pd.isna(dif_fr) and dif_fr * 100 >= 5 and fr_peor is not None:
        alertas.append({"sev": "Media", "titulo": f"La franja {fr_peor.franja.lower()} concentra más retrasos",
                        "que": f"{pct(fr_peor.tasa)} de retraso vs. {pct(fr_mejor.tasa)} en la franja {fr_mejor.franja.lower()}.",
                        "por_que": "La tasa de retraso crece a lo largo del día; es compatible con congestión acumulada, aunque estos datos no lo comprueban.",
                        "si_no": "Las salidas de esa franja seguirán mostrando más retrasos si nada cambia en su programación.",
                        "accion": "Evaluar holguras y programación en esa franja con el área de itinerarios."})
    if not pd.isna(psi_max) and psi_max < R["psi_estable"]:
        alertas.append({"sev": "Informativa", "titulo": "El cambio no viene de la mezcla de vuelos sino del nivel de retrasos",
                        "que": f"Índice de estabilidad poblacional (PSI) máximo de las variables: {psi_max:.3f} (estable si es menor que {R['psi_estable']}).",
                        "por_que": "Vigilar solo las características de los vuelos no detectaría el problema.",
                        "si_no": "Un monitor basado solo en la composición daría falsa tranquilidad.",
                        "accion": "Monitorear la tasa diaria real contra la estimada (como hace el monitor del proyecto)."})
    if fa is not None:
        alertas.append({"sev": "Informativa", "titulo": "El monitor de cambios es confiable",
                        "que": f"En pruebas sin cambios reales, el monitor dio falsas alarmas en {fa['falsas_alarmas_nivel≥1_%']:.1f} % de los bloques (criterio < 15 %).",
                        "por_que": "Las alertas diarias en producción reflejan cambios reales, no ruido del sistema.",
                        "si_no": "—", "accion": "Tomar en serio las alertas del monitor y atenderlas con la estrategia de mantenimiento."})

    # ---------------- recomendaciones
    recs = [
        {"accion": "Revisar cada mañana la lista de vuelos de riesgo alto antes de iniciar la operación", "prioridad": "Alta",
         "responsable": "Centro de Control de Operaciones",
         "justificacion": f"{num(m['n_alto'])} vuelos de riesgo alto se retrasaron {pct(m['tasa_alto'])} de las veces (promedio: {pct(m['tasa'])}); concentran el {pct(m['cob_alto'])} de los retrasos.",
         "resultado": "Asignar holguras, puertas, tripulaciones y aeronaves de reserva donde la probabilidad de retraso es mayor.",
         "precaucion": "La probabilidad no explica la causa del retraso; es apoyo a la decisión, no una regla automática."},
    ]
    if len(top_ori) or len(top_aer):
        nombres = ", ".join(list(top_ori["origen"].head(2)) + list(top_aer["aerolinea"].head(2)))
        recs.append({"accion": f"Reforzar la operación en los puntos con más retrasos ({nombres})", "prioridad": "Alta",
                     "responsable": "Planeación de red y operación en aeropuertos",
                     "justificacion": "Su tasa supera al promedio en 10 pp o más, con volumen suficiente (ver Alertas).",
                     "resultado": "Reducir la concentración de retrasos y su repetición al día siguiente.",
                     "precaucion": "La asociación puede reflejar clima o congestión no medidos; validar con datos operativos antes de invertir."})
    if not pd.isna(dif_pers) and dif_pers > 0.05:
        recs.append({"accion": "Usar los retrasos del día anterior como señal de alerta temprana", "prioridad": "Media",
                     "responsable": "Centro de Control de Operaciones / Analítica",
                     "justificacion": f"Aeropuertos con muchos retrasos ayer: {pct(pers['tasa'].iloc[-1])} hoy; con pocos: {pct(pers['tasa'].iloc[0])}.",
                     "resultado": "Anticipar ajustes operativos desde el inicio del día.",
                     "precaucion": "Es una relación estadística entre días; no demuestra propagación de retrasos dentro del mismo día."})
    if fr_peor is not None and not pd.isna(dif_fr) and dif_fr * 100 >= 5:
        recs.append({"accion": f"Revisar la programación y las holguras de la franja {fr_peor.franja.lower()}", "prioridad": "Media",
                     "responsable": "Programación de itinerarios",
                     "justificacion": f"{pct(fr_peor.tasa)} de retraso vs. {pct(fr_mejor.tasa)} en la franja {fr_mejor.franja.lower()}.",
                     "resultado": "Posible reducción de retrasos en las franjas tardías, a confirmar con una prueba piloto.",
                     "precaucion": "Requiere evaluar capacidad y costos, información que no está en este tablero."})
    if pol:
        t = pol["tabla"]
        recs.append({"accion": f"Mantener el modelo con la estrategia '{pol['mejor']}'", "prioridad": "Alta",
                     "responsable": "Equipo de analítica / ciencia de datos",
                     "justificacion": f"{POLITICAS_TXT.get(pol['mejor'], '')}: error de calibración {pts(t.loc[pol['base'], 'ECE'])} → {pts(t.loc[pol['mejor'], 'ECE'])}; "
                                      f"costo total ≈ {t.loc[pol['mejor'], 'seg_en_intervenciones']:.0f} s de cómputo en 10 días.",
                     "resultado": "Probabilidades confiables aunque el nivel de retrasos cambie día con día.",
                     "precaucion": "Evaluado en 10 días; confirmar con un periodo más largo antes de automatizar."})
    recs.append({"accion": "Incorporar clima, rotación de aeronaves y minutos de retraso al análisis", "prioridad": "Media",
                 "responsable": "Tecnologías de información / Datos",
                 "justificacion": "Sin esos datos no es posible confirmar el efecto dominó ni cuantificar el costo de los retrasos.",
                 "resultado": "Explicar causas y medir impacto económico, no solo probabilidad.",
                 "precaucion": "Requiere integrar nuevas fuentes y validar su calidad."})

    # ---------------- gráficas
    graficas = []
    if len(dias) >= 2:
        fig = fig_real_vs_estimado(dias.index, dias["tasa"], dias["est"], "Día del periodo analizado", ref=train,
                                   ref_txt=f"Promedio durante el entrenamiento: {pct(train)}", hover_x="Día %{x}")
        mx = abs(dias["brecha"]).max()
        graficas.append({"titulo": f"La tasa de retraso varió de {pct(dias['tasa'].min())} a {pct(dias['tasa'].max())} entre días; "
                                   + ("el modelo siguió de cerca el nivel real" if mx < 0.05 else "el modelo no siempre siguió el nivel real"),
                         "como_leer": "Cada punto es un día. Azul: % real de vuelos retrasados. Naranja: % que el modelo esperaba. Líneas juntas = probabilidades confiables.",
                         "hallazgo": f"Diferencia máxima de {mx * 100:.1f} pp (día {int(abs(dias['brecha']).idxmax())}); todos los días quedaron por encima del promedio de entrenamiento."
                         if (dias["tasa"] > train).all() else f"Diferencia máxima de {mx * 100:.1f} pp (día {int(abs(dias['brecha']).idxmax())}).",
                         "fig": fig, "tabla": dias.reset_index().rename(columns={"dia": "Día", "n": "Vuelos", "tasa": "Real", "est": "Estimado", "brecha": "Diferencia"})})
    if len(hora) >= 6:
        h = hora.copy()
        h["hora"] = h["hora"].astype(int)
        op = h[h["hora"] >= 5]                       # jornada operativa (las salidas de 0–4 h son pocas)
        sube = len(op) >= 4 and np.corrcoef(op["hora"], op["tasa"])[0, 1] > 0.5
        pico, valle = op.sort_values("tasa").iloc[-1], op.sort_values("tasa").iloc[0]
        fig = fig_real_vs_estimado(h["hora"], h["tasa"], h["estimado"], "Hora de salida programada", tipo="barras", hover_x="%{x} h", categorico=False)
        graficas.append({"titulo": (f"Los retrasos aumentan conforme avanza el día: de {pct(valle.tasa)} a las {int(valle.hora)} h a {pct(pico.tasa)} a las {int(pico.hora)} h"
                                    if sube else f"El retraso varía según la hora de salida; el pico está a las {int(pico.hora)} h ({pct(pico.tasa)})"),
                         "como_leer": "Barras: % real de vuelos retrasados por hora de salida (horas con ≥ 100 vuelos). Rombos: % estimado por el modelo.",
                         "hallazgo": "Patrón compatible con congestión acumulada durante el día (no comprobado con estos datos): conviene revisar holguras en franjas tardías." if sube
                         else "No se observa un patrón creciente claro con estos filtros.",
                         "fig": fig, "tabla": h.rename(columns={"hora": "Hora", "n": "Vuelos", "eventos": "Retrasados", "tasa": "Real", "estimado": "Estimado"})})
    if pers is not None and len(pers) == 3:
        fig = fig_real_vs_estimado(list(pers.index.astype(str)), pers["tasa"], pers["est"], "Retrasos del aeropuerto de origen el día anterior", tipo="barras")
        graficas.append({"titulo": (f"Los retrasos de ayer se asocian con los de hoy: {pct(pers['tasa'].iloc[-1])} vs. {pct(pers['tasa'].iloc[0])}"
                                    if dif_pers > 0.05 else "Con estos filtros los retrasos de ayer no anticipan claramente los de hoy"),
                         "como_leer": "Vuelos agrupados en tres partes iguales según qué tan mal le fue ayer a su aeropuerto de origen. Barras: % real de retraso hoy.",
                         "hallazgo": "La persistencia entre días permite actuar desde la mañana; los datos no permiten afirmar propagación dentro del mismo día.",
                         "fig": fig, "tabla": pers.reset_index().rename(columns={"t": "Retrasos ayer en origen", "n": "Vuelos", "tasa": "Real", "est": "Estimado"})})
    conc = tasa_por(df, "origen", 0).sort_values("eventos", ascending=False)
    if len(conc):
        top10 = conc.head(10)
        share10 = top10["eventos"].sum() / max(m["eventos"], 1)
        fig = fig_barras_h(top10["origen"], top10["eventos"].to_numpy(), texto=[f"{num(e)} · {pct(t)}" for e, t in zip(top10["eventos"], top10["tasa"])],
                           hover=[f"{o}: {num(e)} retrasos de {num(n)} vuelos ({pct(t)})" for o, e, n, t in zip(top10["origen"], top10["eventos"], top10["n"], top10["tasa"])],
                           x_pct=False, x_titulo="Vuelos retrasados")
        graficas.append({"titulo": f"10 aeropuertos de origen concentran el {pct(share10)} de los vuelos retrasados",
                         "como_leer": "Barras: número de vuelos retrasados por aeropuerto de origen. Etiqueta: retrasos · % de sus vuelos.",
                         "hallazgo": f"{top10.iloc[0]['origen']} encabeza en volumen ({num(top10.iloc[0]['eventos'])} retrasos). Volumen alto y tasa alta = prioridad operativa.",
                         "fig": fig, "tabla": top10.rename(columns={"origen": "Aeropuerto", "n": "Vuelos", "eventos": "Retrasados", "tasa": "Tasa real", "estimado": "Tasa estimada"})})

    # ---------------- casos: lista operativa del último día del periodo filtrado
    ultimo = int(df["dia"].max())
    base_casos = df[df["dia"] == ultimo] if (df["dia"] == ultimo).sum() >= 15 else df
    top = top_casos(base_casos, 15)
    casos = pd.DataFrame({"Día": top["dia"], "Vuelo": top["vuelo"], "Ruta": top["ruta"],
                          "Salida": [f"{int(t) // 60:02d}:{int(t) % 60:02d}" for t in top["minuto_salida"]],
                          "Probabilidad de retraso": top["p"].astype(float), "Nivel": top["banda"],
                          "Principales razones (según el modelo)": [razones(r, ctx["grupos"], FRASES_AIRLINE) for _, r in top.iterrows()],
                          "Resultado real": np.where(top["y"] == 1, "Retrasado", "A tiempo")})

    resumen = {
        "situacion": f"En {desc}, {pct(m['tasa'])} de los {num(m['n'])} vuelos llegó con retraso ({pp(m['tasa'] - train, 0)} frente al periodo de entrenamiento).",
        "hallazgo": (f"Los retrasos de ayer se asocian con los de hoy: cuando el aeropuerto de origen tuvo muchos retrasos el día anterior, se retrasó el "
                     f"{pct(pers['tasa'].iloc[-1])} de sus vuelos, contra {pct(pers['tasa'].iloc[0])} cuando tuvo pocos."
                     if not pd.isna(dif_pers) and dif_pers > 0.05 else
                     f"El factor con más peso en la estimación es '{imp.index[0].lower()}' ({pct(imp['peso'].iloc[0])})." if len(imp) else "—"),
        "riesgo": (f"{num(m['n_alto'])} vuelos ({pct(m['share_alto'])}) tienen riesgo alto y se retrasó el {pct(m['tasa_alto'])} de ellos"
                   + (f"; {top_ori.iloc[0]['origen']} es el aeropuerto con mayor tasa ({pct(top_ori.iloc[0]['tasa'])})." if len(top_ori) else ".")),
        "oportunidad": f"Atender primero los vuelos de riesgo alto cubre el {pct(m['cob_alto'])} de los retrasos revisando solo el {pct(m['share_alto'])} de la operación.",
        "accion": "Revisar cada mañana la lista de vuelos de riesgo alto, reforzar los puntos con más retrasos y mantener el modelo "
                  f"con la estrategia '{pol['mejor'] if pol else 'híbrida'}'.",
    }
    return {"m": m, "resumen": resumen, "kpis": kpis, "diagnostico": diagnostico, "alertas": alertas, "recs": recs,
            "graficas": graficas, "imp": imp, "casos": casos, "pol": pol,
            "titulo_casos": (f"Vuelos que requieren atención el día {ultimo} (último día del periodo filtrado; 15 con mayor probabilidad)"
                             if len(base_casos) < len(df) else "Vuelos que requieren atención (15 con mayor probabilidad)")}


# =============================================================================
# 9. HOSPITAL · REINGRESOS
# =============================================================================
FRASES_DIABETES = {
    "Uso previo de servicios hospitalarios": "hospitalizaciones o visitas previas frecuentes",
    "Destino al alta": "destino al alta asociado a más reingresos",
    "Diagnósticos (cantidad y tipo)": "varios diagnósticos o diagnóstico de mayor riesgo",
    "Estancia y procedimientos": "estancia larga o muchos procedimientos",
    "Medicación y control de la diabetes": "tratamiento complejo o ajustes de medicación",
    "Ingreso (tipo, origen y especialidad)": "tipo u origen de ingreso asociado a más reingresos",
    "Edad, sexo y grupo étnico": "grupo de edad con más reingresos",
}


def cat_hosp(n):
    return "Ninguna" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4 o más"


def filtrar_diabetes(df, f):
    m = (df["bloque"] >= f["bloques"][0]) & (df["bloque"] <= f["bloques"][1])
    for col, key in [("edad", "edades"), ("destino_alta", "destinos"), ("diagnostico_principal", "diagnosticos"), ("banda", "niveles")]:
        if f.get(key):
            m &= df[col].isin(f[key])
    if f.get("buscar"):
        q = f["buscar"].strip()
        m &= df["encuentro"].astype(str).str.contains(q, regex=False) | df["paciente"].astype(str).str.contains(q, regex=False)
    return df[m]


def analizar_diabetes(df, todo, ctx, art, desc):
    R = REGLAS
    m, mt = metricas(df), metricas(todo)
    if m["n"] == 0:
        return None
    train = ctx["tasa_train"]
    cut_lo, cut_hi = ctx["cortes_riesgo"]
    min_seg = min(R["min_segmento"]["diabetes"], max(50, int(0.02 * m["n"])))
    pol = politicas(art, ctx)
    d = df.assign(hosp=df["hospitalizaciones_previas"].map(cat_hosp))
    hosp = tasa_por(d, "hosp", 30, orden=["Ninguna", "1", "2", "3", "4 o más"])
    dest = tasa_por(df, "destino_alta", min_seg)
    diag = tasa_por(df, "diagnostico_principal", min_seg)
    edad = tasa_por(df, "edad", min_seg)
    bloq = df.groupby("bloque").agg(n=("y", "size"), tasa=("y", "mean"), est=("p", "mean"))
    bloq = bloq[bloq["n"] >= 200]
    imp = influencia(df, ctx["grupos"])
    oe = m["oe"]
    err_rel = abs(1 - oe) if not pd.isna(oe) else np.nan
    sobre = (m["p_media"] > m["tasa"])
    h2 = d[d["hospitalizaciones_previas"] >= 2]
    h0 = d[d["hospitalizaciones_previas"] == 0]
    r_h2 = h2["y"].mean() if len(h2) >= 30 else np.nan
    r_h0 = h0["y"].mean() if len(h0) >= 30 else np.nan
    psi = art.get("psi_splits")
    psi_alto = psi[psi["train→producción"] >= R["psi_relevante"]].sort_values("train→producción", ascending=False) if psi is not None else pd.DataFrame()
    falt = art.get("faltantes_train_vs_produccion")
    fa = falsas_alarmas(art)
    met = fila_metricas_modelo(art, ctx)
    auc = valor_num(met["AUC"]) if met is not None else np.nan
    dest_alto = dest[(dest["tasa"] >= R["segmento_veces"][0] * m["tasa"])].head(3)

    kpis = [
        {"nombre": "Pacientes con riesgo alto de reingreso", "valor": num(m["n_alto"]), "unidad": f"({pct(m['share_alto'])} de las altas)",
         "interpretacion": f"Riesgo estimado ≥ {pct(cut_hi)}. Reingresó el {pct(m['tasa_alto'])}: {m['tasa_alto'] / m['tasa']:.1f} veces la tasa general."
         if m["tasa"] > 0 and not pd.isna(m["tasa_alto"]) else f"Riesgo estimado ≥ {pct(cut_hi)}.",
         "nivel": nivel_mayor_peor(m["share_alto"], R["share_alto"]["diabetes"]),
         "comparacion": f"Periodo completo: {num(mt['n_alto'])} pacientes ({pct(mt['share_alto'])}).",
         "ayuda": f"Pacientes cuya probabilidad estimada de reingresar en menos de 30 días es de {pct(cut_hi)} o más (aprox. el doble de la tasa base)."},
        {"nombre": "Reingresos en menos de 30 días", "valor": pct(m["tasa"], 1), "unidad": f"({num(m['eventos'])} de {num(m['n'])})",
         "interpretacion": f"{de_cada(m['tasa']).capitalize()} pacientes volvieron a hospitalizarse antes de 30 días.",
         "nivel": nivel_mayor_peor((m["tasa"] - train) * 100, R["cambio_tasa_pp"]["diabetes"]),
         "comparacion": f"Periodo de entrenamiento del modelo: {pct(train, 1)} ({pp(m['tasa'] - train)}).",
         "ayuda": "Indicador de calidad de atención: una readmisión temprana suele señalar un alta prematura o seguimiento insuficiente."},
        {"nombre": "Cobertura del seguimiento priorizado", "valor": pct(m["cob_alto_medio"]), "unidad": "de los reingresos",
         "interpretacion": f"Dar seguimiento a riesgo medio y alto ({pct(m['share_alto'] + m['share_medio'])} de los pacientes) alcanzaría a esa proporción de reingresos.",
         "nivel": nivel_mayor_mejor(m["cob_alto_medio"] / max(m["share_alto"] + m["share_medio"], 1e-9), (1.5, 1.2, 1.0)),
         "comparacion": f"Solo riesgo alto: {pct(m['share_alto'])} de pacientes → {pct(m['cob_alto'])} de reingresos. Al azar se cubriría lo mismo que se atiende.",
         "ayuda": "Qué parte de los reingresos ocurre en los pacientes que el modelo prioriza. Mientras más supere al % de pacientes atendidos, más útil es la priorización."},
        {"nombre": "Desajuste entre lo estimado y lo real", "valor": pct(err_rel), "unidad": "de desviación relativa",
         "interpretacion": f"Estimado {pct(m['p_media'], 1)} vs. real {pct(m['tasa'], 1)}: el modelo {'sobreestima' if sobre else 'subestima'} el riesgo.",
         "nivel": nivel_mayor_peor(err_rel, R["brecha_relativa"]),
         "comparacion": (f"Con mantenimiento ('{pol['mejor']}'), reingresos reales/esperados: {pol['oe'].get(pol['base'], np.nan):.2f} → {pol['oe'].get(pol['mejor'], np.nan):.2f} (ideal 1.00)."
                         if pol and pol["oe"] else "—"),
         "ayuda": "Diferencia relativa entre los reingresos que esperaba el modelo y los que ocurrieron: |1 − observados/esperados|. Si el modelo sobreestima, se programan seguimientos de más."},
    ]

    seg = []
    if not pd.isna(r_h2) and not pd.isna(r_h0):
        seg.append(f"Con 2 o más hospitalizaciones en el año previo: {pct(r_h2)} reingresa, contra {pct(r_h0)} sin hospitalizaciones previas ({num(len(h2))} pacientes).")
    if len(dest):
        seg.append("Destinos al alta con mayor tasa: " + ", ".join(f"{r.destino_alta} ({pct(r.tasa)}, {num(r.n)})" for r in dest.head(3).itertuples()) + ".")
    if len(diag):
        seg.append(f"Diagnóstico principal con mayor tasa: {diag.iloc[0].diagnostico_principal} ({pct(diag.iloc[0].tasa)}).")
    if len(edad):
        seg.append(f"Grupo de edad con mayor tasa: {edad.iloc[0].edad} ({pct(edad.iloc[0].tasa)}).")
    por_que = []
    if not pd.isna(r_h2) and r_h2 > 1.3 * m["tasa"]:
        por_que.append("Los pacientes que ya se hospitalizaron varias veces reingresan más: es compatible con enfermedad crónica más compleja, "
                       "pero los datos no permiten afirmar la causa.")
    if len(psi_alto):
        nombres = ", ".join(sin_parentesis(d_).lower() for d_ in psi_alto["descripción"].head(4)) if "descripción" in psi_alto else ", ".join(psi_alto["variable"].head(4))
        por_que.append(f"Cambió la forma de registrar la información entre el entrenamiento y la evaluación ({nombres}).")
    if falt is not None:
        f_ = falt.set_index("columna")
        if {"payer_code", "medical_specialty"} <= set(f_.index):
            por_que.append(f"Ejemplo: la aseguradora faltaba en {f_.loc['payer_code', 'faltante_train_%']:.0f} % de los registros de entrenamiento y en "
                           f"{f_.loc['payer_code', 'faltante_producción_%']:.0f} % de los evaluados; la especialidad, de {f_.loc['medical_specialty', 'faltante_train_%']:.0f} % "
                           f"a {f_.loc['medical_specialty', 'faltante_producción_%']:.0f} %.")
    if m["tasa"] < train:
        por_que.append(f"La tasa de reingreso bajó respecto del entrenamiento ({pct(train, 1)} → {pct(m['tasa'], 1)}); es consistente con que el modelo, sin ajuste, sobreestime.")
    impacto = [f"{num(m['eventos'])} reingresos tempranos en {num(m['n'])} pacientes.",
               f"Los pacientes de riesgo medio y alto concentran el {pct(m['cob_alto_medio'])} de los reingresos.",
               "No hay datos de costos ni de días-cama: el impacto se expresa en número de reingresos."]
    continua = []
    if len(bloq) >= 4:
        mitad = len(bloq) // 2
        oe1 = (bloq["tasa"] * bloq["n"]).iloc[:mitad].sum() / (bloq["est"] * bloq["n"]).iloc[:mitad].sum()
        oe2 = (bloq["tasa"] * bloq["n"]).iloc[mitad:].sum() / (bloq["est"] * bloq["n"]).iloc[mitad:].sum()
        continua.append(f"Relación reingresos reales/esperados: {oe1:.2f} en la primera mitad del periodo y {oe2:.2f} en la segunda (ideal 1.00)"
                        + ("; la sobreestimación tiende a crecer." if oe2 < oe1 and oe2 < 1 else "."))
    continua.append("Si la forma de registrar datos sigue cambiando y el modelo no se ajusta, las prioridades de seguimiento perderán precisión "
                    "(recursos dirigidos a pacientes que no los necesitan).")
    if pol and pol["oe"]:
        continua.append(f"Con la estrategia '{pol['mejor']}' la relación real/esperado mejora de {pol['oe'].get(pol['base'], np.nan):.2f} a "
                        f"{pol['oe'].get(pol['mejor'], np.nan):.2f} (evaluación sin filtros).")
    diagnostico = [
        ("¿Qué está ocurriendo?", [f"En {desc}, {pct(m['tasa'], 1)} de {num(m['n'])} pacientes reingresó en menos de 30 días (entrenamiento: {pct(train, 1)}).",
                                   f"El modelo esperaba {pct(m['p_media'], 1)}: {'sobreestima' if sobre else 'subestima'} el riesgo en {pct(err_rel)}."]),
        ("¿Qué pacientes están más afectados?", seg),
        ("¿Qué factores pesan más en la estimación?", [f"{g}: {pct(r.peso)} del peso total." for g, r in imp.head(3).iterrows()]
         + ["Son asociaciones que usa el modelo para estimar riesgo; no son causas comprobadas."]),
        ("¿Por qué podría estar ocurriendo?", por_que),
        ("¿Qué impacto puede tener?", impacto),
        ("¿Qué podría ocurrir si continúa?", continua),
    ]

    alertas = []
    if not pd.isna(r_h2) and m["tasa"] > 0 and r_h2 >= R["segmento_veces"][0] * m["tasa"]:
        alertas.append({"sev": "Crítica" if r_h2 >= R["segmento_veces"][1] * m["tasa"] else "Alta",
                        "titulo": f"Pacientes con 2 o más hospitalizaciones previas reingresan {r_h2 / m['tasa']:.1f} veces más",
                        "que": f"{pct(r_h2)} de reingreso en {num(len(h2))} pacientes vs. {pct(m['tasa'])} general.",
                        "por_que": "Es el grupo identificable con mayor concentración de reingresos.",
                        "si_no": "Este grupo seguirá concentrando reingresos y uso adicional de camas.",
                        "accion": "Gestión de casos y seguimiento post-alta intensivo para este grupo."})
    if len(dest_alto):
        r0 = dest_alto.iloc[0]
        alertas.append({"sev": "Crítica" if r0.tasa >= R["segmento_veces"][1] * m["tasa"] else "Alta",
                        "titulo": f"Destino al alta con tasa elevada: {r0.destino_alta}",
                        "que": f"{pct(r0.tasa)} de reingreso ({num(r0.n)} pacientes), {r0.tasa / m['tasa']:.1f} veces la tasa general.",
                        "por_que": "La transición del hospital al siguiente lugar de cuidado es un punto de riesgo.",
                        "si_no": "Los reingresos de este grupo se mantendrán por encima del promedio.",
                        "accion": "Revisar con coordinación de altas la información y el seguimiento que acompañan esta transición."})
    if not pd.isna(err_rel) and err_rel >= R["brecha_relativa"][0]:
        alertas.append({"sev": "Alta" if err_rel >= R["brecha_relativa"][1] else "Media",
                        "titulo": f"El modelo {'sobreestima' if sobre else 'subestima'} el riesgo de reingreso en {pct(err_rel)}",
                        "que": f"Esperados {num(m['esperados'])} reingresos; ocurrieron {num(m['eventos'])}.",
                        "por_que": "Las listas de seguimiento se dimensionan con estas probabilidades.",
                        "si_no": "Se asignarían recursos de seguimiento de más (o de menos).",
                        "accion": "Recalibrar el modelo con las altas más recientes cuya etiqueta ya maduró (30 días)."})
    if len(psi_alto):
        alertas.append({"sev": "Alta", "titulo": "Cambió la forma de registrar información clínica y administrativa",
                        "que": f"{len(psi_alto)} variables con cambio fuerte de distribución (PSI ≥ {R['psi_relevante']}), p. ej. "
                               + ", ".join(sin_parentesis(x).lower() for x in (psi_alto["descripción"] if "descripción" in psi_alto else psi_alto["variable"]).head(3)) + ".",
                        "por_que": "El modelo aprende de cómo se registran los datos; si el registro cambia, su precisión se degrada.",
                        "si_no": "Las estimaciones pueden desviarse sin que cambie el estado real de los pacientes.",
                        "accion": "Estandarizar la captura en el expediente y documentar cambios de sistema."})
    if not pd.isna(auc):
        alertas.append({"sev": "Informativa", "titulo": "Capacidad de discriminación moderada: útil para priorizar, no para decidir caso por caso",
                        "que": f"En {pct(auc)} de las parejas, el modelo asigna más riesgo al paciente que reingresa que al que no reingresa (AUC).",
                        "por_que": "Con señal moderada habrá pacientes de riesgo bajo que reingresen.",
                        "si_no": "—", "accion": "Usarlo como apoyo a la priorización junto con el juicio clínico."})
    alertas.append({"sev": "Informativa", "titulo": "El modelo usa edad, sexo y grupo étnico como variables",
                    "que": "Estas variables forman parte de la estimación (ver Metodología: revisión por grupo).",
                    "por_que": "Las decisiones de atención deben ser equitativas entre grupos de pacientes.",
                    "si_no": "Podrían perpetuarse diferencias no justificadas clínicamente.",
                    "accion": "Revisar periódicamente la exactitud del modelo por grupo con el comité de calidad o ética."})

    recs = [
        {"accion": "Implementar seguimiento post-alta priorizado por nivel de riesgo (alto primero, después medio)", "prioridad": "Alta",
         "responsable": "Dirección médica / gestión de casos / enfermería de enlace",
         "justificacion": f"Riesgo alto: {pct(m['tasa_alto'])} de reingreso; riesgo medio y alto concentran el {pct(m['cob_alto_medio'])} de los reingresos con el {pct(m['share_alto'] + m['share_medio'])} de los pacientes.",
         "resultado": "Contactar primero (llamada o cita temprana) a quienes concentran más reingresos con los mismos recursos.",
         "precaucion": "Distinción moderada del modelo: no sustituye el juicio clínico ni excluye a pacientes de riesgo bajo."},
    ]
    if not pd.isna(r_h2):
        recs.append({"accion": "Gestión de casos para pacientes con hospitalizaciones previas frecuentes", "prioridad": "Alta",
                     "responsable": "Gestión de casos / medicina interna",
                     "justificacion": f"Con 2 o más hospitalizaciones previas: {pct(r_h2)} de reingreso vs. {pct(r_h0)} sin hospitalizaciones previas.",
                     "resultado": "Reducir reingresos en el grupo de mayor riesgo identificable antes del alta.",
                     "precaucion": "Asociación observada en datos de 1999–2008; validar con la población actual del hospital."})
    if len(dest_alto):
        recs.append({"accion": "Revisar la planeación del alta en los destinos con mayor tasa de reingreso", "prioridad": "Media",
                     "responsable": "Coordinación de altas / trabajo social",
                     "justificacion": "; ".join(f"{r.destino_alta}: {pct(r.tasa)}" for r in dest_alto.itertuples()) + f" (general {pct(m['tasa'])}).",
                     "resultado": "Transiciones de cuidado más seguras y menos reingresos.",
                     "precaucion": "El destino puede reflejar la gravedad del paciente; no implica que el destino cause el reingreso."})
    recs.append({"accion": "Estandarizar la captura de especialidad, aseguradora y destino al alta", "prioridad": "Alta" if len(psi_alto) else "Media",
                 "responsable": "Calidad de datos / expediente clínico / TI",
                 "justificacion": "Los cambios en el registro fueron los más fuertes detectados por el monitor (ver Alertas).",
                 "resultado": "Datos comparables en el tiempo y un modelo más estable.",
                 "precaucion": "Documentar cualquier cambio de sistema para recalibrar el modelo a tiempo."})
    if pol and pol["oe"]:
        recs.append({"accion": f"Mantener el modelo con la estrategia '{pol['mejor']}'", "prioridad": "Media",
                     "responsable": "Equipo de analítica",
                     "justificacion": f"{POLITICAS_TXT.get(pol['mejor'], '')}: reingresos reales/esperados {pol['oe'].get(pol['base'], np.nan):.2f} → {pol['oe'].get(pol['mejor'], np.nan):.2f}.",
                     "resultado": "Listas de seguimiento dimensionadas con probabilidades confiables.",
                     "precaucion": "Las etiquetas tardan 30 días en conocerse: el ajuste siempre llega con ese retraso."})

    graficas = []
    if len(hosp) >= 2:
        fig = fig_real_vs_estimado(hosp["hosp"].astype(str), hosp["tasa"], hosp["estimado"], "Hospitalizaciones en el año previo", tipo="barras")
        graficas.append({"titulo": f"Más hospitalizaciones previas, más reingresos: de {pct(hosp['tasa'].iloc[0])} ({hosp['hosp'].iloc[0].lower()}) a {pct(hosp['tasa'].iloc[-1])} ({hosp['hosp'].iloc[-1]})",
                         "como_leer": "Barras: % real de pacientes que reingresaron, según hospitalizaciones en el año previo. Rombos: % estimado por el modelo.",
                         "hallazgo": "El historial de hospitalizaciones es la señal más clara para priorizar seguimiento.",
                         "fig": fig, "tabla": hosp.rename(columns={"hosp": "Hospitalizaciones previas", "n": "Pacientes", "eventos": "Reingresos", "tasa": "Real", "estimado": "Estimado"})})
    if len(dest) >= 2:
        dd = dest.head(7)
        fig = fig_barras_h(dd["destino_alta"], dd["tasa"].to_numpy(), texto=[pct(v) for v in dd["tasa"]],
                           hover=[f"{a}: {pct(t)} de {num(n)} pacientes" for a, t, n in zip(dd["destino_alta"], dd["tasa"], dd["n"])],
                           x_titulo="% de pacientes que reingresaron")
        graficas.append({"titulo": f"El destino al alta marca diferencias: '{dd.iloc[0]['destino_alta']}' tiene la mayor tasa ({pct(dd.iloc[0]['tasa'])})",
                         "como_leer": f"% real de reingreso por destino al alta (destinos con ≥ {num(min_seg)} pacientes). Promedio general: {pct(m['tasa'])}.",
                         "hallazgo": "Las transiciones a otros centros de cuidado son puntos de riesgo a revisar.",
                         "fig": fig, "tabla": dd.rename(columns={"destino_alta": "Destino al alta", "n": "Pacientes", "eventos": "Reingresos", "tasa": "Real", "estimado": "Estimado"})})
    bandas = ["Bajo", "Medio", "Alto"]
    bt = df.groupby("banda").agg(n=("y", "size"), ev=("y", "sum")).reindex(bandas).fillna(0)
    if m["eventos"] > 0:
        fig = fig_concentracion(bandas, (bt["n"] / m["n"]).tolist(), (bt["ev"] / m["eventos"]).tolist(), "pacientes", "reingresos")
        graficas.append({"titulo": f"El nivel alto reúne al {pct(m['share_alto'])} de los pacientes y al {pct(m['cob_alto'])} de los reingresos",
                         "como_leer": "Para cada nivel de riesgo estimado: azul = % de pacientes; naranja = % de los reingresos que ocurrieron en ese nivel.",
                         "hallazgo": "Cuando la barra naranja supera a la azul, ese nivel concentra más reingresos de los que le tocarían: ahí conviene priorizar.",
                         "fig": fig, "tabla": bt.reset_index().rename(columns={"banda": "Nivel", "n": "Pacientes", "ev": "Reingresos"})})
    if len(bloq) >= 3:
        n_sobre = int((bloq["est"] > bloq["tasa"]).sum())
        fig = fig_real_vs_estimado(bloq.index, bloq["tasa"], bloq["est"], "Periodo de altas (bloques de 1,000 en orden cronológico)", hover_x="Bloque %{x}")
        graficas.append({"titulo": f"El modelo sobreestimó el riesgo en {n_sobre} de {len(bloq)} periodos: necesita ajuste periódico",
                         "como_leer": "Cada punto es un bloque de altas consecutivas. Azul: % real de reingresos. Naranja: % esperado por el modelo (sin mantenimiento).",
                         "hallazgo": "La línea naranja por encima de la azul indica que se esperan más reingresos de los que ocurren.",
                         "fig": fig, "tabla": bloq.reset_index().rename(columns={"bloque": "Bloque", "n": "Pacientes", "tasa": "Real", "est": "Estimado"})})

    top = top_casos(df, 15)
    casos = pd.DataFrame({"Encuentro": top["encuentro"].astype(str), "Edad": top["edad"], "Sexo": top["sexo"],
                          "Diagnóstico principal": top["diagnostico_principal"], "Destino al alta": top["destino_alta"],
                          "Hosp. previas": top["hospitalizaciones_previas"],
                          "Probabilidad de reingreso": top["p"].astype(float), "Nivel": top["banda"],
                          "Principales razones (según el modelo)": [razones(r, ctx["grupos"], FRASES_DIABETES) for _, r in top.iterrows()],
                          "Resultado real": np.where(top["y"] == 1, "Reingresó", "No reingresó")})
    resumen = {
        "situacion": f"En {desc}, {pct(m['tasa'], 1)} de los {num(m['n'])} pacientes reingresó en menos de 30 días (entrenamiento: {pct(train, 1)}).",
        "hallazgo": (f"Los pacientes con 2 o más hospitalizaciones en el año previo reingresan {pct(r_h2)}, frente a {pct(r_h0)} de quienes no tuvieron ninguna."
                     if not pd.isna(r_h2) and not pd.isna(r_h0) else f"El factor con más peso es '{imp.index[0].lower()}'." if len(imp) else "—"),
        "riesgo": (f"El modelo {'sobreestima' if sobre else 'subestima'} el riesgo en {pct(err_rel)} y la forma de registrar datos cambió: "
                   "sin mantenimiento, las prioridades de seguimiento pierden precisión." if len(psi_alto) else
                   f"El modelo {'sobreestima' if sobre else 'subestima'} el riesgo en {pct(err_rel)}."),
        "oportunidad": f"Dar seguimiento a los pacientes de riesgo medio y alto ({pct(m['share_alto'] + m['share_medio'])}) alcanzaría al {pct(m['cob_alto_medio'])} de los reingresos.",
        "accion": "Implementar seguimiento post-alta priorizado por nivel de riesgo, con gestión de casos para pacientes con hospitalizaciones frecuentes, y recalibrar el modelo periódicamente.",
    }
    return {"m": m, "resumen": resumen, "kpis": kpis, "diagnostico": diagnostico, "alertas": alertas, "recs": recs,
            "graficas": graficas, "imp": imp, "casos": casos, "pol": pol, "titulo_casos": "Pacientes para seguimiento prioritario (15 con mayor probabilidad)"}


# =============================================================================
# 10. FINANZAS · QUIEBRAS
# =============================================================================
HORIZONTES = {2: "2 años antes del desenlace", 1: "1 año antes del desenlace"}
# Nombres ejecutivos cortos de los 64 indicadores (la fórmula completa del notebook se muestra en tablas y en Metodología)
NOMBRE_CORTO = {
    "Attr1": "Rentabilidad sobre activos (ROA)", "Attr2": "Deuda total / activos", "Attr3": "Capital de trabajo / activos",
    "Attr4": "Liquidez corriente", "Attr5": "Días de gastos cubiertos con activos líquidos", "Attr6": "Utilidades retenidas / activos",
    "Attr7": "EBIT / activos", "Attr8": "Capital contable / deuda total", "Attr9": "Ventas / activos", "Attr10": "Capital contable / activos",
    "Attr11": "Utilidad bruta ampliada / activos", "Attr12": "Utilidad bruta / pasivo de corto plazo", "Attr13": "Margen bruto con depreciación",
    "Attr14": "Utilidad bruta e intereses / activos", "Attr15": "Días para pagar la deuda con el flujo", "Attr16": "Flujo bruto / deuda total",
    "Attr17": "Activos / deuda total", "Attr18": "Utilidad bruta / activos", "Attr19": "Margen bruto", "Attr20": "Días de inventario (sobre ventas)",
    "Attr21": "Crecimiento de ventas", "Attr22": "Utilidad de operación / activos", "Attr23": "Margen neto", "Attr24": "Utilidad bruta de 3 años / activos",
    "Attr25": "Capital ganado / activos", "Attr26": "Flujo neto / deuda total", "Attr27": "Cobertura de gastos financieros",
    "Attr28": "Capital de trabajo / activo fijo", "Attr29": "Tamaño (logaritmo de activos)", "Attr30": "Deuda neta de efectivo / ventas",
    "Attr31": "Margen bruto con intereses", "Attr32": "Días de pago de pasivo circulante", "Attr33": "Gastos operativos / pasivo de corto plazo",
    "Attr34": "Gastos operativos / deuda total", "Attr35": "Utilidad en ventas / activos", "Attr36": "Rotación de activos totales",
    "Attr37": "Activo líquido / pasivo de largo plazo", "Attr38": "Capital permanente / activos", "Attr39": "Margen de utilidad en ventas",
    "Attr40": "Liquidez inmediata", "Attr41": "Deuda / flujo operativo", "Attr42": "Margen operativo", "Attr43": "Ciclo de cobro e inventario (días)",
    "Attr44": "Días de cobro a clientes", "Attr45": "Utilidad neta / inventario", "Attr46": "Prueba ácida", "Attr47": "Días de inventario (sobre costo)",
    "Attr48": "EBITDA / activos", "Attr49": "Margen EBITDA", "Attr50": "Activo circulante / deuda total", "Attr51": "Pasivo de corto plazo / activos",
    "Attr52": "Días de pasivo de corto plazo (costo)", "Attr53": "Capital contable / activo fijo", "Attr54": "Capital permanente / activo fijo",
    "Attr55": "Capital de trabajo (monto)", "Attr56": "Margen bruto en ventas", "Attr57": "Liquidez operativa", "Attr58": "Costos totales / ventas",
    "Attr59": "Pasivo de largo plazo / capital", "Attr60": "Rotación de inventario", "Attr61": "Rotación de cuentas por cobrar",
    "Attr62": "Días de pasivo de corto plazo (ventas)", "Attr63": "Ventas / pasivo de corto plazo", "Attr64": "Rotación de activo fijo",
    "n_missing": "Indicadores financieros faltantes",
}


def filtrar_polish(df, f):
    m = pd.Series(True, index=df.index)
    if f.get("horizontes"):
        m &= df["horizonte_anios"].isin(f["horizontes"])
    if f.get("niveles"):
        m &= df["banda"].isin(f["niveles"])
    if f.get("informacion") == "Completa":
        m &= df["n_faltantes"] == 0
    elif f.get("informacion") == "Incompleta":
        m &= df["n_faltantes"] > 0
    if f.get("buscar"):
        m &= df["empresa"].str.contains(f["buscar"].strip(), case=False, regex=False)
    return df[m]


def senales_financieras(df, ctx, top=6):
    """Indicadores con más peso + valor típico en empresas que quebraron vs. que siguieron operando + dirección del riesgo."""
    et = ctx.get("etiquetas", {})
    cols = [c for c in df.columns if c.startswith("c__Attr")]
    if not cols:
        return pd.DataFrame()
    a = df[cols].abs().mean().sort_values(ascending=False).head(top)
    filas = []
    for c in a.index:
        v = c[3:]
        x = df[v].astype(float)
        ok = x.notna()
        direccion = "sin datos suficientes"
        if ok.sum() >= 250 and df.loc[ok, "y"].sum() >= 10:
            q = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
            t = df.loc[ok, "y"].groupby(q).mean()            # tasa real de quiebra por quintil (Q1 = valores más bajos)
            if t.idxmax() == 0 and t.iloc[0] >= 1.5 * max(t.iloc[-1], 1e-9):
                direccion = "valores bajos se asocian con más quiebras"
            elif t.idxmax() == 4 and t.iloc[-1] >= 1.5 * max(t.iloc[0], 1e-9):
                direccion = "valores altos se asocian con más quiebras"
            else:
                direccion = f"relación no lineal: más quiebras en el quintil Q{int(t.idxmax()) + 1}"
        cat = next((g for g, vs in ctx["grupos"].items() if v in vs), "—")
        filas.append({"Indicador": NOMBRE_CORTO.get(v, et.get(v, v)), "Fórmula": et.get(v, v), "Código": v, "Categoría": cat,
                      "Peso": a[c] / df[cols].abs().mean().sum(),
                      "Valor típico · quebraron": df.loc[df["y"] == 1, v].median(), "Valor típico · siguieron operando": df.loc[df["y"] == 0, v].median(),
                      "Dirección": direccion})
    return pd.DataFrame(filas)


def analizar_polish(df, todo, ctx, art, desc):
    R = REGLAS
    m, mt = metricas(df), metricas(todo)
    if m["n"] == 0:
        return None
    train = ctx["tasa_train"]
    cut_lo, cut_hi = ctx["cortes_riesgo"]
    pol = politicas(art, ctx)
    hz = df.groupby("horizonte_anios").agg(n=("y", "size"), tasa=("y", "mean"), est=("p", "mean"), ev=("y", "sum"))
    hz["oe"] = hz["tasa"] / hz["est"]
    falt_cat = pd.cut(df["n_faltantes"], [-1, 0, 2, 100], labels=["Completa (0 faltantes)", "1–2 indicadores faltantes", "3 o más faltantes"])
    falt = df.assign(f=falt_cat).groupby("f", observed=True).agg(n=("y", "size"), tasa=("y", "mean"), est=("p", "mean"))
    falt = falt[falt["n"] >= 30]
    r_inc = df.loc[df["n_faltantes"] > 0, "y"].mean() if (df["n_faltantes"] > 0).sum() >= 30 else np.nan
    r_com = df.loc[df["n_faltantes"] == 0, "y"].mean() if (df["n_faltantes"] == 0).sum() >= 30 else np.nan
    imp = influencia(df, ctx["grupos"])
    sen = senales_financieras(df, ctx)
    oe = m["oe"]
    err_rel = abs(1 - oe) if not pd.isna(oe) else np.nan
    sub = m["p_media"] < m["tasa"]
    psi = art.get("psi_splits")
    psi_alto = psi[psi["train→producción"] >= R["psi_relevante"]].sort_values("train→producción", ascending=False) if psi is not None else pd.DataFrame()
    et = ctx.get("etiquetas", {})

    kpis = [
        {"nombre": "Empresas en riesgo alto de quiebra", "valor": num(m["n_alto"]), "unidad": f"({pct(m['share_alto'], 1)} de las empresas)",
         "interpretacion": f"Probabilidad estimada ≥ {pct(cut_hi)}; de ellas quebró el {pct(m['tasa_alto'])}.",
         "nivel": nivel_mayor_peor(m["share_alto"], R["share_alto"]["polish"]),
         "comparacion": f"Concentran {num(m['cob_alto'] * m['eventos'])} de las {num(m['eventos'])} quiebras ({pct(m['cob_alto'])}).",
         "ayuda": f"Empresas cuya probabilidad estimada de quiebra es de {pct(cut_hi)} o más (unas 4 veces la tasa base)."},
        {"nombre": "Tasa de quiebra observada", "valor": pct(m["tasa"], 1), "unidad": f"({num(m['eventos'])} de {num(m['n'])})",
         "interpretacion": f"{de_cada(m['tasa']).capitalize()} empresas evaluadas quebraron.",
         "nivel": nivel_mayor_peor((m["tasa"] - train) * 100, R["cambio_tasa_pp"]["polish"]),
         "comparacion": f"Empresas del entrenamiento (más lejos del desenlace): {pct(train, 1)} ({pp(m['tasa'] - train)}).",
         "ayuda": "Proporción real de empresas que quebraron. El entrenamiento usó empresas observadas 3–5 años antes del desenlace; la evaluación, 1–2 años antes."},
        {"nombre": "Quiebras detectadas con alerta", "valor": pct(m["sensibilidad"]), "unidad": "de las quiebras (sensibilidad)",
         "interpretacion": f"De las empresas que recibieron alerta, {pct(m['precision'])} efectivamente quebró (precisión).",
         "nivel": nivel_mayor_mejor(min_valido(m["sensibilidad"], m["precision"]), R["anticipacion"]),
         "comparacion": f"Si las alertas se asignaran al azar, acertarían {pct(m['tasa'], 1)} de las veces.",
         "ayuda": f"Alerta = probabilidad ≥ {pct(ctx['umbral_alerta'], 1)} (umbral elegido para no omitir quiebras). Sensibilidad: % de quiebras con alerta; precisión: % de alertas correctas."},
        {"nombre": "Desajuste entre lo estimado y lo real", "valor": pct(err_rel), "unidad": "de desviación relativa",
         "interpretacion": f"Estimado {pct(m['p_media'], 1)} vs. real {pct(m['tasa'], 1)}: el modelo {'subestima' if sub else 'sobreestima'} las quiebras.",
         "nivel": nivel_mayor_peor(err_rel, R["brecha_relativa"]),
         "comparacion": (f"Con mantenimiento ('{pol['mejor']}'), quiebras reales/esperadas: {pol['oe'].get(pol['base'], np.nan):.2f} → {pol['oe'].get(pol['mejor'], np.nan):.2f} (ideal 1.00)."
                         if pol and pol["oe"] else "—"),
         "ayuda": "Diferencia relativa entre las quiebras que esperaba el modelo y las que ocurrieron: |1 − observadas/esperadas|. Subestimar implica reservas o garantías insuficientes."},
    ]
    sen_txt = [f"{r['Indicador']}: valor típico {r['Valor típico · quebraron']:.3g} en empresas que quebraron vs. {r['Valor típico · siguieron operando']:.3g} en las que siguieron operando ({r['Dirección']})."
               for _, r in sen.head(3).iterrows()
               if not pd.isna(r["Valor típico · quebraron"]) and not pd.isna(r["Valor típico · siguieron operando"])]
    if not sen_txt:
        sen_txt = ["Con estos filtros no hay suficientes empresas que quebraron y que siguieron operando para comparar indicadores."]
    seg = [f"Riesgo alto: {num(m['n_alto'])} empresas, {pct(m['tasa_alto'])} quebró; riesgo medio: {num(m['n_medio'])} empresas, {pct(m['tasa_medio'])} quebró; "
           f"riesgo bajo: {pct(m['tasa_bajo'], 1)}."]
    if not pd.isna(r_inc) and not pd.isna(r_com):
        seg.append(f"Empresas con información financiera incompleta: {pct(r_inc, 1)} de quiebra vs. {pct(r_com, 1)} con información completa.")
    if len(hz) == 2:
        seg.append(f"A 1 año del desenlace quiebra el {pct(hz.loc[1, 'tasa'], 1)}; a 2 años, el {pct(hz.loc[2, 'tasa'], 1)}.")
    diagnostico = [
        ("¿Qué está ocurriendo?", [f"En {desc}, {pct(m['tasa'], 1)} de {num(m['n'])} empresas quebró; en el entrenamiento la tasa fue {pct(train, 1)}.",
                                   f"El modelo esperaba {pct(m['p_media'], 1)}: {'subestima' if sub else 'sobreestima'} las quiebras en {pct(err_rel)}."]),
        ("¿Qué empresas requieren atención?", seg),
        ("¿Qué factores pesan más en la estimación?", [f"{g}: {pct(r.peso)} del peso total." for g, r in imp.head(3).iterrows()]
         + ["Indicadores individuales con más peso: " + "; ".join(sen["Indicador"].head(3)) + "." if len(sen) else ""]),
        ("¿Qué señales de alerta deben vigilarse?", sen_txt + ["Son asociaciones observadas en los datos, no relaciones causales comprobadas."]),
        ("¿Qué impacto puede tener?", [f"{num(m['eventos'])} quiebras en {num(m['n'])} empresas; el {pct(m['cob_alto'])} estaba en riesgo alto.",
                                       (f"Con las alertas del modelo se revisan {num(m['n_alerta'])} empresas para encontrar {num(m['precision'] * m['n_alerta'])} quiebras; "
                                        f"eligiendo al azar harían falta ≈ {num(m['precision'] * m['n_alerta'] / m['tasa'])} revisiones."
                                        if m["n_alerta"] and m["tasa"] > 0 and not pd.isna(m["precision"]) else None),
                                       "No hay montos de exposición ni sector: el impacto se expresa en número de empresas."]),
        ("¿Qué podría ocurrir si continúa?", [
            (f"Conforme las empresas se acercan al desenlace la tasa sube ({pct(hz.loc[2, 'tasa'], 1)} → {pct(hz.loc[1, 'tasa'], 1)}) y la relación real/esperado pasa de "
             f"{hz.loc[2, 'oe']:.2f} a {hz.loc[1, 'oe']:.2f}: sin mantenimiento, la subestimación tiende a crecer." if len(hz) == 2 and hz.loc[1, "oe"] > hz.loc[2, "oe"] > 0 else
             "La tasa de quiebra cambia con la cercanía al desenlace; el modelo requiere ajuste periódico."),
            (f"Con la estrategia '{pol['mejor']}', el error de calibración bajó de {pts(pol['tabla'].loc[pol['base'], 'ECE'], 2)} a "
             f"{pts(pol['tabla'].loc[pol['mejor'], 'ECE'], 2)} y la capacidad de discriminación (AUC) subió de {pol['tabla'].loc[pol['base'], 'AUC']:.3f} a {pol['tabla'].loc[pol['mejor'], 'AUC']:.3f}."
             if pol else None)]),
    ]

    alertas = []
    if m["n_alto"] > 0:
        alertas.append({"sev": "Crítica" if m["tasa_alto"] >= 0.5 else "Alta",
                        "titulo": f"{num(m['n_alto'])} empresas en riesgo alto requieren revisión inmediata",
                        "que": f"Probabilidad estimada ≥ {pct(cut_hi)}; en la evaluación quebró el {pct(m['tasa_alto'])} de ellas.",
                        "por_que": f"Concentran el {pct(m['cob_alto'])} de las quiebras observadas.",
                        "si_no": "Exposición crediticia o comercial a empresas con alta probabilidad de incumplimiento.",
                        "accion": "Revisión por el comité de crédito: límites, garantías o plan de salida según la exposición."})
    if not pd.isna(r_inc) and not pd.isna(r_com) and r_com > 0 and r_inc >= R["segmento_veces"][0] * r_com:
        alertas.append({"sev": "Alta", "titulo": "La información financiera incompleta se asocia con más quiebras",
                        "que": f"{pct(r_inc, 1)} de quiebra con indicadores faltantes vs. {pct(r_com, 1)} con información completa ({r_inc / r_com:.1f} veces).",
                        "por_que": "La falta de reportes puede ser en sí misma una señal de deterioro.",
                        "si_no": "Empresas riesgosas podrían pasar inadvertidas si los faltantes se ignoran.",
                        "accion": "Exigir estados financieros completos y tratar los faltantes como señal de alerta."})
    if not pd.isna(err_rel) and err_rel >= R["brecha_relativa"][0]:
        alertas.append({"sev": "Alta" if err_rel >= R["brecha_relativa"][1] else "Media",
                        "titulo": f"El modelo {'subestima' if sub else 'sobreestima'} las quiebras en {pct(err_rel)}",
                        "que": f"Esperadas {num(m['esperados'])}; ocurrieron {num(m['eventos'])}.",
                        "por_que": "Las reservas y garantías se calculan con estas probabilidades.",
                        "si_no": "Reservas insuficientes si la tendencia continúa al acercarse el desenlace." if sub else "Restricciones de crédito más severas de lo necesario.",
                        "accion": "Recalibrar y reentrenar el modelo con información reciente (estrategia recomendada)."})
    if (m["tasa"] - train) * 100 >= R["cambio_tasa_pp"]["polish"][1]:
        alertas.append({"sev": "Media", "titulo": "La tasa de quiebra es mayor que en el periodo de entrenamiento",
                        "que": f"{pct(m['tasa'], 1)} vs. {pct(train, 1)} ({pp(m['tasa'] - train)}).",
                        "por_que": "El modelo aprendió con empresas más lejanas al desenlace.",
                        "si_no": "Mayor riesgo de subestimar en las empresas más cercanas al desenlace.",
                        "accion": "Incorporar observaciones recientes en el reentrenamiento."})
    if len(psi_alto):
        v0 = psi_alto.iloc[0]["variable"]
        alertas.append({"sev": "Informativa", "titulo": f"Cambió la disponibilidad del indicador '{NOMBRE_CORTO.get(v0, et.get(v0, v0))}'",
                        "que": f"Índice de estabilidad poblacional (PSI) de {psi_alto.iloc[0]['train→producción']:.2f} (cambio fuerte a partir de {R['psi_relevante']}); el resto de indicadores es estable.",
                        "por_que": "Es un cambio de calidad de datos (cuánto se reporta), no necesariamente del desempeño de las empresas.",
                        "si_no": "—", "accion": "Verificar con el área de información financiera la disponibilidad de ese dato."})
    alertas.append({"sev": "Informativa", "titulo": "Datos anónimos sin sector, tamaño de exposición ni fechas calendario",
                    "que": "Las empresas son manufactureras polacas identificadas solo por archivo y fila; los archivos son horizontes de predicción.",
                    "por_que": "Limita traducir el riesgo a montos o a sectores específicos.",
                    "si_no": "—", "accion": "En uso real, unir el modelo con la cartera (montos, sector, garantías)."})

    recs = [{"accion": f"Revisar de inmediato la relación con las {num(m['n_alto'])} empresas en riesgo alto", "prioridad": "Crítica" if m["tasa_alto"] >= 0.5 else "Alta",
             "responsable": "Riesgo de crédito / comité de crédito",
             "justificacion": f"{pct(m['tasa_alto'])} de ellas quebró y concentran el {pct(m['cob_alto'])} de las quiebras.",
             "resultado": "Reducir pérdidas ajustando límites, garantías o condiciones antes del deterioro.",
             "precaucion": f"El {pct(1 - m['tasa_alto'])} de las empresas de riesgo alto no quebró: la decisión debe incluir análisis del caso."}]
    if len(sen):
        recs.append({"accion": "Vigilar trimestralmente los indicadores con más peso: " + "; ".join(sen["Indicador"].head(3)), "prioridad": "Alta",
                     "responsable": "Análisis financiero / seguimiento de cartera",
                     "justificacion": "Son los indicadores con mayor peso en la estimación y muestran valores típicos distintos entre empresas que quebraron y las que no.",
                     "resultado": "Detectar deterioro antes de que la empresa llegue al nivel alto.",
                     "precaucion": "Los valores típicos son referencias de este conjunto de datos, no umbrales regulatorios."})
    if not pd.isna(r_inc) and not pd.isna(r_com):
        recs.append({"accion": "Exigir información financiera completa y tratar los faltantes como señal de alerta", "prioridad": "Media",
                     "responsable": "Originación / cumplimiento",
                     "justificacion": f"Información incompleta: {pct(r_inc, 1)} de quiebra vs. {pct(r_com, 1)} con información completa.",
                     "resultado": "Menos empresas riesgosas sin evaluar adecuadamente.",
                     "precaucion": "Algunos faltantes son administrativos; confirmar antes de penalizar."})
    if m["n_medio"] > 0 and not pd.isna(m["tasa_medio"]) and m["tasa"] > 0:
        recs.append({"accion": f"Mantener en lista de vigilancia a las {num(m['n_medio'])} empresas de riesgo medio", "prioridad": "Media",
                     "responsable": "Seguimiento de cartera",
                     "justificacion": f"Su tasa de quiebra ({pct(m['tasa_medio'])}) es {m['tasa_medio'] / m['tasa']:.1f} veces la general.",
                     "resultado": "Intervención temprana antes de que el riesgo escale.",
                     "precaucion": "Revisión menos intensiva que el nivel alto para no saturar al equipo."})
    if pol:
        t = pol["tabla"]
        recs.append({"accion": f"Mantener el modelo con la estrategia '{pol['mejor']}'", "prioridad": "Media",
                     "responsable": "Riesgo de modelo / analítica",
                     "justificacion": f"{POLITICAS_TXT.get(pol['mejor'], '')}: error de calibración {pts(t.loc[pol['base'], 'ECE'], 2)} → {pts(t.loc[pol['mejor'], 'ECE'], 2)}.",
                     "resultado": "Probabilidades confiables conforme cambia la cercanía al desenlace.",
                     "precaucion": "Documentar cada reentrenamiento (gobierno de modelos)."})

    graficas = []
    bandas = ["Bajo", "Medio", "Alto"]
    bt = df.groupby("banda").agg(n=("y", "size"), ev=("y", "sum")).reindex(bandas).fillna(0)
    if m["eventos"] > 0:
        fig = fig_concentracion(bandas, (bt["n"] / m["n"]).tolist(), (bt["ev"] / m["eventos"]).tolist(), "empresas", "quiebras")
        graficas.append({"titulo": f"El riesgo está concentrado: el nivel alto reúne al {pct(m['share_alto'])} de las empresas y al {pct(m['cob_alto'])} de las quiebras",
                         "como_leer": "Para cada nivel de riesgo estimado: azul = % de empresas; naranja = % de las quiebras que ocurrieron en ese nivel.",
                         "hallazgo": "Revisar un grupo pequeño de empresas permite cubrir la mayoría de las quiebras.",
                         "fig": fig, "tabla": bt.reset_index().rename(columns={"banda": "Nivel", "n": "Empresas", "ev": "Quiebras"})})
    if len(hz) >= 1:
        h = hz.reset_index().sort_values("horizonte_anios", ascending=False)
        fig = fig_real_vs_estimado([HORIZONTES.get(int(x), str(x)) for x in h["horizonte_anios"]], h["tasa"], h["est"], "Distancia al desenlace", tipo="barras")
        graficas.append({"titulo": (f"Más cerca del desenlace, más quiebras: {pct(hz.loc[2, 'tasa'], 1)} a 2 años y {pct(hz.loc[1, 'tasa'], 1)} a 1 año"
                                    if len(hz) == 2 else "Tasa de quiebra según distancia al desenlace"),
                         "como_leer": "Barras: % real de empresas que quebraron. Rombos: % estimado por el modelo.",
                         "hallazgo": "Cuando el rombo queda por debajo de la barra, el modelo subestima: es la señal para recalibrar.",
                         "fig": fig, "tabla": h.rename(columns={"horizonte_anios": "Años al desenlace", "n": "Empresas", "tasa": "Real", "est": "Estimado", "ev": "Quiebras", "oe": "Real/estimado"})})
    if len(falt) >= 2:
        fig = fig_real_vs_estimado(list(falt.index.astype(str)), falt["tasa"], falt["est"], "Calidad de la información financiera", tipo="barras")
        graficas.append({"titulo": f"Las empresas con información incompleta quiebran más ({pct(falt['tasa'].iloc[-1], 1)} vs. {pct(falt['tasa'].iloc[0], 1)})",
                         "como_leer": "Empresas agrupadas por número de indicadores financieros faltantes. Barras: % real de quiebra; rombos: % estimado.",
                         "hallazgo": "La ausencia de información es una señal de alerta en sí misma.",
                         "fig": fig, "tabla": falt.reset_index().rename(columns={"f": "Información", "n": "Empresas", "tasa": "Real", "est": "Estimado"})})
    if len(sen) >= 2 and m["n"] >= 300:
        k = min(4, len(sen))
        def partir(s, n=22):
            pal, lin, out = s.split(), "", []
            for w in pal:
                if len(lin) + len(w) + 1 > n and lin:
                    out.append(lin)
                    lin = w
                else:
                    lin = (lin + " " + w).strip()
            return "<br>".join(out + [lin])
        fig = make_subplots(rows=1, cols=k, shared_yaxes=True, horizontal_spacing=0.05,
                            subplot_titles=[partir(s) for s in sen["Indicador"].head(k)])
        filas_tab = []
        for i, r in enumerate(sen.head(k).itertuples(), 1):
            x = df[r.Código]
            ok = x.notna()
            q = pd.qcut(x[ok].rank(method="first"), 5, labels=["Q1 (más bajo)", "Q2", "Q3", "Q4", "Q5 (más alto)"])
            t = df.loc[ok].assign(q=q).groupby("q", observed=True)["y"].mean()
            fig.add_bar(x=["Q1", "Q2", "Q3", "Q4", "Q5"], y=t.to_numpy(), marker=dict(color=AZUL, cornerradius=4), row=1, col=i, showlegend=False,
                        hovertemplate=f"{esc(r.Indicador)}<br>%{{x}}: %{{y:.1%}} de quiebra<extra></extra>")
            filas_tab += [{"Indicador": r.Indicador, "Quintil": qq, "Tasa de quiebra": vv} for qq, vv in t.items()]
        fig = estilo(fig, alto=340, leyenda=False)
        fig.update_layout(margin=dict(t=58))
        fig.update_annotations(font=dict(size=11, color=INK))
        fig.update_xaxes(title_text="", type="category")
        tq = pd.DataFrame(filas_tab)
        rango = tq.groupby("Indicador")["Tasa de quiebra"].agg(lambda s: s.max() / max(s.min(), 1e-9))
        graficas.append({"titulo": (f"La tasa de quiebra cambia hasta {rango.max():.0f} veces según el valor de '{rango.idxmax()}'"
                                    if rango.max() >= 2 else "Cómo cambia la tasa de quiebra según el valor de los indicadores con más peso"),
                         "como_leer": "Para cada indicador, las empresas se ordenan en 5 grupos iguales (Q1 = valores más bajos, Q5 = más altos). Barra = % que quebró.",
                         "hallazgo": "; ".join(f"{r['Indicador']}: {r['Dirección']}" for _, r in sen.head(min(k, 3)).iterrows()) + ".",
                         "fig": fig, "tabla": pd.DataFrame(filas_tab)})

    top = top_casos(df, 15)

    def razones_emp(r):
        cc = {c: float(r[c]) for c in df.columns if c.startswith("c__Attr")}
        pos = sorted(((v, c) for c, v in cc.items() if v > 0), reverse=True)[:2]
        txt = [NOMBRE_CORTO.get(c[3:], et.get(c[3:], c[3:])) for _, c in pos]
        if float(r.get("c__n_missing", 0)) > 0 and r["n_faltantes"] > 0:
            txt.append(f"{int(r['n_faltantes'])} indicadores faltantes")
        return " · ".join(txt) or "Sin un factor dominante"
    casos = pd.DataFrame({"Empresa": top["empresa"], "Horizonte": top["horizonte_anios"].map(HORIZONTES),
                          "Probabilidad de quiebra": top["p"].astype(float), "Nivel": top["banda"], "Indicadores faltantes": top["n_faltantes"],
                          "Señales con más peso (según el modelo)": [razones_emp(r) for _, r in top.iterrows()],
                          "Resultado real": np.where(top["y"] == 1, "Quebró", "Siguió operando")})
    resumen = {
        "situacion": f"En {desc}, {pct(m['tasa'], 1)} de las {num(m['n'])} empresas quebró (entrenamiento: {pct(train, 1)})"
                     + (f"; la tasa es mayor a 1 año del desenlace ({pct(hz.loc[1, 'tasa'], 1)}) que a 2 años ({pct(hz.loc[2, 'tasa'], 1)})." if len(hz) == 2 else "."),
        "hallazgo": f"El riesgo está muy concentrado: el {pct(m['share_alto'], 1)} de empresas en nivel alto reúne el {pct(m['cob_alto'])} de las quiebras ({pct(m['tasa_alto'])} de ellas quebró).",
        "riesgo": f"{num(m['n_alto'])} empresas en riesgo alto requieren revisión; el modelo {'subestima' if sub else 'sobreestima'} las quiebras en {pct(err_rel)}"
                  + (f"; las señales con más peso son {', '.join(s.lower() for s in sen['Indicador'].head(3))}." if len(sen) else "."),
        "oportunidad": f"Con el umbral de alerta se anticipa el {pct(m['sensibilidad'])} de las quiebras, con {pct(m['precision'])} de alertas correctas (al azar: {pct(m['tasa'], 1)}).",
        "accion": "Revisar de inmediato la exposición con las empresas de riesgo alto, vigilar trimestralmente sus indicadores clave y exigir información financiera completa.",
    }
    return {"m": m, "resumen": resumen, "kpis": kpis, "diagnostico": diagnostico, "alertas": alertas, "recs": recs,
            "graficas": graficas, "imp": imp, "casos": casos, "pol": pol, "senales": sen,
            "titulo_casos": "Empresas que requieren atención (15 con mayor probabilidad)"}


# =============================================================================
# 11. ORQUESTACIÓN: filtros → análisis (en caché) → vista
# =============================================================================
FILTRAR = {"airline": filtrar_airline, "diabetes": filtrar_diabetes, "polish": filtrar_polish}
ANALIZAR = {"airline": analizar_airline, "diabetes": analizar_diabetes, "polish": analizar_polish}


def filtros_por_defecto(ds, df):
    if ds == "airline":
        return {"dias": [int(df["dia"].min()), int(df["dia"].max())]}
    if ds == "diabetes":
        return {"bloques": [int(df["bloque"].min()), int(df["bloque"].max())]}
    return {}


def describir_filtros(ds, f, df):
    partes = []
    if ds == "airline":
        d0, d1 = f["dias"]
        partes.append(f"los días {d0}–{d1}" if d0 != d1 else f"el día {d0}")
        if f.get("aerolineas"):
            partes.append("aerolíneas " + ", ".join(f["aerolineas"][:4]) + ("…" if len(f["aerolineas"]) > 4 else ""))
        if f.get("origenes"):
            partes.append("origen " + ", ".join(f["origenes"][:4]) + ("…" if len(f["origenes"]) > 4 else ""))
        if f.get("franjas"):
            partes.append(", ".join(x.split(" (")[0].lower() for x in f["franjas"]))
    elif ds == "diabetes":
        b0, b1 = f["bloques"]
        partes.append("todo el periodo evaluado" if (b0, b1) == (int(df["bloque"].min()), int(df["bloque"].max())) else
                      (f"el bloque de altas {b0}" if b0 == b1 else f"los bloques de altas {b0}–{b1}"))
        if f.get("edades"):
            partes.append("edad " + ", ".join(f["edades"][:3]))
        if f.get("destinos"):
            partes.append(f"{len(f['destinos'])} destino(s) al alta")
        if f.get("diagnosticos"):
            partes.append("diagnóstico " + ", ".join(f["diagnosticos"][:3]).lower())
    else:
        partes.append("las empresas evaluadas" if not f.get("horizontes") else
                      "empresas observadas " + " y ".join(HORIZONTES[h] for h in sorted(f["horizontes"], reverse=True)))
        if f.get("informacion") in ("Completa", "Incompleta"):
            partes.append(f"información {f['informacion'].lower()}")
    if f.get("niveles"):
        partes.append("nivel de riesgo " + "/".join(x.lower() for x in f["niveles"]))
    if f.get("buscar"):
        partes.append(f"búsqueda «{f['buscar']}»")
    return " · ".join(partes) if ds != "airline" else ", ".join(partes)


@st.cache_data(show_spinner="Calculando diagnóstico…", max_entries=64)
def analisis(ds: str, raiz: str, filtros_json: str):
    filas, ctx, art = cargar(ds, raiz)
    f = json.loads(filtros_json)
    df = FILTRAR[ds](filas, f)
    desc = describir_filtros(ds, f, filas)
    return ANALIZAR[ds](df, filas, ctx, art, desc)


def widgets_filtros(ds, filas):
    """Filtros sencillos y relevantes; actualizan resumen, indicadores, alertas, recomendaciones y gráficas."""
    f = filtros_por_defecto(ds, filas)
    with st.expander("Filtros", expanded=False):
        if ds == "airline":
            c1, c2, c3 = st.columns([1.2, 1, 1])
            d0, d1 = int(filas["dia"].min()), int(filas["dia"].max())
            f["dias"] = list(c1.slider("Periodo (días del mes analizado)", d0, d1, (d0, d1), key="a_dias",
                                       help="El conjunto cubre 31 días consecutivos sin fechas calendario; se analizan los días de evaluación del modelo."))
            vol_a = filas["aerolinea"].value_counts()
            f["aerolineas"] = c2.multiselect("Aerolínea", list(vol_a.index), key="a_aer", placeholder="Todas")
            vol_o = filas["origen"].value_counts()
            f["origenes"] = c3.multiselect("Aeropuerto de origen", list(vol_o.index), key="a_ori", placeholder="Todos",
                                           help="Ordenados de mayor a menor volumen de vuelos.")
            c4, c5, c6 = st.columns([1.2, 1, 1])
            f["franjas"] = c4.multiselect("Franja horaria", FRANJAS, key="a_fr", placeholder="Todas")
            f["niveles"] = c5.multiselect("Nivel de riesgo estimado", ["Alto", "Medio", "Bajo"], key="a_niv", placeholder="Todos")
            f["buscar"] = c6.text_input("Buscar vuelo o ruta", key="a_bus", placeholder="Ej. WN 1500  ·  ORD → DFW")
        elif ds == "diabetes":
            c1, c2, c3 = st.columns([1.2, 1, 1])
            b0, b1 = int(filas["bloque"].min()), int(filas["bloque"].max())
            f["bloques"] = list(c1.slider("Periodo (bloques de 1,000 altas en orden cronológico)", b0, b1, (b0, b1), key="d_blo",
                                          help="El conjunto no tiene fechas; el orden de los encuentros es cronológico (verificado en el notebook)."))
            f["edades"] = c2.multiselect("Grupo de edad", sorted(filas["edad"].unique(), key=lambda s: int(s.split("–")[0])), key="d_eda", placeholder="Todos")
            f["destinos"] = c3.multiselect("Destino al alta", list(filas["destino_alta"].value_counts().index), key="d_des", placeholder="Todos")
            c4, c5, c6 = st.columns([1.2, 1, 1])
            f["diagnosticos"] = c4.multiselect("Diagnóstico principal", list(filas["diagnostico_principal"].value_counts().index), key="d_dia", placeholder="Todos")
            f["niveles"] = c5.multiselect("Nivel de riesgo estimado", ["Alto", "Medio", "Bajo"], key="d_niv", placeholder="Todos")
            f["buscar"] = c6.text_input("Buscar paciente o encuentro (ID)", key="d_bus", placeholder="Ej. 176718774")
        else:
            c1, c2, c3, c4 = st.columns(4)
            hz = c1.multiselect("Distancia al desenlace", [2, 1], format_func=lambda h: HORIZONTES[h], key="p_hz", placeholder="Ambas")
            f["horizontes"] = hz
            f["niveles"] = c2.multiselect("Nivel de riesgo estimado", ["Alto", "Medio", "Bajo"], key="p_niv", placeholder="Todos")
            f["informacion"] = c3.selectbox("Información financiera", ["Todas", "Completa", "Incompleta"], key="p_inf")
            f["buscar"] = c4.text_input("Buscar empresa", key="p_bus", placeholder="Ej. 5year · fila 120")
    return {k: v for k, v in f.items() if v not in (None, "", [], "Todas")}


def render_tecnico(ds, A, ctx, art, df_filtrado_fn):
    """Sección desplegable: el contenido técnico no ocupa la vista principal."""
    with st.expander("Metodología y detalle técnico", expanded=False):
        meta = art.get("metadata", {})
        st.markdown(f"**Modelo utilizado:** {ctx['modelo']} (modelo recomendado por el notebook; menor log-loss validada en el split de calibración). "
                    f"Corrida `{ctx.get('run_id')}`. Umbral de alerta: {ctx['umbral_alerta']:.4f} · regla {ctx['regla_umbral']}. "
                    f"Niveles de riesgo: Bajo < {ctx['cortes_riesgo'][0]:.0%} ≤ Medio < {ctx['cortes_riesgo'][1]:.0%} ≤ Alto.")
        st.markdown("**Cómo se generan las cifras del tablero.** Se reconstruye exactamente el periodo de evaluación (producción) de cada notebook, "
                    "se puntúa con el modelo y el calibrador exportados y se calculan las contribuciones por variable (SHAP nativo de XGBoost). "
                    "Los resultados reales son conocidos porque se trata de una evaluación retrospectiva.")
        ver = ctx.get("verificacion", {})
        st.markdown(f"**Verificación de reproducibilidad:** {'✅' if ver.get('coincide') else '⚠️'} {ver.get('detalle', '—')} "
                    + ("Datos crudos idénticos a los del notebook (SHA-256)." if ctx.get("datos", {}).get("coincide_hash") else
                       "El hash de los datos crudos no coincide o no está disponible; las cifras pueden diferir ligeramente."))
        met = fila_metricas_modelo(art, ctx)
        if met is not None:
            filas_m = [("AUC", "Probabilidad de que un caso con evento reciba mayor riesgo que uno sin evento (1 = perfecto, 0.5 = azar)."),
                       ("AP", "Precisión promedio sobre todos los umbrales (referencia al azar = tasa del evento)."),
                       ("Brier", "Error cuadrático medio de la probabilidad (0 = perfecto)."),
                       ("ECE", "Diferencia promedio entre probabilidad estimada y frecuencia observada (criterio del protocolo ≤ 0.05)."),
                       ("O/E", "Eventos observados / esperados (ideal 1)."),
                       ("Pendiente", "Pendiente de calibración: 1 = dispersión correcta; < 1 = probabilidades demasiado extremas.")]
            st.markdown("**Desempeño en el periodo de evaluación (sin filtros, intervalos al 95 % por bootstrap):**")
            st.dataframe(pd.DataFrame([{"Métrica": k, "Valor": str(met[k]), "Qué significa": v} for k, v in filas_m if k in met.index]), hide_index=True)
        pol = A.get("pol")
        if pol:
            t = pol["tabla"].reset_index()
            keep = [c for c in ["política", "AUC", "AP", "ECE", "LogLoss", "recalibraciones", "reentrenamientos", "seg_en_intervenciones", pol["col_alerta"]] if c and c in t.columns]
            t = t[keep].copy()
            t.insert(1, "En palabras simples", t["política"].map(POLITICAS_TXT))
            if pol["oe"]:
                t["Real/esperado"] = t["política"].map(pol["oe"])
            st.markdown(f"**Estrategias de mantenimiento del modelo (simulación del notebook §4.6).** Recomendada en el tablero: **{pol['mejor']}** "
                        "(mejor posición conjunta en log-loss y ECE entre A, B y C; desempate por AUC).")
            st.dataframe(t.round(4), hide_index=True)
        fa, lat = art.get("monitor_falsas_alarmas"), art.get("monitor_latencia")
        if fa is not None:
            st.markdown("**Validación del monitor de cambios (escenarios simulados donde se sabe si hubo cambio; criterio: falsas alarmas < 15 %):**")
            st.dataframe(fa.round(2).rename(columns={"señal": "Señal", "bloques_sin_drift": "Bloques sin cambio",
                                                      "falsas_alarmas_nivel≥1_%": "% falsas alarmas (nivel ≥ 1)",
                                                      "falsas_alarmas_nivel2_%": "% falsas alarmas (nivel 2)"}), hide_index=True)
        if lat is not None:
            st.markdown("**Volumen de observaciones para detección** (observaciones nuevas hasta la primera alerta tras un cambio simulado; criterio ≤ 500):")
            st.dataframe(lat.round(1).rename(columns={
                "aumento_de_tasa_pp": "Aumento de la tasa (pp)", "detección_N1_%": "% detectado (nivel ≥ 1)",
                "latencia_mediana_N1": "Observaciones hasta la 1.ª alerta, nivel ≥ 1 (mediana)",
                "detección_N2_%": "% detectado (nivel 2)",
                "latencia_mediana_N2": "Observaciones hasta la 1.ª alerta, nivel 2 (mediana)"}), hide_index=True)
        iv = influencia_variables(df_filtrado_fn(), ctx.get("etiquetas", {}))
        if len(iv):
            st.markdown("**Variables individuales con más peso en la estimación (con los filtros actuales):**")
            st.dataframe(iv, hide_index=True, column_config={"Peso en la estimación": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=float(iv["Peso en la estimación"].max()))})
        if ds == "polish" and A.get("senales") is not None and len(A["senales"]):
            st.markdown("**Señales financieras (valores típicos = medianas):**")
            st.dataframe(A["senales"].round(4), hide_index=True)
        if ds == "diabetes":
            d = df_filtrado_fn()
            eq = pd.concat([d.groupby("sexo").agg(Pacientes=("y", "size"), Real=("y", "mean"), Estimado=("p", "mean")).assign(Grupo="Sexo"),
                            d.groupby("grupo_etnico").agg(Pacientes=("y", "size"), Real=("y", "mean"), Estimado=("p", "mean")).assign(Grupo="Grupo étnico")])
            st.markdown("**Revisión por grupo (equidad):** tasa real vs. estimada por sexo y grupo étnico.")
            st.dataframe(eq.reset_index(names="Categoría")[["Grupo", "Categoría", "Pacientes", "Real", "Estimado"]].round(4), hide_index=True)
        st.markdown("**Reglas del tablero** (criterios de presentación, ajustables en `REGLAS` dentro de app.py): niveles de atención por proporción en riesgo alto, "
                    "cambio de la tasa frente al entrenamiento, desajuste entre lo estimado y lo real, y capacidad de detección (el menor valor "
                    "entre sensibilidad y precisión); segmentos con volumen mínimo "
                    f"({REGLAS['min_segmento'][ds]} casos o menos con filtros) y tasa ≥ promedio + 10 pp (Aerolínea) o ≥ 1.5 veces el promedio (Hospital, Finanzas).")
        st.markdown("**Supuestos y limitaciones:**\n" + "\n".join(f"- {x}" for x in LIMITACIONES[ds]))
        if meta.get("entorno"):
            e = meta["entorno"]
            st.caption(f"Entorno de la corrida: Python {e.get('python')} · scikit-learn {e.get('scikit-learn')} · xgboost {e.get('xgboost')} · {e.get('cpus_logicos')} CPU.")


LIMITACIONES = {
    "airline": ["31 días consecutivos sin fechas calendario; el día se reconstruyó a partir del orden de las filas (§1.6 del notebook).",
                "Sin clima, rotación de aeronaves, tripulaciones, minutos de retraso ni costos: no se puede confirmar el efecto dominó ni cuantificar pérdidas.",
                "El historial de retrasos usa solo días anteriores (sin fuga de información).",
                "Los pesos de los factores (SHAP) describen cómo estima el modelo; no son efectos causales."],
    "diabetes": ["Datos de 130 hospitales de EE. UU. (1999–2008); se usa el primer encuentro por paciente y se excluyen altas por defunción u hospicio.",
                 "Se excluyó el periodo final con etiqueta incompleta (readmisiones posteriores a la extracción no registradas).",
                 "El orden de encuentros se usa como eje temporal (verificado en §1.7); no hay fechas.",
                 "Sin costos ni datos de intervenciones post-alta: no se puede estimar el efecto de una acción, solo priorizar."],
    "polish": ["Empresas manufactureras polacas; los archivos 1year–5year son horizontes de predicción, no fechas (§1.5).",
               "Se eliminaron 401 duplicados y se aleatorizó el orden dentro de cada horizonte con semilla 42 (§1.7).",
               "Sin sector, montos de exposición ni identidad de las empresas.",
               "Los valores típicos de los indicadores son medianas del conjunto, no umbrales regulatorios."],
}
NOMBRE_TAB = {"airline": "Aerolínea · Retrasos", "diabetes": "Hospital · Reingresos", "polish": "Finanzas · Quiebras"}
PREGUNTA = {"airline": "¿Qué vuelos tienen más probabilidad de llegar con retraso y qué lo está impulsando?",
            "diabetes": "¿Qué pacientes con diabetes tienen más riesgo de reingresar en menos de 30 días?",
            "polish": "¿Qué empresas tienen más riesgo de quiebra y qué señales financieras lo anticipan?"}


def render_dataset(ds):
    ok, err = disponible(ds)
    if not ok:
        st.error(f"No hay datos preparados para esta pestaña. {err}")
        st.markdown(f"Falló la preparación de **{NOMBRE_TAB[ds]}**: revisa la tabla de estado de la sección 7 del notebook "
                    "(artefactos `.zip` de ese notebook TT2 y datos crudos). Las demás pestañas siguen funcionando.")
        return None
    filas, ctx, art = cargar(ds, str(EX_RAIZ))
    st.markdown(f'<div class="tt-nota">{esc(PREGUNTA[ds])} · Periodo de evaluación del modelo con resultados reales conocidos '
                f'({num(len(filas))} {ctx["unidad_plural"]}).</div>', unsafe_allow_html=True)
    f = widgets_filtros(ds, filas)
    A = analisis(ds, str(EX_RAIZ), json.dumps(f, sort_keys=True, ensure_ascii=False, default=str))
    if A is None:
        st.warning("No hay registros con los filtros seleccionados. Amplía el periodo o quita algún filtro.")
        return None
    st.markdown(f'<div class="tt-filtros">Mostrando: {esc(describir_filtros(ds, {**filtros_por_defecto(ds, filas), **f}, filas))} · '
                f'{num(A["m"]["n"])} {ctx["unidad"] if A["m"]["n"] == 1 else ctx["unidad_plural"]}</div>', unsafe_allow_html=True)
    aviso_muestra(A["m"], ds)

    encabezado_seccion("01", "Resumen ejecutivo", "Lectura de un minuto")
    tarjeta_resumen(A["resumen"], nivel_general(A["alertas"], A["kpis"]))

    encabezado_seccion("02", "Indicadores clave", "Pasa el cursor sobre una tarjeta para ver su definición")
    tarjetas_kpi(A["kpis"])

    encabezado_seccion("03", "Diagnóstico", "Qué ocurre, dónde, por qué podría ocurrir y qué pasaría si continúa")
    bloque_diagnostico(A["diagnostico"])
    if len(A["imp"]):
        imp = A["imp"]
        mostrar_grafica({"titulo": f"'{imp.index[0]}' es el factor con más peso en la estimación ({pct(imp['peso'].iloc[0])})",
                         "como_leer": "Peso relativo de cada grupo de factores en las estimaciones del modelo (contribuciones SHAP agregadas). Suma 100 %.",
                         "hallazgo": "Indica en qué se apoya el modelo; un peso alto es asociación, no prueba de causa.",
                         "fig": fig_influencia(imp), "tabla": imp[["peso"]].reset_index().rename(columns={"index": "Factor", "peso": "Peso"})}, key=f"ex_{ds}_imp")

    encabezado_seccion("04", "Alertas prioritarias", "Solo hallazgos que requieren atención")
    tarjetas_alertas(A["alertas"])

    encabezado_seccion("05", "Recomendaciones ejecutivas", "Sugerencias de apoyo a la decisión, derivadas de los datos")
    tarjetas_recomendaciones(A["recs"])

    encabezado_seccion("06", "Visualizaciones esenciales", "Cada gráfica responde una pregunta de negocio")
    gs = A["graficas"]
    for i in range(0, len(gs), 2):
        cols = st.columns(2)
        for j, g in enumerate(gs[i:i + 2]):
            with cols[j]:
                mostrar_grafica(g, key=f"ex_{ds}_g{i + j}")

    st.markdown(f'<div class="tt-gtit" style="margin-top:1rem">{esc(A["titulo_casos"])}</div>', unsafe_allow_html=True)
    prob_col = [c for c in A["casos"].columns if c.startswith("Probabilidad")][0]
    st.dataframe(A["casos"], hide_index=True,
                 column_config={prob_col: st.column_config.ProgressColumn(prob_col, format="percent", min_value=0.0, max_value=1.0)})
    st.download_button("Descargar lista (CSV)", A["casos"].to_csv(index=False).encode("utf-8-sig"), file_name=f"tt2_{ds}_casos_prioritarios.csv",
                       mime="text/csv", key=f"ex_{ds}_dl")
    st.markdown('<div class="tt-nota">"Resultado real" se conoce porque el análisis es retrospectivo; en operación diaria esa columna no estaría disponible.</div>',
                unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    render_tecnico(ds, A, ctx, art, lambda: FILTRAR[ds](filas, f if ds == "polish" else {**filtros_por_defecto(ds, filas), **f}))
    return A


def render_portada():
    st.markdown('<div class="tt-nota" style="margin-bottom:.6rem">Visión de conjunto de los tres casos sin filtros. '
                'Cada pestaña detalla el diagnóstico, las alertas y las recomendaciones, y permite filtrar.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    mejoras = []
    for c, ds in zip(cols, ["airline", "diabetes", "polish"]):
        with c:
            ok, err = disponible(ds)
            if not ok:
                st.markdown(f'<div class="tt-ov"><div class="t">{NOMBRE_TAB[ds]}</div><div class="q">Sin datos preparados.</div></div>', unsafe_allow_html=True)
                continue
            filas, ctx, art = cargar(ds, str(EX_RAIZ))
            A = analisis(ds, str(EX_RAIZ), json.dumps(filtros_por_defecto(ds, filas), sort_keys=True))
            nivel = nivel_general(A["alertas"], A["kpis"])
            alerta = sorted(A["alertas"], key=lambda a: ORDEN_SEV[a["sev"]])[0] if A["alertas"] else None
            st.markdown(
                f'<div class="tt-ov"><div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start">'
                f'<div class="t">{NOMBRE_TAB[ds]}</div>{badge(nivel, NIVEL_TXT.get(nivel, "—"))}</div>'
                f'<div class="q">{esc(PREGUNTA[ds])}</div>'
                f'<div class="k">Situación</div><div class="v">{esc(A["resumen"]["situacion"])}</div>'
                f'<div class="k">Hallazgo principal</div><div class="v">{esc(A["resumen"]["hallazgo"])}</div>'
                + (f'<div class="k">Alerta más relevante</div><div class="v">{esc(alerta["titulo"])}</div>' if alerta else "")
                + f'<div class="k">Acción prioritaria</div><div class="v">{esc(A["resumen"]["accion"])}</div></div>', unsafe_allow_html=True)
            pol = A.get("pol")
            if pol:
                t = pol["tabla"]
                mejoras.append((NOMBRE_TAB[ds], t.loc[pol["base"], "ECE"], t.loc[pol["mejor"], "ECE"], pol["mejor"]))
    if mejoras:
        encabezado_seccion("", "Conclusión común a los tres casos", "Error de calibración con y sin mantenimiento del modelo")
        todas = all(b < a for _, a, b, _ in mejoras)
        st.markdown(f'<div class="tt-card" style="font-size:.95rem">'
                    + ("En los tres casos, mantener el modelo (reajustar probabilidades ante alertas y reentrenar periódicamente) "
                       "redujo el error de calibración —la diferencia promedio entre la probabilidad estimada y la frecuencia real— "
                       "frente a dejarlo fijo: " if todas else
                       "El efecto del mantenimiento del modelo varía entre casos: ")
                    + "; ".join(f"<b>{n}</b> {pts(a, 2)} → {pts(b, 2)} ({esc(p)})" for n, a, b, p in mejoras)
                    + ". Un modelo que nace bien calibrado se desajusta cuando el entorno cambia; el monitor detecta el cambio y la estrategia de "
                      "mantenimiento lo corrige.</div>", unsafe_allow_html=True)
    encabezado_seccion("", "Cómo leer este tablero")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("- **Probabilidad estimada:** lo que el modelo espera que ocurra (p. ej., 60 % = de cada 100 casos similares, unos 60 presentan el evento).\n"
                    "- **Nivel de riesgo (bajo, medio, alto):** agrupa los casos según esa probabilidad.\n"
                    "- **pp (puntos porcentuales):** diferencia entre dos porcentajes; de 20 % a 23 % hay 3 pp.\n"
                    "- **Factores con más peso:** en qué se apoya el modelo; indican **asociación**, no causa.\n"
                    "- **Recomendaciones:** sugerencias de apoyo a la decisión; no sustituyen el criterio experto.")
    with c2:
        st.markdown(" ".join([badge("bajo", "Bajo"), badge("medio", "Medio"), badge("alto", "Alto"), badge("critico", "Crítico"), badge("info", "Informativa")])
                    + '<div class="tt-nota" style="margin-top:8px">Niveles de atención: bajo = sin acción inmediata · medio = vigilar · '
                      'alto = actuar en el corto plazo · crítico = actuar de inmediato.</div>', unsafe_allow_html=True)


# =============================================================================
# 12. DATOS DE LA VISTA EJECUTIVA: búsqueda automática, carga por .zip y vista previa
# =============================================================================
EX_DATASETS = ("airline", "diabetes", "polish")
EX_RAIZ: Path | None = None                     # carpeta datos_tablero/ encontrada en esta ejecución (o None)
_AQUI = Path(__file__).resolve().parent


def _raices_candidatas() -> list[Path]:
    env = os.environ.get("TT2_TABLERO_DATOS")
    raices = [Path(env)] if env else []
    return raices + [_AQUI / "datos_tablero", Path("/content/tt2_tablero/datos_tablero"), Path("/content/datos_tablero"),
                     Path("/content/drive/MyDrive/TT2/datos_tablero"), Path("/content/drive/MyDrive/TT2/tt2_tablero/datos_tablero")]


def _zips_candidatos() -> list[Path]:
    """tt2_tablero_paquete.zip (o cualquier *tablero*.zip) junto a app.py, en /content o en Drive (si está montado)."""
    out = []
    for carpeta in (_AQUI, Path("/content"), Path("/content/drive/MyDrive/TT2"), Path("/content/drive/MyDrive")):
        try:
            if carpeta.is_dir():
                out += sorted(carpeta.glob("*tablero*.zip"), key=lambda q: q.stat().st_mtime, reverse=True)
        except OSError:
            continue
    return list(dict.fromkeys(out))


def _tiene_datos(raiz: Path) -> bool:
    return any((raiz / ds / "filas_produccion.parquet").is_file() for ds in EX_DATASETS)


def extraer_paquete(origen, destino: Path) -> int:
    """Extrae del .zip solo <dataset>/filas_produccion.parquet, contexto.json y artefactos/* (con o sin carpeta
    datos_tablero/ al inicio). Ignora cualquier otro archivo y rutas sospechosas. Devuelve cuántos archivos extrajo."""
    n = 0
    with zipfile.ZipFile(origen) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            partes = [x for x in info.filename.replace("\\", "/").split("/") if x]
            if ".." in partes:
                continue
            pos = next((i for i, x in enumerate(partes) if x in EX_DATASETS), None)
            if pos is None:
                continue
            rel = partes[pos:]
            valido = (len(rel) == 2 and rel[1] in ("filas_produccion.parquet", "contexto.json")) or \
                     (len(rel) == 3 and rel[1] == "artefactos")
            if not valido:
                continue
            dst = destino.joinpath(*rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            n += 1
    return n


def localizar_datos_ejecutivos() -> Path | None:
    for raiz in _raices_candidatas():
        if _tiene_datos(raiz):
            return raiz
    destino = _AQUI / "datos_tablero"
    for z in _zips_candidatos():
        try:
            if extraer_paquete(z, destino) and _tiene_datos(destino):
                return destino
        except (zipfile.BadZipFile, OSError):
            continue
    return None


def vista_previa_ejecutiva():
    """Mientras no se cargan los registros de producción: lectura agregada con las tablas de las corridas finales
    (las mismas que usa la pestaña técnica). Usa la misma regla de 'estrategia recomendada' que la vista completa."""
    encabezado_seccion("", "Vista previa", "Cifras agregadas de las corridas finales (sin filtros)")
    cols = st.columns(3)
    mejoras = []
    for c, ds in zip(cols, ["airline", "diabetes", "polish"]):
        part = M(ds)["particion"]
        tasa = {k: next(v[t] for t in ("prevalencia", "tasa_retraso", "tasa_readmision_30d") if t in v) for k, v in part.items()}
        pol = politicas({"politicas_resumen": T(ds, "politicas_resumen"), "politicas_bloques": T(ds, DS[ds]["tabla_t"])},
                        {"tipo_modelo": "xgb"})
        t = pol["tabla"]
        mejoras.append((NOMBRE_TAB[ds], t.loc[pol["base"], "ECE"], t.loc[pol["mejor"], "ECE"], pol["mejor"]))
        with c:
            st.markdown(
                f'<div class="tt-ov"><div class="t">{NOMBRE_TAB[ds]}</div><div class="q">{esc(PREGUNTA[ds])}</div>'
                f'<div class="k">Tasa del evento</div><div class="v">Entrenamiento {pct(tasa["train"], 1)} → evaluación '
                f'{pct(tasa["production"], 1)} ({pp(tasa["production"] - tasa["train"])}).</div>'
                f'<div class="k">Error de calibración</div><div class="v">Sin mantenimiento {pts(t.loc[pol["base"], "ECE"], 2)} → '
                f'con mantenimiento {pts(t.loc[pol["mejor"], "ECE"], 2)}.</div>'
                f'<div class="k">Estrategia recomendada</div><div class="v">{esc(pol["mejor"])}: '
                f'{esc(POLITICAS_TXT.get(pol["mejor"], ""))}.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="tt-nota" style="margin-top:8px">El diagnóstico completo (indicadores con filtros, alertas, '
                'recomendaciones y listas de casos) aparece en cuanto se cargan los registros de producción.</div>',
                unsafe_allow_html=True)


def pantalla_sin_datos():
    st.markdown(
        '<div class="tt-aviso"><b>Aún no hay registros de producción para la vista ejecutiva.</b> Se generan en las '
        'secciones 6 y 7 del notebook a partir de los artefactos de los notebooks TT2 (carpeta <code>MyDrive/TT2</code>). '
        'Revisa la tabla de estado de la sección 7, corrige lo que falte y vuelve a ejecutar las secciones 7 a 9. '
        'Si ya tienes <code>tt2_tablero_paquete.zip</code> de una corrida anterior, también puedes subirlo aquí.</div>',
        unsafe_allow_html=True)
    up = st.file_uploader("Subir tt2_tablero_paquete.zip (o un .zip que contenga la carpeta datos_tablero/)", type=["zip"],
                          key="ex_zip")
    if up is not None:
        destino = _AQUI / "datos_tablero"
        try:
            n = extraer_paquete(io.BytesIO(up.getvalue()), destino)
        except zipfile.BadZipFile:
            st.error("El archivo no es un .zip válido.")
        else:
            if n and _tiene_datos(destino):
                cargar.clear()
                analisis.clear()
                st.success(f"Paquete cargado ({n} archivos). Actualizando la vista…")
                st.rerun()
            else:
                st.error("El .zip no contiene registros de producción (carpetas airline/, diabetes/ o polish/ con "
                         "filas_produccion.parquet). Verifica que sea tt2_tablero_paquete.zip.")
    vista_previa_ejecutiva()


def vista_ejecutiva():
    st.markdown('<div class="tt-nota" style="margin-bottom:.4rem">Diagnóstico y apoyo a decisiones para dirección, a partir de '
                'los modelos con monitoreo de <i>calibration drift</i>. La evidencia técnica (métricas, calibración, deriva y '
                'estrategias) está en la pestaña <b>Monitoreo, resultados y detalles</b>.</div>', unsafe_allow_html=True)
    if EX_RAIZ is None:
        pantalla_sin_datos()
        return
    tabs = st.tabs(["Resumen general", NOMBRE_TAB["airline"], NOMBRE_TAB["diabetes"], NOMBRE_TAB["polish"]])
    with tabs[0]:
        render_portada()
    for tab, ds in zip(tabs[1:], ["airline", "diabetes", "polish"]):
        with tab:
            try:
                render_dataset(ds)
            except Exception as err:  # una pestaña con problemas no tumba el tablero
                st.error(f"Ocurrió un problema al construir esta pestaña ({type(err).__name__}: {err}).")
                st.caption("Revisa que el paquete de datos incluya este conjunto y esté completo.")


# #####################################################################################################
# PÁGINA PRINCIPAL
# #####################################################################################################
def main():
    global PACK, EX_RAIZ
    if not os.path.exists(DATA_PACK):
        st.error(f"No se encontró el paquete de datos «{DATA_PACK}». Ejecuta en Colab la celda 4 (paquete de resultados) "
                 "y vuelve a cargar la página.")
        st.stop()
    PACK = load_pack(DATA_PACK)
    EX_RAIZ = localizar_datos_ejecutivos()
    st.markdown('<div class="app-head"><div class="t">Calibration Drift Monitor</div>'
                '<div class="d">Detección y mitigación de <i>calibration drift</i> en modelos de clasificación · '
                'Trabajo Terminal 2026-B080 · ESCOM-IPN</div></div>', unsafe_allow_html=True)
    tab_ejecutivo, tab_tecnico = st.tabs(["Resumen Ejecutivo", "Monitoreo, resultados y detalles"])
    with tab_ejecutivo:
        vista_ejecutiva()
    with tab_tecnico:
        vista_tecnica()
    st.markdown('<div class="foot">Detección y Mitigación de Calibration Drift en Modelos de Clasificación · Trabajo Terminal '
                '2026-B080 · Casas Aguilar José David y Robles Cervantes Pedro Iván · ESCOM-IPN</div>', unsafe_allow_html=True)


main()
