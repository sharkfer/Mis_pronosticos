import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LogiPredict Pro | Enterprise", layout="wide")

# --- SIDEBAR: ASISTENTE Y CARGA DE DATOS ---
with st.sidebar:
    st.title("🧠 LogiPredict Assistant")
    
    st.header("1. Carga tus datos")
    uploaded_file = st.file_bar = st.file_uploader("Sube tu Excel o CSV", type=["csv", "xlsx"])
    
    st.divider()
    st.header("2. ¿Qué método usar?")
    q1 = st.radio("¿Tienes datos de Precio o Marketing?", ["No", "Sí"])
    q2 = st.checkbox("¿Tu demanda sube o baja constantemente (Tendencia)?")
    q3 = st.checkbox("¿Vendes más en temporadas fijas?")

    if q1 == "Sí":
        st.success("👉 Sugerencia: Usa **Regresión Múltiple** (Tab 3)")
    elif q3:
        st.success("👉 Sugerencia: El **Modo Torneo** elegirá **Holt-Winters**")
    else:
        st.success("👉 Sugerencia: El **Modo Torneo** comparará los mejores para ti.")

# --- PROCESAMIENTO DE DATOS ---
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
        st.session_state.df = df_input
    except Exception as e:
        st.error("Error al leer el archivo. Revisa el formato.")
else:
    # Datos de ejemplo si no hay archivo
    if 'df' not in st.session_state:
        fechas = pd.date_range(start='2023-01-01', periods=15, freq='ME')
        st.session_state.df = pd.DataFrame({
            'Fecha': fechas,
            'Demanda': [100, 110, 105, 115, 120, 130, 140, 155, 160, 165, 170, 180, 190, 200, 210],
            'Precio': [50, 50, 50, 48, 48, 45, 45, 42, 40, 40, 40, 38, 35, 35, 30],
            'Mkt_Inversion': [5, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 35, 40, 50]
        })

# Asegurar que Demanda sea decimal para evitar errores
st.session_state.df['Demanda'] = st.session_state.df['Demanda'].astype(float)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🧹 1. Limpieza GIGO", "🏆 2. Modo Torneo", "📈 3. Regresión Múltiple", "📥 4. Reporte Final"])

# --- TAB 1: LIMPIEZA ---
with tab1:
    st.header("Módulo de Limpieza (Regla GIGO)")
    df_clean = st.session_state.df.copy()
    mean_val = df_clean['Demanda'].mean()
    std_val = df_clean['Demanda'].std()
    df_clean['Outlier'] = (np.abs(df_clean['Demanda'] - mean_val) > (1.5 * std_val))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['Demanda'], name="Demanda Actual"))
    outliers = df_clean[df_clean['Outlier']]
    fig.add_trace(go.Scatter(x=outliers.index, y=outliers['Demanda'], mode='markers', name="Anomalía", marker=dict(color='red', size=10)))
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Normalizar Datos (Limpiar Basura)"):
        st.session_state.df.loc[df_clean['Outlier'], 'Demanda'] = float(mean_val)
        st.rerun()

# --- TAB 2: MODO TORNEO (TODOS LOS MÉTODOS) ---
with tab2:
    st.header("🏆 Modo Torneo: Competencia de Métodos")
    y = st.session_state.df['Demanda'].values
    
    # Calculamos todos los métodos del PDF
    pms = st.session_state.df['Demanda'].rolling(window=3).mean().shift(1).fillna(y[0]).values
    ses = ExponentialSmoothing(y, trend=None).fit(smoothing_level=0.3).fittedvalues
    holt = ExponentialSmoothing(y, trend='add').fit().fittedvalues
    
    # Medimos quién es mejor
    def get_mape(real, pred): return np.mean(np.abs((real - pred) / real)) * 100
    
    scores = [
        {"Método": "Promedio Móvil Simple", "MAPE": get_mape(y, pms), "Pred": pms},
        {"Método": "Suavización Exponencial (SES)", "MAPE": get_mape(y, ses), "Pred": ses},
        {"Método": "Método de Holt (Tendencia)", "MAPE": get_mape(y, holt), "Pred": holt}
    ]
    
    mejor = min(scores, key=lambda x: x['MAPE'])
    st.success(f"🥇 El ganador es: **{mejor['Método']}** con un error de {mejor['MAPE']:.2f}%")
    
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(y=y, name="Real", line=dict(color='black')))
    fig_t.add_trace(go.Scatter(y=mejor['Pred'], name="Pronóstico Ganador", line=dict(color='green')))
    st.plotly_chart(fig_t, use_container_width=True)

# --- TAB 3: REGRESIÓN MÚLTIPLE ---
with tab3:
    st.header("📈 Regresión Lineal Múltiple")
    cols = st.session_state.df.columns.tolist()
    features = st.multiselect("Selecciona variables para predecir (X)", [c for c in cols if c != 'Demanda'], default=[cols[2], cols[3]] if len(cols)>3 else [])
    
    if features:
        X = st.session_state.df[features]
        model = LinearRegression().fit(X, y)
        pred_rm = model.predict(X)
        st.write(f"Precisión del modelo: {model.score(X, y)*100:.2f}%")
        fig_rm = go.Figure()
        fig_rm.add_trace(go.Scatter(y=y, name="Real"))
        fig_rm.add_trace(go.Scatter(y=pred_rm, name="Regresión Múltiple"))
        st.plotly_chart(fig_rm, use_container_width=True)

# --- TAB 4: REPORTE ---
with tab4:
    st.header("📥 Descargar Resultados")
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar Reporte en CSV", data=csv, file_name="pronostico_final.csv")
