import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LogiPredict Pro | Enterprise", layout="wide")

# --- SIDEBAR: ASISTENTE ---
with st.sidebar:
    st.title("🧠 LogiPredict Assistant")
    fuente = st.radio("Fuente de Datos", ["Ejemplo / Manual", "Subir Archivo Excel/CSV"])
    df_source = None
    
    if fuente == "Subir Archivo Excel/CSV":
        uploaded_file = st.file_uploader("Sube tu archivo", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                df_source = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            except Exception: st.error("Error al leer el archivo.")
    else:
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = pd.DataFrame({
                'Mes': range(1, 13),
                'Demanda': [100, 110, 105, 115, 120, 130, 140, 155, 160, 165, 170, 180]
            })
        df_source = st.session_state.manual_data

# --- CUERPO PRINCIPAL ---
st.title("📊 LogiPredict Pro: Inteligencia Logística")

if df_source is not None:
    df_working = st.data_editor(df_source, num_rows="dynamic", use_container_width=True)
    cols = df_working.columns.tolist()
    col_demanda = st.selectbox("🎯 Selecciona la columna de DEMANDA (Y):", cols, index=cols.index('Demanda') if 'Demanda' in cols else 0)
    df_working[col_demanda] = pd.to_numeric(df_working[col_demanda], errors='coerce').fillna(0).astype(float)
else:
    st.stop()

# Inicializar resultados para el reporte
if 'resultados' not in st.session_state: st.session_state.resultados = {}

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🧹 1. Limpieza GIGO", "🏆 2. Modo Torneo", "📈 3. Regresión Múltiple", "📥 4. Reporte Final"])

# --- TAB 1: LIMPIEZA ---
with tab1:
    mean_val = df_working[col_demanda].mean()
    std_val = df_working[col_demanda].std()
    outliers = (np.abs(df_working[col_demanda] - mean_val) > (1.5 * std_val))
    if st.button("Aplicar Limpieza GIGO"):
        df_working.loc[outliers, col_demanda] = float(mean_val)
        st.rerun()

# --- TAB 2: MODO TORNEO ---
with tab2:
    y = df_working[col_demanda].values
    if len(y) >= 3:
        pms = df_working[col_demanda].rolling(window=2).mean().shift(1).fillna(y[0]).values
        ses = ExponentialSmoothing(y, trend=None).fit(smoothing_level=0.3, optimized=False).fittedvalues
        
        def get_mape(real, pred): return np.mean(np.abs((real - pred) / np.where(real==0, 1, real))) * 100
        
        scores = [
            {"Metodo": "Promedio Móvil", "MAPE": get_mape(y, pms), "Pred": pms},
            {"Metodo": "Suavización Exponencial", "MAPE": get_mape(y, ses), "Pred": ses}
        ]
        mejor = min(scores, key=lambda x: x['MAPE'])
        st.success(f"Mejor método: {mejor['Metodo']} ({100-mejor['MAPE']:.2f}% precisión)")
        st.session_state.resultados['mejor_metodo'] = mejor
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y, name="Real"))
        fig.add_trace(go.Scatter(y=mejor['Pred'], name="Pronóstico"))
        st.plotly_chart(fig)

# --- TAB 3: REGRESIÓN ---
with tab3:
    features = st.multiselect("Variables predictoras (X):", [c for c in cols if c != col_demanda])
    if features:
        X = df_working[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        model = LinearRegression().fit(X, y)
        pred_rm = model.predict(X)
        st.session_state.resultados['regresion'] = {"Pred": pred_rm, "R2": model.score(X, y)}
        st.metric("Confiabilidad R²", f"{model.score(X, y)*100:.2f}%")

# --- TAB 4: REPORTE FINAL ---
with tab4:
    st.header("📥 Generador de Reporte Ejecutivo")
    st.write("Este archivo incluye los datos originales, los pronósticos calculados y las métricas de error.")
    
    if st.button("🚀 Generar Excel Profesional"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Hoja 1: Datos y Pronósticos
            df_final = df_working.copy()
            if 'mejor_metodo' in st.session_state.resultados:
                df_final['Pronostico_Serie_Tiempo'] = st.session_state.resultados['mejor_metodo']['Pred']
            if 'regresion' in st.session_state.resultados:
                df_final['Pronostico_Regresion'] = st.session_state.resultados['regresion']['Pred']
            
            df_final.to_excel(writer, sheet_name='Pronósticos', index=False)
            
            # Hoja 2: Resumen Ejecutivo
            resumen_data = []
            if 'mejor_metodo' in st.session_state.resultados:
                resumen_data.append(["Mejor Modelo Serie Tiempo", st.session_state.resultados['mejor_metodo']['Metodo']])
                resumen_data.append(["Precisión (100-MAPE)", f"{100-st.session_state.resultados['mejor_metodo']['MAPE']:.2f}%"])
            if 'regresion' in st.session_state.resultados:
                resumen_data.append(["Confiabilidad Regresión (R²)", f"{st.session_state.resultados['regresion']['R2']*100:.2f}%"])
            
            pd.DataFrame(resumen_data, columns=["Indicador", "Valor"]).to_excel(writer, sheet_name='Resumen Ejecutivo', index=False)
            
            # Formateo visual rápido
            workbook = writer.book
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            
        st.download_button(
            label="📥 Descargar Reporte para Dirección (Excel)",
            data=output.getvalue(),
            file_name="Reporte_Logistico_PRO.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
