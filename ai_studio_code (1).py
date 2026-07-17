import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LogiPredict Pro | Universal", layout="wide")

# --- SIDEBAR: ASISTENTE Y FUENTE DE DATOS ---
with st.sidebar:
    st.title("🧠 LogiPredict Assistant")
    
    st.header("1. Fuente de Datos")
    fuente = st.radio("¿Cómo quieres ingresar los datos?", ["Ejemplo / Manual", "Subir Archivo Excel/CSV"])
    
    df_source = None
    
    if fuente == "Subir Archivo Excel/CSV":
        uploaded_file = st.file_uploader("Sube tu archivo", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                df_source = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            except Exception:
                st.error("Error al leer el archivo.")
    else:
        st.info("✍️ Edita la tabla en el centro de la pantalla para ingresar datos manualmente.")
        # Creamos un template inicial para la entrada manual
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = pd.DataFrame({
                'Fecha': pd.date_range(start='2024-01-01', periods=6, freq='ME').strftime('%Y-%m-%d'),
                'Demanda': [100.0, 120.0, 110.0, 130.0, 150.0, 140.0],
                'Precio': [50.0, 48.0, 49.0, 45.0, 44.0, 45.0],
                'Marketing': [10.0, 15.0, 12.0, 20.0, 25.0, 22.0]
            })
        df_source = st.session_state.manual_data

    st.divider()
    st.header("2. Configuración del Asistente")
    q1 = st.radio("¿Tienes variables externas?", ["No", "Sí"])
    q2 = st.checkbox("¿Ves una tendencia clara?")
    q3 = st.checkbox("¿Es un producto estacional?")

# --- CUERPO PRINCIPAL ---
st.title("📊 LogiPredict Pro: Inteligencia Logística")

if fuente == "Ejemplo / Manual":
    st.subheader("📝 Editor Manual de Datos")
    st.write("Puedes editar las celdas, agregar filas al final o pegar datos desde Excel directamente aquí abajo:")
    # Usamos st.data_editor para que sea como un Excel
    df_working = st.data_editor(df_source, num_rows="dynamic", use_container_width=True, key="editor_manual")
    st.session_state.manual_data = df_working # Guardamos cambios
else:
    if df_source is not None:
        df_working = df_source
        st.success("✅ Archivo cargado correctamente.")
    else:
        st.warning("⚠️ Por favor, sube un archivo en la barra lateral.")
        st.stop()

# --- VALIDACIÓN INTELIGENTE DE COLUMNAS ---
cols = df_working.columns.tolist()
col_demanda = st.selectbox("🎯 Selecciona la columna de DEMANDA (Ventas):", cols, index=cols.index('Demanda') if 'Demanda' in cols else 0)

# Limpieza técnica inmediata: convertir a número y quitar errores
df_working[col_demanda] = pd.to_numeric(df_working[col_demanda], errors='coerce').fillna(0).astype(float)

# --- TABS DE ANÁLISIS ---
tab1, tab2, tab3, tab4 = st.tabs(["🧹 1. Limpieza GIGO", "🏆 2. Modo Torneo", "📈 3. Regresión Múltiple", "📥 4. Reporte Final"])

# --- TAB 1: LIMPIEZA ---
with tab1:
    st.header("Módulo de Limpieza (Normalización)")
    mean_val = df_working[col_demanda].mean()
    std_val = df_working[col_demanda].std()
    # Identificar outliers (PDF pág 2: Variaciones Aleatorias)
    outliers = (np.abs(df_working[col_demanda] - mean_val) > (1.5 * std_val))
    
    fig_clean = go.Figure()
    fig_clean.add_trace(go.Scatter(y=df_working[col_demanda], name="Datos actuales", line=dict(color='#3366CC')))
    fig_clean.add_trace(go.Scatter(y=df_working[col_demanda].where(outliers), mode='markers', name="Anomalía (Ruido)", marker=dict(color='red', size=12)))
    fig_clean.update_layout(title="Detección de Variaciones Aleatorias", template="plotly_white")
    st.plotly_chart(fig_clean, use_container_width=True)
    
    if st.button("Aplicar Limpieza GIGO"):
        df_working.loc[outliers, col_demanda] = float(mean_val)
        st.success("Datos normalizados con el promedio histórico.")
        st.rerun()

# --- TAB 2: MODO TORNEO ---
with tab2:
    st.header("🏆 Torneo de Pronósticos")
    y = df_working[col_demanda].values
    if len(y) < 3:
        st.error("Se necesitan al menos 3 datos para calcular pronósticos.")
    else:
        # Métodos del PDF
        pms = df_working[col_demanda].rolling(window=2).mean().shift(1).fillna(y[0]).values
        ses = ExponentialSmoothing(y, trend=None).fit(smoothing_level=0.3, optimized=False).fittedvalues
        holt = ExponentialSmoothing(y, trend='add').fit().fittedvalues
        
        def mape(real, pred): return np.mean(np.abs((real - pred) / np.where(real==0, 1, real))) * 100
        
        scores = [
            {"Método": "Promedio Móvil Simple", "Error (MAPE)": mape(y, pms), "Pred": pms},
            {"Método": "Suavización Exponencial (SES)", "Error (MAPE)": mape(y, ses), "Pred": ses},
            {"Método": "Método de Holt (Tendencia)", "Error (MAPE)": mape(y, holt), "Pred": holt}
        ]
        mejor = min(scores, key=lambda x: x['Error (MAPE)'])
        
        st.success(f"🥇 El mejor método es: **{mejor['Método']}** (Acierto: {100-mejor['Error (MAPE) Susan']:.2f}%)")
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(y=y, name="Real (Histórico)", line=dict(color='black', width=2)))
        fig_t.add_trace(go.Scatter(y=mejor['Pred'], name="Pronóstico Sugerido", line=dict(color='green', dash='dash')))
        st.plotly_chart(fig_t, use_container_width=True)

# --- TAB 3: REGRESIÓN MÚLTIPLE ---
with tab3:
    st.header("📈 Regresión Lineal Múltiple")
    st.write("Ideal para nivel Dirección: ¿Cómo afectan las otras variables a mi demanda?")
    features = st.multiselect("Selecciona las variables independientes (X):", [c for c in cols if c != col_demanda])
    
    if features:
        X = df_working[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        model = LinearRegression().fit(X, y)
        pred_rm = model.predict(X)
        
        st.metric("Confiabilidad Estratégica (R²)", f"{model.score(X, y)*100:.2f}%")
        
        fig_rm = go.Figure()
        fig_rm.add_trace(go.Scatter(y=y, name="Real"))
        fig_rm.add_trace(go.Scatter(y=pred_rm, name="Modelo Multivariable", line=dict(color='purple')))
        st.plotly_chart(fig_rm, use_container_width=True)
    else:
        st.info("Selecciona columnas como 'Precio' o 'Marketing' para ver la regresión.")

# --- TAB 4: REPORTE ---
with tab4:
    st.header("📥 Descargar Resultados")
    csv = df_working.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Descargar Datos y Pronóstico (CSV)", data=csv, file_name="logipredict_report.csv", mime="text/csv")
    st.success("Reporte listo para ser presentado en Excel.")
