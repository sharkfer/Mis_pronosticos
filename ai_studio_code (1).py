import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="LogiPredict Pro | Enterprise", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .best-model { border: 2px solid #28a745; background-color: #e9f7ef; padding: 20px; border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CÁLCULO ---
def calcular_metricas(real, pronostico):
    real, pronostico = np.array(real), np.array(pronostico)
    errores = real - pronostico
    mad = np.mean(np.abs(errores))
    mape = np.mean(np.abs(errores / real)) * 100
    # Tracking Signal (PDF pág 6): Suma errores / MAD
    ts = np.sum(errores) / mad if mad != 0 else 0
    return round(mad, 2), round(mape, 2), round(ts, 2)

# --- SIDEBAR: TEORÍA Y AYUDA ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1622/1622857.png", width=100)
    st.title("Centro de Expertos")
    
    with st.expander("📖 Conceptos Clave (PDF)"):
        st.write("**Regla GIGO:** Basura entra, basura sale. La limpieza es vital.")
        st.write("**Tracking Signal:** Límite aceptable [-4, +4].")
        st.write("**MAD:** Desviación media en unidades.")
        st.write("**MAPE:** Error en porcentaje (Ideal < 10%).")
    
    st.info("Nivel: Estudiante / Dirección / CEO")

# --- TABS PRINCIPALES ---
tab1, tab2, tab3, tab4 = st.tabs(["🧹 1. Limpieza GIGO", "🏆 2. Modo Torneo", "📈 3. Regresión Múltiple", "📥 4. Reporte Final"])

# --- GENERACIÓN DE DATOS DE EJEMPLO ---
if 'df' not in st.session_state:
    fechas = pd.date_range(start='2023-01-01', periods=15, freq='M')
    data = {
        'Fecha': fechas,
        'Demanda': [100, 110, 5, 115, 120, 130, 140, 155, 160, 250, 170, 180, 190, 200, 210],
        'Precio': [50, 50, 50, 48, 48, 45, 45, 42, 40, 40, 40, 38, 35, 35, 30],
        'Mkt_Inversion': [5, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 35, 40, 50]
    }
    st.session_state.df = pd.DataFrame(data)

# --- TAB 1: MÓDULO DE LIMPIEZA ---
with tab1:
    st.header("Módulo de Limpieza de Datos (Normalización)")
    st.write("Detectamos anomalías (desabastos o picos atípicos) según la Unidad 4.2 del PDF.")
    
    df_clean = st.session_state.df.copy()
    
    # Detección simple de Outliers (Z-score > 2)
    mean = df_clean['Demanda'].mean()
    std = df_clean['Demanda'].std()
    df_clean['Outlier'] = (np.abs(df_clean['Demanda'] - mean) > (1.5 * std))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Visualización de Anomalías")
        fig_clean = go.Figure()
        fig_clean.add_trace(go.Scatter(x=df_clean['Fecha'], y=df_clean['Demanda'], name="Demanda Original"))
        outliers = df_clean[df_clean['Outlier']]
        fig_clean.add_trace(go.Scatter(x=outliers['Fecha'], y=outliers['Demanda'], mode='markers', name="Anomalía Detectada", marker=dict(color='red', size=12)))
        st.plotly_chart(fig_clean, use_container_width=True)
    
    with col2:
        st.subheader("Acción Sugerida")
        if outliers.empty:
            st.success("No se detectan anomalías críticas.")
        else:
            st.warning(f"Se detectaron {len(outliers)} puntos sospechosos.")
            if st.button("Normalizar Datos (Limpieza GIGO)"):
                # Reemplazar con el promedio móvil de los vecinos
                df_clean.loc[df_clean['Outlier'], 'Demanda'] = mean
                st.session_state.df = df_clean
                st.rerun()
        st.info("Nota: Un valor muy bajo puede ser desabasto (limpiar). Un valor muy alto puede ser una promoción única.")

# --- TAB 2: MODO TORNEO ---
with tab2:
    st.header("Modo Torneo: Competencia de Modelos")
    st.write("Ejecutamos múltiples modelos de series de tiempo y elegimos el mejor basándonos en el **MAPE**.")
    
    df_t = st.session_state.df.copy()
    y = df_t['Demanda'].values
    
    modelos_resultados = {}
    
    # 1. Promedio Móvil Simple (n=3)
    df_t['PMS'] = df_t['Demanda'].rolling(window=3).mean().shift(1).fillna(y[0])
    modelos_resultados['Promedio Móvil Simple'] = df_t['PMS']
    
    # 2. Suavización Exponencial Simple (SES)
    ses = ExponentialSmoothing(y, trend=None).fit(smoothing_level=0.3, optimized=False)
    df_t['SES'] = ses.fittedvalues
    modelos_resultados['Suavización Exponencial (SES)'] = df_t['SES']
    
    # 3. Holt (Tendencia)
    holt = ExponentialSmoothing(y, trend='add').fit()
    df_t['Holt'] = holt.fittedvalues
    modelos_resultados['Holt (Tendencia)'] = df_t['Holt']

    # Ranking de Torneo
    ranking = []
    for nombre, pred in modelos_resultados.items():
        mad, mape, ts = calcular_metricas(df_t['Demanda'], pred)
        ranking.append({"Modelo": nombre, "MAPE": mape, "MAD": mad, "Tracking Signal": ts, "Prediccion": pred})
    
    rank_df = pd.DataFrame(ranking).sort_values("MAPE")
    ganador = rank_df.iloc[0]
    
    st.markdown(f"""
        <div class="best-model">
            <h3>🏆 Modelo Ganador: {ganador['Modelo']}</h3>
            <p>Este modelo tiene la mayor precisión para tu demanda actual.</p>
            <p><b>MAPE:</b> {ganador['MAPE']}% | <b>Confianza:</b> {100-ganador['MAPE']}%</p>
        </div>
    """, unsafe_allow_html=True)
    
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df_t['Fecha'], y=df_t['Demanda'], name="Real", line=dict(color='black', width=2)))
    for m in modelos_resultados:
        fig_t.add_trace(go.Scatter(x=df_t['Fecha'], y=modelos_resultados[m], name=m, line=dict(dash='dot')))
    st.plotly_chart(fig_t, use_container_width=True)
    
    st.table(rank_df[["Modelo", "MAPE", "MAD", "Tracking Signal"]])

# --- TAB 3: REGRESIÓN MULTIPLE ---
with tab3:
    st.header("Análisis de Regresión Lineal Múltiple")
    st.write("Nivel Dirección: ¿Cómo influyen el Precio y el Marketing en la demanda?")
    
    features = ['Precio', 'Mkt_Inversion']
    X = st.session_state.df[features]
    y = st.session_state.df['Demanda']
    
    model_rm = LinearRegression()
    model_rm.fit(X, y)
    pronostico_rm = model_rm.predict(X)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Ecuación del Negocio")
        for i, f in enumerate(features):
            st.metric(f"Impacto de: {f}", f"{model_rm.coef_[i]:.2f}", 
                      delta="Positivo" if model_rm.coef_[i] > 0 else "Negativo")
        st.write(f"**Intercepción (Base):** {model_rm.intercept_:.2f}")
    
    with c2:
        st.subheader("Predicción vs Realidad")
        fig_rm = go.Figure()
        fig_rm.add_trace(go.Scatter(x=st.session_state.df['Fecha'], y=y, name="Real"))
        fig_rm.add_trace(go.Scatter(x=st.session_state.df['Fecha'], y=pronostico_rm, name="Regresión Múltiple", line=dict(color='green')))
        st.plotly_chart(fig_rm, use_container_width=True)

# --- TAB 4: EXPORTACIÓN ---
with tab4:
    st.header("Exportación de Resultados")
    st.write("Genera el archivo para presentar a Dirección o para tu estudio.")
    
    # Consolidar datos
    df_final = st.session_state.df.copy()
    df_final['Pronostico_Ganador'] = ganador['Prediccion']
    df_final['Error_Absoluto'] = np.abs(df_final['Demanda'] - df_final['Pronostico_Ganador'])
    
    buffer = io.BytesIO()
    df_final.to_csv(buffer, index=False)
    
    st.download_button(
        label="💾 Descargar Excel de Pronósticos (CSV)",
        data=buffer.getvalue(),
        file_name="Reporte_Logistico_Pro.csv",
        mime="text/csv"
    )
    
    st.subheader("Resumen Ejecutivo")
    st.write(f"- **Estado del sistema:** Limpieza aplicada.")
    st.write(f"- **Mejor método:** {ganador['Modelo']}.")
    st.write(f"- **Precisión General:** {100-ganador['MAPE']:.2f}%.")
    
    if abs(ganador['Tracking Signal']) > 4:
        st.error(f"⚠️ Alerta: El Tracking Signal es {ganador['Tracking Signal']}. El modelo tiene sesgo (PDF pág 6).")
    else:
        st.success(f"✅ El modelo es estable (Tracking Signal: {ganador['Tracking Signal']}).")