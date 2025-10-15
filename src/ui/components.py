import streamlit as st
from .styles import create_kpi_card

def render_sidebar():
    """Renderiza la barra lateral con navegación y estado"""
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h2 style='color: white; margin: 0;'>🤖 ML Dashboard</h2>
                <p style='color: rgba(255,255,255,0.7); font-size: 12px;'>Sistema Predictivo Avanzado</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Selector de páginas
        page = st.radio(
            "📊 Navegación",
            ["🏠 Inicio", "📁 Carga de Datos", "🔍 Análisis Exploratorio", 
             "⚙️ Preprocesamiento", "🎯 Entrenamiento", "📈 Evaluación", "🔮 Predicción"],
            label_visibility="visible"
        )
        
        st.markdown("---")
        render_pipeline_status()
        st.markdown("---")
        render_project_info()
        
        return page

def render_pipeline_status():
    """Renderiza el estado del pipeline"""
    st.markdown("<h3 style='color: white; font-size: 16px;'>📋 Estado del Pipeline</h3>", unsafe_allow_html=True)
    
    status_items = [
        ("Datos cargados", st.session_state.get('data_loaded', False)),
        ("Preprocesamiento", st.session_state.get('preprocessing_done', False)),
        ("Modelo entrenado", st.session_state.get('model_trained', False))
    ]
    
    for label, status in status_items:
        icon = "✅" if status else "⏳"
        color = "#10b981" if status else "#f59e0b"
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; 
                    margin: 8px 0; border-left: 3px solid {color};'>
            <span style='color: white; font-size: 14px;'>{icon} {label}</span>
        </div>
        """, unsafe_allow_html=True)

def render_project_info():
    """Renderiza la información del proyecto"""
    with st.expander("ℹ️ Información", expanded=False):
        st.markdown("""
        **Proyecto ML Supervisado**
        
        **Algoritmos:**
        - Regresión Lineal
        - Regresión Logística  
        - Ridge & Lasso
        - Árboles CART
        - KNN Regresión
        - Redes Neuronales
        
        **Autor:** Tu Nombre
        **Fecha:** Oct 2025
        """, unsafe_allow_html=True)

def render_kpi_metrics(metrics):
    """Renderiza métricas KPI en columnas"""
    cols = st.columns(len(metrics))
    for col, (title, value, subtitle) in zip(cols, metrics):
        with col:
            st.markdown(create_kpi_card(title, value, subtitle, gradient=True), 
                       unsafe_allow_html=True)