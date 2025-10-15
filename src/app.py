import streamlit as st
import sys
from pathlib import Path

# Configurar el path para importar módulos
sys.path.append(str(Path(__file__).parent))

# Importaciones absolutas
from ui.styles import get_custom_css
from ui.components import render_sidebar
from ui.layouts import render_home_page, render_data_loading_page, render_eda_page
from ui.layouts import render_preprocessing_page, render_training_page, render_evaluation_page, render_prediction_page

# Configuración de la página
st.set_page_config(
    page_title="ML Supervisado - Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar CSS personalizado
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Inicializar estado de sesión
def initialize_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False

def main():
    initialize_session_state()
    
    # Renderizar sidebar y obtener página seleccionada
    page = render_sidebar()
    
    # Navegación de páginas
    if page == "🏠 Inicio":
        render_home_page()
    
    elif page == "📁 Carga de Datos":
        render_data_loading_page()
    
    elif page == "🔍 Análisis Exploratorio":
        render_eda_page(st.session_state.df)
    
    elif page == "⚙️ Preprocesamiento":
        render_preprocessing_page(st.session_state.df)
    
    elif page == "🎯 Entrenamiento":
        render_training_page(st.session_state.df)
    
    elif page == "📈 Evaluación":
        render_evaluation_page()
    
    elif page == "🔮 Predicción":
        render_prediction_page()
    
    # Footer
    render_footer()

def render_footer():
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64748b; padding: 20px;'>
            <p style='font-weight: 600;'>🤖 Dashboard ML Supervisado | Proyecto Parcial 2025</p>
            <p style='font-size: 13px; margin-top: 10px;'>Desarrollado con Streamlit, Python & ❤️</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()