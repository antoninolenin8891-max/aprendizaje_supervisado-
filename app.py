import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Configurar el path para importar módulos
sys.path.append(str(Path(__file__).parent / 'src'))

# Configuración de la página
st.set_page_config(
    page_title="ML Supervisado - Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para un diseño profesional
st.markdown("""
    <style>
    /* Estilo general */
    .main {
        background: #f8fafc;
    }
    
    /* KPI Cards personalizadas */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        text-align: center;
        color: white;
        margin: 10px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
    }
    
    .kpi-title {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    
    .kpi-value {
        font-size: 36px;
        font-weight: 800;
        margin: 10px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .kpi-subtitle {
        font-size: 12px;
        opacity: 0.8;
        margin-top: 5px;
    }
    
    /* KPI alternativo - fondo blanco */
    .kpi-card-white {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .kpi-card-white:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .kpi-value-dark {
        font-size: 36px;
        font-weight: 800;
        color: #1e293b;
        margin: 10px 0;
    }
    
    .kpi-title-dark {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Botones */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: white;
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        background-color: #f1f5f9;
        border: 2px solid transparent;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #667eea;
    }
    
    /* Dataframes mejorados */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5em;
    }
    
    h2 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.8em;
    }
    
    h3 {
        color: #334155;
        font-weight: 600;
        font-size: 1.3em;
    }
    
    /* Cards personalizadas */
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    /* Sección de métricas */
    .metrics-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
    }
    
    /* Tablas estilizadas */
    .styled-table {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Alert personalizada */
    .custom-alert {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #f59e0b;
        color: #92400e;
        font-weight: 500;
    }
    
    /* Success alert */
    .success-alert {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #10b981;
        color: #065f46;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Función para crear KPI cards
def create_kpi_card(title, value, subtitle="", gradient=True):
    if gradient:
        return f"""
        <div class='kpi-card'>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-subtitle'>{subtitle}</div>
        </div>
        """
    else:
        return f"""
        <div class='kpi-card-white'>
            <div class='kpi-title-dark'>{title}</div>
            <div class='kpi-value-dark'>{value}</div>
            <div class='kpi-subtitle'>{subtitle}</div>
        </div>
        """

# Inicializar estado de sesión
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False

# =============== SIDEBAR ===============
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
    
    # Estado del pipeline
    st.markdown("<h3 style='color: white; font-size: 16px;'>📋 Estado del Pipeline</h3>", unsafe_allow_html=True)
    
    status_items = [
        ("Datos cargados", st.session_state.data_loaded),
        ("Preprocesamiento", st.session_state.preprocessing_done),
        ("Modelo entrenado", st.session_state.model_trained)
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
    
    st.markdown("---")
    
    # Información del proyecto
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

# =============== CONTENIDO PRINCIPAL ===============

if page == "🏠 Inicio":
    # Header principal
    st.markdown("""
        <div style='text-align: center; padding: 40px 0 20px 0;'>
            <h1 style='font-size: 3em; margin-bottom: 10px;'>
                🤖 Dashboard de Machine Learning Supervisado
            </h1>
            <p style='font-size: 1.2em; color: #64748b; font-weight: 500;'>
                Sistema Completo de Análisis y Predicción con IA
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Métricas principales estilo KPI
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_kpi_card("Algoritmos ML", "6+", "Modelos disponibles"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_kpi_card("Precisión Objetivo", "95%", "Target accuracy"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_kpi_card("Tiempo Real", "< 1s", "Predicción instantánea"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_kpi_card("POO", "100%", "Código modular"), unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Sección de características
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
            <div class='info-card'>
                <h3>🎯 Características Principales</h3>
                <br>
                <table style='width: 100%; color: #334155;'>
                    <tr>
                        <td style='padding: 12px 0; border-bottom: 1px solid #e2e8f0;'>
                            <b>📊 Carga de Datos</b><br>
                            <small style='color: #64748b;'>Soporte para CSV, Excel, JSON y APIs</small>
                        </td>
                    </tr>
                    <tr>
                        <td style='padding: 12px 0; border-bottom: 1px solid #e2e8f0;'>
                            <b>🔍 Análisis Exploratorio</b><br>
                            <small style='color: #64748b;'>Visualizaciones interactivas y estadísticas</small>
                        </td>
                    </tr>
                    <tr>
                        <td style='padding: 12px 0; border-bottom: 1px solid #e2e8f0;'>
                            <b>⚙️ Preprocesamiento</b><br>
                            <small style='color: #64748b;'>Limpieza automática y feature engineering</small>
                        </td>
                    </tr>
                    <tr>
                        <td style='padding: 12px 0; border-bottom: 1px solid #e2e8f0;'>
                            <b>🎯 Entrenamiento</b><br>
                            <small style='color: #64748b;'>6+ algoritmos con ajuste automático</small>
                        </td>
                    </tr>
                    <tr>
                        <td style='padding: 12px 0; border-bottom: 1px solid #e2e8f0;'>
                            <b>📈 Evaluación</b><br>
                            <small style='color: #64748b;'>Métricas completas y visualizaciones</small>
                        </td>
                    </tr>
                    <tr>
                        <td style='padding: 12px 0;'>
                            <b>🔮 Predicción</b><br>
                            <small style='color: #64748b;'>Interface intuitiva para nuevos datos</small>
                        </td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-card'>
                <h3>📊 Pipeline de Trabajo</h3>
                <br>
        """, unsafe_allow_html=True)
        
        # Pipeline visual con progress
        steps = [
            ("1️⃣", "Carga de Datos", "Importa tu dataset desde múltiples fuentes"),
            ("2️⃣", "Análisis Exploratorio", "Explora y visualiza patrones en los datos"),
            ("3️⃣", "Preprocesamiento", "Limpia, transforma y prepara los datos"),
            ("4️⃣", "Entrenamiento", "Selecciona y entrena el mejor modelo"),
            ("5️⃣", "Evaluación", "Analiza métricas y rendimiento"),
            ("6️⃣", "Predicción", "Realiza predicciones en tiempo real")
        ]
        
        for i, (num, title, desc) in enumerate(steps):
            st.markdown(f"""
                <div style='padding: 15px; margin: 10px 0; background: #f8fafc; 
                            border-radius: 10px; border-left: 4px solid #667eea;'>
                    <div style='color: #667eea; font-size: 20px; font-weight: bold;'>{num} {title}</div>
                    <div style='color: #64748b; font-size: 13px; margin-top: 5px;'>{desc}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Call to action
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; border-radius: 20px; text-align: center; 
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
            <h2 style='color: white; margin: 0; font-size: 2em;'>🚀 ¿Listo para comenzar?</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 15px 0; font-size: 1.1em;'>
                Utiliza el menú lateral para navegar por las diferentes etapas del pipeline de ML
            </p>
            <div style='margin-top: 20px;'>
                <span style='background: rgba(255,255,255,0.2); padding: 10px 20px; 
                             border-radius: 20px; color: white; font-weight: 600;'>
                    👈 Comienza con "Carga de Datos"
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif page == "📁 Carga de Datos":
    st.title("📁 Carga de Datos")
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📤 Subir Archivo", "🔗 Desde URL"])
    
    with tab1:
        st.markdown("""
            <div class='info-card'>
                <h3>📂 Sube tu Dataset</h3>
                <p style='color: #64748b;'>Formatos soportados: <b>CSV</b>, <b>Excel</b> (.xlsx, .xls), <b>JSON</b></p>
                <p style='color: #64748b; font-size: 13px;'>Tamaño máximo: 200 MB</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Arrastra tu archivo aquí o haz clic para seleccionar",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Selecciona un archivo con tu dataset"
        )
        
        if uploaded_file:
            try:
                # Determinar tipo de archivo y cargar
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                with st.spinner('🔄 Cargando y procesando datos...'):
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_extension in ['xlsx', 'xls']:
                        df = pd.read_excel(uploaded_file)
                    elif file_extension == 'json':
                        df = pd.read_json(uploaded_file)
                    
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                
                st.markdown(f"""
                    <div class='success-alert'>
                        <b>✅ ¡Datos cargados exitosamente!</b><br>
                        Se cargaron <b>{df.shape[0]:,}</b> filas y <b>{df.shape[1]}</b> columnas
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Métricas principales del dataset
                st.markdown("<h3>📊 Resumen del Dataset</h3>", unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(create_kpi_card("Total Filas", f"{df.shape[0]:,}", 
                                              "Registros", gradient=False), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(create_kpi_card("Total Columnas", f"{df.shape[1]}", 
                                              "Variables", gradient=False), unsafe_allow_html=True)
                
                with col3:
                    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
                    st.markdown(create_kpi_card("Numéricas", f"{numeric_cols}", 
                                              "Variables", gradient=False), unsafe_allow_html=True)
                
                with col4:
                    cat_cols = df.select_dtypes(include=['object']).shape[1]
                    st.markdown(create_kpi_card("Categóricas", f"{cat_cols}", 
                                              "Variables", gradient=False), unsafe_allow_html=True)
                
                with col5:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                    st.markdown(create_kpi_card("Valores Nulos", f"{missing_pct:.1f}%", 
                                              "Del total", gradient=False), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Vista previa del dataset
                with st.expander("👁️ Vista Previa del Dataset (primeras 10 filas)", expanded=True):
                    st.markdown("<div class='styled-table'>", unsafe_allow_html=True)
                    st.dataframe(
                        df.head(10), 
                        use_container_width=True,
                        height=400
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Información detallada
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📋 Información de Columnas", expanded=False):
                        info_df = pd.DataFrame({
                            'Columna': df.columns,
                            'Tipo de Dato': df.dtypes.values,
                            'Valores Únicos': [df[col].nunique() for col in df.columns],
                            'Nulos': df.isnull().sum().values,
                            '% Nulos': (df.isnull().sum().values / len(df) * 100).round(2)
                        })
                        st.dataframe(info_df, use_container_width=True, height=400)
                
                with col2:
                    with st.expander("📊 Estadísticas por Tipo", expanded=False):
                        type_stats = pd.DataFrame({
                            'Tipo': df.dtypes.value_counts().index.astype(str),
                            'Cantidad': df.dtypes.value_counts().values
                        })
                        
                        fig = px.pie(
                            type_stats, 
                            values='Cantidad', 
                            names='Tipo',
                            title='Distribución de Tipos de Datos',
                            color_discrete_sequence=px.colors.sequential.Purples_r
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.markdown(f"""
                    <div class='custom-alert'>
                        <b>❌ Error al cargar el archivo</b><br>
                        {str(e)}
                    </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
            <div class='info-card'>
                <h3>🔗 Cargar desde URL</h3>
                <p style='color: #64748b;'>Ingresa la URL directa de un archivo CSV público</p>
            </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input(
            "URL del dataset:", 
            placeholder="https://ejemplo.com/datos.csv",
            help="Debe ser una URL pública accesible"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            load_btn = st.button("🔗 Cargar desde URL", use_container_width=True, type="primary")
        
        if load_btn:
            if url:
                try:
                    with st.spinner('🔄 Descargando y procesando datos...'):
                        df = pd.read_csv(url)
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                    
                    st.markdown(f"""
                        <div class='success-alert'>
                            <b>✅ ¡Datos cargados exitosamente desde URL!</b><br>
                            Se cargaron <b>{df.shape[0]:,}</b> filas y <b>{df.shape[1]}</b> columnas
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(df.head(), use_container_width=True)
                    
                except Exception as e:
                    st.markdown(f"""
                        <div class='custom-alert'>
                            <b>❌ Error al cargar desde URL</b><br>
                            {str(e)}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Por favor ingresa una URL válida")

elif page == "🔍 Análisis Exploratorio":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")
    
    if not st.session_state.data_loaded:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay datos cargados</b><br>
                Por favor, primero carga un dataset desde la sección "Carga de Datos"
            </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.df
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Panel de métricas generales
        st.markdown("<h3>📊 Métricas Generales del Dataset</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown(create_kpi_card("Registros", f"{len(df):,}", "", False), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_kpi_card("Variables", f"{df.shape[1]}", "", False), unsafe_allow_html=True)
        
        with col3:
            duplicados = df.duplicated().sum()
            st.markdown(create_kpi_card("Duplicados", f"{duplicados:,}", "", False), unsafe_allow_html=True)
        
        with col4:
            memoria = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown(create_kpi_card("Memoria", f"{memoria:.1f} MB", "", False), unsafe_allow_html=True)
        
        with col5:
            nulos_total = df.isnull().sum().sum()
            st.markdown(create_kpi_card("Nulos Total", f"{nulos_total:,}", "", False), unsafe_allow_html=True)
        
        with col6:
            completitud = ((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
            st.markdown(create_kpi_card("Completitud", f"{completitud:.1f}%", "", False), unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Tabs para diferentes análisis
        tabs = st.tabs([
            "📊 Resumen Estadístico",
            "📈 Distribuciones",
            "🔗 Correlaciones",
            "📉 Valores Atípicos"
        ])
        
        with tabs[0]:
            st.markdown("<h3>📋 Estadísticas Descriptivas</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<div class='styled-table'>", unsafe_allow_html=True)
                stats_df = df.describe().round(3)
                st.dataframe(stats_df, use_container_width=True, height=400)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<h4>🎯 Información Detallada</h4>", unsafe_allow_html=True)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_var = st.selectbox("Selecciona variable:", numeric_cols)
                    
                    st.markdown(f"""
                        <div class='info-card'>
                            <h4>{selected_var}</h4>
                            <table style='width: 100%; margin-top: 15px;'>
                                <tr>
                                    <td style='padding: 8px; color: #64748b;'>Media:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].mean():.3f}</td>
                                </tr>
                                <tr style='background: #f8fafc;'>
                                    <td style='padding: 8px; color: #64748b;'>Mediana:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].median():.3f}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px; color: #64748b;'>Desv. Est.:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].std():.3f}</td>
                                </tr>
                                <tr style='background: #f8fafc;'>
                                    <td style='padding: 8px; color: #64748b;'>Mínimo:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].min():.3f}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px; color: #64748b;'>Máximo:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].max():.3f}</td>
                                </tr>
                                <tr style='background: #f8fafc;'>
                                    <td style='padding: 8px; color: #64748b;'>Asimetría:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].skew():.3f}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px; color: #64748b;'>Curtosis:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].kurtosis():.3f}</td>
                                </tr>
                            </table>
                        </div>
                    """, unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("<h3>📊 Distribución de Variables</h3>", unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    selected_col = st.selectbox("Selecciona variable:", numeric_cols, key='dist_var')
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(create_kpi_card("Media", f"{df[selected_col].mean():.2f}", "", False), 
                               unsafe_allow_html=True)
                    st.markdown(create_kpi_card("Mediana", f"{df[selected_col].median():.2f}", "", False), 
                               unsafe_allow_html=True)
                    st.markdown(create_kpi_card("Desv. Est.", f"{df[selected_col].std():.2f}", "", False), 
                               unsafe_allow_html=True)
                
                with col2:
                    # Crear subplot con histograma y boxplot
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(f'Histograma - {selected_col}', f'Box Plot - {selected_col}'),
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4]
                    )
                    
                    # Histograma
                    fig.add_trace(
                        go.Histogram(
                            x=df[selected_col],
                            nbinsx=50,
                            name='Frecuencia',
                            marker_color='#667eea',
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
                    
                    # Box plot
                    fig.add_trace(
                        go.Box(
                            x=df[selected_col],
                            name=selected_col,
                            marker_color='#764ba2',
                            boxmean='sd'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        height=600,
                        showlegend=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de todas las distribuciones
                st.markdown("<br><h4>📊 Todas las Variables Numéricas</h4>", unsafe_allow_html=True)
                
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    subplot_titles=numeric_cols
                )
                
                for idx, col in enumerate(numeric_cols):
                    row = idx // n_cols + 1
                    col_pos = idx % n_cols + 1
                    
                    fig.add_trace(
                        go.Histogram(
                            x=df[col],
                            name=col,
                            marker_color='#667eea',
                            opacity=0.7,
                            showlegend=False
                        ),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(
                    height=300 * n_rows,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No hay variables numéricas en el dataset")
        
        with tabs[2]:
            st.markdown("<h3>🔗 Análisis de Correlaciones</h3>", unsafe_allow_html=True)
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                
                # Heatmap de correlación
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    title="Matriz de Correlación de Pearson",
                    labels=dict(color="Correlación")
                )
                
                fig.update_layout(
                    height=max(600, len(corr_matrix) * 40),
                    font=dict(size=10),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Top correlaciones
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("<h4>🔝 Correlaciones Más Fuertes (Positivas)</h4>", unsafe_allow_html=True)
                    
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlación': corr_matrix.iloc[i, j]
                            })
                    
                    corr_df = pd.DataFrame(corr_pairs)
                    top_positive = corr_df.sort_values('Correlación', ascending=False).head(10)
                    top_positive['Correlación'] = top_positive['Correlación'].round(3)
                    
                    st.dataframe(top_positive, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("<h4>🔻 Correlaciones Más Fuertes (Negativas)</h4>", unsafe_allow_html=True)
                    
                    top_negative = corr_df.sort_values('Correlación', ascending=True).head(10)
                    top_negative['Correlación'] = top_negative['Correlación'].round(3)
                    
                    st.dataframe(top_negative, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Se necesitan al menos 2 variables numéricas para calcular correlaciones")
        
        with tabs[3]:
            st.markdown("<h3>📉 Detección de Valores Atípicos (Outliers)</h3>", unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_var = st.selectbox("Selecciona variable para análisis:", numeric_cols, key='outlier')
                
                # Calcular outliers con IQR
                Q1 = df[selected_var].quantile(0.25)
                Q3 = df[selected_var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[selected_var] < lower_bound) | (df[selected_var] > upper_bound)]
                outliers_pct = len(outliers) / len(df) * 100
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(create_kpi_card("Total Outliers", f"{len(outliers):,}", "", False), 
                               unsafe_allow_html=True)
                
                with col2:
                    st.markdown(create_kpi_card("Porcentaje", f"{outliers_pct:.2f}%", "del dataset", False), 
                               unsafe_allow_html=True)
                
                with col3:
                    st.markdown(create_kpi_card("Límite Inferior", f"{lower_bound:.2f}", "", False), 
                               unsafe_allow_html=True)
                
                with col4:
                    st.markdown(create_kpi_card("Límite Superior", f"{upper_bound:.2f}", "", False), 
                               unsafe_allow_html=True)
                
                with col5:
                    st.markdown(create_kpi_card("Método", "IQR 1.5", "", False), 
                               unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualización
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure()
                    
                    # Box plot principal
                    fig.add_trace(go.Box(
                        y=df[selected_var],
                        name=selected_var,
                        marker_color='#667eea',
                        boxmean='sd',
                        boxpoints='outliers'
                    ))
                    
                    # Líneas de límites
                    fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                                 annotation_text=f"Límite Inferior: {lower_bound:.2f}")
                    fig.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                                 annotation_text=f"Límite Superior: {upper_bound:.2f}")
                    
                    fig.update_layout(
                        title=f"Detección de Outliers: {selected_var}",
                        yaxis_title=selected_var,
                        showlegend=False,
                        height=500,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("<h4>📊 Resumen Estadístico</h4>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class='info-card'>
                            <table style='width: 100%;'>
                                <tr>
                                    <td style='padding: 8px; color: #64748b;'>Q1 (25%):</td>
                                    <td style='padding: 8px; font-weight: bold;'>{Q1:.3f}</td>
                                </tr>
                                <tr style='background: #f8fafc;'>
                                    <td style='padding: 8px; color: #64748b;'>Q2 (50%):</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].median():.3f}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px; color: #64748b;'>Q3 (75%):</td>
                                    <td style='padding: 8px; font-weight: bold;'>{Q3:.3f}</td>
                                </tr>
                                <tr style='background: #f8fafc;'>
                                    <td style='padding: 8px; color: #64748b;'>IQR:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{IQR:.3f}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px; color: #64748b;'>Mínimo:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].min():.3f}</td>
                                </tr>
                                <tr style='background: #f8fafc;'>
                                    <td style='padding: 8px; color: #64748b;'>Máximo:</td>
                                    <td style='padding: 8px; font-weight: bold;'>{df[selected_var].max():.3f}</td>
                                </tr>
                            </table>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Tabla de outliers si existen
                if len(outliers) > 0:
                    st.markdown("<br><h4>📋 Registros con Outliers (primeros 20)</h4>", unsafe_allow_html=True)
                    st.dataframe(outliers.head(20), use_container_width=True)
            else:
                st.info("ℹ️ No hay variables numéricas para analizar outliers")

elif page == "⚙️ Preprocesamiento":
    st.title("⚙️ Preprocesamiento de Datos")
    
    if not st.session_state.data_loaded:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay datos cargados</b><br>
                Por favor, primero carga un dataset desde la sección "Carga de Datos"
            </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.df
        
        st.markdown("""
            <div class='info-card'>
                <h3>🔧 Herramientas de Preprocesamiento</h3>
                <p style='color: #64748b;'>Limpia, transforma y prepara tus datos para el modelado</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tabs = st.tabs([
            "🧹 Limpieza de Datos",
            "🔢 Codificación",
            "📏 Escalado",
            "✂️ División Train/Test"
        ])
        
        with tabs[0]:
            st.markdown("<h3>🧹 Limpieza de Datos</h3>", unsafe_allow_html=True)
            
            # Métricas de calidad
            col1, col2, col3, col4 = st.columns(4)
            
            total_nulls = df.isnull().sum().sum()
            total_duplicates = df.duplicated().sum()
            completeness = (1 - total_nulls / (df.shape[0] * df.shape[1])) * 100
            
            with col1:
                st.markdown(create_kpi_card("Valores Nulos", f"{total_nulls:,}", "", False), 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_kpi_card("Duplicados", f"{total_duplicates:,}", "", False), 
                           unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_kpi_card("Completitud", f"{completeness:.1f}%", "", False), 
                           unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_kpi_card("Filas Limpias", f"{len(df.dropna()):,}", "", False), 
                           unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Opciones de limpieza
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h4>📊 Valores Nulos por Columna</h4>", unsafe_allow_html=True)
                
                null_df = pd.DataFrame({
                    'Columna': df.columns,
                    'Nulos': df.isnull().sum().values,
                    '% Nulos': (df.isnull().sum().values / len(df) * 100).round(2)
                })
                null_df = null_df[null_df['Nulos'] > 0].sort_values('Nulos', ascending=False)
                
                if len(null_df) > 0:
                    st.dataframe(null_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No hay valores nulos en el dataset")
            
            with col2:
                st.markdown("<h4>🔧 Acciones de Limpieza</h4>", unsafe_allow_html=True)
                
                st.info("""
                **Próximamente disponible:**
                - Eliminar valores nulos
                - Imputar con media/mediana/moda
                - Eliminar duplicados
                - Filtrar outliers
                
                *Integrar con tu clase DataPreprocessor*
                """)
        
        with tabs[1]:
            st.markdown("<h3>🔢 Codificación de Variables</h3>", unsafe_allow_html=True)
            st.info("🚧 Sección para integrar con tu clase DataPreprocessor - Codificación One-Hot, Label Encoding, etc.")
        
        with tabs[2]:
            st.markdown("<h3>📏 Escalado de Características</h3>", unsafe_allow_html=True)
            st.info("🚧 Sección para integrar con tu clase DataPreprocessor - StandardScaler, MinMaxScaler, etc.")
        
        with tabs[3]:
            st.markdown("<h3>✂️ División Train/Test</h3>", unsafe_allow_html=True)
            st.info("🚧 Sección para integrar con tu clase DataPreprocessor - División de datos")

elif page == "🎯 Entrenamiento":
    st.title("🎯 Entrenamiento del Modelo")
    
    if not st.session_state.data_loaded:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay datos cargados</b><br>
                Por favor, primero carga y preprocesa los datos
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='info-card'>
                <h3>🤖 Selección y Entrenamiento de Modelos</h3>
                <p style='color: #64748b;'>Elige el algoritmo más adecuado para tu problema</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Selección de algoritmo
        st.markdown("<h3>🎯 Selecciona el Algoritmo</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='info-card'>
                    <h4>📊 Regresión</h4>
                    <ul style='color: #64748b; line-height: 2;'>
                        <li>Regresión Lineal</li>
                        <li>Ridge & Lasso</li>
                        <li>Árboles CART</li>
                        <li>KNN Regresión</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='info-card'>
                    <h4>🎯 Clasificación</h4>
                    <ul style='color: #64748b; line-height: 2;'>
                        <li>Regresión Logística</li>
                        <li>Árboles de Decisión</li>
                        <li>KNN Clasificación</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='info-card'>
                    <h4>🧠 Deep Learning</h4>
                    <ul style='color: #64748b; line-height: 2;'>
                        <li>Redes Neuronales</li>
                        <li>Personalizable</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.info("🚧 Esta sección se integrará con tu clase ModelTrainer")

elif page == "📈 Evaluación":
    st.title("📈 Evaluación del Modelo")
    
    if not st.session_state.model_trained:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay modelo entrenado</b><br>
                Por favor, primero entrena un modelo en la sección "Entrenamiento"
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("🚧 Esta sección se integrará con tu clase ModelEvaluator")
        
        # Ejemplo de cómo se verán las métricas
        st.markdown("<h3>📊 Métricas del Modelo</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_kpi_card("Accuracy", "95.4%", "Precisión general"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_kpi_card("Precision", "93.2%", "Precisión positiva"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_kpi_card("Recall", "94.8%", "Sensibilidad"), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_kpi_card("F1-Score", "94.0%", "Media armónica"), unsafe_allow_html=True)

elif page == "🔮 Predicción":
    st.title("🔮 Realizar Predicciones")
    
    if not st.session_state.model_trained:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay modelo entrenado</b><br>
                Por favor, primero entrena un modelo en la sección "Entrenamiento"
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("🚧 Esta sección se integrará con tu clase Predictor")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p style='font-weight: 600;'>🤖 Dashboard ML Supervisado | Proyecto Parcial 2025</p>
        <p style='font-size: 13px; margin-top: 10px;'>Desarrollado con Streamlit, Python & ❤️</p>
    </div>
""", unsafe_allow_html=True)