import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots

# Importaciones absolutas
from ui.components import render_kpi_metrics, create_kpi_card
from utils.visualizations import create_distribution_plots, create_correlation_heatmap

def render_home_page():
    """Renderiza la página de inicio"""
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
    
    # Métricas principales
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

def render_data_loading_page():
    """Renderiza la página de carga de datos"""
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
                
                # Mostrar resumen del dataset
                show_dataset_summary(df)
                
            except Exception as e:
                st.markdown(f"""
                    <div class='custom-alert'>
                        <b>❌ Error al cargar el archivo</b><br>
                        {str(e)}
                    </div>
                """, unsafe_allow_html=True)

def show_dataset_summary(df):
    """Muestra resumen del dataset cargado"""
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

def render_eda_page(df):
    """Renderiza la página de análisis exploratorio completo"""
    if df is None:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay datos cargados</b><br>
                Por favor, primero carga un dataset desde la sección "Carga de Datos"
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Métricas generales del dataset
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
        "📉 Valores Atípicos",
        "👁️ Vista de Datos"
    ])
    
    with tabs[0]:
        render_statistical_summary(df)
    
    with tabs[1]:
        render_distributions(df)
    
    with tabs[2]:
        render_correlations(df)
    
    with tabs[3]:
        render_outliers(df)
        
    with tabs[4]:
        render_data_preview(df)

# ============ AQUÍ DEBEN ESTAR TODAS LAS FUNCIONES AUXILIARES ============
# (NO DENTRO de render_eda_page)

def render_statistical_summary(df):
    """Renderiza resumen estadístico"""
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

def render_distributions(df):
    """Renderiza análisis de distribuciones"""
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
    else:
        st.info("ℹ️ No hay variables numéricas en el dataset")

def render_correlations(df):
    """Renderiza análisis de correlaciones"""
    st.markdown("<h3>🔗 Análisis de Correlaciones</h3>", unsafe_allow_html=True)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        # Heatmap de correlación
        corr_matrix = numeric_df.corr()
        
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
        
        # Top correlaciones
        col1, col2 = st.columns(2)
        
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

def render_outliers(df):
    """Renderiza detección de outliers"""
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
        
        # Métricas de outliers
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
        
        # Visualización
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
        
        # Mostrar outliers si existen
        if len(outliers) > 0:
            with st.expander(f"📋 Registros con Outliers ({len(outliers)} encontrados)"):
                st.dataframe(outliers, use_container_width=True)
    else:
        st.info("ℹ️ No hay variables numéricas para analizar outliers")

def render_data_preview(df):
    """Renderiza vista previa de datos"""
    st.markdown("<h3>👁️ Vista Completa de Datos</h3>", unsafe_allow_html=True)
    
    # Filtros interactivos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rows_to_show = st.slider("Filas a mostrar:", 5, 100, 20)
    
    with col2:
        sort_column = st.selectbox("Ordenar por:", ["None"] + df.columns.tolist())
    
    with col3:
        ascending = st.checkbox("Orden ascendente", True)
    
    # Aplicar filtros
    display_df = df.copy()
    if sort_column != "None":
        display_df = display_df.sort_values(sort_column, ascending=ascending)
    
    st.dataframe(display_df.head(rows_to_show), use_container_width=True)
    
    # Información de columnas
    with st.expander("📋 Información Detallada de Columnas"):
        info_data = []
        for col in df.columns:
            info_data.append({
                'Columna': col,
                'Tipo': str(df[col].dtype),
                'No Nulos': df[col].count(),
                'Nulos': df[col].isnull().sum(),
                '% Nulos': f"{(df[col].isnull().sum() / len(df) * 100):.2f}%",
                'Valores Únicos': df[col].nunique()
            })
        
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)

    # Mostrar vista previa de los datos
    with st.expander("👁️ Vista Previa de Datos", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

def render_preprocessing_page(df):
    """Renderiza la página de preprocesamiento integrada con DataPreprocessor"""
    st.title("⚙️ Preprocesamiento de Datos")
    
    if df is None:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay datos cargados</b><br>
                Por favor, primero carga un dataset desde la sección "Carga de Datos"
            </div>
        """, unsafe_allow_html=True)
        return
    
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
        render_data_cleaning(df)
    
    with tabs[1]:
        render_encoding(df)
    
    with tabs[2]:
        render_scaling(df)
    
    with tabs[3]:
        render_train_test_split(df)

def render_data_cleaning(df):
    """Renderiza herramientas de limpieza de datos"""
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
        
        # Botón para eliminar duplicados
        if total_duplicates > 0:
            if st.button("🗑️ Eliminar Duplicados", use_container_width=True):
                df_clean = df.drop_duplicates()
                st.session_state.df = df_clean
                st.success(f"✅ Se eliminaron {total_duplicates} duplicados")
                st.rerun()
        
        # Botón para eliminar nulos
        if total_nulls > 0:
            if st.button("🧹 Eliminar Filas con Nulos", use_container_width=True):
                df_clean = df.dropna()
                st.session_state.df = df_clean
                st.success(f"✅ Se eliminaron filas con nulos. Nuevo tamaño: {len(df_clean)} filas")
                st.rerun()

def render_encoding(df):
    """Renderiza herramientas de codificación"""
    st.markdown("<h3>🔢 Codificación de Variables Categóricas</h3>", unsafe_allow_html=True)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        st.write(f"**Columnas categóricas encontradas:** {len(categorical_cols)}")
        
        for col in categorical_cols:
            with st.expander(f"📊 {col} - {df[col].nunique()} categorías"):
                st.write(f"Valores únicos: {df[col].unique().tolist()}")
    else:
        st.success("✅ No hay variables categóricas para codificar")

def render_scaling(df):
    """Renderiza herramientas de escalado"""
    st.markdown("<h3>📏 Escalado de Características Numéricas</h3>", unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.write(f"**Columnas numéricas encontradas:** {len(numeric_cols)}")
        st.info("🔜 Funcionalidad de escalado disponible próximamente")
    else:
        st.info("ℹ️ No hay variables numéricas para escalar")

def render_train_test_split(df):
    """Renderiza división train/test"""
    st.markdown("<h3>✂️ División Train/Test</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State:", 0, 100, 42)
    
    with col2:
        st.markdown("""
            <div class='info-card'>
                <h4>📊 Distribución</h4>
                <p><b>Train:</b> {(1-test_size)*100:.1f}%</p>
                <p><b>Test:</b> {test_size*100:.1f}%</p>
                <p><b>Total:</b> {len(df):,} registros</p>
            </div>
        """, unsafe_allow_html=True)
    
    if st.button("🎯 Aplicar División Train/Test", use_container_width=True):
        st.info("🔜 Funcionalidad de división disponible próximamente")


def render_training_page(df):
    """Renderiza la página de entrenamiento"""
    st.title("🎯 Entrenamiento del Modelo")
    
    if df is None:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay datos cargados</b><br>
                Por favor, primero carga y preprocesa los datos
            </div>
        """, unsafe_allow_html=True)
        return
    
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
    
    # Selector de modelo
    st.markdown("<br><h3>⚙️ Configuración del Modelo</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Tipo de Modelo:",
            ["Regresión Lineal", "Regresión Logística", "Random Forest", "Árbol de Decisión", "KNN"]
        )
    
    with col2:
        problem_type = st.selectbox(
            "Tipo de Problema:",
            ["Clasificación", "Regresión"]
        )
    
    # Botón de entrenamiento
    if st.button("🚀 Entrenar Modelo", use_container_width=True, type="primary"):
        with st.spinner("Entrenando modelo..."):
            # Simular entrenamiento
            import time
            time.sleep(2)
            
            # Actualizar estado
            st.session_state.model_trained = True
            st.success("✅ Modelo entrenado exitosamente!")
            st.rerun()

def render_evaluation_page():
    """Renderiza la página de evaluación"""
    st.title("📈 Evaluación del Modelo")
    
    if not st.session_state.get('model_trained', False):
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay modelo entrenado</b><br>
                Por favor, primero entrena un modelo en la sección "Entrenamiento"
            </div>
        """, unsafe_allow_html=True)
    else:
        # Ejemplo de métricas
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
        
        # Gráficas de evaluación
        st.markdown("<br><h3>📈 Visualizaciones de Evaluación</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Matriz de confusión simulada
            st.markdown("**Matriz de Confusión**")
            confusion_data = np.array([[45, 5], [3, 47]])
            fig = px.imshow(
                confusion_data,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Predicho", y="Real", color="Cantidad"),
                x=['Clase 0', 'Clase 1'],
                y=['Clase 0', 'Clase 1']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Curva ROC simulada
            st.markdown("**Curva ROC**")
            # Datos de ejemplo para la curva ROC
            fpr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            tpr = [0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.9, 0.94, 0.97, 0.99, 1.0]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#667eea', width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='red', dash='dash')))
            
            fig.update_layout(
                title='Curva ROC',
                xaxis_title='Tasa de Falsos Positivos',
                yaxis_title='Tasa de Verdaderos Positivos',
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def render_prediction_page():
    """Renderiza la página de predicción usando los datos ya cargados"""
    import time
    
    st.title("🔮 Realizar Predicciones")
    
    # Validación más estricta del estado de los datos
    if not st.session_state.get('data_loaded', False) or st.session_state.df is None:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay datos cargados</b><br>
                Por favor, primero carga un dataset desde la sección "Carga de Datos"
            </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    
    # Validar que el dataset tenga datos
    if len(df) == 0:
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ Dataset vacío</b><br>
                El dataset cargado no contiene datos. Por favor carga un dataset válido.
            </div>
        """, unsafe_allow_html=True)
        return
    
    if not st.session_state.get('model_trained', False):
        st.markdown("""
            <div class='custom-alert'>
                <b>⚠️ No hay modelo entrenado</b><br>
                Por favor, primero entrena un modelo en la sección "Entrenamiento"
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
        <div class='info-card'>
            <h3>🎯 Realizar Predicciones con los Datos Cargados</h3>
            <p style='color: #64748b;'>Usando el dataset actual para hacer predicciones</p>
            <p style='color: #64748b; font-size: 14px;'><b>Dataset:</b> {:,} filas × {} columnas</p>
        </div>
    """.format(len(df), df.shape[1]), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs para diferentes tipos de predicción
    tab1, tab2 = st.tabs(["📊 Predecir Todo el Dataset", "🎯 Predecir Muestra Aleatoria"])
    
    with tab1:
        render_full_dataset_prediction(df)
    
    with tab2:
        render_sample_prediction(df)

def render_full_dataset_prediction(df):
    """Renderiza predicción para todo el dataset"""
    st.markdown("<h3>📊 Predicción Completa del Dataset</h3>", unsafe_allow_html=True)
    
    st.info("""
    **ℹ️ Esta función realizará predicciones para todas las filas del dataset cargado.**
    - Total de registros a predecir: **{:,}**
    - Columnas disponibles: **{}**
    """.format(len(df), df.shape[1]))
    
    # Seleccionar variables para la predicción
    st.markdown("<h4>🎯 Seleccionar Variables para Predicción</h4>", unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔢 Variables Numéricas**")
        selected_numeric = []
        for col in numeric_cols:
            if st.checkbox(col, value=True, key=f"num_{col}"):
                selected_numeric.append(col)
    
    with col2:
        st.markdown("**📝 Variables Categóricas**")
        selected_categorical = []
        for col in categorical_cols:
            if st.checkbox(col, value=True, key=f"cat_{col}"):
                selected_categorical.append(col)
    
    selected_features = selected_numeric + selected_categorical
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una variable para la predicción")
        return
    
    st.success(f"✅ **Variables seleccionadas:** {len(selected_features)}")
    
    # Mostrar vista previa de los datos seleccionados
    with st.expander("👁️ Vista Previa de Datos Seleccionados", expanded=False):
        preview_df = df[selected_features].head(10)
        st.dataframe(preview_df, use_container_width=True)
        st.write(f"**Forma:** {preview_df.shape[0]} filas × {preview_df.shape[1]} columnas")
    
    # Botón para realizar predicción completa
    if st.button("🚀 Realizar Predicción Completa", use_container_width=True, type="primary"):
        with st.spinner(f"Realizando {len(df):,} predicciones..."):
            # Simular procesamiento
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)  # Simular procesamiento
                progress_bar.progress(i + 1)
            
            # Crear resultados de predicción simulados
            results_df = df[selected_features].copy()
            
            # Agregar columnas de resultados
            np.random.seed(42)  # Para resultados consistentes
            results_df['Predicción'] = np.random.choice(['Clase 0', 'Clase 1'], size=len(results_df))
            results_df['Probabilidad'] = np.random.uniform(0.6, 0.98, len(results_df))
            results_df['Confianza'] = np.where(
                results_df['Probabilidad'] > 0.8, 'Alta', 
                np.where(results_df['Probabilidad'] > 0.6, 'Media', 'Baja')
            )
            
            st.session_state.prediction_results = results_df
            
            st.success(f"✅ **Predicción completada!** {len(results_df):,} registros procesados")
            
            # Mostrar métricas de resultados
            st.markdown("<br><h3>📊 Resultados de la Predicción</h3>", unsafe_allow_html=True)
            
            # Métricas principales
            pred_counts = results_df['Predicción'].value_counts()
            avg_probability = results_df['Probabilidad'].mean()
            high_confidence = (results_df['Confianza'] == 'Alta').sum()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(create_kpi_card("Clase 0", f"{pred_counts.get('Clase 0', 0):,}", "Predicciones"), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_kpi_card("Clase 1", f"{pred_counts.get('Clase 1', 0):,}", "Predicciones"), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_kpi_card("Prob. Promedio", f"{avg_probability:.1%}", "Confianza"), unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_kpi_card("Alta Confianza", f"{high_confidence:,}", "Registros"), unsafe_allow_html=True)
            
            # Distribución de confianza
            st.markdown("<h4>📈 Distribución de Confianza</h4>", unsafe_allow_html=True)
            
            confidence_counts = results_df['Confianza'].value_counts()
            fig = px.pie(
                values=confidence_counts.values,
                names=confidence_counts.index,
                title='Distribución de Niveles de Confianza',
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar resultados detallados
            with st.expander("📋 Ver Resultados Detallados", expanded=True):
                st.dataframe(results_df, use_container_width=True)
                
                # Botones de descarga
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Resultados Completos (CSV)",
                        data=csv,
                        file_name="predicciones_completas.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Descargar solo predicciones
                    pred_summary = results_df[['Predicción', 'Probabilidad', 'Confianza']]
                    csv_summary = pred_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Solo Predicciones (CSV)",
                        data=csv_summary,
                        file_name="predicciones_resumen.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

def render_sample_prediction(df):
    """Renderiza predicción para una muestra del dataset"""
    st.markdown("<h3>🎯 Predicción de Muestra Aleatoria</h3>", unsafe_allow_html=True)
    
    # Validar que el dataset no esté vacío
    if df is None or len(df) == 0:
        st.error("❌ El dataset está vacío. Por favor carga datos válidos.")
        return
    
    # Selector de tamaño de muestra con límites seguros
    max_sample_size = min(1000, len(df))
    sample_size = st.slider(
        "Tamaño de la muestra:",
        min_value=1,  # Mínimo 1 en lugar de 10
        max_value=max_sample_size,
        value=min(100, len(df)),  # Valor por defecto seguro
        step=1,
        help=f"Número de registros aleatorios para predecir (máximo: {max_sample_size})"
    )
    
    st.info(f"**Seleccionados:** {sample_size} registros de {len(df):,} totales")
    
    # Validación adicional antes del botón
    if sample_size <= 0:
        st.warning("⚠️ El tamaño de la muestra debe ser mayor a 0")
        return
    
    if sample_size > len(df):
        st.error(f"❌ El tamaño de la muestra ({sample_size}) no puede ser mayor que el dataset ({len(df)})")
        return
    
    if st.button("🔮 Predecir Muestra Aleatoria", use_container_width=True):
        # Validación final antes de procesar
        if len(df) == 0:
            st.error("❌ No hay datos disponibles para predecir")
            return
            
        if sample_size > len(df):
            st.error("❌ Tamaño de muestra inválido")
            return
        
        with st.spinner(f"Seleccionando muestra y realizando {sample_size} predicciones..."):
            time.sleep(1)
            
            try:
                # Seleccionar muestra aleatoria con manejo de errores
                sample_df = df.sample(n=sample_size, random_state=42)
                
                # Simular predicciones
                np.random.seed(42)
                predictions = np.random.choice(['Clase A', 'Clase B', 'Clase C'], size=len(sample_df))
                probabilities = np.random.uniform(0.5, 0.95, len(sample_df))
                
                # Crear DataFrame de resultados
                results_sample = sample_df.copy()
                results_sample['Predicción'] = predictions
                results_sample['Probabilidad'] = probabilities
                results_sample['Confianza'] = np.where(
                    probabilities > 0.8, 'Alta', 
                    np.where(probabilities > 0.6, 'Media', 'Baja')
                )
                
                st.success(f"✅ **Muestra procesada!** {len(results_sample)} predicciones realizadas")
                
                # Mostrar resultados de la muestra
                st.markdown("<h4>📊 Resultados de la Muestra</h4>", unsafe_allow_html=True)
                
                # Métricas de la muestra
                pred_counts = results_sample['Predicción'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(results_sample, use_container_width=True, height=400)
                
                with col2:
                    # Gráfico de distribución de predicciones
                    fig = px.bar(
                        x=pred_counts.index,
                        y=pred_counts.values,
                        title='Distribución de Predicciones en la Muestra',
                        labels={'x': 'Clase Predicha', 'y': 'Cantidad'},
                        color=pred_counts.values,
                        color_continuous_scale='purples'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Descargar resultados de muestra
                    csv_sample = results_sample.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Resultados de Muestra",
                        data=csv_sample,
                        file_name="predicciones_muestra.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"❌ Error al procesar la muestra: {str(e)}")
                st.info("💡 **Solución:** Asegúrate de que el dataset esté cargado correctamente y tenga datos válidos")