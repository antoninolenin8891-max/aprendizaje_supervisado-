import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_distribution_plots(df, selected_col):
    """Crea gráficas de distribución para una columna"""
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
    
    return fig

def create_correlation_heatmap(df):
    """Crea heatmap de correlaciones"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) <= 1:
        return None
    
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
    
    return fig

def get_top_correlations(df, ascending=True, top_n=10):
    """Obtiene las top correlaciones"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) <= 1:
        return pd.DataFrame()
    
    corr_matrix = numeric_df.corr()
    
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Correlación': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    sorted_df = corr_df.sort_values('Correlación', ascending=ascending).head(top_n)
    sorted_df['Correlación'] = sorted_df['Correlación'].round(3)
    
    return sorted_df

def create_outlier_plot(df, selected_var, lower_bound, upper_bound):
    """Crea gráfica de detección de outliers"""
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
    
    return fig