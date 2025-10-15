"""
Funciones Auxiliares y Utilidades
Herramientas comunes usadas en todo el proyecto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== MANEJO DE DATOS ====================

def load_dataset(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Carga un dataset desde diferentes formatos
    
    Args:
        filepath: Ruta del archivo
        **kwargs: Argumentos adicionales para pandas
    
    Returns:
        DataFrame con los datos
    """
    extension = Path(filepath).suffix.lower()
    
    try:
        if extension == '.csv':
            return pd.read_csv(filepath, **kwargs)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(filepath, **kwargs)
        elif extension == '.json':
            return pd.read_json(filepath, **kwargs)
        elif extension == '.parquet':
            return pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Formato no soportado: {extension}")
    except Exception as e:
        raise Exception(f"Error al cargar el archivo: {str(e)}")

def save_dataset(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Guarda un DataFrame en diferentes formatos"""
    extension = Path(filepath).suffix.lower()
    
    if extension == '.csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif extension in ['.xlsx', '.xls']:
        df.to_excel(filepath, index=False, **kwargs)
    elif extension == '.json':
        df.to_json(filepath, **kwargs)
    elif extension == '.parquet':
        df.to_parquet(filepath, index=False, **kwargs)
    else:
        raise ValueError(f"Formato no soportado: {extension}")

def get_dataset_info(df: pd.DataFrame) -> Dict:
    """
    Obtiene información completa del dataset
    
    Returns:
        Diccionario con estadísticas del dataset
    """
    return {
        'shape': df.shape,
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
        'total_nulls': df.isnull().sum().sum(),
        'null_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicates': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict(),
    }

def print_dataset_info(df: pd.DataFrame) -> None:
    """Imprime información formateada del dataset"""
    info = get_dataset_info(df)
    
    print("=" * 60)
    print("📊 INFORMACIÓN DEL DATASET")
    print("=" * 60)
    print(f"📏 Dimensiones: {info['rows']:,} filas × {info['columns']} columnas")
    print(f"📋 Variables numéricas: {info['numeric_columns']}")
    print(f"📝 Variables categóricas: {info['categorical_columns']}")
    print(f"📅 Variables datetime: {info['datetime_columns']}")
    print(f"❌ Valores nulos: {info['total_nulls']:,} ({info['null_percentage']:.2f}%)")
    print(f"🔁 Duplicados: {info['duplicates']:,}")
    print(f"💾 Memoria: {info['memory_usage_mb']:.2f} MB")
    print("=" * 60)

# ==================== ANÁLISIS EXPLORATORIO ====================

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza valores faltantes por columna"""
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data_Type': df.dtypes.values
    })
    
    missing = missing[missing['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    return missing.reset_index(drop=True)

def detect_outliers_iqr(df: pd.DataFrame, column: str, 
                       threshold: float = 1.5) -> Dict:
    """
    Detecta outliers usando el método IQR
    
    Args:
        df: DataFrame
        column: Columna a analizar
        threshold: Multiplicador IQR (default 1.5)
    
    Returns:
        Diccionario con información de outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_count': len(outliers),
        'outliers_percentage': (len(outliers) / len(df)) * 100,
        'outliers_indices': outliers.index.tolist(),
    }

def get_correlation_pairs(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Obtiene pares de variables con alta correlación
    
    Args:
        df: DataFrame
        threshold: Umbral mínimo de correlación (en valor absoluto)
    
    Returns:
        DataFrame con pares correlacionados
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                pairs.append({
                    'Variable_1': corr_matrix.columns[i],
                    'Variable_2': corr_matrix.columns[j],
                    'Correlation': corr_value,
                    'Abs_Correlation': abs(corr_value)
                })
    
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values('Abs_Correlation', ascending=False)
    
    return pairs_df

def analyze_categorical_variables(df: pd.DataFrame) -> Dict:
    """Analiza variables categóricas del dataset"""
    cat_columns = df.select_dtypes(include=['object']).columns
    
    analysis = {}
    for col in cat_columns:
        analysis[col] = {
            'unique_values': df[col].nunique(),
            'most_frequent': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
            'value_counts': df[col].value_counts().to_dict(),
            'null_count': df[col].isnull().sum(),
        }
    
    return analysis

# ==================== PREPROCESAMIENTO ====================

def remove_outliers(df: pd.DataFrame, column: str, 
                   threshold: float = 1.5) -> pd.DataFrame:
    """Elimina outliers de una columna usando IQR"""
    outliers_info = detect_outliers_iqr(df, column, threshold)
    
    clean_df = df[
        (df[column] >= outliers_info['lower_bound']) & 
        (df[column] <= outliers_info['upper_bound'])
    ].copy()
    
    return clean_df

def cap_outliers(df: pd.DataFrame, column: str, 
                threshold: float = 1.5) -> pd.DataFrame:
    """Limita outliers a los valores límite (capping)"""
    outliers_info = detect_outliers_iqr(df, column, threshold)
    
    df_capped = df.copy()
    df_capped[column] = df_capped[column].clip(
        lower=outliers_info['lower_bound'],
        upper=outliers_info['upper_bound']
    )
    
    return df_capped

def encode_categorical_target(y: pd.Series) -> Tuple[np.ndarray, Dict]:
    """
    Codifica variable objetivo categórica
    
    Returns:
        Tupla con array codificado y diccionario de mapeo
    """
    unique_values = y.unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    reverse_mapping = {idx: val for val, idx in mapping.items()}
    
    y_encoded = y.map(mapping).values
    
    return y_encoded, {'mapping': mapping, 'reverse': reverse_mapping}

def split_features_target(df: pd.DataFrame, 
                         target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa características y variable objetivo"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# ==================== VISUALIZACIÓN ====================

def plot_missing_values(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
    """Visualiza valores faltantes por columna"""
    missing = analyze_missing_values(df)
    
    if missing.empty:
        print("✅ No hay valores faltantes en el dataset")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#ef4444' if x > 50 else '#f59e0b' if x > 20 else '#10b981' 
              for x in missing['Missing_Percentage']]
    
    ax.barh(missing['Column'], missing['Missing_Percentage'], color=colors)
    ax.set_xlabel('Porcentaje de Valores Faltantes (%)')
    ax.set_title('Valores Faltantes por Columna', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Agregar valores en las barras
    for i, (col, pct) in enumerate(zip(missing['Column'], missing['Missing_Percentage'])):
        ax.text(pct + 1, i, f'{pct:.1f}%', va='center')
    
    plt.tight_layout()
    return fig

def plot_distribution_comparison(df: pd.DataFrame, columns: List[str], 
                                figsize: Tuple[int, int] = (15, 5)):
    """Compara distribuciones de múltiples columnas"""
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    if n_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, columns):
        df[col].hist(bins=30, ax=ax, color='#667eea', edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribución: {col}', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, 
                            figsize: Tuple[int, int] = (12, 10),
                            annotate: bool = True):
    """Visualiza matriz de correlación"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("⚠️ Se necesitan al menos 2 variables numéricas")
        return
    
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=annotate,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlación'},
        ax=ax
    )
    
    ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_plotly_histogram(df: pd.DataFrame, column: str, 
                           title: str = None) -> go.Figure:
    """Crea histograma interactivo con Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=50,
        name=column,
        marker_color='#667eea',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=title or f'Distribución de {column}',
        xaxis_title=column,
        yaxis_title='Frecuencia',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_plotly_boxplot(df: pd.DataFrame, column: str, 
                         title: str = None) -> go.Figure:
    """Crea boxplot interactivo con Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=df[column],
        name=column,
        marker_color='#764ba2',
        boxmean='sd'
    ))
    
    fig.update_layout(
        title=title or f'Box Plot de {column}',
        yaxis_title=column,
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

# ==================== MÉTRICAS Y EVALUACIÓN ====================

def calculate_regression_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas para problemas de regresión"""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error
    )
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (evitar división por cero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    else:
        mape = np.nan
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
    }

def calculate_classification_metrics(y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     y_pred_proba: np.ndarray = None) -> Dict:
    """Calcula métricas para problemas de clasificación"""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    
    # ROC-AUC solo si hay probabilidades
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='weighted'
                )
        except:
            metrics['roc_auc'] = None
    
    return metrics

def print_metrics(metrics: Dict, title: str = "Métricas del Modelo"):
    """Imprime métricas formateadas"""
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)
    
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"\n{key}:")
            print(value)
        elif isinstance(value, (int, float, np.number)):
            if np.isnan(value):
                print(f"{key}: N/A")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 60 + "\n")

# ==================== GUARDAR Y CARGAR MODELOS ====================

def save_model(model: Any, filepath: str, metadata: Dict = None) -> None:
    """
    Guarda un modelo entrenado con metadata
    
    Args:
        model: Modelo a guardar
        filepath: Ruta donde guardar
        metadata: Información adicional del modelo
    """
    model_data = {
        'model': model,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat(),
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Modelo guardado en: {filepath}")

def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Carga un modelo guardado
    
    Returns:
        Tupla con el modelo y su metadata
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data.get('metadata', {})

def save_results(results: Dict, filepath: str) -> None:
    """Guarda resultados en formato JSON"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"✅ Resultados guardados en: {filepath}")

def load_results(filepath: str) -> Dict:
    """Carga resultados desde JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

# ==================== UTILIDADES GENERALES ====================

def create_timestamp() -> str:
    """Crea un timestamp legible"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_number(num: float, decimals: int = 2) -> str:
    """Formatea un número con separadores de miles"""
    return f"{num:,.{decimals}f}"

def calculate_percentage(part: float, total: float) -> float:
    """Calcula porcentaje de forma segura"""
    if total == 0:
        return 0.0
    return (part / total) * 100

def get_memory_usage(df: pd.DataFrame) -> Dict:
    """Obtiene uso de memoria detallado del DataFrame"""
    mem_usage = df.memory_usage(deep=True)
    
    return {
        'total_mb': mem_usage.sum() / 1024**2,
        'by_column': {
            col: f"{mem_usage[col] / 1024**2:.2f} MB" 
            for col in df.columns
        }
    }

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce el uso de memoria optimizando tipos de datos"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f"Memoria inicial: {start_mem:.2f} MB")
        print(f"Memoria final: {end_mem:.2f} MB")
        print(f"Reducción: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df

def timer(func):
    """Decorador para medir tiempo de ejecución"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} ejecutado en {end - start:.2f} segundos")
        return result
    
    return wrapper

# ==================== GENERACIÓN DE REPORTES ====================

def generate_summary_report(df: pd.DataFrame, output_path: str = None) -> Dict:
    """Genera un reporte completo del dataset"""
    report = {
        'general_info': get_dataset_info(df),
        'missing_values': analyze_missing_values(df).to_dict('records'),
        'categorical_analysis': analyze_categorical_variables(df),
        'numeric_summary': df.describe().to_dict(),
        'generated_at': datetime.now().isoformat(),
    }
    
    if output_path:
        save_results(report, output_path)
    
    return report

# ==================== VALIDACIÓN DE DATOS ====================

def validate_dataset(df: pd.DataFrame, 
                    min_rows: int = 100,
                    max_null_percentage: float = 50.0) -> Tuple[bool, List[str]]:
    """
    Valida que un dataset cumpla con requisitos mínimos
    
    Returns:
        Tupla con (es_válido, lista_de_errores)
    """
    errors = []
    
    # Validar número mínimo de filas
    if len(df) < min_rows:
        errors.append(f"El dataset debe tener al menos {min_rows} filas. Actual: {len(df)}")
    
    # Validar porcentaje de nulos
    null_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if null_pct > max_null_percentage:
        errors.append(f"Demasiados valores nulos: {null_pct:.1f}% (máximo {max_null_percentage}%)")
    
    # Validar que haya al menos una columna numérica
    if df.select_dtypes(include=[np.number]).shape[1] == 0:
        errors.append("El dataset debe tener al menos una columna numérica")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors

# ==================== EXPORTAR FUNCIONES ====================

__all__ = [
    'load_dataset',
    'save_dataset',
    'get_dataset_info',
    'print_dataset_info',
    'analyze_missing_values',
    'detect_outliers_iqr',
    'get_correlation_pairs',
    'analyze_categorical_variables',
    'remove_outliers',
    'cap_outliers',
    'encode_categorical_target',
    'split_features_target',
    'plot_missing_values',
    'plot_distribution_comparison',
    'plot_correlation_heatmap',
    'create_plotly_histogram',
    'create_plotly_boxplot',
    'calculate_regression_metrics',
    'calculate_classification_metrics',
    'print_metrics',
    'save_model',
    'load_model',
    'save_results',
    'load_results',
    'create_timestamp',
    'format_number',
    'calculate_percentage',
    'get_memory_usage',
    'reduce_memory_usage',
    'timer',
    'generate_summary_report',
    'validate_dataset',
]