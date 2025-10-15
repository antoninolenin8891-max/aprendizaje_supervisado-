"""
Archivo de Configuración del Proyecto ML
Centraliza todas las configuraciones, rutas y parámetros
"""

import os
from pathlib import Path

# ==================== RUTAS DEL PROYECTO ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
LOGS_DIR = BASE_DIR / 'logs'

# Crear directorios si no existen
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== CONFIGURACIÓN DE DATOS ====================
DATA_CONFIG = {
    'train_size': 0.7,          # 70% para entrenamiento
    'validation_size': 0.15,    # 15% para validación
    'test_size': 0.15,          # 15% para prueba
    'random_state': 42,         # Semilla para reproducibilidad
    'shuffle': True,            # Mezclar datos
}

# ==================== PREPROCESAMIENTO ====================
PREPROCESSING_CONFIG = {
    'missing_values': {
        'numeric_strategy': 'mean',     # 'mean', 'median', 'mode', 'drop'
        'categorical_strategy': 'mode', # 'mode', 'constant', 'drop'
        'threshold': 0.5,               # % máximo de nulos permitidos
    },
    'encoding': {
        'method': 'onehot',             # 'onehot', 'label', 'target'
        'drop_first': True,             # Evitar multicolinealidad
    },
    'scaling': {
        'method': 'standard',           # 'standard', 'minmax', 'robust'
        'apply_to': 'numeric_only',     # Solo a columnas numéricas
    },
    'outliers': {
        'method': 'iqr',                # 'iqr', 'zscore', 'isolation'
        'threshold': 1.5,               # Multiplicador IQR
        'action': 'cap',                # 'cap', 'remove', 'none'
    }
}

# ==================== MODELOS DISPONIBLES ====================
MODELS_CONFIG = {
    'linear_regression': {
        'name': 'Regresión Lineal',
        'type': 'regression',
        'params': {
            'fit_intercept': True,
        },
        'tuning_params': {}
    },
    'ridge': {
        'name': 'Ridge Regression',
        'type': 'regression',
        'params': {
            'alpha': 1.0,
            'random_state': 42,
        },
        'tuning_params': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    'lasso': {
        'name': 'Lasso Regression',
        'type': 'regression',
        'params': {
            'alpha': 1.0,
            'random_state': 42,
        },
        'tuning_params': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    'logistic_regression': {
        'name': 'Regresión Logística',
        'type': 'classification',
        'params': {
            'random_state': 42,
            'max_iter': 1000,
        },
        'tuning_params': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'decision_tree': {
        'name': 'Árbol de Decisión (CART)',
        'type': 'both',
        'params': {
            'random_state': 42,
            'max_depth': None,
        },
        'tuning_params': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'knn_regression': {
        'name': 'KNN Regresión',
        'type': 'regression',
        'params': {
            'n_neighbors': 5,
            'weights': 'uniform',
        },
        'tuning_params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    'knn_classification': {
        'name': 'KNN Clasificación',
        'type': 'classification',
        'params': {
            'n_neighbors': 5,
            'weights': 'uniform',
        },
        'tuning_params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    'neural_network': {
        'name': 'Red Neuronal',
        'type': 'both',
        'params': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'random_state': 42,
            'max_iter': 1000,
        },
        'tuning_params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        }
    }
}

# ==================== VALIDACIÓN ====================
VALIDATION_CONFIG = {
    'cross_validation': {
        'enabled': True,
        'cv_folds': 5,              # K-fold cross validation
        'scoring': 'auto',          # 'auto', 'r2', 'accuracy', etc.
    },
    'hyperparameter_tuning': {
        'enabled': True,
        'method': 'grid',           # 'grid', 'random', 'bayesian'
        'cv_folds': 3,
        'n_iter': 10,               # Para random search
        'n_jobs': -1,               # Usar todos los cores
    }
}

# ==================== MÉTRICAS ====================
METRICS_CONFIG = {
    'regression': [
        'mae',      # Mean Absolute Error
        'mse',      # Mean Squared Error
        'rmse',     # Root Mean Squared Error
        'r2',       # R-squared
        'mape',     # Mean Absolute Percentage Error
    ],
    'classification': [
        'accuracy',     # Exactitud
        'precision',    # Precisión
        'recall',       # Sensibilidad
        'f1',          # F1-Score
        'roc_auc',     # ROC-AUC
        'confusion_matrix',
    ]
}

# ==================== VISUALIZACIÓN ====================
VISUALIZATION_CONFIG = {
    'style': 'seaborn',
    'figsize': (12, 6),
    'dpi': 100,
    'color_palette': 'husl',
    'plotly_theme': 'plotly_white',
    'colors': {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#0ea5e9',
    }
}

# ==================== DASHBOARD STREAMLIT ====================
DASHBOARD_CONFIG = {
    'page_title': 'ML Supervisado - Dashboard',
    'page_icon': '🤖',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'theme': {
        'primaryColor': '#667eea',
        'backgroundColor': '#f8fafc',
        'secondaryBackgroundColor': '#ffffff',
        'textColor': '#1e293b',
    }
}

# ==================== LOGGING ====================
LOGGING_CONFIG = {
    'level': 'INFO',            # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'ml_project.log',
    'console_output': True,
}

# ==================== LÍMITES Y RESTRICCIONES ====================
LIMITS = {
    'max_file_size_mb': 200,        # Tamaño máximo de archivo
    'max_rows': 1000000,            # Máximo de filas a procesar
    'max_features': 100,            # Máximo de características
    'min_samples': 100,             # Mínimo de muestras requeridas
    'timeout_seconds': 300,         # Timeout para operaciones largas
}

# ==================== MENSAJES ====================
MESSAGES = {
    'es': {
        'data_loaded': '✅ Datos cargados exitosamente',
        'data_error': '❌ Error al cargar los datos',
        'preprocessing_complete': '✅ Preprocesamiento completado',
        'model_trained': '✅ Modelo entrenado exitosamente',
        'prediction_complete': '✅ Predicción completada',
        'no_data': '⚠️ No hay datos cargados',
        'no_model': '⚠️ No hay modelo entrenado',
    }
}

# ==================== INFORMACIÓN DEL PROYECTO ====================
PROJECT_INFO = {
    'name': 'Dashboard ML Supervisado',
    'version': '1.0.0',
    'author': 'Tu Nombre',
    'description': 'Sistema completo de análisis y predicción con Machine Learning',
    'university': 'Tu Universidad',
    'course': 'Machine Learning Supervisado',
    'date': 'Octubre 2025',
    'github': 'https://github.com/tu-usuario/tu-proyecto',
}

# ==================== FUNCIONES AUXILIARES ====================

def get_model_config(model_name):
    """Obtiene la configuración de un modelo específico"""
    return MODELS_CONFIG.get(model_name, None)

def get_metric_config(task_type):
    """Obtiene las métricas para un tipo de tarea"""
    return METRICS_CONFIG.get(task_type, [])

def validate_config():
    """Valida que la configuración sea correcta"""
    # Verificar que train_size + validation_size + test_size = 1.0
    total = (DATA_CONFIG['train_size'] + 
             DATA_CONFIG['validation_size'] + 
             DATA_CONFIG['test_size'])
    
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"La suma de train/validation/test debe ser 1.0, actualmente es {total}")
    
    return True

# Validar configuración al importar
validate_config()

# ==================== EXPORTAR CONFIGURACIONES ====================
__all__ = [
    'DATA_DIR',
    'MODELS_DIR',
    'REPORTS_DIR',
    'LOGS_DIR',
    'DATA_CONFIG',
    'PREPROCESSING_CONFIG',
    'MODELS_CONFIG',
    'VALIDATION_CONFIG',
    'METRICS_CONFIG',
    'VISUALIZATION_CONFIG',
    'DASHBOARD_CONFIG',
    'LOGGING_CONFIG',
    'LIMITS',
    'MESSAGES',
    'PROJECT_INFO',
    'get_model_config',
    'get_metric_config',
]