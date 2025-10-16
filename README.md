🎓 Dashboard Interactivo - Aprendizaje Supervisado
📋 Descripción del Proyecto
Dashboard interactivo desarrollado en Python que integra y documenta todos los algoritmos de aprendizaje supervisado abordados durante la Unidad I. Este proyecto demuestra la implementación práctica de modelos de machine learning utilizando Programación Orientada a Objetos y una interfaz amigable construida con Streamlit.

🚀 Características Principales
🤖 Algoritmos Implementados
📈 Regresión Lineal - Predicción de valores continuos

🎯 Regresión Ridge y Lasso - Regularización para mejorar generalización

📊 Regresión Logística - Clasificación binaria y multiclase

👥 K-Nearest Neighbors (KNN) - Clasificación y regresión basada en instancias

🌳 Árboles de Decisión - Modelos interpretables para clasificación y regresión

🧠 Redes Neuronales - Perceptrón multicapa para problemas complejos

🏡 Proyecto Principal: Predicción de Precios de Casas
Dataset: King County (más de 21,000 propiedades)

Características: 21 variables predictoras (numéricas y categóricas)

Objetivo: Predecir el precio de venta de propiedades

Pipeline completo: Desde carga de datos hasta deployment

🛠️ Tecnologías Utilizadas
Backend
Python 3.8+

scikit-learn - Algoritmos de machine learning

pandas - Manipulación de datos

numpy - Cálculos numéricos

matplotlib/seaborn - Visualizaciones

Frontend
Streamlit - Dashboard interactivo

Plotly - Gráficos interactivos

Arquitectura
Programación Orientada a Objetos (POO)

Patrón MVC (Modelo-Vista-Controlador)

Pipeline modular de machine learning

📁 Estructura del Proyecto
text
ml-dashboard/
├── main_dashboard.py          # Aplicación principal Streamlit
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Este archivo
├── data/
│   └── king_county.csv       # Dataset principal
└── scripts/
    ├── __init__.py
    ├── data_loader.py        # Clase DataLoader
    ├── data_preprocessor.py  # Clase DataPreprocessor
    ├── feature_engineer.py   # Clase FeatureEngineer
    ├── model_trainer.py      # Clase ModelTrainer
    ├── model_evaluator.py    # Clase ModelEvaluator
    └── predictor.py          # Clase Predictor
🏗️ Arquitectura POO
Clases Principales
DataLoader
Responsabilidad: Carga y gestión de datasets

Métodos principales: load_data(), split_data()

DataPreprocessor
Responsabilidad: Limpieza y transformación de datos

Métodos: handle_missing_values(), encode_categorical(), scale_features()

FeatureEngineer
Responsabilidad: Creación y selección de características

Métodos: create_new_features(), select_features()

ModelTrainer
Responsabilidad: Entrenamiento y ajuste de modelos

Métodos: train(), cross_validate(), hyperparameter_tuning()

ModelEvaluator
Responsabilidad: Evaluación y métricas de modelos

Métodos: evaluate(), plot_results(), confusion_matrix()

Predictor
Responsabilidad: Realizar predicciones en nuevos datos

Métodos: predict(), predict_proba()

📊 Métricas de Evaluación
Para Regresión
MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² (Coefficient of Determination)

MAPE (Mean Absolute Percentage Error)

Para Clasificación
Accuracy (Exactitud)

Precision (Precisión)

Recall (Sensibilidad)

F1-Score (Media armónica)

ROC-AUC (Curva ROC)

Matriz de Confusión

🚀 Instalación y Uso
1. Clonar el repositorio
bash
git clone https://github.com/tu-usuario/ml-dashboard.git
cd ml-dashboard
2. Crear entorno virtual (recomendado)
bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
3. Instalar dependencias
bash
pip install -r requirements.txt
4. Ejecutar la aplicación
bash
streamlit run main_dashboard.py
5. Acceder al dashboard
Abrir http://localhost:8501 en el navegador

📝 Requisitos del Sistema
Python 3.8 o superior

4GB RAM mínimo

500MB espacio en disco

Navegador web moderno

🧪 Casos de Uso
🔧 Para Estudiantes
Aprender algoritmos de ML de manera interactiva

Probar diferentes configuraciones de hiperparámetros

Visualizar el impacto del preprocesamiento

🔬 Para Desarrolladores
Base de código modular y extensible

Implementación de mejores prácticas POO

Ejemplos de pipeline completo de ML

📚 Para Educadores
Material didáctico interactivo

Ejemplos prácticos con datos reales

Visualizaciones explicativas

🎯 Contextos de Aplicación por Algoritmo
Regresión Lineal
Aplicación: Predicción de precios, ventas, demanda

Cuándo usar: Relaciones lineales, interpretabilidad importante

Ventajas: Simple, interpretable, rápido

Regresión Logística
Aplicación: Clasificación binaria (spam, fraude, diagnóstico)

Cuándo usar: Probabilidades de clase, decisiones binarias

Ventajas: Probabilístico, interpretable

K-Nearest Neighbors
Aplicación: Reconocimiento de patrones, sistemas de recomendación

Cuándo usar: Datos con estructura local, pocas características

Ventajas: No paramétrico, simple de implementar

Árboles de Decisión
Aplicación: Clasificación médica, análisis de riesgo

Cuándo usar: Interpretabilidad crucial, datos heterogéneos

Ventajas: Muy interpretable, maneja mixed data types

Redes Neuronales
Aplicación: Imágenes, texto, series temporales complejas

Cuándo usar: Patrones no lineales, grandes volúmenes de datos

Ventajas: Alta capacidad de modelado, features automáticas

🔍 Validación de Modelos
Técnicas Implementadas
Train/Validation/Test Split - División estratificada

Cross-Validation - Validación cruzada k-fold

Hyperparameter Tuning - Búsqueda en grid y random

Learning Curves - Detección de overfitting/underfitting

Prevención de Overfitting
Regularización (L1/L2)

Early stopping

Validación cruzada

Pruning (árboles)

📈 Resultados Destacados
Proyecto Principal - Predicción de Precios
R²: 0.85+ en conjunto de test

RMSE: Menos del 15% del precio promedio

Características más importantes: Metros cuadrados, ubicación, antigüedad

🤝 Contribución
Estructura de Código
Seguir principios SOLID

Documentar con docstrings

Mantener coherencia en naming

Incluir tests unitarios

Flujo de Trabajo
Fork del proyecto

Crear rama feature (git checkout -b feature/AmazingFeature)

Commit cambios (git commit -m 'Add AmazingFeature')

Push a la rama (git push origin feature/AmazingFeature)

Abrir Pull Request