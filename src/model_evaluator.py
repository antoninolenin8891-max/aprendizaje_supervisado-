from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelEvaluator:
    def __init__(self, y_true, y_pred, problem_type="regression"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.problem_type = problem_type

    def calculate_metrics(self):
        """Calcula métricas según el tipo de problema"""
        if self.problem_type == "regression":
            return self._regression_metrics()
        else:
            return self._classification_metrics()

    def _regression_metrics(self):
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        
        return {
            "MSE": mse,
            "RMSE": rmse, 
            "MAE": mae,
            "R2 Score": r2
        }

    def _classification_metrics(self):
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

    def plot_predictions(self, y_true, y_pred):
        """Grafica según el tipo de problema"""
        if self.problem_type == "regression":
            self._plot_regression_results(y_true, y_pred)
        else:
            self._plot_classification_results(y_true, y_pred)

    def _plot_regression_results(self, y_true, y_pred):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs Reales - Regresión')
        
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuales')
        plt.title('Análisis de Residuales')
        
        plt.tight_layout()
        plt.show()

    def _plot_classification_results(self, y_true, y_pred):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicciones')
        plt.ylabel('Reales')
        plt.title('Matriz de Confusión')
        
        plt.subplot(1, 2, 2)
        metrics = self.calculate_metrics()
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())
        
        plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('Métricas de Clasificación')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()