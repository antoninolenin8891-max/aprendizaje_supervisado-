import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Carga datos desde archivo CSV con manejo de errores"""
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"El archivo {self.file_path} no existe")
            
            self.data = pd.read_csv(self.file_path)
            print(f"✅ Datos cargados exitosamente: {self.data.shape[0]} filas, {self.data.shape[1]} columnas")
            return self.data
            
        except Exception as e:
            print(f"❌ Error cargando el archivo: {e}")
            return None

    def get_data_info(self):
        """Retorna información básica del dataset"""
        if self.data is None:
            return "No hay datos cargados"
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': self.data.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        return info

    def display_basic_info(self):
        """Muestra información básica del dataset"""
        if self.data is None:
            print("No hay datos cargados")
            return
        
        print("\n" + "="*50)
        print("📊 INFORMACIÓN DEL DATASET")
        print("="*50)
        print(f"📁 Archivo: {self.file_path}")
        print(f"📏 Dimensiones: {self.data.shape[0]} filas x {self.data.shape[1]} columnas")
        print(f"🏷️ Columnas: {list(self.data.columns)}")
        print(f"🔢 Columnas numéricas: {self.data.select_dtypes(include=['number']).columns.tolist()}")
        print(f"📝 Columnas categóricas: {self.data.select_dtypes(include=['object', 'category']).columns.tolist()}")
        print(f"❓ Valores faltantes: {self.data.isnull().sum().sum()} total")