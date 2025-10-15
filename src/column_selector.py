class ColumnSelector:
    def __init__(self, data):
        self.data = data
        self.feature_columns = []
        self.target_column = None

    def display_columns(self):
        """Muestra todas las columnas disponibles"""
        print("\n" + "="*50)
        print("🏷️ COLUMNAS DISPONIBLES")
        print("="*50)
        
        for i, column in enumerate(self.data.columns, 1):
            dtype = self.data[column].dtype
            missing = self.data[column].isnull().sum()
            print(f"{i:2d}. {column} ({dtype}) - Missing: {missing}")

    def select_target_column(self):
        """Permite al usuario seleccionar la columna target"""
        self.display_columns()
        
        while True:
            try:
                choice = int(input("\n🔍 Selecciona el NÚMERO de la columna TARGET (variable a predecir): "))
                if 1 <= choice <= len(self.data.columns):
                    self.target_column = self.data.columns[choice - 1]
                    
                    # Verificar si es adecuada para target
                    if self.data[self.target_column].dtype == 'object' and self.data[self.target_column].nunique() > 10:
                        print("⚠️  Advertencia: Columna target tiene muchos valores únicos para ser categórica")
                    
                    print(f"✅ Target seleccionado: {self.target_column}")
                    break
                else:
                    print("❌ Número fuera de rango. Intenta nuevamente.")
            except ValueError:
                print("❌ Entrada inválida. Ingresa un número.")

    def select_feature_columns(self):
        """Permite al usuario seleccionar las columnas features"""
        available_columns = [col for col in self.data.columns if col != self.target_column]
        
        print("\n" + "="*50)
        print("🎯 SELECCIÓN DE CARACTERÍSTICAS (FEATURES)")
        print("="*50)
        
        for i, column in enumerate(available_columns, 1):
            dtype = self.data[column].dtype
            print(f"{i:2d}. {column} ({dtype})")

        while True:
            try:
                choices = input("\n🔍 Ingresa los NÚMEROS de las características a usar (separados por coma, 'all' para todas): ")
                
                if choices.lower() == 'all':
                    self.feature_columns = available_columns
                    break
                else:
                    selected_indices = [int(x.strip()) - 1 for x in choices.split(',')]
                    self.feature_columns = [available_columns[i] for i in selected_indices]
                    
                    # Validar selección
                    if all(0 <= i < len(available_columns) for i in selected_indices):
                        break
                    else:
                        print("❌ Algunos números están fuera de rango.")
            except ValueError:
                print("❌ Entrada inválida. Usa números separados por coma.")

        print(f"✅ Características seleccionadas: {self.feature_columns}")

    def get_selected_data(self):
        """Retorna X e y según la selección"""
        if not self.target_column or not self.feature_columns:
            raise ValueError("No se han seleccionado target y features")
        
        X = self.data[self.feature_columns]
        y = self.data[self.target_column]
        
        return X, y