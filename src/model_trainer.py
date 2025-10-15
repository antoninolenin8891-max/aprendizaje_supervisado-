from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

class ModelTrainer:
    def __init__(self, problem_type="regression", model_type="linear"):
        self.problem_type = problem_type
        self.model_type = model_type
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Inicializa el modelo según el tipo de problema"""
        if self.problem_type == "regression":
            if self.model_type == "linear":
                self.model = LinearRegression()
            elif self.model_type == "random_forest":
                self.model = RandomForestRegressor(random_state=42)
            elif self.model_type == "decision_tree":
                self.model = DecisionTreeRegressor(random_state=42)
            elif self.model_type == "knn":
                self.model = KNeighborsRegressor()
        else:  # classification
            if self.model_type == "linear":
                self.model = LogisticRegression(random_state=42)
            elif self.model_type == "random_forest":
                self.model = RandomForestClassifier(random_state=42)
            elif self.model_type == "decision_tree":
                self.model = DecisionTreeClassifier(random_state=42)
            elif self.model_type == "knn":
                self.model = KNeighborsClassifier()

    def train(self, X_train, y_train):
        """Entrena el modelo"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Realiza predicciones"""
        return self.model.predict(X_test)

    def set_model_type(self, model_type):
        """Cambia el tipo de modelo"""
        self.model_type = model_type
        self._initialize_model()