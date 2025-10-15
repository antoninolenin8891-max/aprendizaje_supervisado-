from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_transform(self, data):
        data_filled = self.imputer.fit_transform(data)
        return self.scaler.fit_transform(data_filled)

    def transform(self, data):
        data_filled = self.imputer.transform(data)
        return self.scaler.transform(data_filled)