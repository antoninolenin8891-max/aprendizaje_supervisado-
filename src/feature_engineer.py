class FeatureEngineer:
    def __init__(self):
        pass

    def add_polynomial_features(self, data, degree=2):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree)
        return poly.fit_transform(data)