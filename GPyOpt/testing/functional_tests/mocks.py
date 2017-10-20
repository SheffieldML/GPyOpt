import numpy as np

from GPyOpt.models.base import BOModel

class MockModel(BOModel):
    def __init__(self):
        self.params = []
        self.X = []
        self.Y = []

    def f(self, x):
        return np.dot(np.insert(x, 0, 1.0), self.params)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        self.X = X_all
        self.Y = Y_all
        ones = np.ones(X_all.shape[0]).reshape(-1, 1)
        X = np.hstack([ones, X_all])
        self.params = np.linalg.lstsq(X, Y_all.flatten())[0]

    def predict(self, X):
        preds = [self.f(x) for x in X]
        return np.atleast_2d(np.mean(preds)), np.atleast_2d(np.std(preds))

    def predict_withGradients(self, X):
        preds = [self.f(x) for x in X]
        return np.atleast_2d(np.mean(preds)), np.atleast_2d(np.std(preds)), X, X

    def get_fmin(self):
        preds = [self.f(x) for x in self.X]
        return min(preds)

    def get_model_parameters(self):
        return np.atleast_2d(self.params)

    def get_model_parameter_namess(self):
        return np.atleast_2d(['b' + i for i, _ in enumerate(self.params)])

class MockModelVectorValuedPredict(MockModel):

    def predict(self, X):
        preds = [self.f(x) for x in X]
        return np.atleast_2d(np.mean(preds)*np.ones(len(preds))), np.atleast_2d(np.std(preds)*np.ones(len(preds)))

    def predict_withGradients(self, X):
        preds = [self.f(x) for x in X]
        return np.atleast_2d(np.mean(preds)*np.ones(len(preds))), np.atleast_2d(np.std(preds)*np.ones(len(preds))), X, X
