# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import BOModel
import numpy as np


class RFModel(BOModel):
    """
    General class for handling a Ramdom Forest in GPyOpt.

    .. Note:: The model has beed wrapper 'as it is' from  Scikit-learn. Check
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    for further details.
    """

    analytical_gradient_prediction = False

    def __init__(self, bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False):

        self.bootstrap = bootstrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

        self.model = None

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        from sklearn.ensemble import RandomForestRegressor
        self.X = X
        self.Y = Y
        self.model = RandomForestRegressor(bootstrap = self.bootstrap,
                                           criterion = self.criterion,
                                           max_depth = self.max_depth,
                                           max_features = self.max_features,
                                           max_leaf_nodes = self.max_leaf_nodes,
                                           min_samples_leaf = self.min_samples_leaf,
                                           min_samples_split = self.min_samples_split,
                                           min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                                           n_estimators = self.n_estimators,
                                           n_jobs = self.n_jobs,
                                           oob_score = self.oob_score,
                                           random_state = self.random_state,
                                           verbose = self.verbose,
                                           warm_start = self.warm_start)

        #self.model = RandomForestRegressor()
        self.model.fit(X,Y.flatten())


    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        self.X = X_all
        self.Y = Y_all
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.fit(X_all, Y_all.flatten())

    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X.
        """
        X = np.atleast_2d(X)
        m = np.empty(shape=(0,1))
        s = np.empty(shape=(0,1))

        for k in range(X.shape[0]):
            preds = []
            for pred in self.model.estimators_:
                preds.append(pred.predict(X[k,:])[0])
            m = np.vstack((m ,np.array(preds).mean()))
            s = np.vstack((s ,np.array(preds).std()))
        return m, s

    def get_fmin(self):
        return self.model.predict(self.X).min()
