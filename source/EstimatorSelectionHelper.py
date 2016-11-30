import pandas as pd
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(estimator=model, param_grid=params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs

    def score_summary(self, X, y):
        for key in self.keys:
            #print(self.grid_searches[k].cv_results_)
            #print(self.grid_searches[key].cv_results_)

            model = self.grid_searches[key]

            #print "Predicting on test set..."
            y_pred = model.predict(X)

            print(key)
            print("Classification report for classifier %s:\n%s\n"
                  % (model, classification_report(y, y_pred)))
            print("Confusion matrix:\n%s" % confusion_matrix(y, y_pred))