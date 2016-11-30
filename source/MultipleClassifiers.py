#Reference http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from EstimatorSelectionHelper import EstimatorSelectionHelper

models1 = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params1 = {
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ]
}

df_path = 'data/train.csv'
sub_df_path = 'data/test.csv'

df = pd.read_csv(df_path)

target = 'type'
predictors = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color']

le = preprocessing.LabelEncoder()
le.fit(df['color'])
df['color'] = le.transform(df['color'])

X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=0)

helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_train, y_train, n_jobs=-1, refit=True)

helper1.score_summary(X_test, y_test)

