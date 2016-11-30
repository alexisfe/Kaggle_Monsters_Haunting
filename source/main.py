import pandas as pd
import numpy as np
import seaborn as sns
from plot_multiple_hist import plot_multiple_hist
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

df_path = 'input/train.csv'
sub_df_path = 'input/test.csv'

df = pd.read_csv(df_path)

target = 'type'
predictors = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color']

#plot_multiple_hist(df, df.columns, 3, 3)

le = LabelEncoder()
le.fit(df['color'])
df['color'] = le.transform(df['color'])

#sns.set_style('whitegrid')
#sns.pairplot(df, hue="type")

X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=1)

ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

svc_param = [{'kernel': ['rbf'], 'gamma': np.arange(1e-5, 5e-4, 1e-5)},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 5000]}]

nn_param = [{'alpha': [1e-2, 1e-3, 1e-4]
                , 'momentum': np.arange(0.3, 0.6, 0.05)
                , 'hidden_layer_sizes': [(10,5), (20,10), (40,20), (60,30), (80,40), (100,50)]
            }]

clf = GridSearchCV(estimator=MLPClassifier(activation='relu', solver='sgd', learning_rate='adaptive', max_iter=10000), cv=4, param_grid=nn_param)
#clf = GridSearchCV(estimator=SVC(cache_size=4000, C=1607), cv=5, param_grid=svc_param)
#clf = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), warm_start=True, n_jobs=-1, max_iter_predict=1000, random_state=1)
print "Fitting model..."
clf.fit(X_train, y_train)

print "Best estimator found by GridSearch: "
print clf.best_estimator_
print "with a score of: "
print clf.best_score_

print "Predicting on test set..."
y_pred = clf.predict(X_test)

print("Classification report for model %s:\n%s\n"
      % (clf, classification_report(y_test, y_pred)))
print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))

print "Creating kaggle submission file..."
sub_df = pd.read_csv(sub_df_path)

sub_df['color'] = le.transform(sub_df['color'])

predictions = clf.predict(ss.transform(sub_df[predictors]))
submission = pd.DataFrame({"id": sub_df['id'], 'type': predictions})
submission.to_csv("output/monsters_haunting_submission.csv", index=False)