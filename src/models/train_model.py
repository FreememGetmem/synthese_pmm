import numpy as np
import pandas as pd
import  pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.append('/synthese_pmm/src/utils')
from utilities import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import log_loss
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import RFE
from numpy import mean
from numpy import std
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Loading data processed
df = pd.read_csv('data/processed/data.csv')

X = df.drop('PoidsNet', axis=1)
Y = df['PoidsNet']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

#Standardisation
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
#Regression
rl = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse', max_features=1.0,
                         min_samples_split=2, min_samples_leaf=1, n_estimators=100, n_jobs=-1, oob_score=False,
                         random_state=123)
cv = KFold(n_splits=10, shuffle=True, random_state=1)
n_scores = cross_val_score(rl, X, Y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')

rl.fit(X_train, Y_train)


y_pred_train = rl.predict(X_train)
y_pred = rl.predict(X_test)

y_pred_proba = rl.predict(X_test)

r2_score_train = r2_score(Y_train, y_pred_train)
r2_score_test = r2_score(Y_test, y_pred)
print(r2_score_test)

#Model dump
pickle.dump(rl,open('models/ExtraTreesRegressor_model.pkl','wb'))

# rl = pickle.load(open('models/ExtraTreesRegressor_model.pkl','rb'))
## Utilisation de RFE pour la selection de feature
rfe_model = RFE(estimator=rl, n_features_to_select=17)
rfe_model.fit(X_train, Y_train)
mask = rfe_model.support_
reduced_X = X.loc[:,mask]

r2_score_train = rfe_model.score(X_train, Y_train)
r2_score_test = rfe_model.score(X_test, Y_test)
print("R2 SCORE: TRAIN=%.4f TEST=%.4f" % (r2_score_train,r2_score_test))
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
print('MAE: %.3f ' % mean_absolute_error(Y_test, y_pred))
