import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Loading data processed
df = pd.read_csv('data/processed/data.csv')


X = df.drop('PoidsNet', axis=1)
Y = df['PoidsNet']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

train_features = X_train
train_labels = Y_train

# rl = pickle.load(open('models/ExtraTreesRegressor_model.pkl', 'rb'))
rl = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='absolute_error', max_features=1.0,
                         min_samples_split=2, min_samples_leaf=1, n_estimators=100, n_jobs=-1, oob_score=False,
                         random_state=123)
cv = KFold(n_splits=10, shuffle=True, random_state=1)
n_scores = cross_val_score(rl, X, Y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
rl.fit(X_train, Y_train)


plt.figure(figsize=(14,6))
var_imp = pd.Series(rl.feature_importances_, index = train_features.columns).sort_values(ascending=False)
sns.barplot(x=var_imp.index, y=var_imp)
plt.xticks(rotation=90)
plt.ylabel("Score d'importance de la variable")
plt.xlabel("Variables")
plt.title('Importance des variables prédictives')
plt.savefig("src/visualization/importances_variables.png")

#FertilizerAmount
seuil = 0.009642
var_selected = var_imp[var_imp > seuil].index.to_list()
train_features = train_features[var_selected]
X_test = X_test[var_selected]

rl.fit(train_features, train_labels)

y_pred_train = rl.predict(train_features)
y_pred = rl.predict(X_test)
r2_score_test = r2_score(Y_test, y_pred)

print(r2_score_test)
print(train_features.columns)
# print(len(var_imp[var_imp > seuil]))


#Model dump
pickle.dump(rl,open('models/ExtraTreesRegressor_model_bon.pkl','wb'))


