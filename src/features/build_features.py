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

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
from fastparquet import ParquetFile

# Loading data processed
df = pd.read_csv('data/processed/data.csv')

X = df.drop('PoidsNet', axis=1)
Y = df['PoidsNet']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

train_features = X_train
train_labels = Y_train

# rl = pickle.load(open('models/ExtraTreesRegressor_model.pkl', 'rb'))
rl = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, max_features=1.0,
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

###### Début changement #######
predictors_df = train_features
target_df = df['PoidsNet']

timestamps = pd.date_range(end = pd.Timestamp.now(), periods= len(df), freq='D').to_frame(name = 'event_timestamp', index = False)

predictors_df = pd.concat(objs=[predictors_df, timestamps], axis=1)
target_df = pd.concat(objs= [target_df, timestamps], axis=1)

dataLen = len(df)
idsList = list(range(dataLen))
pmm_ids = pd.DataFrame(data=idsList, columns= ['pmm_id'])

predictors_df = pd.concat(objs=[predictors_df, pmm_ids], axis=1)
target_df = pd.concat(objs= [target_df, pmm_ids], axis=1)

# predictors_df.to_pickle('feature_repo/feature_repo/data/predictors.pkl')
# target_df.to_pickle('feature_repo/feature_repo/data/target.pkl')

# pickle.dump('feature_repo/feature_repo/data/predictors.pkl')
# pickle.dump('feature_repo/feature_repo/data/target.pkl')

predictors_df.to_parquet(path='feature_repo/feature_repo/data/predictors.parquet')
target_df.to_parquet(path='feature_repo/feature_repo/data/target.parquet')

# store = FeatureStore(repo_path='feature_repo/feature_repo/')
#
# entity_df = pd.read_parquet('feature_repo/feature_repo/data/target.parquet')
# training_data = store.get_historical_features(
#    entity_df=entity_df,
#    features = [ "predictors_df_feature_view:RendementLbParHa",
#                 "predictors_df_feature_view:SuperficieCultiveeHa",
#                 "predictors_df_feature_view:RejetDeclassement",
#                 "predictors_df_feature_view:NbrePlateauxTotal",
#                 "predictors_df_feature_view:Rejet_Dimensions",
#                 "predictors_df_feature_view:SuperficieTotaleHa",
#                 "predictors_df_feature_view:QtePlantsRequis",
#                 "predictors_df_feature_view:AnneeProduction",
#                 "predictors_df_feature_view:QteSemencesMillegrains",
#                 "predictors_df_feature_view:EmplacementChenille",
#                 "predictors_df_feature_view:Rejet_AutresDefauts",
#                 "predictors_df_feature_view:PesticideAmount",
#                 "predictors_df_feature_view:TypePlateaux",
#                 "predictors_df_feature_view:duree_visee",
#                 "predictors_df_feature_view:PopulationViseeParHa",
#                 "predictors_df_feature_view:duree_obtenue",
#                 "predictors_df_feature_view:Rejet_Matiere_Etrangere",
#                 "predictors_df_feature_view:SurplusPourcent",
#                 "predictors_df_feature_view:StadeCulture",
#                 "predictors_df_feature_view:FertilizerAmount",
#                 "predictors_df_feature_view:NbPlantsObserves",
#                 "predictors_df_feature_view:FournisseurPlant",
#                 "predictors_df_feature_view:NbChenilleObserve",
#                 "predictors_df_feature_view:Grosseur"
#                 ]
# )
#
# dataset = store.create_saved_dataset(
#     from_ = training_data,
#     name = "diabetes_dataset",
#     storage = SavedDatasetFileStorage('feature_repo/feature_repo/data/diabetes_dataset.parquet')
# )

# print(training_data.to_df())

###### Fin Changement #########


rl.fit(train_features, train_labels)

y_pred_train = rl.predict(train_features)
y_pred = rl.predict(X_test)
r2_score_test = r2_score(Y_test, y_pred)

print(r2_score_test)
print(train_features.columns)
# print(len(var_imp[var_imp > seuil]))


#Model dump
pickle.dump(rl,open('models/ExtraTreesRegressor_model_bon.pkl','wb'))

rl2 = pickle.load(open('models/ExtraTreesRegressor_model_bon.pkl', 'rb'))

val = rl2.predict(train_features)
pmm_data = train_features
pmm_data['target'] = df['PoidsNet']
# pmm_data.rename(columns={'PoidsNet': 'target'}, inplace=True)

pmm_data['prediction'] = val

reference = pmm_data.sample(n=280, replace=False)
current = pmm_data.sample(n=280, replace=False)

report = Report(metrics=[
    DataDriftPreset(),
])

report.run(reference_data=reference, current_data=current)

report.save_html('src/visualization/report.html')

tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
])

tests.run(reference_data=reference, current_data=current)
tests.save_html('src/visualization//tests.html')