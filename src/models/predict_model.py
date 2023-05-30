import numpy as np
import  pickle
import warnings
import mlflow
warnings.filterwarnings("ignore")

import sys
sys.path.append('/synthese_pmm/src/utils')
from utilities import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Loading data processed
df = pd.read_csv('/data/processed/data.csv')

X = df.drop('PoidsNet', axis=1)
Y = df['PoidsNet']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

mlflow.set_experiment(experiment_name='experimentCI_CD')
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.autolog()

train_features = X_train
train_labels = Y_train

cols_selected = ['Rejet_Dimensions', 'SuperficieCultiveeHa', 'SuperficieTotaleHa',
       'NbrePlateauxTotal', 'Rejet_AutresDefauts', 'QtePlantsRequis',
       'EmplacementChenille', 'RejetDeclassement', 'PesticideAmount',
       'QteSemencesMillegrains', 'duree_obtenue', 'duree_visee',
       'AnneeProduction', 'RendementLbParHa', 'TypePlateaux',
       'Rejet_Matiere_Etrangere', 'PopulationViseeParHa', 'SurplusPourcent',
       'StadeCulture', 'FertilizerAmount', 'NbChenilleObserve',
       'NbPlantsObserves', 'QuantiteMaladie', 'Grosseur', 'EmplacementMaladie',
       'CouleurPlant', 'FournisseurPlant', 'Maladie', 'StadeAlternaria',
       'NbPlantsMaladie']

rl = pickle.load(open('models/ExtraTreesRegressor_model_bon.pkl', 'rb'))


with mlflow.start_run():
    y_pred = rl.predict(X_test[cols_selected])
    r2_score_test = r2_score(Y_test, y_pred)
    # Metrics
    mae = mean_absolute_error(Y_test, y_pred)
    msqe = mean_squared_error(Y_test, y_pred)
    sqr_msqe = np.sqrt(mean_squared_error(Y_test, y_pred))
    # Score
    score = r2_score_test

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("msqe", msqe)
    mlflow.log_metric("sqr_msqe", sqr_msqe)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(rl, "model")

    with open("src/models/metrics.txt", 'w') as outfile:
        outfile.write("LES METRICS \n")
        outfile.write("Mean absolute error : %2.2f\n" % mae)
        outfile.write("Mean square error : %2.2f\n" % msqe)
        outfile.write("Sqr error : %2.2f\n" % sqr_msqe)
        outfile.write("Score : %2.2f\n" % r2_score_test)
        # outfile.write("R2 Score : %2.2f%%\n" % r2_score)

print("LES METRICS")
print("Mean absolute error : %2.2f" % mae)
print("Mean square error : %2.2f" % msqe)
print("Sqr error : %2.2f" % sqr_msqe)
print("Score : %2.2f" % r2_score_test)
