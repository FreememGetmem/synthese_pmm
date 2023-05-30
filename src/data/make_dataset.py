# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/synthese_pmm/src/utils')
from utilities import *
# import sys
# sys.path.append('/src/utils')
# import

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    datas = input_filepath.replace("\\", '/')
    # datas2 = datas.replace('.','')
    data = pd.read_excel(datas)
    df = data[data['AnneeProduction']>2018].copy()

    features = ['NoEspece', 'NoProducteur', 'NoCultivar', 'EntreRangPo', 'RangPo']
    drop_columns(df, features)

    val_old = ['0.0']
    val_new = ['Aucune']
    replace_chaine(df, 'NbPlantsVersGris', val_old, val_new)
    val_old = ['Autre tâche']
    val_new = ['AT- Autres taches']
    replace_chaine(df, 'Maladie', val_old, val_new)
    val_old = [1, 2, 3, 4, 5, 8, 7, 6, 9, 60, 20, 10, 13, 15, 40, 12, 33, 14, 35, 17, 18, 0]
    val_new = ['FAIBLE (1 plant)', 'MOYEN (2-5 plants)', 'MOYEN (2-5 plants)', 'MOYEN (2-5 plants)',
               'MOYEN (2-5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'AUCUN']
    replace_chaine(df, 'NbPlantsMaladie', val_old, val_new)
    val_old = ['B- Feuille bas', 'Tête', 'P-Partout', 'Feuille tˆte', 'Feuille carenc‚e']
    val_new = ['Feuille bas', 'Feuille tête', 'Partout', 'Feuille tête', 'Feuille carence']
    replace_chaine(df, 'EmplacementMaladie', val_old, val_new)
    val_old = ['Tˆte']
    val_new = ['Tête']
    replace_chaine(df, 'EmplacementChenille', val_old, val_new)
    val_old = ['Tˆte', 'Feuille tˆte']
    val_new = ['Tête', 'Feuille tête']
    replace_chaine(df, 'MaladieEmplacementSecondaire', val_old, val_new)
    val_old = ['Tƒches concentriques', "D‚but d'infection", 'Tƒches sŠches']
    val_new = ['Tâches concentriques', "Début d'infection", 'Tâches sèches']
    replace_chaine(df, 'StadeAlternaria', val_old, val_new)
    val_old = ['LÉGER (1-2 taches)', 'SÉVÈRE (>5 taches)']
    val_new = ['LÉGER (1-2 taches)', 'SÉVÈRE (>5 taches)']
    replace_chaine(df, 'QuantiteMaladieSecondaire', val_old, val_new)
    val_old = ['Vert pƒle']
    val_new = ['Vert pâle']
    replace_chaine(df, 'CouleurPlant', val_old, val_new)
    val_old = ['SVÔRE +++(>5 taches)']
    val_new = ['SÉVÈRE +++(>5 taches)']
    replace_chaine(df, 'QuantiteMaladie', val_old, val_new)
    val_old = ['Autre tƒche']
    val_new = ['Autre tâche']
    replace_chaine(df, 'MaladieSecondaire', val_old, val_new)
    val_old = ['Vert pƒle']
    val_new = ['Vert pâle']
    replace_chaine(df, 'Carence', val_old, val_new)
    val_old = ['Petit  (< 0,5 cm)', 'Moyen (0,5-1 cm)']
    val_new = ['Mini (< 0,5 cm)', 'Petit (0,5-1 cm)']
    replace_chaine(df, 'Grosseur', val_old, val_new)
    val_old = ['2 pouce']
    val_new = [2]
    replace_chaine(df, 'StadeCulture', val_old, val_new)
    val_old = [0, 2, 1, 3, 4, 5, 8, 7, 6, 9, 60, 20, 10, 13, 15, 40, 12, 33, 14, 35, 17,
               27, 24, 22, 16, 11, 47, 37, 57, 25, 19, 27, 18, 'LEVE (> 5 plants)', 0.27, 0.11, 0.01,
               0.03, 0.16, 0.07, 0.02, 0.47, 0.15, 0.05, 0.22, 0.04, 0.09, 0.25, 0.4, 0.57, 0.1, 0.06, 0.17]
    val_new = ['AUCUNE', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'MOYEN (2-5 plants)', 'MOYEN (2-5 plants)',
               'MOYEN (2-5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'FAIBLE (1 plant)',
               'FAIBLE (1 plant)',
               'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)',
               'FAIBLE (1 plant)',
               'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)',
               'FAIBLE (1 plant)'
        , 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)']
    replace_chaine(df, 'NbChenilleObserve', val_old, val_new)
    val_old = [0, 2, '1', 3, 4, '5', 8, '7', '6', '9', '60', '20', '10', '13', '15', '40', '12', '33', '14', '35', '17'
        , '18', '27', '24', '22', '16', '11', '47', '37', '57', '25', '19', '27', '18', 'LEVE (> 5 plants)']
    val_new = ['AUCUNE', 'FAIBLE (1 plant)', 'FAIBLE (1 plant)', 'MOYEN (2-5 plants)', 'MOYEN (2-5 plants)',
               'MOYEN (2-5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)'
        , 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)', 'ÉLEVÉE (> 5 plants)',
               'ÉLEVÉE (> 5 plants)']
    replace_chaine(df, 'NbPlantsCecidomyie', val_old, val_new)

    replace_chaine(df, 'DebutRecolte', '0', df['DebutRecolte'].mode()[0])
    replace_chaine(df, 'DebutRecolte', np.nan, df['DebutRecolte'].mode()[0])
    replace_chaine(df, 'FinRecolte', '0', df['DebutRecolte'].mode()[0])
    replace_chaine(df, 'FinRecolte', np.nan, df['DebutRecolte'].mode()[0])

    cols_date = ['DateImplantation', 'DateRecolteVisee', 'DebutRecolte', 'DateDepistage', 'FinRecolte']
    for col in cols_date:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['duree_visee'] = (df['DateRecolteVisee'] - df['DateImplantation']).dt.days
    df['duree_obtenue'] = (df['DebutRecolte'] - df['DateImplantation']).dt.days

    drop_columns(df, cols_date)
    val = ['NomSemis', 'NomEspece', 'NomCultivar', 'NomChamp']
    df3 = df[val]
    drop_columns(df, val)

    cat = []
    val_cat = variables_cat(df, cat)
    num = []
    val_num = variables_num(df, num)
    colonnes = show_columns30(df[num])['Champs'].to_list()
    for feature in colonnes:
        replace_with = df[feature].mean()
        df[feature].fillna(replace_with, inplace=True)
    cols = show_mv(df[num])['Champs'].to_list()
    features_to_drop = cat + cols
    cols_to_predict = show_mv(df[cat])['Champs'].to_list()

    sc = RobustScaler()
    appended_data = []
    for col_to_predict in cols_to_predict:
        features_to_drop.remove(col_to_predict)
        df_cleaned = df.drop(features_to_drop, axis=1).copy()
        df_with_val_to_predict = df_cleaned[df_cleaned[col_to_predict].notna()]
        df_no_val_to_predict = df_cleaned[df_cleaned[col_to_predict].isna()]

        X = df_with_val_to_predict.drop(col_to_predict, axis=1).values
        Y = df_with_val_to_predict[col_to_predict].values

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        forest = RandomForestClassifier(n_estimators=45, max_depth=25, random_state=False,
                                        max_features=0.6, min_samples_leaf=3, n_jobs=-1)
        forest.fit(X_train, Y_train)
        # y_pred_train = forest.predict(X_train)
        # y_pred = forest.predict(X_test)
        df_no_val_to_predict = df_no_val_to_predict.drop(col_to_predict, axis=1)
        prediction = forest.predict(df_no_val_to_predict)
        df_no_val_to_predict.insert(0, col_to_predict, prediction)

        frames = [df_with_val_to_predict, df_no_val_to_predict]
        df_final = pd.concat(frames)

        frames2 = [df[features_to_drop], df_final]
        df = pd.concat(frames2, axis=1, join='inner')
        appended_data.append(df[col_to_predict])
        df = df.drop(col_to_predict, axis=1).copy()

    df1 = pd.concat(appended_data, axis=1)
    df = pd.concat([df3, df1, df[features_to_drop], df_final.drop(col_to_predict, axis=1)], axis=1, join='inner')

    for feature in num:
        impute_outliers(df, feature)

    le = LabelEncoder()
    df[cat] = df[cat].apply(le.fit_transform)
    df = pd.concat([df[cat], df[num]], axis=1)
    # output_filepath = 'ml-datas/PMM_wodate_final_clean.csv'
    df.to_csv(output_filepath)
    # print(df)
    # print(df[cat])
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
