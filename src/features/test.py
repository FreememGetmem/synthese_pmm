import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

store = FeatureStore(repo_path='feature_repo/feature_repo')

entity_df = pd.read_parquet('feature_repo/feature_repo/data/predictors.parquet')
training_data = store.get_historical_features(
   entity_df=entity_df,
   features = [ "predictors_df_feature_view:RendementLbParHa",
                "predictors_df_feature_view:SuperficieCultiveeHa",
                "predictors_df_feature_view:RejetDeclassement",
                "predictors_df_feature_view:NbrePlateauxTotal",
                "predictors_df_feature_view:Rejet_Dimensions",
                "predictors_df_feature_view:SuperficieTotaleHa",
                "predictors_df_feature_view:QtePlantsRequis",
                "predictors_df_feature_view:AnneeProduction",
                "predictors_df_feature_view:QteSemencesMillegrains",
                "predictors_df_feature_view:EmplacementChenille",
                "predictors_df_feature_view:Rejet_AutresDefauts",
                "predictors_df_feature_view:PesticideAmount",
                "predictors_df_feature_view:TypePlateaux",
                "predictors_df_feature_view:duree_visee",
                "predictors_df_feature_view:PopulationViseeParHa",
                "predictors_df_feature_view:duree_obtenue",
                "predictors_df_feature_view:Rejet_Matiere_Etrangere",
                "predictors_df_feature_view:SurplusPourcent",
                "predictors_df_feature_view:StadeCulture",
                "predictors_df_feature_view:FertilizerAmount",
                "predictors_df_feature_view:NbPlantsObserves",
                "predictors_df_feature_view:FournisseurPlant",
                "predictors_df_feature_view:NbChenilleObserve",
                "predictors_df_feature_view:Grosseur",
                ]
)

# dataset = store.create_saved_dataset(
#     from_ = training_data,
#     name = "diabetes_dataset",
#     storage = SavedDatasetFileStorage('feature_repo/feature_repo/data/diabetes_dataset.parquet')
# )


print(training_data.to_df())