# This is an example feature definition file

from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, Int32

pmm = Entity(name = "pmm_id",
                     value_type = ValueType.INT64,
                 description = "ID of the pmm")

## Predictors Feature View
file_source = FileSource(path = r"data/predictors.parquet",
                         event_timestamp_column = "event_timestamp",)

df1_fv = FeatureView(
    name = "predictors_df_feature_view",
    ttl = timedelta(seconds = 86400*2),
    entities = [pmm],
    schema = [
    Field(name = "RendementLbParHa", dtype = Float64),
    Field(name = "SuperficieCultiveeHa", dtype = Float64),
    Field(name = "RejetDeclassement", dtype = Float64),
    Field(name = "NbrePlateauxTotal", dtype = Float64),
    Field(name = "Rejet_Dimensions", dtype = Float64),
    Field(name = "SuperficieTotaleHa", dtype = Float64),
    Field(name = "QtePlantsRequis", dtype = Float64),
    Field(name = "AnneeProduction", dtype = Int64),
    Field(name = "QteSemencesMillegrains", dtype = Float64),
    Field(name = "EmplacementChenille", dtype = Int32),
    Field(name = "Rejet_AutresDefauts", dtype = Float64),
    Field(name = "PesticideAmount", dtype = Float64),
    Field(name = "TypePlateaux", dtype = Float64),
    Field(name = "duree_visee", dtype = Float64),
    Field(name = "PopulationViseeParHa", dtype = Float64),
    Field(name = "duree_obtenue", dtype = Float64),
    Field(name = "Rejet_Matiere_Etrangere", dtype = Float64),
    Field(name = "SurplusPourcent", dtype = Float64),
    Field(name = "StadeCulture", dtype = Float64),
    Field(name = "FertilizerAmount", dtype = Float64),
    Field(name = "NbPlantsObserves", dtype = Float64),
    Field(name = "FournisseurPlant", dtype = Int32),
    Field(name = "NbChenilleObserve", dtype = Int32),
    Field(name = "Grosseur", dtype = Int32)
    ],
    source = file_source,
    online = True,
    tags= {},
)

## Target FEature View

target_source = FileSource(path = r"data/target.parquet",
                         event_timestamp_column = "event_timestamp",)

target_fv = FeatureView(
    name = "target_df_feature_view",
    ttl = timedelta(seconds = 86400*2),
    entities = [pmm],
    schema = [
    Field(name = "PoidsNet", dtype = Float64),
    ],
    source = target_source,
    online = True,
    tags= {},
)
