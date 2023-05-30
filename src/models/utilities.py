import pandas as pd
def drop_columns(df, features):
    for feature in features:
        df.drop(feature, axis=1, inplace=True)

def variables_cat(df, val_cat=[]):
    for i in df.columns:
        if df[i].dtype == 'object':
            val_cat.append(i)
def variables_num(df, val_num=[]):
    for i in df.columns:
        if df[i].dtype == 'int64' or df[i].dtype == 'float64':
            val_num.append(i)

def replacement(df, val_old, val_new):
    return df.replace(val_old, val_new)

def replace_mv(feature, df):
    df[feature].fillna(df[feature].value_counts().index[0], inplace=True)

def replace_virgule(df, features):
    for i in features:
        df[i] = df[i].str.replace(',', '.')

def replace_virgule2(df, features):
    for i in features:
        df[i] = df[i].str.replace(',', '')

def replace_pourcentage(df, features):
    for i in features:
        df[i] = df[i].str.replace('%', '')

def conv_float(df, features):
    for i in features:
        df[i] = df[i].astype('float')

def conv_int(df, features):
    for i in features:
        df[i] = df[i].astype(int)

def conv_longint(df, features):
    for i in features:
        df[i] = pd.to_numeric(df[i])

def replace_chaine(df, feature, val_old, val_new):
    df[feature] = df[feature].replace(val_old, val_new)

def show_mv(df):
    df_nan= pd.DataFrame(round(((df.isna().sum()/df.shape[0])*100),1).to_dict().items(), columns=['Champs', 'Val NaN %'])
    return df_nan[df_nan['Val NaN %']>0]

def show_columns30(df):
    df_nan= pd.DataFrame(round(((df.isna().sum()/df.shape[0])*100),1).to_dict().items(), columns=['Champs', 'Val NaN %'])
    return df_nan[(df_nan['Val NaN %']>0) & (df_nan['Val NaN %']<30)]

def impute_outliers(df, feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    df.loc[df[feature] < lower_bound, feature] = lower_bound
    df.loc[df[feature] > upper_bound, feature] = upper_bound