import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.stats import skew
import config

import os
import joblib

def identify_correlated_cols(df,threshold=0.7, mode="train"):
    """
    :param df : pandas dataframe with train/test data
    :param threshold: integer 
    :return: dataframe with new features
    """
    try:
        if mode == 'train':
            # Create correlation matrix
            corr_matrix = df.corr().abs()
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

            dic = {'Feature_1':[],'Feature_2':[],'val':[]}
            for col in upper.columns:
                corl = list(filter(lambda x: x >= threshold, upper[col] ))
                if len(corl) > 0:
                    inds = [round(x,4) for x in corl]
                    for ind in inds:
                        col2 = upper[col].index[list(upper[col].apply(lambda x: round(x,4))).index(ind)]
                        dic['Feature_1'].append(col)
                        dic['Feature_2'].append(col2)
                        dic['val'].append(ind) 
            df_corr = pd.DataFrame(dic).sort_values(by="val", ascending=False)
            corr_cols = df_corr['Feature_1'].values.tolist()
            df.drop(corr_cols, axis=1, inplace=True)        
            joblib.dump(corr_cols, os.path.join(config.MODELS_DIR , 'corr_columns.pkl'))
        else:
            corr_cols = joblib.load(os.path.join(config.MODELS_DIR , 'corr_columns.pkl'))
            df.drop(corr_cols, axis=1, inplace=True)
        
    except:
        raise
    
    return df 
    

def manipulate_df(df):
    cat_feats = df.select_dtypes(include='int64')
    cols = [col for col in cat_feats.columns.tolist() if cat_feats[col].nunique() < 2000]
    cat_feats = cat_feats[cols]
    cat_feats[cols] = cat_feats[cols].astype('str')
    return cat_feats

def FeatureHashing(df):
    """
    :param df : pandas dataframe with train/test data
    :param cat_feats: cat cols list 
    :return: dataframe with encoded cat features
    """
    cat_feats = manipulate_df(df)
    cat_cols = cat_feats.columns.tolist()
    for col_name in cat_feats.columns:
        df[col_name]=df[col_name].astype('category') 
    return df


def handle_missing_cols(df):
    missing_df= df.isnull().sum().reset_index()
    na_cols = missing_df['index'][missing_df[0] > 0].tolist()
    print(na_cols)
    n_neighbors = [5]
    for col_name in na_cols:
        for k in n_neighbors:
            knn_imp = KNNImputer()
            df[col_name] = knn_imp.fit_transform(df[[col_name]])
    return df


def check_skewness(df):
    for col_name in df.columns.tolist():
        skew_cols=[]
        norm_cols=[]
        if col_name not in config.TRAIN_DROP_COLS:
            res = skew(df[col_name])
            if res > 5.0: 
                skew_cols.append(res)
            else:
                ss =StandardScaler()
                norm_cols.append(res)    
    return skew_cols, norm_cols 


def preprocess(df, mode):
    try:
        df.pipe(handle_missing_cols)\
          .pipe(identify_correlated_cols, threshold=config.CORR_THRESHOLD, mode=mode)
    except:
        print("EXCEPTION CAUGHT IN preprocess FUNC")
        raise 
    return df