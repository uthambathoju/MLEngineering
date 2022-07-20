import os
import joblib
import pandas as pd

from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import make_column_selector,make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction import FeatureHasher
from sklearn import model_selection

#models
import config
import model_dispatcher as dispatcher
from preprocessor import *

def train_old(df, MODEL):
    """
    :param df: pandas dataframe with train/test data
    :return score: f1_score
    """
    for fold in range(5):
        #features
        df_train = df[df["kfold"] != fold].reset_index(drop=True)
        df_valid = df[df["kfold"] == fold].reset_index(drop=True)
        #target
        ytrain = df_train['LICENSE STATUS'].values
        yvalid = df_valid['LICENSE STATUS'].values

        drop_cols = ["ID","LICENSE STATUS","kfold"]
        df_train = df_train.drop(drop_cols, axis=1)
        df_valid = df_valid.drop(drop_cols, axis=1)
        #model
        clf = dispatcher.MODELS[MODEL]
        clf = clf.fit(df_train, ytrain)
        preds = clf.predict_proba(df_valid)[:, 1]
        score = metrics.f1_score(yvalid, preds)

        #save model , cols
        joblib.dump(clf, f'./models/{MODEL}_{fold}.pkl')
        joblib.dump(df_train.columns, f'./models/{MODEL}_{fold}_columns.pkl')

    
    return cat_feats

def train_func(df, MODEL):
    cat_feats = manipulate_df(df)
    print(cat_feats.columns)
    print(cat_feats.info())
    """
    df.pipe(explode, column="A")\
      .pipe(explode, column="C")\
      .pipe(fill_na, -99)\
      .pipe(encode)
    """
    corr_est = FunctionTransformer(func = identify_correlated_cols, kw_args={"df" : df, "threshold" : config.CORR_THRESHOLD})
    #enc_est  = FunctionTransformer(func = FeatureHashing, kw_args={"cat_feats" : cat_feats.columns.tolist()})
    skew_est  = FunctionTransformer(func = check_skewness, kw_args={"df" : df})

    
    estimator = ColumnTransformer(
        transformers = [
            ("corr", corr_est),
            ("skew", skew_est)
        ]
        )
   
    custom_pipe = Pipeline(steps = [
        ("preprocess", estimator),
        ("base", MODEL),
        ]
                     )
    X_train = df.drop('target', axis=1)
    y_train =  df['target'].values
    custom_pipe.fit(X_train, y_train)
    
def train_almost(df, MODEL,X_train, y_train, missing_cols):
    cat_feats = manipulate_df(X_train)
    categorical_cols = cat_feats.columns.tolist()
    print(cat_feats.nunique())
    
    #Build numeric processor
    #num_feats = df.drop(categorical_cols + ['target', 'kfold'], axis=1)
    num_feats = df.drop(categorical_cols + ['target'], axis=1)
    print("herhe :: ", num_feats.columns)
    skew_cols, norm_cols = check_skewness(num_feats)
    imputer_pipe = make_pipeline(KNNImputer())
    num_power_pipe = make_pipeline(PowerTransformer())
    num_std_pipe = make_pipeline(StandardScaler())
    #processor
    #("categorical", categorical_pipe, [categorical_cols]),
    custom_proc = ColumnTransformer(
        transformers=[
            ("imputer", imputer_pipe, missing_cols),
            ("power_transform", num_power_pipe, skew_cols),
            ("standardization", num_std_pipe, norm_cols)
        ]
    )

    #pipeline
    custom_pipe = Pipeline(
        steps = [
            ("preprocess", custom_proc),
            ("model", MODEL)
        ]
    )
    model_res = custom_pipe.fit(X_train, y_train)


def train(df, MODEL):

    
    for fold in range(1):
        #features
        X_train = df[df["kfold"] != fold].reset_index(drop=True)
        X_valid = df[df["kfold"] == fold].reset_index(drop=True)
        #target
        ytrain = X_train['target'].values
        yvalid = X_valid['target'].values

        X_train = X_train.drop(config.TRAIN_DROP_COLS, axis=1)
        X_valid = X_valid.drop(config.TRAIN_DROP_COLS, axis=1)
        missing_df= X_train.isnull().sum().reset_index()
        na_cols = missing_df['index'][missing_df[0] > 0].tolist()
        print(na_cols)
        #model
        clf = dispatcher.MODELS[MODEL]
        cat_feats = manipulate_df(X_train)
        categorical_cols = cat_feats.columns.tolist()
        print(cat_feats.nunique())
        
        #Build numeric processor
        num_feats = X_train.drop(categorical_cols, axis=1)
        skew_cols, norm_cols = check_skewness(num_feats)
        imputer_pipe = make_pipeline(KNNImputer())
        num_power_pipe = make_pipeline(PowerTransformer())
        num_std_pipe = make_pipeline(StandardScaler())
        #processor
        #("categorical", categorical_pipe, [categorical_cols]),
        #imputer_pipe.fit(X_train, y_train)

        custom_proc = ColumnTransformer(
            transformers=[
                ("power_transform", num_power_pipe, skew_cols),
                ("standardization", num_std_pipe, norm_cols)
            ]
        )

        #pipeline
        custom_pipe = Pipeline(
            steps = [
                ("preprocess", custom_proc),
                ("model", MODEL)
            ]
        )
        custom_pipe.fit(X_train, ytrain)    


def train__(df,MODEL):

    for fold in range(1):
        #features
        X_train = df[df["kfold"] != fold].reset_index(drop=True)
        X_valid = df[df["kfold"] == fold].reset_index(drop=True)
        #target
        ytrain = X_train['target'].values
        yvalid = X_valid['target'].values

        X_train = X_train.drop(config.TRAIN_DROP_COLS, axis=1)
        X_valid = X_valid.drop(config.TRAIN_DROP_COLS, axis=1)
        missing_df= X_train.isnull().sum().reset_index()
        na_cols = missing_df['index'][missing_df[0] > 0].tolist()
        print(na_cols)
        #model
        clf = dispatcher.MODELS[MODEL]
        cat_feats = manipulate_df(X_train)
        print(cat_feats.columns)
        num_cols = X_train.columns.tolist()
        cat_cols = cat_feats.columns.tolist()
        #num_cols = [item for item in num_cols if item not in cat_cols]
        print(X_train.shape)
        categorical_cols = cat_feats.columns.tolist()
        corr_est = FunctionTransformer(identify_correlated_cols)
        enc_est = FunctionTransformer(FeatureHashing)
        impt_est = FunctionTransformer(handle_missing_cols)
        #imp = make_pipeline(KNNImputer())
        preprocess = make_column_transformer(
            (impt_est, na_cols),
            (corr_est, num_cols),
            (enc_est, categorical_cols),
            remainder='passthrough'
        )
        #X_train = pd.DataFrame(preprocess.fit_transform(X_train), columns=num_cols)
        #X_train = preprocess.fit_transform(X_train)
        
        
        pipe = make_pipeline(preprocess, clf)
        #X_train = X_train[['f_2','f_3','f_40']]
        pipe.fit(X_train, ytrain)
        """
        preds = pipe.predict(X_valid)
        print(set(preds))
        score = metrics.f1_score(yvalid, preds, average='macro')
        print(score)
        #save model , cols
        #joblib.dump(clf, f'./models/{MODEL}_{fold}.pkl')
        #joblib.dump(df_train.columns, f'./models/{MODEL}_{fold}_columns.pkl')
        """

if __name__ == "__main__":
    MODEL = "randomforest"
    #MODEL = dispatcher.MODELS[MODEL]
    df = pd.read_parquet(config.TRAINING_DATA, engine='fastparquet')
    fixed_cols = ['target', 'kfold']
    f_cols = ["f_" + col for col in df.columns.tolist() if col not in fixed_cols]
    df.columns = f_cols + fixed_cols
    #handle_missing_cols(df)
    #identify_correlated_cols(df)
    train(df, MODEL)