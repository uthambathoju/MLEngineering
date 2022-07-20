import os
import joblib
import pandas as pd

from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.compose import make_column_selector,make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction import FeatureHasher
from sklearn import model_selection

from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
#models
import config
import model_dispatcher as dispatcher
from preprocessor import *
from sklearn.decomposition import PCA

def train(df, MODEL):
    for fold in range(config.NUM_FOLDS):
        #features
        X_train = df[df["kfold"] != fold].reset_index(drop=True)
        X_valid = df[df["kfold"] == fold].reset_index(drop=True)
        #target
        ytrain = X_train['target'].values
        yvalid = X_valid['target'].values

        X_train = X_train.drop(config.TRAIN_DROP_COLS, axis=1)
        X_valid = X_valid.drop(config.TRAIN_DROP_COLS, axis=1)
       
        #X_train= preprocess(X_train, "train")
        #X_valid= preprocess(X_valid, "valid")
        na_cols = ['f_2']
        impt_est = FunctionTransformer(handle_missing_cols)
        preprocess = make_column_transformer(
            (impt_est, na_cols)
        )
        
        clf = dispatcher.MODELS[MODEL]      
        pipe = make_pipeline(preprocess, 
                            MODEL)     
        pipe.fit(X_train, ytrain)
        preds = clf.predict(X_valid)
        for metric_name in config.METRICS:
            print(metric_name)
            if metric_name != 'f1_score':
                sc = metrics.get_scorer(metric_name)._score_func(yvalid, preds) 
            else:
                sc = metrics.get_scorer(metric_name)._score_func(yvalid, preds)
        print(sc)
        score = metrics.f1_score(yvalid, preds, average='macro')
        print(score)
        
        #save model , cols
        joblib.dump(clf, os.path.join(config.MODELS_DIR, f'{MODEL}_{fold}.pkl'))
        
if __name__ == "__main__":
    #MODEL = "randomforest"
    MODEL = "randomforest"
    #MODEL = "logisticRegression"
    #MODEL = dispatcher.MODELS[MODEL]
    df = pd.read_parquet(config.TRAINING_DATA, engine='fastparquet')
    fixed_cols = ['target', 'kfold']
    f_cols = ["f_" + col for col in df.columns.tolist() if col not in fixed_cols]
    df.columns = f_cols + fixed_cols
    train(df, MODEL)