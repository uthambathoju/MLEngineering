import os
import joblib
import pandas as pd
from sklearn import metrics

from preprocessor import *
import config

def predict(df, MODEL):
    predictions= None

    for fold in range(config.NUM_FOLDS):
        #load model and columns
        #clf = joblib.load(os.path.join("./models/", f'{MODEL}_{fold}.pkl'))
        clf = joblib.load(os.path.join(config.MODELS_DIR , f'{MODEL}.pkl'))
        X_test = df.drop('target', axis=1)
        y_test = df['target'].values
        X_test= preprocess(X_test, "test")
        preds = clf.predict(X_test)
        
        #append preds in each fold
        if fold == 0:
            predictions = preds
        else:
            predictions += preds
        score = metrics.f1_score(y_test, predictions, average='macro')
        print(score)
    if config.NUM_FOLDS > 1:
        #avg of all folds preds
        predictions /= config.NUM_FOLDS
    sub = pd.DataFrame(predictions, columns=['PREDICTIONS'])
    return sub



if __name__ == "__main__":
    df = pd.read_parquet(config.TEST_DATA, engine='fastparquet')
    fixed_cols = ['target']
    f_cols = ["f_" + col for col in df.columns.tolist() if col not in fixed_cols]
    df.columns = f_cols + fixed_cols
    print(df.shape)
    sub = predict(df, config.MODEL)
    
    sub.to_csv(f'./models/{config.MODEL}.csv', index=False)