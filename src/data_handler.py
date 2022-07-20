import pandas as pd
from sklearn import model_selection
import config



if __name__ == "__main__":
    df = pd.read_csv(config.DATA_PATH)
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    X = df.drop("target", axis=1)
    #create test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y)
    X_train["target"] = y_train
    X_test["target"] = y_test
    #Stratified Split 
    X_train['kfold'] = -1
    X_train = X_train.sample(frac=1).reset_index(drop=True)
    target = X_train.target.values
    #initiate kfold
    kf = model_selection.StratifiedKFold(n_splits=5)
    #fill the new kfold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_train, y=target)):
        X_train.loc[val_idx, 'kfold'] = fold
    
    #save the file
    X_train.to_parquet(config.TRAINING_DATA, index=False)
    X_test.to_parquet(config.TEST_DATA, index=False)