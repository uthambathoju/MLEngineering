#config.py

#Input data related settings
DATA_PATH = "D:/Utham/Micron/data/Q6_data.csv"
TRAINING_DATA = "D:/Utham/Micron/data/train_folds.parquet"
TEST_DATA = "D:/Utham/Micron/data/test.parquet"
MODELS_DIR = "D:/Utham/Micron/models/"
TEST_SIZE = 0.2
SEED = 42
#correlation
CORR_THRESHOLD = 0.7
#training
METRICS = ['accuracy', 'precision','recall']
TRAIN_DROP_COLS = ['kfold', 'target']
NUM_FOLDS = 1
HYPERPARAM_TUNE =False
CUSTOM_WEIGHTS = {0:1.0, 1:2.0, 2:2.0, 3:2.0, 4:2.0}
MODEL="randomforest_0.68"
