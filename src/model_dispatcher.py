from sklearn import ensemble
from sklearn import linear_model
import config


MODELS = {
    "randomforest_plain" : ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "randomforest_custom":ensemble.RandomForestClassifier(n_estimators=200, 
                                                       n_jobs=-1, 
                                                       verbose=2,
                                                       class_weight=config.CUSTOM_WEIGHTS),
    "randomforest" :ensemble.RandomForestClassifier(n_estimators= 200, 
                                                    min_samples_split= 2, 
                                                    min_samples_leaf= 1, 
                                                    max_features= 'auto', 
                                                    max_depth= 40, 
                                                    bootstrap= False),
    "logisticRegression": linear_model.LogisticRegression(random_state=0, multi_class='ovr'),
    "extratrees" :ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2)
}
 