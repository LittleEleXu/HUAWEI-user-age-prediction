# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import gc
from tools import to_categorical, timer, batch_generator, batch_generatorp, scale_data, create_submission
from data_process import xgb1_data


train, test, y_train = xgb1_data()
gc.collect()
lable_group = LabelEncoder()
Y = lable_group.fit_transform(y_train)

NFOLDS = 5
SEED = 0
print("{},{}".format(train.shape, test.shape))

x_train = train.values
ntrain = train.shape[0]
x_test = test.values
ntest = test.shape[0]

kf = StratifiedKFold(n_splits=NFOLDS, random_state=SEED)


params = {
    "objective": "multi:softprob",
    'min_child_weight': 1,
    "num_class": 6,
    "booster": "gbtree",
    'colsample_bytree': 0.5,  
    'subsample': 0.8,
    "max_depth": 4,
    "eval_metric": "mlogloss",
    "eta": 0.01,
    "silent": 1,
    "alpha": 1,
    'gamma': 0,
    'seed': SEED
    }
oof_train = np.zeros((ntrain, 6))
oof_test = np.zeros((ntest, 6))


for i, (train_index, test_index) in enumerate(kf.split(train, y_train)):
    print('\n Fold %d\n' % (i + 1))
    X_train, X_val = x_train[train_index], x_train[test_index]
    y_train, y_val = Y[train_index], Y[test_index]
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]

    clf = xgb.train(params,
                    d_train,
                    10000,
                    evals=watchlist,
                    early_stopping_rounds=20)

    oof_test[:] += clf.predict(xgb.DMatrix(x_test), ntree_limit=clf.best_iteration)
    oof_train[test_index] = clf.predict(xgb.DMatrix(X_val), ntree_limit=clf.best_iteration)

oof_test /= NFOLDS

xgb1_predictions_test = pd.DataFrame(oof_test)
xgb1_prediction_train = pd.DataFrame(oof_train)

xgb1_predictions_test.to_csv('xgb1_predictions_test.csv', index=None)
xgb1_prediction_train.to_csv('xgb1_predictions_train.csv', index=None)


