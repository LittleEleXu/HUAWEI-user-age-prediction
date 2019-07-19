# coding: utf-8
import gc
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from data_process import Et_Rf_data


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    def predict(self, x):
        return self.clf.predict_proba(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


# 数据太大，不搞交叉验证
# def get_oof(clf):
#     oof_train = np.zeros((ntrain, 6))
#     oof_test = np.zeros((ntest, 6))
#
#     for i, (train_index, val_index) in enumerate(kf.split(train, tra_label)):
#         x_tr = x_train[train_index]
#         y_tr = tra_label[train_index]
#         x_val = x_train[val_index]
#
#         clf.train(x_tr, y_tr)
#
#         oof_train[val_index] = clf.predict(x_val)
#         oof_test[:] += clf.predict(x_test)
#
#     oof_test[:] /= NFOLDS
#     return oof_train, oof_test

def get_oof(clf):
    oof_train = np.zeros((ntrain, 6))
    oof_test = np.zeros((ntest, 6))

    # x_tr = x_train[train_index]
    #     y_tr = tra_label[train_index]
    #     x_val = x_train[val_index]

    clf.train(x_train, tra_label)

    oof_train += clf.predict(x_train)
    oof_test += clf.predict(x_test)

    return oof_train, oof_test


# 程序开始
train, tra_label, test = Et_Rf_data()
print("1:{},{}".format(train.shape, test.shape))

gc.collect()
lable_group = LabelEncoder()
Y = lable_group.fit_transform(tra_label)

NFOLDS = 5
SEED = 0

x_train = train.values
ntrain = train.shape[0]
x_test = test.values
ntest = test.shape[0]

# 取小数据试试
# x_train = train[0:10000].values
# ntrain = x_train.shape[0]
# x_test = test[0:10000].values
# ntest = x_test.shape[0]
# tra_label = tra_label[0:10000]


kf = StratifiedKFold(n_splits=NFOLDS, random_state=SEED)

et_params = {'n_jobs': 2,
             'n_estimators': 300,
             'max_features': 0.5,
             'max_depth': 7,
             'min_samples_leaf': 2}

rf_params = {'n_jobs': 2,
             'n_estimators': 300,
             'max_features': 0.4,
             'max_depth': 7,
             'min_samples_leaf': 2}

print('2:start train=========================================')

et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

print('3:start train=========================================')

# start training the RF and ET models
et_oof_train, et_oof_test = get_oof(et)
print("4:{},{}".format(et_oof_train.shape, et_oof_test.shape))
rf_oof_train, rf_oof_test = get_oof(rf)


# print the CV results
print("5:ET-CV: {}".format(log_loss(tra_label, et_oof_train)))
print("5:RF-CV: {}".format(log_loss(tra_label, rf_oof_train)))


# i concatenate the train prediction with the target value so to use it in the next step : Stacking which will be our final model.
x_train = np.concatenate((et_oof_train, rf_oof_train, pd.DataFrame(tra_label)), axis=1)     # 6+6+1=13
x_test = np.concatenate((et_oof_test, rf_oof_test), axis=1)                                 # 6+6=12


rf_et_prediction_train = pd.DataFrame(x_train)
rf_et_predictions_test = pd.DataFrame(x_test)


rf_et_prediction_train.to_csv('rf_et_prediction_train.csv', index=None)
rf_et_predictions_test.to_csv('rf_et_predictions_test.csv', index=None)


# print the resulting train and data set's shape
print("6:{},{}".format(x_train.shape, x_test.shape))

