
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from tools import to_categorical, timer, batch_generator, batch_generatorp, scale_data, create_submission
import gc


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# def timer(start_time=None):
#     if not start_time:
#         start_time = datetime.now()
#         return start_time
#     elif start_time:
#         tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
#         print(' Time taken: %i minutes and %s seconds.' %
#               (tmin, round(tsec, 2)))


# In[ ]:

# def scale_data(X, scaler=None):
#     if not scaler:
#         scaler = StandardScaler()
#         scaler.fit(X)
#     X = scaler.transform(X)
#     return X, scaler




# ------------- the stacked model: load all the predictions (6 model) and build on top of it an xgboost model-------------------------#

# load xgboost predctions 
xgb1_train = pd.read_csv('second_data/xgb2_predictions_train.csv', header=0)
xgb1_test = pd.read_csv('second_data/xgb2_predictions_test.csv', header=0)


# load xgboost predictions with no events
xgb2_train = pd.read_csv('second_data/xgb2_predictions_train.csv', header=0)
xgb2_test = pd.read_csv('second_data/xgb2_predictions_test.csv', header=0)


# load keras predictions with events :
nn1_train = pd.read_csv('second_data/nn1_predictions_train.csv', header=0)
nn1_test = pd.read_csv('second_data/nn1_predictions_test.csv', header=0)


# load keras prediction with no events
nn2_train = pd.read_csv('second_data/nn2_train.csv', header=0)
nn2_test = pd.read_csv('second_data/nn2_test.csv', header=0)



# load extra tree end RF prediction on the data with events which also has the raget variable
rf_et_predictions_test = pd.read_csv('second_data/rf_et_predictions_test.csv', header=0)
rf_et_prediction_train = pd.read_csv('second_data/rf_et_predictions_train.csv', header=0)
columns = rf_et_prediction_train.columns.tolist()
Target_name = columns[len(columns)-1]           # 最后一列名字
target = rf_et_prediction_train[Target_name]
rf_et_prediction_train.drop(Target_name, axis=1, inplace=True)
print('target\n', target)

# concat all train and test data in the same order
train = pd.concat((xgb1_train, xgb2_train, nn1_train, nn2_train, rf_et_prediction_train), axis=1)       # 2010000*36
test = pd.concat((xgb1_test, xgb2_test, nn1_test, nn2_test, rf_et_predictions_test), axis=1)            # 502500*36

train.columns = [i for i in range(36)]
test.columns = [i for i in range(36)]

print('train\n', train)
print('test\n', test)


# 小数据试试
train = train[:10000]
target = target[:10000]
test = test[:10000]


# Actual Run Code
lable_group = LabelEncoder()
Y = lable_group.fit_transform(target)       # 变成一个列表
classes_ = lable_group.classes_
print('Y\n', Y)
print('classes_\n', classes_)


# enter the number of folds from xgb.cv
ntest = test.shape[0]
folds = 5
early_stopping = 50
oof_test = np.zeros((ntest, 6))
start_time = timer(None)



# Load data set and target values
# test = test.loc[:, ~test.columns.duplicated()]
# print('test\n', test)


d_test = xgb.DMatrix(test.values)


# In[ ]:

# set up KFold that matches xgb.cv number of folds
kf = StratifiedKFold(n_splits=folds, random_state=0)

# Start the CV
acc = []
for i, (train_index, test_index) in enumerate(kf.split(train, Y)):
    print('\n Fold %d\n' % (i + 1))
    X_train, X_val = train.values[train_index], train.values[test_index]
    y_train, y_val = Y[train_index], Y[test_index]

    # Define  variables
    params = {}
    params["booster"] = "gbtree"
    params['objective'] = "multi:softprob"
    params['eval_metric'] = 'mlogloss'
    params['num_class'] = 6
    params['eta'] = 0.01
    params['gamma'] = 0.1
    params['min_child_weight'] = 1
    params['colsample_bytree'] = 0.5
    params['subsample'] = 0.8
    params['max_depth'] = 7  
    params['silent'] = 1
    params['random_state'] = 0

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]


    #  Build Model
    clf = xgb.train(params,
                    d_train,
                    10,
                    watchlist,
                    early_stopping_rounds=early_stopping) 
    timer(start_time)

    #  Evaluate Model and Predict
    oof_test[:] += clf.predict(d_test, ntree_limit=clf.best_iteration)
    oof_val = clf.predict(d_valid, ntree_limit=clf.best_iteration)
    temp1 = np.argmax(oof_val, axis=1)
    score = accuracy_score(y_val, temp1)
    print(' eval-accuracy: %.6f' % score)
    acc.append(score)

print('acc:', acc)

oof_test /= folds

#  Make a submision
temp = np.argmax(oof_test, axis=1) + 1          # 列表
temp = pd.DataFrame(temp, columns=['label'], dtype='int32')

test = pd.read_csv('../../data/age_test.csv', header=0, names=['uId'])
test = test.sort_values(by='uId')

result = test
result = pd.concat((result, temp), axis=1)
result.columns = ['id', 'label']

now = datetime.now()
sub_file = 'submission_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
result.to_csv(sub_file, index=None)

