# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tools import to_categorical, timer, batch_generator, batch_generatorp, scale_data, create_submission
from data_process import xgb2_data

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


train, test, y_train = xgb2_data()

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
    "objective": "multi:softprob",      # 定义学习任务及相应的学习目标，可选的目标函数多种，“multi:softprob” –和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量reshape成ndata行nclass列的矩阵。没行数据表示样本所属于每个类别的概率。
    'min_child_weight': 1,              # 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
    "num_class": 6,                     # 类别
    "booster": "gbtree",                # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认值为gbtree。
    'colsample_bytree': 0.5,            # 在建立树时对特征采样的比例。缺省值为1
    'subsample': 0.8,                   # 用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的冲整个样本集合中随机的抽取出50%的子样本建立树模型，这能够防止过拟合
    "max_depth": 6,                     # 数的最大深度。缺省值为6
    "eval_metric": "mlogloss",          # Multiclass logloss
    "eta": 0.01,                        # 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守。0-1之间缺省值为0.3
    "silent": 0,                        # 取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息。缺省值为0
    "alpha": 1,
    'gamma': 0,                         # minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be.【0，正无穷】
    'seed': SEED
    }
oof_train = np.zeros((ntrain, 6))
oof_test = np.zeros((ntest, 6))

for i, (train_index, val_index) in enumerate(kf.split(train, y_train)):
    print('\n Fold %d\n' % (i + 1))
    X_train, X_val = x_train[train_index], x_train[val_index]
    y_train, y_val = Y[train_index], Y[val_index]
    d_train = xgb.DMatrix(X_train, label=y_train)           # 训练数据
    d_valid = xgb.DMatrix(X_val, label=y_val)               # 验证数据
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    
    # train
    clf = xgb.train(params,
                    d_train,                                # 训练数据
                    10000,                                  # 生成多少基模型
                    evals=watchlist,                        # 这是一个列表，用于对训练过程中进行评估列表中的元素。形式是evals = [(dtrain,'train'),(dval,'val')]或者是evals = [(dtrain,'train')]，对于第一种情况，它使得我们可以在训练过程中观察验证集的效果
                    early_stopping_rounds=20)

    oof_test[:] += clf.predict(xgb.DMatrix(x_test), ntree_limit=clf.best_iteration)             # 用最好的那次迭代来预测
    oof_train[val_index] = clf.predict(xgb.DMatrix(X_val), ntree_limit=clf.best_iteration)
    
    
oof_test /= NFOLDS

xgb_predictions_test = pd.DataFrame(oof_test)
xgb_prediction_train = pd.DataFrame(oof_train)

xgb_predictions_test.to_csv('xgb2_predictions_test.csv', index=None)
xgb_prediction_train.to_csv('xgb2_prediction_train.csv', index=None)


