# coding: utf-8

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from tools import to_categorical, timer, batch_generator, batch_generatorp, scale_data, create_submission
from data_process import nn1_data


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.random.seed(1991)


train, test, y = nn1_data()

ntrain = train.shape[0]
print('ntrain=>', ntrain, '\n')
train_test = pd.concat((train, test), axis=0)                   # 列对齐，train test数据接起来
print('tr_te:', '\n')
print(train_test, '\n')

sparse_data = []
features = ['gender', 'city', 'prodName', 'ramCapacity', 
            'ramLeftRation', 'romCapacity', 'romLeftRation', 
            'color', 'fontSize', 'ct', 'carrier', 'os']

print(train_test['ramCapacity'].astype('category'))
print(pd.get_dummies(train_test['ramCapacity'].astype('category')))
print(csr_matrix(pd.get_dummies(train_test['ramCapacity'].astype('category'))))


for f in features:
    dummy = pd.get_dummies(train_test[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

print('sparse_data:\n', sparse_data)

# del(tr_te, train, test)

x_train_test = hstack(sparse_data, format='csr')                # 横向合并
print('x_train_test:\n', x_train_test)
xtrain = x_train_test[0:ntrain, :]
xtest = x_train_test[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)
# del(xtr_te, sparse_data, tmp)


# neural net
def nn_model():
    model = Sequential()
    model.add(Dense(1500, kernel_regularizer=regularizers.l2(0.001), input_dim=xtrain.shape[1], init='he_normal'))     # He正态分布初始化
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(800, init='he_normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(500, init='he_normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(6, init='he_normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(model)


# cv-folds
# nfolds = 5
# cv = StratifiedKFold(n_splits=nfolds, random_state=0)      # StratifiedKFold 分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同。

# train models
i = 0
# nbags = 10
nepochs = 1
# y = y[0:512]
number_class = y.nunique()

print('xtrain.shape[0]:', xtrain.shape[0])
print('xtest.shape[0]:', xtest.shape[0])

pred_oob = np.zeros((xtrain.shape[0], number_class))        # 2010000*6=>100000*6
pred_test = np.zeros((xtest.shape[0], number_class))        # 502500*6=>20000*6


# print('y:\n', y)
y_onehot = to_categorical(y.values)                         # 0-5进行独热编码
print('y_onehot:\n', y_onehot)


'''
# start training
for (indexTr, indexTe) in cv.split(xtrain, y):        # indexTr, indexTe是每一折所选数据的序号,5折迭代
    xtr = xtrain[indexTr]                             # 本折训练数据
    ytr = y_onehot[indexTr]                           # 本折训练数据标签
    xval = xtrain[indexTe]                            # 本折验证数据
    yval = y_onehot[indexTe]                          # 本折验证数据标签
    pred = np.zeros((xval.shape[0], number_class))    # 验证数据的预测值数组20000*6
    for j in range(nbags):                            # 创建多次模型
        model = nn_model()
        fit = model.fit_generator(generator=batch_generator(xtr, ytr, 128, True),       # 运行nepochs轮
                                  nb_epoch=nepochs,
                                  samples_per_epoch=xtr.shape[0],
                                  verbose=1,
                                  validation_data=(xval.todense(), yval))               # todense returns a matrix

        pred += model.predict_generator(generator=batch_generatorp(xval, 256, False), val_samples=xval.shape[0])                    # 本折验证集的预测矩阵
        pred_test += model.predict_generator(generator=batch_generatorp(xtest, 512, False), val_samples=xtest.shape[0])             # 测试集的预测矩阵
    pred /= nbags           # n个模型的预测结果求平均
    pred_oob[indexTe] = pred        # pred_oob是100000行，此时存放这一折的20000行验证集预测平均结果
    
    score = accuracy_score(yval, pred)      # 验证精度
    
    i += 1
    print('Fold ', i, '- accuracy:', score)
    
total_score = accuracy_score(y, pred_oob)    # 求总共100000行的验证精度

pred_test /= (nfolds * nbags)                  # 因为每一折，每个模型都求了pred_test，所以求平均，500000*6矩阵，表示对每行样本的预测向量

print('Total - - accuracy:', total_score)
'''

# start training,直接训练跟测试
# pred = np.zeros((xval.shape[0], number_class))  # 验证数据的预测值数组20000*6



# 建模
model = nn_model()

# xtrain = xtrain[0:10000, :]                     # 小数据试试
# y_onehot = y_onehot[0:10000, :]
# xtest = xtest[0:10000, :]

pred_oob = np.zeros((xtrain.shape[0], number_class))        # 2010000*6=>100000*6
pred_test = np.zeros((xtest.shape[0], number_class))        # 502500*6=>20000*6

fit = model.fit_generator(generator=batch_generator(xtrain, y_onehot, 500, True),  # 运行nepochs轮
                          nb_epoch=nepochs,
                          samples_per_epoch=xtrain.shape[0],
                          verbose=1)  # 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

model.save('nn_model1.h5')

pred_test += model.predict_generator(generator=batch_generatorp(xtest, 500, False),
                                     steps=1005,        # 测试集的预测矩阵
                                     verbose=1)


pred_oob += model.predict_generator(generator=batch_generatorp(xtrain, 500, False),
                                    steps=4020,         # 训练集的预测矩阵
                                    verbose=1)

keras_predictions_test = pd.DataFrame(pred_test)            # 测试集的预测矩阵
keras_prediction_train = pd.DataFrame(pred_oob)             # 训练集汇总的预测矩阵

keras_predictions_test.to_csv('nn1_predictions_test.csv', index=None)
keras_prediction_train.to_csv('nn1_prediction_train.csv', index=None)

