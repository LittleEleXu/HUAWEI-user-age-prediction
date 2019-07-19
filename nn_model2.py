# coding: utf-8

import os
import numpy as np
import pandas as pd
from keras import regularizers
from keras.models import Sequential
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from tools import to_categorical, timer, batch_generator, batch_generatorp, scale_data, create_submission
from data_process import nn2_data

np.random.seed(1991)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


train_discrete, label, train_continuous, test_discrete, test_continuous = nn2_data()

# 先处理离散值
ntrain = train_discrete.shape[0]
train_test = pd.concat((train_discrete, test_discrete), axis=0)

sparse_data = []
features = ['gender', 'city', 'prodName', 'ramCapacity',
            'ramLeftRation', 'romCapacity', 'romLeftRation',
            'color', 'fontSize', 'ct', 'carrier', 'os', ]

# 删掉一些无关特征
# features = ['gender', 'ramCapacity',
#             'romCapacity',
#             'color', 'fontSize', 'ct', 'carrier', 'os', ]

for f in features:
    dummy = pd.get_dummies(train_test[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

del(train_test, train_discrete, test_discrete)

# 得到独热编码好的xtrain，xtest
xtrain_test = hstack(sparse_data, format='csr')
xtrain = xtrain_test[:ntrain, :]
xtest = xtrain_test[ntrain:, :]


# 处理离散的数据，并合并
train_continuous = train_continuous.as_matrix()
train_continuous = csr_matrix(train_continuous)                 # ndarray转csr_matrix
xtrain = hstack([xtrain, train_continuous], format='csr')
test_continuous = test_continuous.as_matrix()
test_continuous = csr_matrix(test_continuous)
xtest = hstack([xtest, test_continuous], format='csr')

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)


def nn_model():
    model = Sequential()
    model.add(Dense(1200, kernel_regularizer=regularizers.l2(0.001), input_dim=xtrain.shape[1], init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(800, init='he_normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(400, init='he_normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(6, init='he_normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return(model)


# cv-folds
nfolds = 4
cv = StratifiedKFold(n_splits=nfolds, random_state=0)

# train models
str = 0
nbags = 1
nepochs = 1
number_class = label.nunique()

# 编码
y_onehot = to_categorical(label.values)

'''
# start training 
for (index_train, index_val) in cv.split(xtrain, y_cat):
    xtr = xtrain[index_train]
    ytr = y_cat[index_train]
    xval = xtrain[index_val]
    yval = y_cat[index_val]
    pred = np.zeros((xval.shape[0], number_class))
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator=batch_generator(xtr, ytr, 256, True),
                                  nb_epoch=nepochs,
                                  samples_per_epoch=xtr.shape[0],
                                  verbose=1,
                                  validation_data=(xval.todense(), yval))
        pred += model.predict_generator(generator=batch_generatorp(xval, 512, False),
                                        val_samples=xval.shape[0],
                                        verbose=1)
        pred_test += model.predict_generator(generator=batch_generatorp(xtest, 512, False),
                                             val_samples=xtest.shape[0],
                                             verbose=1)
   
        # score = accuracy_score(yte, pred)
    
        # str += 1
        # print('Fold ', str, 'accuracy:', score)

    pred_oob[index_val] = pred
# total_score = accuracy_score(yte, pred_oob)

pred_test /= (nfolds*nbags)

# print('Total - accuracy:', total_score)
'''

# 取小数据调试
# xtrain = xtrain[:10000, :]
# y_onehot = y_onehot[:10000, :]
# xtest = xtest[:10000, :]

pred_oob = np.zeros((xtrain.shape[0], number_class))
pred_test = np.zeros((xtest.shape[0], number_class))

# 建模
model = nn_model()
fit = model.fit_generator(generator=batch_generator(xtrain, y_onehot, 500, True),
                          nb_epoch=nepochs,
                          samples_per_epoch=xtrain.shape[0],
                          verbose=1)
pred_oob += model.predict_generator(generator=batch_generatorp(xtrain, 500, False),
                                    steps=4020,
                                    verbose=1)
pred_test += model.predict_generator(generator=batch_generatorp(xtest, 500, False),
                                     steps=1005,
                                     verbose=1)

keras_predictions_test = pd.DataFrame(pred_test)
keras_prediction_train = pd.DataFrame(pred_oob)

keras_predictions_test.to_csv('nn2_test.csv', index=None)
keras_prediction_train.to_csv('nn2_train.csv', index=None)

