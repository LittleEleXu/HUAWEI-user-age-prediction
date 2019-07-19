import numpy as np
import pandas as pd
import os

# 本地路径
DATA_TRAIN_PATH = '../../data/age_train.csv'
DATA_TEST_PATH = '../../data/age_test.csv'
USER_BASIC_INFO_PATH = '../../data/user_basic_info.csv'
USER_BEHAVIOR_INFO_PATH = '../../data/user_behavior_info.csv'
APP_INFO_PATH = '../../data/app_info.csv'
USER_APP_ACTIVED_PATH = '../../data/user_app_actived.csv'
USER_APP_USAGE_PATH = '../../data/user_app_usage/user_app_usage.csv'

# 服务器路径
# DATA_TRAIN_PATH = '/home/xuwenxiang/xu/data/age_train.csv'
# DATA_TEST_PATH = '/home/xuwenxiang/xu/data/age_test.csv'
# USER_BASIC_INFO_PATH = '/home/xuwenxiang/xu/data/user_basic_info.csv'
# USER_BEHAVIOR_INFO_PATH = '/home/xuwenxiang/xu/data/user_behavior_info.csv'
# APP_INFO_PATH = '/home/xuwenxiang/xu/data/app_info.csv'
# USER_APP_ACTIVED_PATH = '/home/xuwenxiang/xu/data/user_app_actived.csv'
# USER_APP_USAGE_PATH = '/home/xuwenxiang/xu/data/user_app_usage/user_app_usage.csv'


def nn1_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH, path_user_basic_info=USER_BASIC_INFO_PATH):
    basic = pd.read_csv(path_user_basic_info, names=['uId', 'gender', 'city', 'prodName', 'ramCapacity',
                                                     'ramLeftRation', 'romCapacity', 'romLeftRation', 'color',
                                                     'fontSize', 'ct', 'carrier', 'os'], dtype={'uId': np.str})
    basic = basic.sort_values(by='uId')
    basic.drop_duplicates('uId', keep='first', inplace=True)

    # 离散值进行数值编码，取其0维的编码值，1维是形状
    basic['gender'] = pd.factorize(basic['gender'], sort=True)[0]
    basic['city'] = pd.factorize(basic['city'], sort=True)[0]
    basic['prodName'] = pd.factorize(basic['prodName'], sort=True)[0]
    basic['ramCapacity'] = pd.factorize(basic['ramCapacity'], sort=True)[0]
    basic['ramLeftRation'] = pd.factorize(basic['ramLeftRation'], sort=True)[0]
    basic['romCapacity'] = pd.factorize(basic['romCapacity'], sort=True)[0]
    basic['romLeftRation'] = pd.factorize(basic['romLeftRation'], sort=True)[0]
    basic['color'] = pd.factorize(basic['color'], sort=True)[0]
    basic['fontSize'] = pd.factorize(basic['fontSize'], sort=True)[0]
    basic['ct'] = pd.factorize(basic['ct'], sort=True)[0]
    basic['carrier'] = pd.factorize(basic['carrier'], sort=True)[0]
    basic['os'] = pd.factorize(basic['os'], sort=True)[0]

    # 删除部分特征
    # basic.drop(['uId', 'age_group'], axis=1, inplace=True)

    train = pd.read_csv(path_train, names=['uId', 'age_group'], dtype={'uId': np.str})
    train = train.sort_values(by='uId')                                     # 1.训练数据里的ID要排序！！！！！要保证去掉id之后是对得上的
    train['age_group'] = pd.factorize(train['age_group'], sort=True)[0]     # 2.标签编码,比实际小了1，变成0-5
    train = pd.merge(train, basic, how='left', on='uId', left_index=True)
    print('train:', '\n')
    print(train, '\n')

    # target
    target = train.age_group
    print('target:', '\n')
    print(target, '\n')

    # train
    train.drop(['uId', 'age_group'], axis=1, inplace=True)
    train.fillna(-1, inplace=True)
    # train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
    print('train:', '\n')
    print(train, '\n')

    # test data
    test_loader = pd.read_csv(path_test, names=['uId'], dtype={'uId': np.str})
    test_loader = test_loader.sort_values(by='uId')
    test = pd.merge(test_loader, basic, how='left', on='uId', left_index=True)
    test.drop('uId', axis=1, inplace=True)
    test.fillna(-1, inplace=True)
    print('test:', '\n')
    print(test, '\n')

    return train, test, target




def nn2_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH, path_user_basic_info=USER_BASIC_INFO_PATH,
              path_user_behavior_info=USER_BEHAVIOR_INFO_PATH, app_info_path=APP_INFO_PATH,
              user_app_avtived_path=USER_APP_ACTIVED_PATH, user_app_usage_path=USER_APP_USAGE_PATH):
    # train.csv 2010000
    train = pd.read_csv(path_train, header=None, names=['uId', 'age_group'], dtype={'uId': np.str})
    train = train.sort_values(by='uId')

    train['age_group'] = pd.factorize(train['age_group'], sort=True)[0]

    # test.csv 502500
    test = pd.read_csv(path_test, names=['uId'], dtype={'uId': np.str})
    test = test.sort_values(by='uId')

    # User basic info 2512500
    basic = pd.read_csv(path_user_basic_info, header=None, names=['uId', 'gender', 'city', 'prodName', 'ramCapacity',
                                                                  'ramLeftRation', 'romCapacity', 'romLeftRation',
                                                                  'color', 'fontSize', 'ct', 'carrier', 'os'],
                        dtype={'uId': np.str})
    basic.drop_duplicates('uId', keep='first', inplace=True)
    basic = basic.sort_values(by='uId')  # uId按序排好
    # 编码
    basic['gender'] = pd.factorize(basic['gender'], sort=True)[0]
    basic['city'] = pd.factorize(basic['city'], sort=True)[0]
    basic['prodName'] = pd.factorize(basic['prodName'], sort=True)[0]
    basic['ramCapacity'] = pd.factorize(basic['ramCapacity'], sort=True)[0]
    basic['ramLeftRation'] = pd.factorize(basic['ramLeftRation'], sort=True)[0]
    basic['romCapacity'] = pd.factorize(basic['romCapacity'], sort=True)[0]
    basic['romLeftRation'] = pd.factorize(basic['romLeftRation'], sort=True)[0]
    basic['color'] = pd.factorize(basic['color'], sort=True)[0]
    basic['fontSize'] = pd.factorize(basic['fontSize'], sort=True)[0]
    basic['ct'] = pd.factorize(basic['ct'], sort=True)[0]
    basic['carrier'] = pd.factorize(basic['carrier'], sort=True)[0]
    basic['os'] = pd.factorize(basic['os'], sort=True)[0]

    # user_behavior_info 2512500
    behavior = pd.read_csv(path_user_behavior_info, header=None, dtype={'uId': np.str},
                           names=['uId', 'bootTimes', 'AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes',
                                  'EFuncTimes', 'FFuncTimes', 'GFuncSums'])
    behavior = behavior.sort_values(by='uId')

    # 每列归一化
    # 临时取出uId列，然后全局归一化
    temp0 = behavior['uId']
    behavior[['uId']] = behavior[['uId']].astype(int)
    behavior = behavior.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    behavior['uId'] = temp0
    behavior.fillna(-1, inplace=True)

    # user_app_actived.csv 2512500
    user_app_actived = pd.read_csv(user_app_avtived_path, header=None, names=['uId', 'appId'], dtype={'uId': np.str})
    user_app_actived = user_app_actived.sort_values(by='uId')
    # print(user_app_actived.describe())
    # print('user_app_actived:\n', user_app_actived)

    # app_info  188864
    app_info = pd.read_csv(app_info_path, header=None, names=['appId', 'category'])
    app_info = app_info.sort_values(by='appId')
    # print(user_app_actived.describe())
    # print('app_info1:\n', app_info)
    app_info['category'] = pd.factorize(app_info['category'], sort=True)[0]  # 40类
    # print('app_info2:\n', app_info)
    # 得到每个用户激活的某类APP次数
    if os.path.exists('appId_vector.csv'):
    # if os.path.exists('/home/xuwenxiang/xu/HUAWEI/demographic/appId_vector.csv'):
        pass
    else:
        # 改成字典
        app_info_dict = {}
        for appId, category in app_info.itertuples(index=False):
            app_info_dict[appId] = category
        # print('app_info_dict type:\n', type(app_info_dict))

        # appId映射40维的向量
        appId_vector = []
        n_category = app_info['category'].nunique()         # 40,0-39
        for str in user_app_actived['appId']:               # 这个dataframe是按uId排的，所以每行产生的向量按uId有序，遍历此列所有行
            # print(str, '\n', type(str))                   # str类型是字符串
            # print(str.split('#'))                         # 字符串转列表

            one_vector = np.zeros(n_category)               # 初始化一个向量
            str_list = str.split('#')
            for i in str_list:
                # print(i, type(i))
                if i in app_info_dict.keys():
                    # print(app_info_dict[i], type(app_info_dict[i]))
                    app_cat_num = app_info_dict[i]
                    one_vector[app_cat_num] += 1

            # print('one_vector:\n', one_vector)

            appId_vector.append(one_vector)
        appId_vector_data = pd.DataFrame(appId_vector)
        appId_vector_data.to_csv('appId_vector.csv', index=None)
        # print('appId_vector_data shape:\n', appId_vector_data.shape)

    # 读取每个用户激活的某类APP次数
    appId_num_data = pd.read_csv('appId_vector.csv')
    # appId_num_data = pd.read_csv('/home/xuwenxiang/xu/HUAWEI/demographic/appId_vector.csv')
    # user_app_actived里先去掉appId
    user_app_actived.drop('appId', axis=1, inplace=True)
    user_app_actived.reset_index(drop=True, inplace=True)
    # 加上appId_num_data数据
    appId_num_data = pd.concat([user_app_actived, appId_num_data], axis=1)
    # 归一化
    # 临时取出uId列，然后全局归一化
    temp1 = appId_num_data['uId']
    appId_num_data[['uId']] = appId_num_data[['uId']].astype(int)
    appId_num_data = appId_num_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    appId_num_data['uId'] = temp1
    appId_num_data.fillna(-1, inplace=True)

    # train_discrete
    train_discrete = pd.merge(train, basic, how='left', on='uId')
    train_discrete.fillna(-1, inplace=True)
    # print('train_discrete:\n', train_discrete)

    # label
    label = train_discrete.age_group
    # print('label:\n', label)

    train_discrete.drop(['uId', 'age_group'], axis=1, inplace=True)
    train_discrete.fillna(-1, inplace=True)

    # train_continuous
    temp = pd.merge(behavior, appId_num_data, how='left', on='uId')
    train_continuous = pd.merge(train, temp, how='left', on='uId')
    train_continuous.drop(['uId', 'age_group'], axis=1, inplace=True)
    train_continuous.fillna(-1, inplace=True)

    # test_discrete
    test_discrete = pd.merge(test, basic, how='left', on='uId', left_index=True)
    test_discrete.drop_duplicates('uId', keep='first', inplace=True)
    test_discrete.drop('uId', axis=1, inplace=True)
    test_discrete.fillna(-1, inplace=True)

    # test_continuous
    test_continuous = pd.merge(test, temp, how='left', on='uId')
    test_continuous.drop_duplicates('uId', keep='first', inplace=True)
    test_continuous.drop('uId', axis=1, inplace=True)
    test_continuous.fillna(-1, inplace=True)

    return train_discrete, label, train_continuous, test_discrete, test_continuous


def xgb1_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH, path_user_basic_info=USER_BASIC_INFO_PATH):
    basic = pd.read_csv(path_user_basic_info, names=['uId', 'gender', 'city', 'prodName', 'ramCapacity',
                                                     'ramLeftRation', 'romCapacity', 'romLeftRation', 'color',
                                                     'fontSize',
                                                     'ct', 'carrier', 'os'], dtype={'uId': np.str})
    basic.drop_duplicates('uId', keep='first', inplace=True)

    basic['gender'] = pd.factorize(basic['city'], sort=True)[0]
    basic['city'] = pd.factorize(basic['city'], sort=True)[0]
    basic['prodName'] = pd.factorize(basic['prodName'], sort=True)[0]
    basic['ramCapacity'] = pd.factorize(basic['ramCapacity'], sort=True)[0]
    basic['ramLeftRation'] = pd.factorize(basic['ramLeftRation'], sort=True)[0]
    basic['romCapacity'] = pd.factorize(basic['romCapacity'], sort=True)[0]
    basic['romLeftRation'] = pd.factorize(basic['romLeftRation'], sort=True)[0]
    basic['color'] = pd.factorize(basic['color'], sort=True)[0]
    basic['fontSize'] = pd.factorize(basic['fontSize'], sort=True)[0]
    basic['ct'] = pd.factorize(basic['ct'], sort=True)[0]
    basic['carrier'] = pd.factorize(basic['carrier'], sort=True)[0]
    basic['os'] = pd.factorize(basic['os'], sort=True)[0]

    train_loader = pd.read_csv(path_train, names=['uId', 'age_group'], dtype={'uId': np.str})
    train = train_loader
    train['age_group'] = pd.factorize(train['age_group'], sort=True)[0]
    train = pd.merge(train, basic, how='left', on='uId', left_index=True)
    # target
    target = train.age_group

    train.drop(['uId', 'age_group'], axis=1, inplace=True)
    train.fillna(-1, inplace=True)
    # train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)

    test_loader = pd.read_csv(path_test, names=['uId'], dtype={'uId': np.str})
    test = pd.merge(test_loader, basic, how='left', on='uId', left_index=True)
    test.drop('uId', axis=1, inplace=True)
    test.fillna(-1, inplace=True)

    return train, test, target


def xgb2_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH, path_user_basic_info=USER_BASIC_INFO_PATH,
              path_user_behavior_info=USER_BEHAVIOR_INFO_PATH, app_info_path=APP_INFO_PATH,
              user_app_avtived_path=USER_APP_ACTIVED_PATH, user_app_usage_path=USER_APP_USAGE_PATH):
    # User basic info
    basic = pd.read_csv(path_user_basic_info, header=0, names=['uId', 'gender', 'city', 'prodName', 'ramCapacity',
                                                               'ramLeftRation', 'romCapacity', 'romLeftRation', 'color',
                                                               'fontSize', 'ct', 'carrier', 'os'], dtype={'uId': np.str})
    basic.drop_duplicates('uId', keep='first', inplace=True)

    basic['gender'] = pd.factorize(basic['gender'], sort=True)[0]
    basic['city'] = pd.factorize(basic['city'], sort=True)[0]
    basic['prodName'] = pd.factorize(basic['prodName'], sort=True)[0]
    basic['ramCapacity'] = pd.factorize(basic['ramCapacity'], sort=True)[0]
    basic['ramLeftRation'] = pd.factorize(basic['ramLeftRation'], sort=True)[0]
    basic['romCapacity'] = pd.factorize(basic['romCapacity'], sort=True)[0]
    basic['romLeftRation'] = pd.factorize(basic['romLeftRation'], sort=True)[0]
    basic['color'] = pd.factorize(basic['color'], sort=True)[0]
    basic['fontSize'] = pd.factorize(basic['fontSize'], sort=True)[0]
    basic['ct'] = pd.factorize(basic['ct'], sort=True)[0]
    basic['carrier'] = pd.factorize(basic['carrier'], sort=True)[0]
    basic['os'] = pd.factorize(basic['os'], sort=True)[0]

    train_loader = pd.read_csv(path_train, header=0, names=['uId', 'age_group'], dtype={'uId': np.str})
    train = train_loader
    train['age_group'] = pd.factorize(train['age_group'], sort=True)[0]

    # merge
    user_app_actived = pd.read_csv(user_app_avtived_path, header=0, names=['uId', 'appId'], dtype={'uId': np.str})
    user_app_actived['appId'] = pd.factorize(user_app_actived['appId'], sort=True)[0]

    app_info = pd.read_csv(app_info_path, header=0, names=['appId', 'category'])
    app_info['appId'] = pd.factorize(app_info['appId'], sort=True)[0]
    app_info['category'] = pd.factorize(app_info['category'], sort=True)[0]

    app = pd.merge(user_app_actived, app_info, how='left', on='appId')
    add1 = pd.merge(basic, app, how='left', on='uId')
    # train
    train = pd.merge(train, add1, how='left', on='uId')
    train.fillna(-1, inplace=True)

    # target
    target = train.age_group

    train.drop(['uId', 'age_group'], axis=1, inplace=True)
    train.fillna(-1, inplace=True)

    test_loader = pd.read_csv(path_test, names=['uId'], dtype={'uId': np.str})
    test = pd.merge(test_loader, add1, how='left', on='uId', left_index=True)
    test.drop_duplicates('uId', keep='first', inplace=True)
    test.drop('uId', axis=1, inplace=True)
    test.fillna(-1, inplace=True)

    return train, test, target


def Et_Rf_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH, path_user_basic_info=USER_BASIC_INFO_PATH,
              path_user_behavior_info=USER_BEHAVIOR_INFO_PATH, app_info_path=APP_INFO_PATH,
              user_app_avtived_path=USER_APP_ACTIVED_PATH, user_app_usage_path=USER_APP_USAGE_PATH):
    # train.csv 2010000
    train = pd.read_csv(path_train, header=None, names=['uId', 'age_group'], dtype={'uId': np.str})
    train = train.sort_values(by='uId')

    train['age_group'] = pd.factorize(train['age_group'], sort=True)[0]

    # test.csv 502500
    test = pd.read_csv(path_test, names=['uId'], dtype={'uId': np.str})
    test = test.sort_values(by='uId')

    # User basic info 2512500
    basic = pd.read_csv(path_user_basic_info, header=None, names=['uId', 'gender', 'city', 'prodName', 'ramCapacity',
                                                                  'ramLeftRation', 'romCapacity', 'romLeftRation',
                                                                  'color', 'fontSize', 'ct', 'carrier', 'os'],
                        dtype={'uId': np.str})
    basic.drop_duplicates('uId', keep='first', inplace=True)
    basic = basic.sort_values(by='uId')  # uId按序排好
    # 编码
    basic['gender'] = pd.factorize(basic['gender'], sort=True)[0]
    basic['city'] = pd.factorize(basic['city'], sort=True)[0]
    basic['prodName'] = pd.factorize(basic['prodName'], sort=True)[0]
    basic['ramCapacity'] = pd.factorize(basic['ramCapacity'], sort=True)[0]
    basic['ramLeftRation'] = pd.factorize(basic['ramLeftRation'], sort=True)[0]
    basic['romCapacity'] = pd.factorize(basic['romCapacity'], sort=True)[0]
    basic['romLeftRation'] = pd.factorize(basic['romLeftRation'], sort=True)[0]
    basic['color'] = pd.factorize(basic['color'], sort=True)[0]
    basic['fontSize'] = pd.factorize(basic['fontSize'], sort=True)[0]
    basic['ct'] = pd.factorize(basic['ct'], sort=True)[0]
    basic['carrier'] = pd.factorize(basic['carrier'], sort=True)[0]
    basic['os'] = pd.factorize(basic['os'], sort=True)[0]

    # user_behavior_info 2512500
    behavior = pd.read_csv(path_user_behavior_info, header=None, dtype={'uId': np.str},
                           names=['uId', 'bootTimes', 'AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes',
                                  'EFuncTimes', 'FFuncTimes', 'GFuncSums'])
    behavior = behavior.sort_values(by='uId')

    # 每列归一化
    # 临时取出uId列，然后全局归一化
    temp0 = behavior['uId']
    behavior[['uId']] = behavior[['uId']].astype(int)
    behavior = behavior.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    behavior['uId'] = temp0
    behavior.fillna(-1, inplace=True)

    # user_app_actived.csv 2512500
    user_app_actived = pd.read_csv(user_app_avtived_path, header=None, names=['uId', 'appId'], dtype={'uId': np.str})
    user_app_actived = user_app_actived.sort_values(by='uId')
    # print(user_app_actived.describe())
    # print('user_app_actived:\n', user_app_actived)

    # app_info  188864
    app_info = pd.read_csv(app_info_path, header=None, names=['appId', 'category'])
    app_info = app_info.sort_values(by='appId')
    # print(user_app_actived.describe())
    # print('app_info1:\n', app_info)
    app_info['category'] = pd.factorize(app_info['category'], sort=True)[0]  # 40类
    # print('app_info2:\n', app_info)
    # 得到每个用户激活的某类APP次数
    # if os.path.exists('appId_vector.csv'):
    if os.path.exists('/home/xuwenxiang/xu/HUAWEI/demographic/appId_vector.csv'):
        pass
    else:
        # 改成字典
        app_info_dict = {}
        for appId, category in app_info.itertuples(index=False):
            app_info_dict[appId] = category
        # print('app_info_dict type:\n', type(app_info_dict))

        # appId映射40维的向量
        appId_vector = []
        n_category = app_info['category'].nunique()         # 40,0-39
        for str in user_app_actived['appId']:               # 这个dataframe是按uId排的，所以每行产生的向量按uId有序，遍历此列所有行
            # print(str, '\n', type(str))                   # str类型是字符串
            # print(str.split('#'))                         # 字符串转列表

            one_vector = np.zeros(n_category)               # 初始化一个向量
            str_list = str.split('#')
            for i in str_list:
                # print(i, type(i))
                if i in app_info_dict.keys():
                    # print(app_info_dict[i], type(app_info_dict[i]))
                    app_cat_num = app_info_dict[i]
                    one_vector[app_cat_num] += 1

            # print('one_vector:\n', one_vector)

            appId_vector.append(one_vector)
        appId_vector_data = pd.DataFrame(appId_vector)
        appId_vector_data.to_csv('appId_vector.csv', index=None)
        # print('appId_vector_data shape:\n', appId_vector_data.shape)

    # 读取每个用户激活的某类APP次数
    # appId_num_data = pd.read_csv('appId_vector.csv')
    appId_num_data = pd.read_csv('/home/xuwenxiang/xu/HUAWEI/demographic/appId_vector.csv')
    # user_app_actived里先去掉appId
    user_app_actived.drop('appId', axis=1, inplace=True)
    user_app_actived.reset_index(drop=True, inplace=True)
    # 加上appId_num_data数据
    appId_num_data = pd.concat([user_app_actived, appId_num_data], axis=1)
    # 归一化
    # 临时取出uId列，然后全局归一化
    temp1 = appId_num_data['uId']
    appId_num_data[['uId']] = appId_num_data[['uId']].astype(int)
    appId_num_data = appId_num_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    appId_num_data['uId'] = temp1
    appId_num_data.fillna(-1, inplace=True)


    # label
    label = train.age_group
    # print('label:\n', label)


    temp = pd.merge(behavior, appId_num_data, how='left', on='uId')
    temp = pd.merge(basic, temp, how='left', on='uId')
    train_ = pd.merge(train, temp, how='left', on='uId')
    train_.drop(['uId', 'age_group'], axis=1, inplace=True)
    train_.fillna(-1, inplace=True)


    test_ = pd.merge(test, temp, how='left', on='uId')
    test_.drop_duplicates('uId', keep='first', inplace=True)
    test_.drop('uId', axis=1, inplace=True)
    test_.fillna(-1, inplace=True)

    return train_, label, test_
