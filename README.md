# -
针对给定数据，对手机用户的年龄段进行预测，是一个六分类问题，评判标准是准确率

https://developer.huawei.com/consumer/cn/activity/devStarAI/algo/competition.html#/preliminary/info/digix-trail-02/introduction

baseline：0.401789

排行榜最高：0.66

模型介绍：

由于数据量比较大，神经网络模型不采取交叉验证。

有一个20多G的用户使用APP的行为数据太大了，无法处理，baseline分数较低

取部分特征分别训练nn1,nn2,xgb1,xgb2,et,rf共六个模型，生成训练数据和测试数据，

将6组数据拼接，在用xgboost模型训练，这就是stacking方法。
