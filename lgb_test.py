# 导入数据包
import pandas as pd
import lightgbm as lgb
import warnings
import matplotlib.pylab as plt
warnings.filterwarnings('ignore')

# 基础配置信息
path = 'files/'
n_splits = 5
seed = 42

# lgb 参数
params={
    "learning_rate":0.1,
    "lambda_l1":0.0,
    "lambda_l2":0.0,
    "max_depth":8,
    "num_leaves":64,
    "objective":"multiclass",
    "num_class":15,
    "silent":True,
}

# 读取数据
train = pd.read_csv(path + 'train_feature.csv')
test = pd.read_csv(path + 'test_feature.csv')

'''
简单分析数据：
user_id 为编码后的数据，大小：
train data shape (612652, 27)
train data of user_id shape 612652
简单的1个用户1条样本的题目,标签的范围 current_service
'''
print('标签',set(train.columns)-set(test.columns))

print('train data shape',train.shape)
print('train data of user_id shape',len(set(train['user_id'])))
print('train data of current_service shape',(set(train['current_service'])))

print('train data shape',test.shape)
print('train data of user_id shape',len(set(test['user_id'])))

# 对标签编码 映射关系
label2current_service = dict(zip(range(0,len(set(train['current_service']))),sorted(list(set(train['current_service'])))))
current_service2label = dict(zip(sorted(list(set(train['current_service']))),range(0,len(set(train['current_service'])))))

# 原始数据的标签映射
train['current_service'] = train['current_service'].map(current_service2label)

# 构造原始数据
y = train.pop('current_service')
train_id = train.pop('user_id')
# 这个字段有点问题
X = train
train_col = train.columns

X_test = test[train_col]
test_id = test['user_id']

# 数据有问题数据
for i in train_col:
    X[i] = X[i].replace("\\N",-1)
    X_test[i] = X_test[i].replace("\\N",-1)

X,y,X_test = X.values,y,X_test.values
weight_dict={6:0.5,3:0.7,4:0.9,13:1,0:1,9:1,7:1,5:1,12:1,11:1,8:1,14:1,10:1,1:1,2:1}
# 采取k折模型方案
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True


X_train,X_valid,Y_train,Y_valid=train_test_split(X,y,test_size=0.2,random_state=1)
Y_train_weight=Y_train.map(weight_dict)
Y_valid_weight=Y_valid.map(weight_dict)

train_data = lgb.Dataset(X_train, label=Y_train,weight=Y_train_weight)
validation_data = lgb.Dataset(X_valid, label=Y_valid)

clf=lgb.train(params,train_data,num_boost_round=200,valid_sets=[validation_data],feval=f1_score_vali,verbose_eval=1)
# plt.figure(figsize=(12,6))
# lgb.plot_importance(clf, max_num_features=30)
# plt.title("Featurertances")
# plt.show()

print(pd.DataFrame({
        'column': train_col,
        'importance': clf.feature_importance(),
    }).sort_values(by='importance'))

# xx_pred = clf.predict(X_test,num_iteration=-1)
#
# xx_pred = [np.argmax(x) for x in xx_pred]
# xx_pred=np.array(xx_pred).reshape(-1,1)
#
# # 保存结果
# df_test = pd.DataFrame()
# df_test['id'] = list(test_id.unique())
# df_test['predict'] = xx_pred
# df_test['predict'] = df_test['predict'].map(label2current_service)
#
# df_test.to_csv('files/baseline2.csv',index=False)
