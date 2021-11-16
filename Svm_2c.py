import numpy as np
import pylab as pl
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


import matplotlib.pyplot as plt
import matplotlib

from utils import load_data_ml,load_data_ml2

import warnings
warnings.filterwarnings("ignore")


def test():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    clf.predict([[2., 2.]])

    print(clf)
    # 支持向量
    print(clf.support_vectors_)
    # 获得支持向量的索引
    print(clf.support_)
    # 为每一个类别获得支持向量的数量
    print(clf.n_support_)
    print(clf.score(X, y))
    # 预测
    print(clf.predict([[2., 2.]]))

    print('-' * 30)


# 创建 40 个点
np.random.seed(0) # 让每次运行程序生成的随机样本点不变
# 生成训练实例并保证是线性可分的
# np._r表示将矩阵在行方向上进行相连
# random.randn(a,b)表示生成 a 行 b 列的矩阵，且随机数服从标准正态分布
# array(20,2) - [2,2] 相当于给每一行的两个数都减去 2

''' 1.读取数据集 '''
x, y, x1, y1, x0, y0 = load_data_ml()
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
y = label_encoder.transform(y)

label_encoder1 = LabelEncoder()
label_encoder1 = label_encoder1.fit(y1)
y1 = label_encoder1.transform(y1)

label_encoder2 = LabelEncoder()
label_encoder2 = label_encoder2.fit(y0)
y0 = label_encoder2.transform(y0)

x_train2, x_test2, y_train2, y_test2 = train_test_split(
        x, y, test_size=0.3, random_state=0)
print(x)
print(y)
from imblearn.over_sampling import SMOTE
from collections import Counter
# smo = SMOTE(sampling_strategy={0: 700, 1:200, 2:150 }, random_state=42)
# X_smo, y_smo = smo.fit(x, y)
x_smo, y_smo = SMOTE().fit_resample(x, y)
print(Counter(y_smo))
x_smo1, y_smo1 = SMOTE().fit_resample(x1, y1)
print(Counter(y_smo1))


''' 2.划分数据与标签 '''
x_train, x_test, y_train, y_test = train_test_split(
        x_smo, y_smo, test_size=0.3, random_state=0)
x_train1, x_test1, y_train1, y_test1 = train_test_split(
        x_smo1, y_smo1, test_size=0.3, random_state=0)

x_train0, x_test0, y_train0, y_test0 = train_test_split(
        x0, y0, test_size=0.3, random_state=0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)


''' 3.训练SVM分类器 '''
# clf = svm.SVC()
# clf2 = svm.SVC()
# clf = svm.SVC(C=1000, kernel='rbf', gamma='auto')
# clf = svm.SVC(C=1000, kernel='rbf', gamma='auto', decision_function_shape='ovo', probability=True, class_weight='balanced')
# clf.fit(x_train, y_train)

clf = XGBClassifier()               # 载入模型（模型命名为model)
clf2 = XGBClassifier()
# clf = XGBClassifier(n_estimators=2000, max_depth=40, random_state=27, scale_pos_weight=50)               # 载入模型（模型命名为model)


# import joblib
# joblib.dump(clf, 'svm_model.m')
# clf = joblib.load("svm_model.m")

# clf.fit(x_train, y_train)
clf.fit(x_train, y_train)
clf2.fit(x_train1, y_train1)


""""""
x3,y3, x4,y4,x5,y5 = load_data_ml2()
label_encoder = label_encoder.fit(y3)
y3 = label_encoder.transform(y3)
label_encoder = label_encoder.fit(y4)
y4 = label_encoder.transform(y4)
label_encoder = label_encoder.fit(y5)
y5 = label_encoder.transform(y5)
x_smo4, y_smo4 = SMOTE().fit_resample(x4, y4)
print(Counter(y_smo4))
x_smo5, y_smo5 = SMOTE().fit_resample(x5, y5)
print(Counter(y_smo5))
x_train3, x_test3, y_train3, y_test3 = train_test_split(
        x3, y3, test_size=0.3, random_state=0)
x_train4, x_test4, y_train4, y_test4 = train_test_split(
        x_smo4, y_smo4, test_size=0.3, random_state=0)
x_train5, x_test5, y_train5, y_test5 = train_test_split(
        x_smo5, y_smo5, test_size=0.3, random_state=0)

clf4 = XGBClassifier()
clf5 = XGBClassifier()
clf4.fit(x_train4, y_train4)
clf5.fit(x_train5, y_train5)



y_pred = clf.predict(x_test0)
# print(y_test4)
# print(y_pred)
for i, key in enumerate(y_pred):
    if key == 1:
        # print(key)
        pred1 = clf2.predict(x_test0[i].reshape(1, -1))+1
        # pred1 = clf4.predict(x_test0[i].reshape(1, -1))+1
        y_pred[i] = pred1
        # if pred1 == 2:
        #     pred2 = clf5.predict(x_test0[i].reshape(1, -1))+2
        #     y_pred[i] = pred2

print(accuracy_score(y_test0, y_pred))
print(classification_report(y_test0, y_pred))

# from xgboost import plot_importance
# fig, ax = plt.subplots(figsize=(15, 15))
# plot_importance(clf, height=0.5, ax=ax, max_num_features=64)
# plt.show()

''' 4.计算svc分类器的准确率 '''
scoring = ['precision_macro', 'recall_macro']
print("训练集score：", clf.score(x_train, y_train))
print("测试集score：", clf.score(x_test, y_test))



"""
def load_data_ml(path="./database/", dataset="1"):
    print('Loading {} dataset...'.format(dataset))
    data = pd.read_csv('{}{}.csv'.format(path, dataset), encoding='utf-8', low_memory=False)
    x0 = np.array(data.iloc[:, 1:-1]).astype('float32')
    y0 = data.iloc[:, -1]
    data2 = data.copy()
    if 'dlt_死亡终点' in data2.columns:
        data2.loc[data2['dlt_死亡终点'] == '短期复发', 'dlt_死亡终点'] = '是'
        data2.loc[data2['dlt_死亡终点'] == '长期复发', 'dlt_死亡终点'] = '是'
    data3 = data.copy()
    data3 = data3.loc[(data['dlt_死亡终点'] == '是') | (data['dlt_死亡终点'] == '短期复发') | (data['dlt_死亡终点'] == '长期复发')]
    # if 'dlt_死亡终点' in data3.columns:
    #     data3.loc[data3['dlt_死亡终点'] == '是', 'dlt_死亡终点'] = 1
    #     data3.loc[data3['dlt_死亡终点'] == '短期复发', 'dlt_死亡终点'] = 2
    #     data3.loc[data3['dlt_死亡终点'] == '长期复发', 'dlt_死亡终点'] = 3
    x = np.array(data2.iloc[:, 1:-1]).astype('float32')
    # x = np.array(data.drop(['dt_吸烟'], axis=1).iloc[:, 1:-1]).astype('float32')
    # features = np.array(data.iloc[:, 1:-1])
    # features = Normalization(features, 0)
    y = data2.iloc[:, -1]

    x1 = np.array(data3.iloc[:, 1:-1]).astype('float32')
    y1 = data3.iloc[:, -1]


    # x = Normalization(x, 0)
    # x = normalize(x)
    # y = data.iloc[:, -1]
    # y = encode_onehot(data.iloc[:, -1])
    # y = torch.LongTensor(np.where(y)[1])
    return x, y, x1, y1, x0, y0


def load_data_ml2(path="./database/", dataset="1"):
    print('Loading {} dataset...'.format(dataset))
    data = pd.read_csv('{}{}.csv'.format(path, dataset), encoding='utf-8', low_memory=False)
    data = data.loc[(data['dlt_死亡终点'] == '是') | (data['dlt_死亡终点'] == '短期复发') | (data['dlt_死亡终点'] == '长期复发')]
    x = np.array(data.iloc[:, 1:-1]).astype('float32')
    y = data.iloc[:, -1]

    data2 = data.copy()
    if 'dlt_死亡终点' in data2.columns:
        data2.loc[data2['dlt_死亡终点'] == '是', 'dlt_死亡终点'] = 1
        data2.loc[data2['dlt_死亡终点'] == '短期复发', 'dlt_死亡终点'] = 2
        data2.loc[data2['dlt_死亡终点'] == '长期复发', 'dlt_死亡终点'] = 2
    x1 = np.array(data2.iloc[:, 1:-1]).astype('float32')
    y1 = data2.iloc[:, -1]

    data3 = data.copy()
    data3 = data3.loc[(data['dlt_死亡终点'] == '短期复发') | (data['dlt_死亡终点'] == '长期复发')]
    if 'dlt_死亡终点' in data3.columns:
        data3.loc[data3['dlt_死亡终点'] == '短期复发', 'dlt_死亡终点'] = 2
        data3.loc[data3['dlt_死亡终点'] == '长期复发', 'dlt_死亡终点'] = 3
    x2 = np.array(data3.iloc[:, 1:-1]).astype('float32')
    y2 = data3.iloc[:, -1]


    return x, y, x1, y1, x2, y2


"""



