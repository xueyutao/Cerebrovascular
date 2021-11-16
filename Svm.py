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

from utils import load_data_ml

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
x, y = load_data_ml()
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
y = label_encoder.transform(y)
print(x)
print(y)

from imblearn.over_sampling import SMOTE
from collections import Counter
# smo = SMOTE(sampling_strategy={0: 700, 1:200, 2:150 }, random_state=42)
# X_smo, y_smo = smo.fit(x, y)
x_smo, y_smo = SMOTE().fit_resample(x, y)
print(Counter(y_smo))


''' 2.划分数据与标签 '''
x_train, x_test, y_train, y_test = train_test_split(
        x_smo, y_smo, test_size=0.3, random_state=0)
x_train0, x_test0, y_train0, y_test0 = train_test_split(
        x, y, test_size=0.3, random_state=0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)


''' 3.训练SVM分类器 '''
# clf = svm.SVC()
# clf = svm.SVC(C=1000, kernel='rbf', gamma='auto', decision_function_shape='ovo', probability=True, class_weight='balanced')

clf = XGBClassifier()               # 载入模型（模型命名为model)
# clf = XGBClassifier(n_estimators=2000, max_depth=40, random_state=27, scale_pos_weight=50)               # 载入模型（模型命名为model)


# import joblib
# joblib.dump(clf, 'svm_model.m')
# clf = joblib.load("svm_model.m")

# clf.fit(x_train, y_train)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test0)

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
scores = cross_val_score(clf, x_train, y_train, cv=5)
print("训练集在交叉验证的指标：", "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)
scores = cross_val_score(clf, x_test, y_test, cv=5)
print("测试集在交叉验证的指标：", "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)
scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_micro')
print("训练集在F1_micro的指标：", "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)
scores = cross_val_score(clf, x_test, y_test, cv=5, scoring='f1_micro')
print("测试集在F1_micro的指标：", "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)
scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_macro')
print("训练集在F1_macro的指标：", "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)
scores = cross_val_score(clf, x_test, y_test, cv=5, scoring='f1_macro')
print("测试集在F1_macro的指标：", "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)


scores = cross_val_score(clf, x_train, y_train, scoring='neg_log_loss')
print("训练集的neg_log_loss：", scores)
scores = cross_val_score(clf, x_test, y_test, scoring='neg_log_loss')
print("测试集的neg_log_loss：", scores)

scores = cross_val_score(clf, x_train, y_train, scoring='roc_auc')
print("训练集的ROC：", "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)
scores = cross_val_score(clf, x_test, y_test, scoring='roc_auc')
print("测试集的ROC：", "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)

scores = cross_val_score(clf, x_train, y_train, scoring='neg_mean_squared_error')
print("训练集的neg_log_loss：", scores)
scores = cross_val_score(clf, x_test, y_test, scoring='neg_mean_squared_error')
print("测试集的neg_log_loss：", scores)
# scores = cross_validate(clf, x_train, y_train, scoring=scoring,
#                         cv=5, return_train_score=False)
# print("训练集的指标：", scores)
# scores = cross_validate(clf, x_test, y_test, scoring=scoring,
#                         cv=5, return_train_score=False)
# print("测试集的指标：", scores)

# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 查看决策函数
# print('train_decision_function:', clf.decision_function(x_train))
# print('predict_result:', clf.predict(x_train))

print('-'*30)
y_train_pred = []
y_test_pred = []
for i in range(len(x_train)):
    y_pred = clf.predict(np.array(x_train[i]).reshape(1, -1))
    y_train_pred.append(y_pred[0])
for i in range(len(x_test)):
    y_pred = clf.predict(np.array(x_test[i]).reshape(1, -1))
    y_test_pred.append(y_pred[0])

from sklearn.metrics import mean_squared_error
print('训练集MSE', mean_squared_error(y_train, y_train_pred))
print('测试集MSE', mean_squared_error(y_test, y_test_pred))

# 与scores等价
# print("训练集acc：", accuracy_score(y_train, y_train_pred))
# print("测试集acc：", accuracy_score(y_test, y_test_pred))

# ''' 5.绘制图形 '''
# # 确定坐标轴范围
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
# x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
# grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# # 指定默认字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# # 设置颜色
# cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
# cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
#
# grid_hat = clf.predict(grid_test)  # 预测分类值
# grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
#
#
#
# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)     # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y[:,0], s=30, cmap=cm_dark)  # 样本
# plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test[:, 0], s=30, edgecolors='k', zorder=2, cmap=cm_dark) #圈中测试集样本点
# plt.xlabel('花萼长度', fontsize=13)
# plt.ylabel('花萼宽度', fontsize=13)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('鸢尾花SVM二特征分类')
# plt.show()


"""



