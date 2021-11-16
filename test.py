import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE



# 矩阵标准化
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


path = './cora/'
dataset = 'cora'
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
# features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
# features = normalize(features)
# features = torch.FloatTensor(np.array(features.todense()))

# features1 = np.array(idx_features_labels[:, 1:-1]).astype('float')
# x = []
# for i in range(features1.shape[1]):
#     temp = (features1[:, i] - np.min(features1[:, i])) / (np.max(features1[:, i]) - np.min(features1[:, i]))
#     x.append(temp)
#

# min_max_scaler = preprocessing.MinMaxScaler()
# x = min_max_scaler.fit_transform(features1)

# for i in range(features1.shape[1]):
#     temp = (features1[:, i] - np.mean(features1[:, i])) / np.std(features1[:, i])
#     x.append(temp)


# x = torch.Tensor(x).T
# print(features1[:, 12])
# print(x.shape)
# print(features)
# pds = pd.DataFrame(idx_features_labels)

# pds.to_csv('./cites.csv', header=True, index=False, encoding='utf_8_sig')


np.random.seed(1234)
rate1 = 0.6
rate2 = 0.2
shuffled_index = np.random.permutation(len(idx))
# print(shuffled_index)
split_index1 = int(len(idx)*rate1)
split_index2 = int(len(idx)*(rate1+rate2))
train_index = shuffled_index[:split_index1]
val_index = shuffled_index[split_index1:split_index2]
test_index = shuffled_index[split_index2:]
print(train_index)
print(val_index)
print(test_index)
print('ok')
