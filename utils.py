import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair


# print(data.shape)
# for index, row in data.iteritems():
#     if (index != 'ID号') and (index != 'dlt_死亡终点'):
#         t = (data[index] - data[index].min()) / (data[index].max() - data[index].min())
#         data[index] = t
# # data.to_csv('2.csv', header=True, index=False, encoding='utf_8_sig')


# 独热编码
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def split_dataset(idx, r1=0.6, r2=0.2):
    np.random.seed(1234)
    shuffled_index = np.random.permutation(len(idx))
    split_index1 = int(len(idx) * r1)
    split_index2 = int(len(idx) * (r1 + r2))
    train_index = shuffled_index[:split_index1]
    val_index = shuffled_index[split_index1:split_index2]
    test_index = shuffled_index[split_index2:]
    return train_index, val_index, test_index


def gen_edges(mx):
    # print(mx)
    edge = []
    edges = []
    for index, row in mx.iterrows():
        if row[1] == '是':
            edge.append(index)
    for index, row in mx.iterrows():
        if row[1] == '短期复发':
            edge.append(index)
    for index, row in mx.iterrows():
        if row[1] == '长期复发':
            edge.append(index)
    # print(edge)
    # print(len(edge))
    for i in range(len(edge)):
        for j in range(len(edge)):
            if i < j:
                edges.append([edge[i], edge[j]])

    # edge = []
    # for index, row in mx.iterrows():
    #     if row[1] == 0:
    #         edge.append(index)
    # for i in range(len(edge)):
    #     for j in range(len(edge)):
    #         if i < j:
    #             edges.append([edge[i], edge[j]])
    # print(len(edge))

    return torch.tensor(edge), torch.tensor(edges)

def load_edges():
    # print('加载图结构')
    feature_edges = np.genfromtxt('./database/graph/c7.txt', dtype=np.int32)
    edges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    return edges
def load_edges2():
    # print('加载图结构')
    feature_edges = np.genfromtxt('./database/graph/c10.txt', dtype=np.int32)
    edges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    return edges


def load_graph(dataset, config):
    print('加载图结构')
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'

    # 加载边缘并构成图
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    # 加载原始图
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj, nfadj


def load_data_ml(path="./database/", dataset="1"):
    print('Loading {} dataset...'.format(dataset))
    data = pd.read_csv('{}{}.csv'.format(path, dataset), encoding='utf-8', low_memory=False)
    x = np.array(data.iloc[:, 1:-1]).astype('float32')
    # x = np.array(data.drop(['dt_吸烟'], axis=1).iloc[:, 1:-1]).astype('float32')
    # x = Normalization(x, 0)
    # x = normalize(x)
    y = data.iloc[:, -1]
    # y = encode_onehot(data.iloc[:, -1])
    # y = torch.LongTensor(np.where(y)[1])
    return x, y


def load_data2(path="./database/", dataset="1"):
    # initgraph()
    print('Loading {} dataset...'.format(dataset))
    data = pd.read_csv('{}{}.csv'.format(path, dataset), encoding='utf-8', low_memory=False)
    # data = data.loc[(data['dlt_死亡终点'] == '是') | (data['dlt_死亡终点'] == '短期复发') | (data['dlt_死亡终点'] == '长期复发')]
    # data = data.reset_index(drop=True)

    from imblearn.over_sampling import SMOTE
    from collections import Counter
    x = np.array(data.iloc[:, 1:-1]).astype('float32')
    y = data.iloc[:, -1]
    x_smo, y_smo = SMOTE().fit_resample(x, y)
    print(x_smo.shape)
    print(Counter(y_smo))
    """ 特征处理 """
    # features = np.array(data.drop(['dt_吸烟'], axis=1).iloc[:, 1:-1]).astype('float32')
    # features = np.array(data.iloc[:, 1:-1]).astype('float32')
    # features = np.loadtxt('./database/' + '1.feature', dtype=float)
    # labels = np.loadtxt('./database/' + '1.label', dtype=float)
    features = Normalization(x_smo, 1)
    features = sp.csr_matrix(torch.Tensor(features), dtype=np.float32)

    """ 标签处理 """
    # data2 = data.copy()
    # if 'dlt_死亡终点' in data2.columns:
    #     data2.loc[data2['dlt_死亡终点'] == '短期复发', 'dlt_死亡终点'] = '是'
    #     data2.loc[data2['dlt_死亡终点'] == '长期复发', 'dlt_死亡终点'] = '是'
    # labels = data.iloc[:, -1]
    # labels = encode_onehot(labels)
    # labels = encode_onehot(data.iloc[:, -1])
    labels = encode_onehot(y_smo)

    """ id处理 （0~4564） """
    idx = np.array(data.iloc[:, 0])
    idx_map = {j: i for i, j in enumerate(idx)}
    idx = [i for i, j in enumerate(idx)]
    idx = torch.tensor(idx)

    """ 邻接矩阵处理 """
    # edge, edges = gen_edges(data.iloc[:, [0, -1]])
    # edge, edges = gen_edges(data.iloc[:, [0, 6]])
    # edges = np.array(edges)
    edges = load_edges()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    """ 标准化，矩阵对称 """
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    # GCN公式，加上单位矩阵，在标准化； eye构造单位矩阵
    adj = normalize(adj + sp.eye(adj.shape[0]))

    """ 训练集验证集测试集划分 """
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    idx_train, idx_val, idx_test = split_dataset(x_smo, 0.6, 0.2)
    # np.random.seed(1234)
    # edge = np.array(edge).tolist()
    # edge = np.random.permutation(len(edge))
    # edge = torch.LongTensor(edge)

    features = torch.FloatTensor(np.array(features.todense()))
    # 原来标签--》独热编码（1 0）--》np.where(labels)生成行和列索引
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print('labels:', labels)
    print('features', features)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_gcn(path="./database/", dataset="1"):
    # initgraph()
    print('Loading {} dataset...'.format(dataset))
    data = pd.read_csv('{}{}.csv'.format(path, dataset), encoding='utf-8', low_memory=False)
    # data = data.loc[(data['dlt_死亡终点'] == '是') | (data['dlt_死亡终点'] == '短期复发') | (data['dlt_死亡终点'] == '长期复发')]
    # data = data.reset_index(drop=True)
    # if 'dlt_死亡终点' in data.columns:
    #     data.loc[data['dlt_死亡终点'] == '短期复发', 'dlt_死亡终点'] = '是'
    #     data.loc[data['dlt_死亡终点'] == '长期复发', 'dlt_死亡终点'] = '是'
    """ 特征处理 """
    # features = np.array(data.drop(['dt_吸烟'], axis=1).iloc[:, 1:-1]).astype('float32')
    features = np.array(data.iloc[:, 1:-1]).astype('float32')
    features = Normalization(features, 1)
    features = sp.csr_matrix(torch.Tensor(features), dtype=np.float32)

    """ 标签处理 """
    # labels = data.iloc[:, -1]
    # labels = encode_onehot(labels)
    labels = encode_onehot(data.iloc[:, -1])

    """ id处理 （0~4564） """
    idx = np.array(data.iloc[:, 0])
    idx_map = {j: i for i, j in enumerate(idx)}
    idx = [i for i, j in enumerate(idx)]
    idx = torch.tensor(idx)

    """ 邻接矩阵处理 """
    # edge, edges = gen_edges(data.iloc[:, [0, -1]])
    # edge, edges = gen_edges(data.iloc[:, [0, 6]])
    # edges = np.array(edges)
    edges = load_edges2()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    """ 标准化，矩阵对称 """
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    # GCN公式，加上单位矩阵，在标准化； eye构造单位矩阵
    adj = normalize(adj + sp.eye(adj.shape[0]))

    """ 训练集验证集测试集划分 """
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    # idx_train, idx_val, idx_test = split_dataset(data, 0.6, 0.2)
    # np.random.seed(1234)
    # edge = np.array(edge).tolist()
    # edge = np.random.permutation(len(edge))
    # edge = torch.LongTensor(edge)

    features = torch.FloatTensor(np.array(features.todense()))
    # 原来标签--》独热编码（1 0）--》np.where(labels)生成行和列索引
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_train = torch.cat([idx_train, edge[:40]], dim=0).numpy()
    # idx_train = list(set(idx_train))
    # idx_train = torch.LongTensor(idx_train)
    # print(idx_train.shape)

    idx_val = torch.LongTensor(idx_val)
    # idx_val = torch.cat([idx_val, edge[40:80]], dim=0).numpy()
    # idx_val = list(set(idx_val))
    # idx_val = torch.LongTensor(idx_val)

    idx_test = torch.LongTensor(idx_test)
    # idx_test = torch.cat([idx_test, edge[99:]], dim=0).numpy()
    # idx_test = list(set(idx_test))
    # idx_test = torch.LongTensor(idx_test)
    print('labels:', labels)
    print('features', features)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data(path="./cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# 矩阵标准化
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj) # 采用三元组(row, col, data)的形式存储稀疏邻接矩阵
    rowsum = np.array(adj.sum(1)) # 按行求和得到rowsum, 即每个节点的度
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # (行和rowsum)^(-1/2)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # isinf部分赋值为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 对角化; 将d_inv_sqrt 赋值到对角线元素上, 得到度矩阵^-1/2
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)


def Normalization(x, k):
    xn = []
    if k == 0:
        """[0,1] normaliaztion"""
        for i in range(x.shape[1]):
            t = x[:, i]
            norm = (t - np.min(t)) / (np.max(t) - np.min(t))
            xn.append(norm)
        # x = (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        """Z-score normaliaztion"""
        for i in range(x.shape[1]):
            norm = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])
            xn.append(norm)
        # x = (x - np.mean(x)) / np.std(x)
    x = np.around(np.array(xn), 6).astype('float')
    return x.T


# 求精确度
def accuracy(output, labels, test=False):
    # preds是所预测的标签，维度（n * c）
    preds = output.max(1)[1].type_as(labels)

    if test:
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(labels.data.cpu().numpy(), preds.data.cpu().numpy()))
        print(confusion_matrix(labels.data.cpu().numpy(), preds.data.cpu().numpy(), labels=[0, 1]))

    # 计算准确率（对的次数/总次数）
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# 稀疏矩阵转为tensor（3种值）
# indices：2*（4564,4564）所有Aij索引， values：邻接矩阵的值，shape：（4564,4564）
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def construct_graph(dataset, features, topk):
    fname = './database/' + '/graph/tmp.txt'
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(dataset):
    for topk in range(2, 10):
        data = np.loadtxt('./database/' + dataset + '.feature', dtype=float)
        # print(data)
        construct_graph(dataset, data, topk)
        f1 = open('./database/' + '/graph/tmp.txt', 'r')
        f2 = open('./database/' + '/graph/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()


def initgraph():
    print('init graph~')
    dataset = '1'
    data = pd.read_csv('{}{}.csv'.format("./database/", dataset), encoding='utf-8', low_memory=False)
    from imblearn.over_sampling import SMOTE
    x = np.array(data.iloc[:, 1:-1]).astype('float32')
    y = data.iloc[:, -1]
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    y = label_encoder.transform(y)
    features, labels = SMOTE().fit_resample(x, y)
    # data = data.loc[(data['dlt_死亡终点'] == '是') | (data['dlt_死亡终点'] == '短期复发') | (data['dlt_死亡终点'] == '长期复发')]
    # data = data.reset_index(drop=True)
    # features = np.array(data.iloc[:, 1:-1])
    # features = normalize(features)
    features = Normalization(features, 0)
    # labels = data.iloc[:, -1]
    if 'dlt_死亡终点' in data.columns:
        data.loc[data['dlt_死亡终点'] == '是', 'dlt_死亡终点'] = 1
        data.loc[data['dlt_死亡终点'] == '否', 'dlt_死亡终点'] = 0
        data.loc[data['dlt_死亡终点'] == '短期复发', 'dlt_死亡终点'] = 2
        data.loc[data['dlt_死亡终点'] == '长期复发', 'dlt_死亡终点'] = 3
    np.savetxt('./database/{}.label'.format(dataset), np.array(labels), fmt='%d')
    np.savetxt('./database/{}.feature'.format(dataset), features, fmt='%f')

    generate_knn(dataset)

if __name__ == '__main__':
    print('begin')
    # load_data2()
    initgraph()

    # load_edges()
